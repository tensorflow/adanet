"""An AdaNet iteration implementation in Tensorflow using a single graph.

Copyright 2018 The AdaNet Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import json
import os

from absl import logging
from adanet import distributed
from adanet import subnetwork
from adanet import tf_compat
from adanet.core.ensemble_builder import _EnsembleSpec
from adanet.core.eval_metrics import _IterationMetrics
from adanet.core.eval_metrics import call_eval_metrics

import numpy as np
import tensorflow as tf


class _TrainManager(object):
  """Manages the training of SubnetworkSpecs and EnsembleSpecs.

  This object maintains a dictionary of states for each SubnetworkSpec and
  EnsembleSpec to coordinate and manage training. Users can check the
  training status of a spec, or request that it stops training.

  It also persists metadata about specs to disk in order to be consistent across
  runs and robust to preemptions.
  """

  def __init__(self, subnetwork_specs, ensemble_specs, train_manager_dir,
               is_chief):
    """Initializes a _TrainManager instance.

    Args:
      subnetwork_specs: List of `_SubnetworkSpec` instances to monitor.
      ensemble_specs: List of `EstimatorSpec` instances to monitor.
      train_manager_dir: Directory for storing metadata about training. When a
        spec should no longer be trained, a JSON file with its name and metadata
        is written to this directory, to persist across runs and preemptions.
      is_chief: Boolean whether the current worker is a chief.
    """

    if not tf.io.gfile.exists(train_manager_dir):
      tf.io.gfile.makedirs(train_manager_dir)
    self._train_manager_dir = train_manager_dir

    self._is_training = {
        spec.name: not self._is_done_training(spec)
        for spec in subnetwork_specs + ensemble_specs
    }

    self._is_chief = is_chief

  def should_train(self, spec):
    """Whether the given spec should keep training."""

    return self._is_training[spec.name]

  def _is_done_training(self, spec):
    """If the file exists, then the candidate is done training."""

    return tf.io.gfile.exists(self._filename_for(spec))

  def _filename_for(self, spec):
    """Returns the filename to identify the spec."""

    return os.path.join(self._train_manager_dir, "{}.json".format(spec.name))

  def request_stop(self, spec, message):
    """Registers that given spec should no longer train."""

    self._is_training[spec.name] = False

    # Only write to disk if chief worker, otherwise there is a risk of conflicts
    # and race conditions during writes.
    if self._is_chief and not self._is_done_training(spec):
      with tf.io.gfile.GFile(self._filename_for(spec), "w") as record_file:
        # TODO: Consider making these messages be some kind of Enum.
        # There # might be a case where we want to parse these files. For
        # example, in iteration n+1, maybe we no longer even want to build
        # NaN candidates.
        message = {"message": message}
        record_file.write(json.dumps(message))

  def is_over(self):
    """Whether all specs are done training and the iteration is over."""

    for k in sorted(self._is_training):
      if self._is_training[k]:
        # Still needs to train.
        return False
    return True


class _NanLossHook(tf_compat.SessionRunHook):
  """Monitors a spec's loss tensor and stops its training if loss is NaN."""

  def __init__(self, train_manager, spec):
    """Initializes a `NanTensorHook`.

    Args:
      train_manager: The current iteration's `_TrainManager`.
      spec: Either a `SubnetworkSpec` or `EnsembleSpec` to monitor.
    """

    self._train_manager = train_manager
    self._spec = spec

  def before_run(self, run_context):
    del run_context  # Unused
    if self._train_manager.should_train(self._spec):
      return tf_compat.SessionRunArgs(self._spec.loss)

  def after_run(self, run_context, run_values):
    loss = run_values.results
    if loss is None or not np.isnan(loss):
      return
    logging.warning("'%s' diverged with loss = NaN.", self._spec.name)
    raise tf_compat.v1.train.NanLossDuringTrainingError
    # TODO: Re-enable once we know that evaluation won't
    # fail from NaNs.
    # self._train_manager.request_stop(self._spec, "NaN loss during training.")


class _TrainingLimitHook(tf_compat.SessionRunHook):
  """Limits a given spec's training to a maximum number of steps.

  Is also responsible for incrementing the spec's step.
  """

  def __init__(self, train_manager, spec, max_steps, increment_step_op):
    """Initializes a _TrainingLimitHook instance.

    Args:
      train_manager: The current iteration's `_TrainManager`.
      spec: Either a `SubnetworkSpec` or `EnsembleSpec` to monitor.
      max_steps: Maximum number steps to train the given spec.
      increment_step_op: That increments the current step and executes one train
        op run.
    """

    self._train_manager = train_manager
    self._spec = spec
    self._max_steps = max_steps
    self._increment_step_op = increment_step_op

  def after_create_session(self, session, coord):
    if not self._train_manager.should_train(self._spec):
      return
    if self._spec.step is None:
      # None for dummy candidates used during round-robin placement.
      self._train_manager.request_stop(self._spec, "Dummy candidate to ignore.")
      return
    step_value = session.run(self._spec.step)
    if self._should_stop(step_value):
      logging.info("Skipping '%s' training which already trained %d steps",
                   self._spec.name, step_value)
      self._train_manager.request_stop(self._spec, "Training already complete.")

  def before_run(self, run_context):
    del run_context  # Unused
    if not self._train_manager.should_train(self._spec):
      return None
    if self._increment_step_op is None:
      # None on TPU.
      return tf_compat.SessionRunArgs(self._spec.step)
    return tf_compat.SessionRunArgs(self._increment_step_op)

  def after_run(self, run_context, run_values):
    step_value = run_values.results
    if step_value is None:
      return
    if self._should_stop(step_value):
      logging.info("Now stopping '%s' training after %d steps", self._spec.name,
                   step_value)
      self._train_manager.request_stop(
          self._spec, "Training complete after {} steps.".format(step_value))

  def _should_stop(self, step):
    return self._max_steps is not None and step >= self._max_steps


class _GlobalStepSetterHook(tf_compat.SessionRunHook):
  """A hook for setting the global step variable.

  Should only be run on CPU and GPU, but not TPU. TPUs run many training steps
  per hook run, so the global step should be incremented in an op along with the
  candidates' train ops.
  """

  def __init__(self, train_manager, subnetwork_specs, base_global_step,
               global_step_combiner_fn):
    """Initializes a _GlobalStepSetterHook instance.

    Args:
      train_manager: The current iteration's `_TrainManager`.
      subnetwork_specs: List of `_SubnetworkSpec` instances for this iteration.
      base_global_step: Integer global step at the beginning of this iteration.
      global_step_combiner_fn: Function for combining each subnetwork's
        iteration step into the global step.
    """

    self._train_manager = train_manager
    self._subnetwork_specs = subnetwork_specs
    self._base_global_step = base_global_step
    self._global_step_combiner_fn = global_step_combiner_fn

  def begin(self):
    steps = [
        self._base_global_step + s.step.read_value()
        for s in self._subnetwork_specs
    ]
    updated_global_step = self._global_step_combiner_fn(steps)
    global_step = tf_compat.v1.train.get_global_step()
    self._assign_global_step_op = global_step.assign(updated_global_step)

  def after_run(self, run_context, run_values):
    # Global step cannot be retrieved via SessionRunArgs and before_run due to
    # race condition in hook execution.
    run_context.session.run(self._assign_global_step_op)


class _TrainingHookRunnerHook(tf_compat.SessionRunHook):
  """Hook wrapper for executing a spec's training hook.

  Will only run the hook according to the current TrainManager.
  """

  def __init__(self, train_manager, spec, hook):
    """Initializes a _TrainingHookRunnerHook instance.

    Only accepts a single hook, since merging hooks is complex and should be
    handled by the MonitoredTrainingSession instead.

    Args:
      train_manager: The current iteration's `_TrainManager`.
      spec: Either a `SubnetworkSpec` or `EnsembleSpec` to train.
      hook: The spec's training hook to execute.
    """

    self._train_manager = train_manager
    self._spec = spec
    self._hook = hook

  def begin(self):
    self._hook.begin()

  @contextlib.contextmanager
  def _session_run_context(self):
    """Intercepts input out of range errors to gracefully stop spec training."""

    try:
      yield
    except (tf.errors.OutOfRangeError, StopIteration) as e:
      logging.info("Now stopping '%s' training after hitting end of input",
                   self._spec.name)
      self._train_manager.request_stop(self._spec,
                                       "OutOfRangeError: {}".format(e))

  def after_create_session(self, session, coord):
    with self._session_run_context():
      self._hook.after_create_session(session, coord)

  def before_run(self, run_context):
    if self._train_manager.should_train(self._spec):
      with self._session_run_context():
        return self._hook.before_run(run_context)

  def after_run(self, run_context, run_values):
    if self._train_manager.should_train(self._spec):
      with self._session_run_context():
        self._hook.after_run(run_context, run_values)

  def end(self, session):
    with self._session_run_context():
      self._hook.end(session)


# TODO: Replace candidates with ensemble_specs.
class _Iteration(
    collections.namedtuple("_Iteration", [
        "number", "candidates", "subnetwork_specs", "estimator_spec",
        "best_candidate_index", "summaries", "train_manager",
        "subnetwork_reports"
    ])):
  """An AdaNet iteration.

  An AdaNet iteration represents the simultaneous training of multiple
  candidates for one iteration of the AdaNet loop, and tracks the best
  candidate's loss, predictions, and evaluation metrics.

  There must be maximum one _Iteration per graph.
  """

  def __new__(cls, number, candidates, subnetwork_specs, estimator_spec,
              best_candidate_index, summaries, train_manager,
              subnetwork_reports):
    """Creates a validated `_Iteration` instance.

    Args:

    Returns:
      A validated `_Iteration` object.

    Args:
      number: The iteration number.
      candidates: List of `_Candidate` instances to track.
      subnetwork_specs: List of `_SubnetworkSpec` instances.
      estimator_spec: `EstimatorSpec` instance.
      best_candidate_index: Int `Tensor` indicating the best candidate's index.
      summaries: List of `adanet.Summary` instances for each candidate.
      train_manager: The current `_TrainManager` for monitoring candidate per
        training.
      subnetwork_reports: Dict mapping string names to `subnetwork.Report`s, one
        per candidate.

    Raises:
      ValueError: If validation fails.
    """

    if not isinstance(number, (int, np.integer)):
      raise ValueError("number must be an integer")
    if number < 0:
      raise ValueError("number must be greater than 0 got %d" % (number))
    if not isinstance(candidates, list) or not candidates:
      raise ValueError("candidates must be a non-empty list")
    if estimator_spec is None:
      raise ValueError("estimator_spec is required")
    if best_candidate_index is None:
      raise ValueError("best_candidate_index is required")
    if not isinstance(subnetwork_reports, dict):
      raise ValueError("subnetwork_reports must be a dict")
    return super(_Iteration, cls).__new__(
        cls,
        number=number,
        candidates=candidates,
        subnetwork_specs=subnetwork_specs,
        estimator_spec=estimator_spec,
        best_candidate_index=best_candidate_index,
        summaries=summaries,
        train_manager=train_manager,
        subnetwork_reports=subnetwork_reports)


def _is_numeric(tensor):
  """Determines if given tensor is a float numeric."""

  if not isinstance(tensor, tf.Tensor):
    return False
  return tensor.dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64]


class _IterationBuilder(object):
  """Builds AdaNet iterations."""

  def __init__(self,
               candidate_builder,
               subnetwork_manager,
               ensemble_builder,
               ensemblers,
               max_steps,
               summary_maker,
               global_step_combiner_fn=tf.math.reduce_mean,
               placement_strategy=distributed.ReplicationStrategy(),
               replicate_ensemble_in_training=False,
               use_tpu=False,
               debug=False,
               enable_ensemble_summaries=True,
               enable_subnetwork_summaries=True):
    """Creates an `_IterationBuilder` instance.

    Args:
      candidate_builder: A `_CandidateBuilder` instance.
      subnetwork_manager: A `_SubnetworkManager` instance.
      ensemble_builder: An `_EnsembleBuilder` instance.
      ensemblers: An iterable of :class:`adanet.ensemble.Ensembler` objects that
        define how to ensemble a group of subnetworks.
      max_steps: Maximum number of steps to train candidate subnetworks.
      summary_maker: A function that constructs an `adanet.Summary` instance
        from (namespace, scope, and skip_summary).
      global_step_combiner_fn: Function for combining each subnetwork's
        iteration step into the global step.
      placement_strategy: A `PlacementStrategy` for assigning subnetworks and
        ensembles to specific workers.
      replicate_ensemble_in_training: Whether to build the frozen subnetworks in
        `training` mode during training.
      use_tpu: Whether AdaNet is running on TPU.
      debug: Boolean to enable debug mode which will check features and labels
        for Infs and NaNs.
      enable_ensemble_summaries: Whether to record summaries to display in
        TensorBoard for each ensemble candidate. Disable to reduce memory and
        disk usage per run.
      enable_subnetwork_summaries: Whether to record summaries to display in
        TensorBoard for each subnetwork. Disable to reduce memory and disk usage
        per run.

    Returns:
      An `_IterationBuilder` object.
    """

    if max_steps is not None and max_steps <= 0:
      raise ValueError("max_steps must be > 0 or None")
    self._candidate_builder = candidate_builder
    self._subnetwork_manager = subnetwork_manager
    self._ensemble_builder = ensemble_builder
    self._ensemblers = ensemblers
    self._max_steps = max_steps
    self._summary_maker = summary_maker
    self._global_step_combiner_fn = global_step_combiner_fn
    self._placement_strategy = placement_strategy
    self._replicate_ensemble_in_training = replicate_ensemble_in_training
    self._use_tpu = use_tpu
    self._debug = debug
    self._enable_ensemble_summaries = enable_ensemble_summaries
    self._enable_subnetwork_summaries = enable_subnetwork_summaries
    super(_IterationBuilder, self).__init__()

  @property
  def placement_strategy(self):
    return self._placement_strategy

  @placement_strategy.setter
  def placement_strategy(self, new_placement_strategy):
    self._placement_strategy = new_placement_strategy

  def _check_numerics(self, features, labels):
    """Checks for NaNs and Infs in input features and labels.

    Args:
      features: Dictionary of `Tensor` objects keyed by feature name.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head). Can be `None`.

    Returns:
      A features and labels tuple with same types and respective inputs, but
      with numeric check ops wrapping them.
    """

    if not self._debug:
      return features, labels

    checked_features, checked_labels = {}, {}
    logging.info("DEBUG: Checking numerics of float features.")
    for name in sorted(features):
      if not _is_numeric(features[name]):
        continue
      logging.info("DEBUG: Checking numerics of float feature '%s'.", name)
      checked_features[name] = tf.debugging.check_numerics(
          features[name], "features '{}'".format(name))
    if isinstance(labels, dict):
      for name in sorted(labels):
        if not _is_numeric(labels[name]):
          continue
        logging.info("DEBUG: Checking numerics of float label '%s'.", name)
        checked_labels[name] = tf.debugging.check_numerics(
            labels[name], "labels '{}'".format(name))
    elif labels is not None and _is_numeric(labels):
      logging.info("DEBUG: Checking numerics of labels.")
      checked_labels = tf.debugging.check_numerics(labels, "'labels'")
    return checked_features, checked_labels

  def build_iteration(self,
                      base_global_step,
                      iteration_number,
                      ensemble_candidates,
                      subnetwork_builders,
                      features,
                      mode,
                      config,
                      labels=None,
                      previous_ensemble_summary=None,
                      previous_ensemble_spec=None,
                      rebuilding=False,
                      rebuilding_ensembler_name=None,
                      best_ensemble_index_override=None):
    """Builds and returns AdaNet iteration t.

    This method uses the generated the candidate subnetworks given the ensemble
    at iteration t-1 and creates graph operations to train them. The returned
    `_Iteration` tracks the training of all candidates to know when the
    iteration is over, and tracks the best candidate's predictions and loss, as
    defined by lowest complexity-regularized loss on the train set.

    Args:
      base_global_step: Integer global step at the beginning of this iteration.
      iteration_number: Integer iteration number.
      ensemble_candidates: Iterable of `adanet.ensemble.Candidate` instances.
      subnetwork_builders: A list of `Builders` for adding ` Subnetworks` to the
        graph. Each subnetwork is then wrapped in a `_Candidate` to train.
      features: Dictionary of `Tensor` objects keyed by feature name.
      mode: Defines whether this is training, evaluation or prediction. See
        `ModeKeys`.
      config: The `tf.estimator.RunConfig` to use this iteration.
      labels: `Tensor` of labels. Can be `None`.
      previous_ensemble_summary: The `adanet.Summary` for the previous ensemble.
      previous_ensemble_spec: Optional `_EnsembleSpec` for iteration t-1.
      rebuilding: Boolean whether the iteration is being rebuilt only to restore
        the previous best subnetworks and ensembles.
      rebuilding_ensembler_name: Optional ensembler to restrict to, only
        relevant when rebuilding is set as True.
      best_ensemble_index_override: Integer index to identify the best ensemble
        candidate instead of computing the best ensemble index dynamically
        conditional on the ensemble AdaNet losses.

    Returns:
      An _Iteration instance.

    Raises:
      ValueError: If subnetwork_builders is empty.
      ValueError: If two subnetworks share the same name.
      ValueError: If two ensembles share the same name.
    """

    self._placement_strategy.config = config

    logging.info("%s iteration %s", "Rebuilding" if rebuilding else "Building",
                 iteration_number)

    if not subnetwork_builders:
      raise ValueError("Each iteration must have at least one Builder.")

    # TODO: Consider moving builder mode logic to ensemble_builder.py.
    builder_mode = mode
    if rebuilding:
      # Build the subnetworks and ensembles in EVAL mode by default. This way
      # their outputs aren't affected by dropout etc.
      builder_mode = tf.estimator.ModeKeys.EVAL
      if mode == tf.estimator.ModeKeys.PREDICT:
        builder_mode = mode

      # Only replicate in training mode when the user requests it.
      if self._replicate_ensemble_in_training and (
          mode == tf.estimator.ModeKeys.TRAIN):
        builder_mode = mode

    features, labels = self._check_numerics(features, labels)

    training = mode == tf.estimator.ModeKeys.TRAIN
    skip_summaries = mode == tf.estimator.ModeKeys.PREDICT or rebuilding
    with tf_compat.v1.variable_scope("iteration_{}".format(iteration_number)):
      seen_builder_names = {}
      candidates = []
      summaries = []
      subnetwork_reports = {}
      previous_ensemble = None

      if previous_ensemble_spec:
        previous_ensemble = previous_ensemble_spec.ensemble
        # Include previous best subnetwork as a candidate so that its
        # predictions are returned until a new candidate outperforms.
        seen_builder_names = {previous_ensemble_spec.name: True}
        previous_best_candidate = self._candidate_builder.build_candidate(
            ensemble_spec=previous_ensemble_spec,
            training=training,
            summary=previous_ensemble_summary)
        candidates.append(previous_best_candidate)
        if self._enable_ensemble_summaries:
          summaries.append(previous_ensemble_summary)

        # Generate subnetwork reports.
        if mode == tf.estimator.ModeKeys.EVAL:
          metrics = call_eval_metrics(previous_ensemble_spec.eval_metrics)
          subnetwork_report = subnetwork.Report(
              hparams={},
              attributes={},
              metrics=metrics,
          )
          subnetwork_report.metrics["adanet_loss"] = tf_compat.v1.metrics.mean(
              previous_ensemble_spec.adanet_loss)
          subnetwork_reports["previous_ensemble"] = subnetwork_report

      for subnetwork_builder in subnetwork_builders:
        if subnetwork_builder.name in seen_builder_names:
          raise ValueError("Two subnetworks have the same name '{}'".format(
              subnetwork_builder.name))
        seen_builder_names[subnetwork_builder.name] = True
      subnetwork_specs = []
      num_subnetworks = len(subnetwork_builders)
      skip_summary = skip_summaries or not self._enable_subnetwork_summaries
      for i, subnetwork_builder in enumerate(subnetwork_builders):
        if not self._placement_strategy.should_build_subnetwork(
            num_subnetworks, i) and not rebuilding:
          continue
        with self._placement_strategy.subnetwork_devices(num_subnetworks, i):
          subnetwork_name = "t{}_{}".format(iteration_number,
                                            subnetwork_builder.name)
          subnetwork_summary = self._summary_maker(
              namespace="subnetwork",
              scope=subnetwork_name,
              skip_summary=skip_summary)
          if not skip_summary:
            summaries.append(subnetwork_summary)
          logging.info("%s subnetwork '%s'",
                       "Rebuilding" if rebuilding else "Building",
                       subnetwork_builder.name)
          subnetwork_spec = self._subnetwork_manager.build_subnetwork_spec(
              name=subnetwork_name,
              subnetwork_builder=subnetwork_builder,
              summary=subnetwork_summary,
              features=features,
              mode=builder_mode,
              labels=labels,
              previous_ensemble=previous_ensemble)
          subnetwork_specs.append(subnetwork_spec)
          # Workers that don't build ensembles need a dummy candidate in order
          # to train the subnetwork.
          # Because only ensembles can be considered candidates, we need to
          # convert the subnetwork into a dummy ensemble and subsequently a
          # dummy candidate. However, this dummy candidate is never considered a
          # true candidate during candidate evaluation and selection.
          # TODO: Eliminate need for candidates.
          if not self._placement_strategy.should_build_ensemble(
              num_subnetworks) and not rebuilding:
            candidates.append(
                self._create_dummy_candidate(subnetwork_spec,
                                             subnetwork_builders,
                                             subnetwork_summary, training))
        # Generate subnetwork reports.
        if mode != tf.estimator.ModeKeys.PREDICT:
          subnetwork_report = subnetwork_builder.build_subnetwork_report()
          if not subnetwork_report:
            subnetwork_report = subnetwork.Report(
                hparams={}, attributes={}, metrics={})
          metrics = call_eval_metrics(subnetwork_spec.eval_metrics)
          for metric_name in sorted(metrics):
            metric = metrics[metric_name]
            subnetwork_report.metrics[metric_name] = metric
          subnetwork_reports[subnetwork_builder.name] = subnetwork_report

      # Create (ensemble_candidate*ensembler) ensembles.
      skip_summary = skip_summaries or not self._enable_ensemble_summaries
      seen_ensemble_names = {}
      for ensembler in self._ensemblers:
        if rebuilding and rebuilding_ensembler_name and (
            ensembler.name != rebuilding_ensembler_name):
          continue
        for ensemble_candidate in ensemble_candidates:
          if not self._placement_strategy.should_build_ensemble(
              num_subnetworks) and not rebuilding:
            continue
          ensemble_name = "t{}_{}_{}".format(iteration_number,
                                             ensemble_candidate.name,
                                             ensembler.name)
          if ensemble_name in seen_ensemble_names:
            raise ValueError(
                "Two ensembles have the same name '{}'".format(ensemble_name))
          seen_ensemble_names[ensemble_name] = True
          summary = self._summary_maker(
              namespace="ensemble",
              scope=ensemble_name,
              skip_summary=skip_summary)
          if not skip_summary:
            summaries.append(summary)
          ensemble_spec = self._ensemble_builder.build_ensemble_spec(
              name=ensemble_name,
              candidate=ensemble_candidate,
              ensembler=ensembler,
              subnetwork_specs=subnetwork_specs,
              summary=summary,
              features=features,
              mode=builder_mode,
              iteration_number=iteration_number,
              labels=labels,
              previous_ensemble_spec=previous_ensemble_spec)
          # TODO: Eliminate need for candidates.
          # TODO: Don't track moving average of loss when rebuilding
          # previous ensemble.
          candidate = self._candidate_builder.build_candidate(
              ensemble_spec=ensemble_spec, training=training, summary=summary)
          candidates.append(candidate)
          # TODO: Move adanet_loss from subnetwork report to a new
          # ensemble report, since the adanet_loss is associated with an
          # ensemble, and only when using a ComplexityRegularizedEnsemblers.
          # Keep adanet_loss in subnetwork report for backwards compatibility.
          if len(ensemble_candidates) != len(subnetwork_builders):
            continue
          if len(ensemble_candidate.subnetwork_builders) > 1:
            continue
          if mode == tf.estimator.ModeKeys.PREDICT:
            continue
          builder_name = ensemble_candidate.subnetwork_builders[0].name
          subnetwork_reports[builder_name].metrics[
              "adanet_loss"] = tf_compat.v1.metrics.mean(
                  ensemble_spec.adanet_loss)

      # Dynamically select the outputs of best candidate.
      best_candidate_index = self._best_candidate_index(
          candidates, best_ensemble_index_override)
      best_predictions = self._best_predictions(candidates,
                                                best_candidate_index)
      best_loss = self._best_loss(candidates, best_candidate_index, mode)
      best_export_outputs = self._best_export_outputs(candidates,
                                                      best_candidate_index,
                                                      mode, best_predictions)
      train_manager_dir = os.path.join(config.model_dir, "train_manager",
                                       "t{}".format(iteration_number))
      train_manager, training_chief_hooks, training_hooks = self._create_hooks(
          base_global_step, subnetwork_specs, candidates, num_subnetworks,
          rebuilding, train_manager_dir, config.is_chief)
      # Iteration summaries.
      summary = self._summary_maker(
          namespace=None, scope=None, skip_summary=skip_summaries)
      summaries.append(summary)
      with summary.current_scope():
        summary.scalar("iteration/adanet/iteration", iteration_number)
        if best_loss is not None:
          summary.scalar("loss", best_loss)
      iteration_metrics = _IterationMetrics(iteration_number, candidates,
                                            subnetwork_specs)
      # All training happens in hooks so we don't need a train op.
      train_op = tf.no_op()
      if self._use_tpu:
        estimator_spec = tf_compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=best_predictions,
            loss=best_loss,
            train_op=self._create_tpu_train_op(base_global_step,
                                               subnetwork_specs, candidates,
                                               mode, num_subnetworks, config),
            eval_metrics=iteration_metrics.best_eval_metrics_tuple(
                best_candidate_index, mode),
            export_outputs=best_export_outputs,
            training_hooks=training_hooks)
      else:
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=best_predictions,
            loss=best_loss,
            train_op=train_op,
            eval_metric_ops=iteration_metrics.best_eval_metric_ops(
                best_candidate_index, mode),
            export_outputs=best_export_outputs,
            training_chief_hooks=training_chief_hooks,
            training_hooks=training_hooks)

      return _Iteration(
          number=iteration_number,
          candidates=candidates,
          subnetwork_specs=subnetwork_specs,
          estimator_spec=estimator_spec,
          best_candidate_index=best_candidate_index,
          summaries=summaries,
          train_manager=train_manager,
          subnetwork_reports=subnetwork_reports)

  def _create_dummy_candidate(self, subnetwork_spec, subnetwork_builders,
                              subnetwork_summary, training):
    """Returns a dummy candidate for the given SubnetworkSpec.

    AdaNet only considers ensembles as candidate models, and ensembles
    are represented as `_Candidates`. When training only subnetworks, such as
    on a subnetwork-worker in the RoundRobinStrategy, then we still need a
    candidate to manage the training of the subnetwork, even if it gets
    discarded, hence the dummy candidate.

    Args:
      subnetwork_spec: The subnetwork spec for the dummy candidate to wrap.
      subnetwork_builders: List of all subnetwork builders generated this
        iteration.
      subnetwork_summary: `_Summary` object to use for TensorBoard.
      training: Whether or not we are currently training.
    """

    dummy_ensemble_spec = _EnsembleSpec(
        name="dummy_{}".format(subnetwork_spec.name),
        ensemble=None,
        architecture=None,
        subnetwork_builders=subnetwork_builders,
        predictions=subnetwork_spec.predictions,
        loss=subnetwork_spec.loss,
        step=None,
        adanet_loss=0.)
    return self._candidate_builder.build_candidate(
        ensemble_spec=dummy_ensemble_spec,
        training=training,
        summary=subnetwork_summary,
        track_moving_average=False)

  def _create_tpu_train_op(self, base_global_step, subnetwork_specs, candidates,
                           mode, num_subnetworks, config):
    """Returns the train op for this set of candidates.

    This train op combines the train ops from all the candidates into a single
    train op. Additionally, it is responsible for incrementing the global step.

    The train op is only non-None during the `TRAIN` mode.

    Args:
      base_global_step: Integer global step at the beginning of this iteration.
      subnetwork_specs: List of `_SubnetworkSpec` instances for this iteration.
      candidates: List of `_Candidate` instances to train.
      mode: Defines whether this is training, evaluation or inference. The train
        op is only non-None during `TRAIN`. See `ModeKeys`.
      num_subnetworks: Integer number of subnetwork builders generated for the
        current iteration.
      config: The `tf.estimator.RunConfig` to use this iteration.

    Returns:
      A `Tensor` train op.
    """

    if mode != tf.estimator.ModeKeys.TRAIN:
      return tf.no_op()
    ensemble_specs = [c.ensemble_spec for c in candidates]
    with tf_compat.v1.variable_scope("train_op"):
      train_ops = []
      if self._placement_strategy.should_train_subnetworks(num_subnetworks):
        for subnetwork_spec in subnetwork_specs:
          if subnetwork_spec.train_op is not None:
            train_ops.append(subnetwork_spec.train_op.train_op)
      for ensemble_spec in ensemble_specs:
        if ensemble_spec.train_op is not None:
          # The train op of a previous ensemble is None even during `TRAIN`.
          train_ops.append(ensemble_spec.train_op.train_op)

      with tf.control_dependencies(train_ops):
        # Increment steps after train ops complete to avoid non-determinism.
        increment_ops = [s.step.assign_add(1) for s in subnetwork_specs]
        increment_ops += [e.step.assign_add(1) for e in ensemble_specs]

        if not config.is_chief:
          return tf.group(*increment_ops)
        # AdaNet's chief worker is responsible for setting the global step, not
        # the candidates it trains. Assigning the global step is the final
        # action performed in the train op.
        with tf.control_dependencies(increment_ops):
          steps = [s.step.read_value() for s in subnetwork_specs]
          global_step = tf_compat.v1.train.get_global_step()
          return global_step.assign(
              tf.cast(
                  base_global_step + self._global_step_combiner_fn(steps),
                  dtype=tf.int64))

  def _create_hooks(self, base_global_step, subnetwork_specs, candidates,
                    num_subnetworks, rebuilding, train_manager_dir, is_chief):
    """Returns the hooks to monitor and train this iteration.

    Args:
      base_global_step: Integer global step at the beginning of this iteration.
      subnetwork_specs: List of `_SubnetworkSpec` instances.
      candidates: List of `_Candidate` instances to compare.
      num_subnetworks: Integer number of subnetwork builders generated for the
        current iteration.
      rebuilding: Boolean whether the iteration is being rebuilt only to restore
        the previous best subnetworks and ensembles.
      train_manager_dir: Directory for the TrainManager to store spec metadata.
      is_chief: Whether the current worker is chief.

    Returns:
      A 3-tuple of a _TrainManager for monitoring training, a list of
      `SessionRunHooks` to run on chief, and a list of `SessionRunHooks` to run
      on all workers.
    """

    training_chief_hooks, training_hooks = [], []
    ensemble_specs = [c.ensemble_spec for c in candidates]
    train_manager = _TrainManager(subnetwork_specs, ensemble_specs,
                                  train_manager_dir, is_chief)
    if not self._use_tpu:
      # On TPU, the global step gets incremented in an op since it doesn't have
      # hook run granularity of CPU and GPU training.
      training_chief_hooks.append(
          _GlobalStepSetterHook(train_manager, subnetwork_specs,
                                base_global_step,
                                self._global_step_combiner_fn))
    should_train_subnetworks = (
        self._placement_strategy.should_train_subnetworks(num_subnetworks))
    for spec in subnetwork_specs:
      if not self._use_tpu:
        training_hooks.append(_NanLossHook(train_manager, spec))
      # We increment the step along with the global step as part of the train
      # op on TPU, whereas on CPU and GPU we use hooks for fine grained control.
      if self._use_tpu or not should_train_subnetworks or spec.train_op is None:
        increment_step_op = None
      else:
        with tf.control_dependencies([spec.train_op.train_op]):
          increment_step_op = spec.step.assign_add(1)
      # TPU also supports uneven training, but up to num_iterations_per_loop.
      training_hooks.append(
          _TrainingLimitHook(
              train_manager,
              spec,
              self._max_steps,
              increment_step_op=increment_step_op))
      if not should_train_subnetworks and not rebuilding:
        continue
      self._add_hooks(spec, train_manager, training_chief_hooks, training_hooks)
    for spec in ensemble_specs:
      if not self._use_tpu:
        training_hooks.append(_NanLossHook(train_manager, spec))
      # See above comment about incrementing the step on CPU vs. TPU.
      if self._use_tpu or spec.train_op is None:
        increment_step_op = None
      else:
        with tf.control_dependencies([spec.train_op.train_op]):
          increment_step_op = spec.step.assign_add(1)
      training_hooks.append(
          _TrainingLimitHook(
              train_manager,
              spec,
              self._max_steps,
              increment_step_op=increment_step_op))
      self._add_hooks(spec, train_manager, training_chief_hooks, training_hooks)
    return train_manager, training_chief_hooks, training_hooks

  def _add_hooks(self, spec, train_manager, training_chief_hooks,
                 training_hooks):
    """Appends spec train hooks to the given hook lists."""

    if not spec.train_op:
      return
    for hook in spec.train_op.chief_hooks:
      training_chief_hooks.append(
          _TrainingHookRunnerHook(train_manager, spec, hook))
    for hook in spec.train_op.hooks:
      training_hooks.append(_TrainingHookRunnerHook(train_manager, spec, hook))

  def _best_candidate_index(self, candidates, best_ensemble_index_override):
    """Returns the index of the best candidate in the list.

    The best candidate is the one with the smallest AdaNet loss, unless
    `best_ensemble_index_override` is given.

    TODO: Best ensemble index should always be static during EVAL
    and PREDICT modes.

    In case a candidate has a NaN loss, their loss is immediately set to
    infinite, so that they are not selected. As long as one candidate ensemble
    has a non-NaN loss during training, the dreaded `NanLossDuringTrainingError`
    should not be raised.

    Args:
      candidates: List of `_Candidate` instances to choose from.
      best_ensemble_index_override: Integer index to return instead of computing
        the best ensemble index dynamically.

    Returns:
      An integer `Tensor` representing the index of the best candidate.
    """

    with tf_compat.v1.variable_scope("best_candidate_index"):
      if best_ensemble_index_override is not None:
        return tf.constant(best_ensemble_index_override)

      if len(candidates) == 1:
        return tf.constant(0)
      adanet_losses = [candidate.adanet_loss for candidate in candidates]
      # Replace NaNs with Infs since so that NaN loss candidates are never
      # chosen.
      adanet_losses = tf.where(
          tf_compat.v1.is_nan(adanet_losses),
          tf.ones_like(adanet_losses) * np.inf, adanet_losses)
      return tf.argmin(input=adanet_losses, axis=0)

  def _best_predictions(self, candidates, best_candidate_index):
    """Returns the best predictions from a set of candidates.

    Args:
      candidates: List of `_Candidate` instances to compare.
      best_candidate_index: `Tensor` index of the best candidate in the list.

    Returns:
      A `Tensor` or dictionary of `Tensor`s representing the best candidate's
      predictions (depending on what the subnetworks return).
    """

    if len(candidates) == 1:
      return candidates[0].ensemble_spec.predictions

    with tf_compat.v1.variable_scope("best_predictions"):
      predictions = None
      for candidate in candidates:
        ensemble_spec = candidate.ensemble_spec
        if isinstance(ensemble_spec.predictions, dict):
          if not predictions:
            predictions = {}
          for key in sorted(ensemble_spec.predictions):
            tensor = ensemble_spec.predictions[key]
            if key in predictions:
              predictions[key].append(tensor)
            else:
              predictions[key] = [tensor]
        else:
          if not predictions:
            predictions = []
          predictions.append(ensemble_spec.predictions)

      if isinstance(predictions, dict):
        best_predictions = {}
        for key in sorted(predictions):
          tensor_list = predictions[key]
          best_predictions[key] = tf.stack(tensor_list)[best_candidate_index]
      else:
        best_predictions = tf.stack(predictions)[best_candidate_index]
      return best_predictions

  def _best_loss(self, candidates, best_candidate_index, mode):
    """Returns the best loss from a set of candidates.

    Args:
      candidates: List of `_Candidate` instances to compare.
      best_candidate_index: `Tensor` index of the best candidate in the list.
      mode: Defines whether this is training, evaluation or inference. Loss is
        always None during inference. See `ModeKeys`.

    Returns:
      Float `Tensor` of the best candidate's loss.
    """

    if mode == tf.estimator.ModeKeys.PREDICT:
      return None
    if len(candidates) == 1:
      return candidates[0].ensemble_spec.loss
    with tf_compat.v1.variable_scope("best_loss"):
      losses = [c.ensemble_spec.loss for c in candidates]
      loss = tf.slice(tf.stack(losses), [best_candidate_index], [1])
      return tf.reshape(loss, [])

  def _best_export_outputs(self, candidates, best_candidate_index, mode,
                           best_predictions):
    """Returns the best `SavedModel` export outputs from a set of candidates.

    Assumes that all candidate ensembles have identical export output keys and
    `ExportOutput` types.

    Args:
      candidates: List of `_Candidate` instances to compare.
      best_candidate_index: `Tensor` index of the best candidate in the list.
      mode: Defines whether this is training, evaluation or inference. Export
        outputs are always None during training and evaluation. See `ModeKeys`.
      best_predictions: A `Tensor` or dictionary of `Tensor`s representing the
        best candidate's predictions (depending on what the subnetworks return).

    Returns:
      A `Tensor` dictionary representing the best candidate's export outputs.

    Raises:
      TypeError: If the `ExportOutput` type is not supported.
    """

    if mode != tf.estimator.ModeKeys.PREDICT:
      return None
    if len(candidates) == 1:
      return candidates[0].ensemble_spec.export_outputs
    with tf_compat.v1.variable_scope("best_export_outputs"):
      # Group tensors by export output key and ExportOutput type.
      export_outputs = {}
      for candidate in candidates:
        ensemble_spec = candidate.ensemble_spec
        for key in sorted(ensemble_spec.export_outputs):
          export_output = ensemble_spec.export_outputs[key]
          if isinstance(export_output,
                        tf.estimator.export.ClassificationOutput):
            if key not in export_outputs:
              export_outputs[key] = ([], [])
            if export_output.scores is not None:
              export_outputs[key][0].append(export_output.scores)
            if export_output.classes is not None:
              export_outputs[key][1].append(export_output.classes)
          elif isinstance(export_output, tf.estimator.export.RegressionOutput):
            if key not in export_outputs:
              export_outputs[key] = []
            export_outputs[key].append(export_output.value)
          elif isinstance(export_output, tf.estimator.export.PredictOutput):
            # Use self._best_predictions() below to get prediction output.
            continue
          else:
            raise TypeError(
                "Values in export_outputs must be ClassificationOutput, "
                "RegressionOutput, or PredictOutput objects. Given: {}".format(
                    export_output))

      # Stack tensor lists into correct ExportOutput type, outputting the
      # correct values based on the best candidate index.
      best_export_outputs = {}
      for key in sorted(candidates[0].ensemble_spec.export_outputs):
        export_output = candidates[0].ensemble_spec.export_outputs[key]
        if isinstance(export_output, tf.estimator.export.ClassificationOutput):
          scores, classes = None, None
          if export_outputs[key][0]:
            scores = tf.stack(export_outputs[key][0])[best_candidate_index]
          if export_outputs[key][1]:
            classes = tf.stack(export_outputs[key][1])[best_candidate_index]
          output = tf.estimator.export.ClassificationOutput(
              scores=scores, classes=classes)
        elif isinstance(export_output, tf.estimator.export.RegressionOutput):
          value = tf.stack(export_outputs[key])[best_candidate_index]
          output = tf.estimator.export.RegressionOutput(value)
        else:
          output = tf.estimator.export.PredictOutput(best_predictions)
        best_export_outputs[key] = output
      return best_export_outputs
