"""An AdaNet estimator implementation in Tensorflow using a single graph.

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

import contextlib
import errno
import inspect
import os
import time

from absl import logging
from adanet import tf_compat
from adanet.core.architecture import _Architecture
from adanet.core.candidate import _CandidateBuilder
from adanet.core.ensemble_builder import _EnsembleBuilder
from adanet.core.ensemble_builder import _SubnetworkManager
from adanet.core.eval_metrics import call_eval_metrics
from adanet.core.iteration import _IterationBuilder
from adanet.core.report_accessor import _ReportAccessor
from adanet.core.summary import _ScopedSummary
from adanet.core.summary import _TPUScopedSummary
from adanet.core.timer import _CountDownTimer
from adanet.distributed import ReplicationStrategy
from adanet.distributed.devices import monkey_patch_default_variable_placement_strategy
from adanet.ensemble import ComplexityRegularizedEnsembler
from adanet.ensemble import GrowStrategy
import numpy as np
import six
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import deprecation
from tensorflow_estimator.python.estimator import util
# pylint: enable=g-direct-tensorflow-import


class _StopAfterTrainingHook(tf_compat.SessionRunHook):
  """Hook that requests stop once iteration is over."""

  def __init__(self, iteration, after_fn):
    """Initializes a `_StopAfterTrainingHook`.

    Args:
      iteration: An `_Iteration` instance.
      after_fn: A function to call after training stopped.

    Returns:
      A `_StopAfterTrainingHook` instance.
    """

    self._iteration = iteration
    self._after_fn = after_fn

  def before_run(self, run_context):
    """See `SessionRunHook`."""

    self._stop_if_is_over(run_context)

  def after_run(self, run_context, run_values):
    """See `SessionRunHook`."""

    self._stop_if_is_over(run_context)

  def _stop_if_is_over(self, run_context):
    """Signals the monitored session to step when the iteration is over."""

    if not self._iteration.train_manager.is_over():
      return
    logging.info("Now stopping iteration %d training", self._iteration.number)
    run_context.request_stop()
    self._after_fn()


class _EvalMetricSaverHook(tf_compat.SessionRunHook):
  """A hook for writing candidate evaluation metrics as summaries to disk."""

  def __init__(self, name, kind, eval_metrics, output_dir):
    """Initializes a `_EvalMetricSaverHook` instance.

    Args:
      name: String name of candidate owner of these metrics.
      kind: The kind of candidate that the metrics belong to (e.g. subnetwork).
      eval_metrics: Tuple of (metric_fn, tensors) which returns a dict of metric
        results keyed by name. The values of the dict are the results of calling
        a metric function, namely a `(metric_tensor, update_op)` tuple.
        `metric_tensor` should be evaluated without any impact on state
        (typically is a pure computation based on variables.). For example, it
        should not trigger the `update_op` or require any input fetching.
      output_dir: Directory for writing evaluation summaries.

    Returns:
      An `_EvalMetricSaverHook` instance.
    """

    self._name = name
    self._kind = kind
    self._eval_metrics = eval_metrics
    self._output_dir = output_dir

  def begin(self):
    """See `SessionRunHook`."""

    # The metric_fn is called with tf.placeholders to simply read the value of
    # the metric variables. The metrics themselves are computed as a result of
    # being returned in the EstimatorSpec by _adanet_model_fn.
    metric_fn, tensors = self._eval_metrics
    tensors = [tf_compat.v1.placeholder(t.dtype, t.shape) for t in tensors]
    eval_metric_ops = metric_fn(*tensors)
    self._eval_metric_tensors = {}
    for key in sorted(eval_metric_ops):
      value = tf_compat.metric_op(eval_metric_ops[key])
      self._eval_metric_tensors[key] = value[0]

  def _dict_to_str(self, dictionary):
    """Get a `str` representation of a `dict`.

    Args:
      dictionary: The `dict` to be represented as `str`.

    Returns:
      A `str` representing the `dictionary`.
    """
    return ", ".join(
        "{} = {}".format(k, v) for k, v in sorted(dictionary.items()))

  def end(self, session):
    """See `SessionRunHook`."""

    # Forked from tensorflow/python/estimator/estimator.py function called
    # _write_dict_to_summary.
    current_global_step = tf_compat.v1.train.get_global_step()
    eval_dict, current_global_step = session.run(
        (self._eval_metric_tensors, current_global_step))

    logging.info("Saving %s '%s' dict for global step %d: %s", self._kind,
                 self._name, current_global_step, self._dict_to_str(eval_dict))
    summary_writer = tf_compat.v1.summary.FileWriterCache.get(self._output_dir)
    summary_proto = tf_compat.v1.summary.Summary()
    for key in eval_dict:
      value = eval_dict[key]
      if isinstance(value, (np.float32, float)):
        summary_proto.value.add(tag=key, simple_value=float(value))
      elif isinstance(value, six.binary_type):
        summ = tf_compat.v1.summary.Summary.FromString(value)
        for i, _ in enumerate(summ.value):
          summ.value[i].tag = "{}/{}".format(key, i)
        summary_proto.value.extend(summ.value)
      else:
        logging.warn(
            "Skipping summary for %s, must be a float, np.float32, "
            "or a serialized string of Summary.", key)
    summary_writer.add_summary(summary_proto, current_global_step)
    summary_writer.flush()
    # Note(b/137672676): Do not explicitly call summary_writer.close() here.
    # This will cause eval summaries to not be written out after the first time
    # in continuous evals.


class _OverwriteCheckpointHook(tf_compat.SessionRunHook):
  """Hook to overwrite the latest checkpoint with next iteration variables."""

  def __init__(self, current_iteration, iteration_number_tensor,
               previous_iteration_vars, config):
    """Initializes an _OverwriteCheckpointHook instance.

    Args:
      current_iteration: Current `_Iteration` object.
      iteration_number_tensor: Int variable `Tensor` storing the current
        iteration number.
      previous_iteration_vars: Variables to restore from the previous iteration
        before overwriting the checkpoint.
      config: The Estimator's RunConfig object.
    """

    self._iteration_number = current_iteration.number
    self._iteration_number_tensor = iteration_number_tensor
    self._previous_iteration_vars = previous_iteration_vars
    self._model_dir = config.model_dir
    self._checkpoint_state = tf.train.get_checkpoint_state(self._model_dir)
    self._keep_checkpoint_max = config.keep_checkpoint_max

    self._update_op = None
    self._overwrite_saver = None
    self._checkpoint_overwritten = False

  def begin(self):
    """Creates the savers and adds ops needed for overwriting the checkpoint.

    Two savers are created, a restore saver which is passed the variables from
    the previous iteration to restore, and an overwrite saver which will
    actually overwrite the checkpoint.
    """

    self._restore_saver = tf_compat.v1.train.Saver(
        sharded=True, var_list=self._previous_iteration_vars)
    # Note: self._iteration_number already contains the value of the next
    # iteration since _OverwriteCheckpointHook should only execute during the
    # graph growing phase.
    self._update_op = self._iteration_number_tensor.assign(
        self._iteration_number)
    self._overwrite_saver = tf_compat.v1.train.Saver(
        sharded=True, max_to_keep=self._keep_checkpoint_max)
    self._overwrite_saver.recover_last_checkpoints(
        self._checkpoint_state.all_model_checkpoint_paths)

  def before_run(self, run_context):
    """Overwrites checkpoint before any calls to session.run().

    This is to ensure that the values of the variables in the overwritten
    checkpoint match those in the pevious iteration checkpoint.

    Args:
      run_context: The tf.train.SessionRunContext passed to the hook.
    """

    if not self._checkpoint_overwritten:
      session = run_context.session
      self._restore_saver.restore(session,
                                  self._checkpoint_state.model_checkpoint_path)
      session.run(self._update_op)
      checkpoint_path = os.path.join(self._model_dir, "increment.ckpt")
      logging.info(
          "Overwriting checkpoint with new graph for iteration %d to %s-%d",
          self._iteration_number, checkpoint_path, self._iteration_number)
      # Specify global_step=self._iteration_number to append the iteration
      # number to the checkpoint name, e.g. <model_dir>/increment.ckpt-1.
      self._overwrite_saver.save(
          session, checkpoint_path, global_step=self._iteration_number)
      self._checkpoint_overwritten = True


class _GraphGrowingHookDecorator(tf_compat.SessionRunHook):
  """Decorates a SessionRunHook to only run begin() and end() methods."""

  def __init__(self, hook):
    """Initializes a _GraphGrowingHookDecorator instance.

    Args:
      hook: The SessionRunHook to decorate.
    """
    self._hook = hook

  def begin(self):
    self._hook.begin()

  def end(self, session):
    self._hook.end(session)


def _delete_directory(directory):
  """Removes directory and handles any folder or file exceptions."""
  if not tf.io.gfile.exists(directory):
    return
  try:
    tf.io.gfile.rmtree(directory)
  except (tf.errors.PermissionDeniedError,
          tf.errors.FailedPreconditionError) as e:
    logging.info("Ignoring folder or file issues: %s '%s'", e.error_code,
                 e.message)


class Estimator(tf.estimator.Estimator):
  # pyformat: disable
  r"""A :class:`tf.estimator.Estimator` for training, evaluation, and serving.

  This implementation uses an :class:`adanet.subnetwork.Generator` as its weak
  learning algorithm for generating candidate subnetworks. These are trained in
  parallel using a single graph per iteration. At the end of each iteration, the
  estimator saves the sub-graph of the best subnetwork ensemble and its weights
  as a separate checkpoint. At the beginning of the next iteration, the
  estimator imports the previous iteration's frozen graph and adds ops for the
  next candidates as part of a new graph and session. This allows the estimator
  have the performance of Tensorflow's static graph constraint (minus the
  performance hit of reconstructing a graph between iterations), while having
  the flexibility of having a dynamic graph.

  NOTE: Subclassing :class:`tf.estimator.Estimator` is only necessary to work
  with :meth:`tf.estimator.train_and_evaluate` which asserts that the estimator
  argument is a :class:`tf.estimator.Estimator` subclass. However, all training
  is delegated to a separate :class:`tf.estimator.Estimator` instance. It is
  responsible for supporting both local and distributed training. As such, the
  :class:`adanet.Estimator` is only responsible for bookkeeping across
  iterations.

  Args:
    head: A :class:`tf.contrib.estimator.Head` instance for computing loss and
      evaluation metrics for every candidate.
    subnetwork_generator: The :class:`adanet.subnetwork.Generator` which defines
      the candidate subnetworks to train and evaluate at every AdaNet iteration.
    max_iteration_steps: Total number of steps for which to train candidates per
      iteration. If :class:`OutOfRange` or :class:`StopIteration` occurs in the
      middle, training stops before `max_iteration_steps` steps. When
      :code:`None`, it will train the current iteration forever.
    ensemblers: An iterable of :class:`adanet.ensemble.Ensembler` objects that
      define how to ensemble a group of subnetworks. If there are multiple,
      each should have a different `name` property.
    ensemble_strategies: An iterable of :class:`adanet.ensemble.Strategy`
      objects that define the candidate ensembles of subnetworks to explore at
      each iteration.
    evaluator: An :class:`adanet.Evaluator` for candidate selection after all
      subnetworks are done training. When :code:`None`, candidate selection uses
      a moving average of their :class:`adanet.Ensemble` AdaNet loss during
      training instead. In order to use the *AdaNet algorithm* as described in
      [Cortes et al., '17], the given :class:`adanet.Evaluator` must be created
      with the same dataset partition used during training. Otherwise, this
      framework will perform *AdaNet.HoldOut* which uses a holdout set for
      candidate selection, but does not benefit from learning guarantees.
    report_materializer: An :class:`adanet.ReportMaterializer`. Its reports are
      made available to the `subnetwork_generator` at the next iteration, so
      that it can adapt its search space. When `None`, the
      `subnetwork_generator` :meth:`generate_candidates` method will receive
      empty Lists for their `previous_ensemble_reports` and `all_reports`
      arguments.
    metric_fn: A function for adding custom evaluation metrics, which should
      obey the following signature:
        - `Args`:
          Can only have the following three arguments in any order:
          - :code:`predictions`: Predictions `Tensor` or dict of `Tensor`
            created by given :code:`head`.
          - :code:`features`: Input `dict` of `Tensor` objects created by
            :code:`input_fn` which is given to :meth:`estimator.evaluate` as an
            argument.
          - :code:`labels`: Labels `Tensor` or dict of `Tensor` (for multi-head)
            created by :code:`input_fn` which is given to
            :meth:`estimator.evaluate` as an argument.
        - `Returns`: Dict of metric results keyed by name. Final metrics are a
          union of this and :code:`head`'s existing metrics. If there is a name
          conflict between this and :code:`head`s existing metrics, this will
          override the existing one. The values of the dict are the results of
          calling a metric function, namely a :code:`(metric_tensor, update_op)`
          tuple.
    force_grow: Boolean override that forces the ensemble to grow by one
      subnetwork at the end of each iteration. Normally at the end of each
      iteration, AdaNet selects the best candidate ensemble according to its
      performance on the AdaNet objective. In some cases, the best ensemble is
      the `previous_ensemble` as opposed to one that includes a newly trained
      subnetwork. When `True`, the algorithm will not select the
      `previous_ensemble` as the best candidate, and will ensure that after n
      iterations the final ensemble is composed of n subnetworks.
    replicate_ensemble_in_training: Whether to rebuild the frozen subnetworks of
      the ensemble in training mode, which can change the outputs of the frozen
      subnetworks in the ensemble. When `False` and during candidate training,
      the frozen subnetworks in the ensemble are in prediction mode, so
      training-only ops like dropout are not applied to them. When `True` and
      training the candidates, the frozen subnetworks will be in training mode
      as well, so they will apply training-only ops like dropout.  This argument
      is useful for regularizing learning mixture weights, or for making
      training-only side inputs available in subsequent iterations. For most
      use-cases, this should be `False`.
    adanet_loss_decay: Float decay for the exponential-moving-average of the
      AdaNet objective throughout training. This moving average is a data-
      driven way tracking the best candidate with only the training set.
    delay_secs_per_worker: Float number of seconds to delay starting the
      i-th worker. Staggering worker start-up during distributed asynchronous
      SGD can improve training stability and speed up convergence. Each worker
      will wait (i+1) * delay_secs_per_worker seconds before beginning training.
    max_worker_delay_secs: Float max number of seconds to delay starting the
      i-th worker. Staggering worker start-up during distributed asynchronous
      SGD can improve training stability and speed up convergence. Each worker
      will wait up to max_worker_delay_secs before beginning training.
    worker_wait_secs: Float number of seconds for workers to wait before
      checking if the chief prepared the next iteration.
    worker_wait_timeout_secs: Float number of seconds for workers to wait for
      chief to prepare the next iteration during distributed training. This is
      needed to prevent workers waiting indefinitely for a chief that may have
      crashed or been turned down. When the timeout is exceeded, the worker
      exits the train loop. In situations where the chief job is much slower
      than the worker jobs, this timeout should be increased.
    model_dir: Directory to save model parameters, graph and etc. This can also
      be used to load checkpoints from the directory into a estimator to
      continue training a previously saved model.
    report_dir: Directory where the
      :class:`adanet.subnetwork.MaterializedReport`s materialized by
      :code:`report_materializer` would be saved. If :code:`report_materializer`
      is :code:`None`, this will not save anything. If :code:`None` or
      empty string, defaults to :code:`<model_dir>/report`.
    config: :class:`RunConfig` object to configure the runtime settings.
    debug: Boolean to enable debug mode which will check features and labels
      for Infs and NaNs.
    enable_ensemble_summaries: Whether to record summaries to display in
      TensorBoard for each ensemble candidate. Disable to reduce memory and disk
      usage per run.
    enable_subnetwork_summaries: Whether to record summaries to display in
      TensorBoard for each subnetwork. Disable to reduce memory and disk usage
      per run.
    global_step_combiner_fn: Function for combining each subnetwork's
      iteration step into the global step.
    max_iterations: Integer maximum number of AdaNet iterations (a.k.a. rounds)
      of generating new subnetworks and ensembles, training them, and evaluating
      them against the current best ensemble. When :code:`None`, AdaNet will
      keep iterating until `Estimator#train` terminates. Otherwise, if
      :code:`max_iteratios` is supplied and is met or exceeded during training,
      training will terminate even before `steps` or `max_steps`.
    **kwargs: Extra keyword args passed to the parent.

  Returns:
    An :class:`adanet.Estimator` instance.

  Raises:
    :code:`ValueError`: If :code:`subnetwork_generator` is :code:`None`.
    :code:`ValueError`: If :code:`max_iteration_steps` is <= 0.
    :code:`ValueError`: If :code:`model_dir` is not specified during distributed
      training.
    :code:`ValueError`: If :code:`max_iterations` is <= 0.
  """
  # pyformat: enable

  class _Keys(object):
    CURRENT_ITERATION = "current_iteration"
    SUBNETWORK_GENERATOR = "subnetwork_generator"

  def __init__(self,
               head,
               subnetwork_generator,
               max_iteration_steps,
               ensemblers=None,
               ensemble_strategies=None,
               evaluator=None,
               report_materializer=None,
               metric_fn=None,
               force_grow=False,
               replicate_ensemble_in_training=False,
               adanet_loss_decay=.9,
               delay_secs_per_worker=5,
               max_worker_delay_secs=60,
               worker_wait_secs=5,
               worker_wait_timeout_secs=7200,
               model_dir=None,
               report_dir=None,
               config=None,
               debug=False,
               enable_ensemble_summaries=True,
               enable_subnetwork_summaries=True,
               global_step_combiner_fn=tf.math.reduce_mean,
               max_iterations=None,
               forward_feature_keys=None,
               sparse_default_values=None,
               **kwargs):
    if subnetwork_generator is None:
      raise ValueError("subnetwork_generator can't be None.")
    if max_iteration_steps is not None and max_iteration_steps <= 0.:
      raise ValueError("max_iteration_steps must be > 0 or None.")
    if max_iterations is not None and max_iterations <= 0.:
      raise ValueError("max_iteration_steps must be > 0 or None.")
    is_distributed_training = config and config.num_worker_replicas > 1
    is_model_dir_specified = model_dir or (config and config.model_dir)
    if is_distributed_training and not is_model_dir_specified:
      # A common model dir for the chief and workers is required for
      # coordination during distributed training.
      raise ValueError(
          "For distributed training, a model_dir must be specified.")

    self._subnetwork_generator = subnetwork_generator

    # Overwrite superclass's assert that members are not overwritten in order
    # to overwrite public methods. Note that we are doing something that is not
    # explicitly supported by the Estimator API and may break in the future.
    tf.estimator.Estimator._assert_members_are_not_overridden = staticmethod(  # pylint: disable=protected-access
        lambda _: None)

    self._evaluator = evaluator
    self._report_materializer = report_materializer

    self._force_grow = force_grow
    self._delay_secs_per_worker = delay_secs_per_worker
    self._max_worker_delay_secs = max_worker_delay_secs
    self._worker_wait_secs = worker_wait_secs
    self._worker_wait_timeout_secs = worker_wait_timeout_secs
    self._max_iterations = max_iterations

    # Added for backwards compatibility.
    default_ensembler_args = [
        "mixture_weight_type", "mixture_weight_initializer",
        "warm_start_mixture_weights", "adanet_lambda", "adanet_beta", "use_bias"
    ]
    default_ensembler_kwargs = {
        k: v for k, v in kwargs.items() if k in default_ensembler_args
    }
    if default_ensembler_kwargs:
      logging.warning(
          "The following arguments have been moved to "
          "`adanet.ensemble.ComplexityRegularizedEnsembler` which can be "
          "specified in the `ensemblers` argument: %s",
          sorted(default_ensembler_kwargs.keys()))
    for key in default_ensembler_kwargs:
      del kwargs[key]

    # Experimental feature.
    placement_strategy_arg = "experimental_placement_strategy"
    placement_strategy = kwargs.pop(placement_strategy_arg, None)
    if placement_strategy:
      logging.warning(
          "%s is an experimental feature. Its behavior is not guaranteed "
          "to be backwards compatible.", placement_strategy_arg)

    # Monkey patch the default variable placement strategy that Estimator uses
    # since it does not support workers having different graphs from the chief.
    # TODO: Consider using `RunConfig.replace` with the new device_fn,
    # but this can cause issues since RunConfig automatically parses TF_CONFIG
    # environment variable.
    with monkey_patch_default_variable_placement_strategy():
      # This `Estimator` is responsible for bookkeeping across iterations, and
      # for training the subnetworks in both a local and distributed setting.
      # Subclassing improves future-proofing against new private methods being
      # added to `tf.estimator.Estimator` that are expected to be callable by
      # external functions, such as in b/110435640.
      super(Estimator, self).__init__(
          model_fn=self._adanet_model_fn,
          params={},
          config=config,
          model_dir=model_dir,
          **kwargs)

    if default_ensembler_kwargs and ensemblers:
      raise ValueError("When specifying the `ensemblers` argument, "
                       "the following arguments must not be given: {}".format(
                           default_ensembler_kwargs.keys()))
    if not ensemblers:
      default_ensembler_kwargs["model_dir"] = self.model_dir
      ensemblers = [ComplexityRegularizedEnsembler(**default_ensembler_kwargs)]

    # These are defined after base Estimator's init so that they can
    # use the same temporary model_dir as the underlying Estimator even if
    # model_dir is not provided.
    self._use_tpu = kwargs.get("use_tpu", False)
    ensemble_builder = _EnsembleBuilder(
        head=head, metric_fn=metric_fn, use_tpu=self._use_tpu)

    # TODO: Merge CandidateBuilder into SubnetworkManager.
    candidate_builder = _CandidateBuilder(adanet_loss_decay=adanet_loss_decay)
    subnetwork_manager = _SubnetworkManager(
        head=head, metric_fn=metric_fn, use_tpu=self._use_tpu)
    if not placement_strategy:
      placement_strategy = ReplicationStrategy()
    self._iteration_builder = _IterationBuilder(
        candidate_builder,
        subnetwork_manager,
        ensemble_builder,
        ensemblers,
        max_iteration_steps,
        self._summary_maker,
        global_step_combiner_fn,
        placement_strategy,
        replicate_ensemble_in_training,
        use_tpu=self._use_tpu,
        debug=debug,
        enable_ensemble_summaries=enable_ensemble_summaries,
        enable_subnetwork_summaries=enable_subnetwork_summaries)
    self._ensemble_strategies = ensemble_strategies or [GrowStrategy()]

    report_dir = report_dir or os.path.join(self._model_dir, "report")
    self._report_accessor = _ReportAccessor(report_dir)

  def _summary_maker(self, scope=None, skip_summary=False, namespace=None):
    """Constructs a `_ScopedSummary`."""
    if self._use_tpu:
      return _TPUScopedSummary(
          logdir=self._model_dir,
          scope=scope,
          skip_summary=skip_summary,
          namespace=namespace)
    else:
      return _ScopedSummary(
          scope=scope, skip_summary=skip_summary, namespace=namespace)

  def _latest_checkpoint_iteration_number(self):
    """Returns the iteration number from the latest checkpoint."""

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if latest_checkpoint is None:
      return 0
    return tf.train.load_variable(latest_checkpoint,
                                  self._Keys.CURRENT_ITERATION)

  def _latest_checkpoint_global_step(self):
    """Returns the global step from the latest checkpoint."""

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if latest_checkpoint is None:
      return 0
    return tf.train.load_variable(latest_checkpoint,
                                  tf_compat.v1.GraphKeys.GLOBAL_STEP)

  def forward_features(self,
                       estimator,
                       forward_feature_keys=None,
                       sparse_default_values=None):
    if forward_feeature_keys is not None:
      result = tf.contrib.estimator.forward_features(super(Estimator, self),
                                                   forward_feature_keys=forward_feature_keys,
                                                   sparse_default_values=sparse_default_values)
      return result
    else:
      raise ValueError("No key/(s) provided to forward features.")

  def train(self,
            input_fn,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None):
    # pyformat: disable
    """Trains a model given training data :code:`input_fn`.

    NOTE: If a given input_fn raises an :code:`OutOfRangeError`, then *all* of
    training will exit. The best practice is to make the training dataset repeat
    forever, in order to perform model search for more than one iteration.

    Args:
      input_fn: A function that provides input data for training as minibatches.
        See [Premade Estimators](
        https://tensorflow.org/guide/premade_estimators#create_input_functions)
        for more information. The function should construct and return one of
        the following:
          * A :code:`tf.data.Dataset` object: Outputs of `Dataset` object must
            be a tuple `(features, labels)` with same constraints as below.
          * A tuple `(features, labels)`: Where `features` is a
            :code:`tf.Tensor` or a dictionary of string feature name to
            `Tensor` and `labels` is a :code:`Tensor` or a dictionary of string
            label name to `Tensor`. Both `features` and `labels` are consumed by
            `model_fn`. They should satisfy the expectation of `model_fn` from
            inputs.
      hooks: List of :code:`tf.train.SessionRunHook` subclass instances. Used
        for callbacks inside the training loop.
      steps: Number of steps for which to train the model. If :code:`None`,
        train forever or train until `input_fn` generates the
        :code:`tf.errors.OutOfRange` error or :code:`StopIteration` exception.
        `steps` works incrementally. If you call two times `train(steps=10)`
        then training occurs in total 20 steps. If :code:`OutOfRange` or
        :code:`StopIteration` occurs in the middle, training stops before 20
        steps. If you don't want to have incremental behavior please set
        `max_steps` instead. If set, `max_steps` must be :code:`None`.
      max_steps: Number of total steps for which to train model. If
        :code:`None`, train forever or train until `input_fn` generates the
        :code:`tf.errors.OutOfRange` error or :code:`StopIteration` exception.
        If set, `steps` must be `None`. If :code:`OutOfRange` or
        :code:`StopIteration` occurs in the middle, training stops before
        `max_steps` steps. Two calls to `train(steps=100)` means 200 training
        iterations. On the other hand, two calls to `train(max_steps=100)`
        means that the second call will not do any iteration since first call
        did all 100 steps.
      saving_listeners: list of :code:`CheckpointSaverListener` objects. Used
        for callbacks that run immediately before or after checkpoint savings.

    Returns:
      `self`, for chaining.

    Raises:
      ValueError: If both `steps` and `max_steps` are not `None`.
      ValueError: If either `steps` or `max_steps <= 0`.
    """
    # pyformat: enable

    if (steps is not None) and (max_steps is not None):
      raise ValueError("Can not provide both steps and max_steps.")
    if steps is not None and steps <= 0:
      raise ValueError("Must specify steps > 0, given: {}".format(steps))

    latest_global_steps = self._latest_checkpoint_global_step()
    if steps is not None:
      max_steps = latest_global_steps + steps

    # Each iteration of this AdaNet loop represents an `_Iteration`. The
    # current iteration number is stored as a variable in the checkpoint so
    # that training can be stopped and started at anytime.
    with monkey_patch_default_variable_placement_strategy():
      while True:
        latest_global_steps = self._latest_checkpoint_global_step()
        current_iteration = self._latest_checkpoint_iteration_number()
        logging.info("Beginning training AdaNet iteration %s",
                     current_iteration)
        self._iteration_ended = False

        # Delegate training to a temporary estimator instead of super to make
        # passing arguments more functional (via params).
        temp_estimator = self._create_temp_estimator(
            self.config, params={
                "is_inside_training_loop": True,
            })
        result = temp_estimator.train(
            input_fn=input_fn,
            hooks=hooks,
            max_steps=max_steps,
            saving_listeners=saving_listeners)
        # In TensorFlow v1.14.0 and below, saving listeners are attached to the
        # first CheckpointSaverHook each time train is called. Instead, we pass
        # in the saving_listeners in the first AdaNet iteration only.
        if not tf_compat.version_greater_or_equal("1.15.0"):
          saving_listeners = None
        logging.info("Finished training Adanet iteration %s", current_iteration)

        # If training ended because the maximum number of training steps
        # occurred, exit training.
        global_steps = self._latest_checkpoint_global_step()
        if max_steps is not None and global_steps >= max_steps:
          logging.info("Training ended after %s global steps", global_steps)
          return result

        # If training ended for any reason other than the iteration ending,
        # exit training.
        if not self._iteration_ended:
          logging.info("Training stop requested")
          return result

        max_iterations = self._max_iterations
        if max_iterations and current_iteration + 1 >= max_iterations:
          logging.info(
              "Training ended after exceeding maximum AdaNet iterations")
          if steps is not None and global_steps - latest_global_steps < steps:
            logging.warning(
                "Both `max_iterations` and `steps` were specified, but "
                "`max_iterations` takes precedence over `steps`")
          return result

        logging.info("Beginning bookkeeping phase for iteration %s",
                     current_iteration)

        # The chief prepares the next AdaNet iteration, and increments the
        # iteration number by 1.
        if self.config.is_chief:
          with self._force_replication_strategy():
            self._execute_bookkeeping_phase(
                input_fn, current_iteration, train_hooks=hooks or [])

        # This inner loop serves mainly for synchronizing the workers with the
        # chief during distributed training. Workers that finish training early
        # wait for the chief to prepare the next iteration and increment the
        # iteration number. Workers that are slow to finish training quickly
        # move onto the next iteration. And workers that go offline and return
        # online after training ended terminate gracefully.
        wait_for_chief = not self.config.is_chief
        timer = _CountDownTimer(self._worker_wait_timeout_secs)
        while wait_for_chief:
          # If the chief hits max_steps, it will stop training itself and not
          # increment the iteration number, so this is how the worker knows to
          # exit if it wakes up and the chief is gone.
          # TODO: Support steps parameter.
          if self._latest_checkpoint_global_step() >= max_steps:
            return result

          # In distributed training, a worker may end training before the chief
          # overwrites the checkpoint with the incremented iteration number. If
          # that is the case, it should wait for the chief to do so. Otherwise
          # the worker will get stuck waiting for its weights to be initialized.
          next_iteration = self._latest_checkpoint_iteration_number()
          if next_iteration > current_iteration:
            break

          # Check timeout when waiting for potentially downed chief.
          if timer.secs_remaining() == 0:
            logging.error(
                "Chief job did not prepare next iteration after %s secs. It "
                "may have been preempted, been turned down, or crashed. This "
                "worker is now exiting training.",
                self._worker_wait_timeout_secs)
            return result
          logging.info("Waiting for chief to finish")
          time.sleep(self._worker_wait_secs)

        # Stagger starting workers to prevent training instability.
        # Mimics behavior of tf.estimator.train_and_evaluate.
        if not self.config.is_chief and self.config.task_type == "worker":
          task_id = self.config.task_id or 0
          # Stagger each worker up to 60 secs.
          delay_secs = min(self._max_worker_delay_secs,
                           (task_id + 1.) * self._delay_secs_per_worker)
          if delay_secs > 0.:
            logging.info("Waiting %d secs before continuing training.",
                         delay_secs)
            time.sleep(delay_secs)

        logging.info("Finished bookkeeping phase for iteration %s",
                     current_iteration)

  def evaluate(self,
               input_fn,
               steps=None,
               hooks=None,
               checkpoint_path=None,
               name=None):
    if not checkpoint_path:
      checkpoint_path = tf.train.latest_checkpoint(self.model_dir)

    # Ensure that the read to get the iteration number and read to restore
    # variable values come from the same checkpoint during evaluation.
    params = {
        "evaluation_checkpoint_path": checkpoint_path,
        "evaluation_name": name,
        "best_ensemble_index": self._compute_best_ensemble_index(),
    }

    # Delegate evaluation to a temporary estimator instead of super to make
    # passing arguments more functional (via params).
    temp_estimator = self._create_temp_estimator(self.config, params=params)
    result = temp_estimator.evaluate(
        input_fn,
        steps=steps,
        hooks=hooks,
        checkpoint_path=checkpoint_path,
        name=name)
    return result

  def predict(self,
              input_fn,
              predict_keys=None,
              hooks=None,
              checkpoint_path=None,
              yield_single_examples=True):
    # Delegate predicting to a temporary estimator instead of super to make
    # passing arguments more functional (via params).
    temp_estimator = self._create_temp_estimator(
        self.config,
        params={
            "best_ensemble_index": self._compute_best_ensemble_index(),
        })
    return temp_estimator.predict(input_fn, predict_keys, hooks,
                                  checkpoint_path, yield_single_examples)

  @deprecation.deprecated(
      None, "This function has been renamed, use `export_saved_model` instead.")
  def export_savedmodel(self,
                        export_dir_base,
                        serving_input_receiver_fn,
                        assets_extra=None,
                        as_text=False,
                        checkpoint_path=None,
                        strip_default_attrs=False):
    # Delegate exporting to a temporary estimator instead of super to make
    # passing arguments more functional (via params).
    temp_estimator = self._create_temp_estimator(
        self.config,
        params={
            "best_ensemble_index": self._compute_best_ensemble_index(),
        })
    with self._force_replication_strategy():
      return temp_estimator.export_savedmodel(
          export_dir_base=export_dir_base,
          serving_input_receiver_fn=serving_input_receiver_fn,
          assets_extra=assets_extra,
          as_text=as_text,
          checkpoint_path=checkpoint_path,
          strip_default_attrs=strip_default_attrs)

  def export_saved_model(self,
                         export_dir_base,
                         serving_input_receiver_fn,
                         assets_extra=None,
                         as_text=False,
                         checkpoint_path=None,
                         experimental_mode=tf.estimator.ModeKeys.PREDICT):
    # Delegate exporting to a temporary estimator instead of super to make
    # passing arguments more functional (via params).
    temp_estimator = self._create_temp_estimator(
        self.config,
        params={
            "best_ensemble_index": self._compute_best_ensemble_index(),
        })
    with self._force_replication_strategy():
      return temp_estimator.export_saved_model(
          export_dir_base=export_dir_base,
          serving_input_receiver_fn=serving_input_receiver_fn,
          assets_extra=assets_extra,
          as_text=as_text,
          checkpoint_path=checkpoint_path,
          experimental_mode=experimental_mode)

  def experimental_export_all_saved_models(self,
                                           export_dir_base,
                                           input_receiver_fn_map,
                                           assets_extra=None,
                                           as_text=False,
                                           checkpoint_path=None):
    # Delegate exporting to a temporary estimator instead of super to make
    # passing arguments more functional (via params).
    temp_estimator = self._create_temp_estimator(
        self.config,
        params={
            "best_ensemble_index": self._compute_best_ensemble_index(),
        })
    with self._force_replication_strategy():
      return temp_estimator.experimental_export_all_saved_models(
          export_dir_base=export_dir_base,
          input_receiver_fn_map=input_receiver_fn_map,
          assets_extra=assets_extra,
          as_text=as_text,
          checkpoint_path=checkpoint_path)

  def _compute_best_ensemble_index(self):
    """Runs the Evaluator to obtain the best ensemble index among candidates."""

    if self._evaluator:
      return self._execute_candidate_evaluation_phase(
          self._evaluator.input_fn, export_best_architecture=False)
    return None

  @contextlib.contextmanager
  def _force_replication_strategy(self):
    """Sets placement_strategy to always be ReplicationStrategy.

    This is useful during the bookkeeping phase and when Estimator's export
    saved model functions are called. In both of these cases, local tf.Sessions
    are created which do not have access to the cluster. Therefore,
    RoundRobinReplicationStrategy will fail when it tries to place ops on
    cluster devices which the local tf.Sessions cannot access.

    Yields:
      Nothing. Simply returns control back to the caller.
    """

    temp_placement_strategy = self._iteration_builder.placement_strategy
    try:
      self._iteration_builder.placement_strategy = ReplicationStrategy()
      yield
    finally:
      self._iteration_builder.placement_strategy = temp_placement_strategy

  @contextlib.contextmanager
  def _call_input_fn_in_new_graph(self, input_fn, mode, config):
    """Calls the given input_fn and yields results within a new graph context.

    Yields features, labels, and hooks from the result of an Estimator input_fn.

    Args:
      input_fn: a function that takes no arguments and returns one of:
        * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
          tuple (features, labels) with same constraints as below.
        * A tuple (features, labels): Where `features` is a `Tensor` or a
          dictionary of string feature name to `Tensor` and `labels` is a
          `Tensor` or a dictionary of string label name to `Tensor`. Both
          `features` and `labels` are consumed by `model_fn`. They should
          satisfy the expectation of `model_fn` from inputs.
      mode: Defines whether this is training, evaluation or prediction. See
        `ModeKeys`.
      config: The current `tf.estimator.RunConfig`.

    Yields:
      Tuple of features, labels, and input_hooks, where features are as
      described above, labels are as described above or None, and input_hooks
      are a list of SessionRunHooks to be included when running.

    Raises:
      ValueError: if the result is a list or tuple of length != 2.
    """

    with tf.Graph().as_default() as g:
      tf_compat.v1.set_random_seed(config.tf_random_seed)
      # Create global step before calling model_fn as does superclass.
      self._create_and_assert_global_step(g)
      with tf.device("/cpu:0"):
        input_fn_outs = input_fn()
      yield util.parse_input_fn_result(input_fn_outs)

  def _create_temp_run_config(self, temp_model_dir):
    """Creates a temp `RunConfig` for the bookkeeping phase."""

    config = self.config
    return tf.estimator.RunConfig(
        model_dir=temp_model_dir,
        tf_random_seed=config.tf_random_seed,
        session_config=config.session_config,
        protocol=config.protocol)

  def _create_temp_estimator(self, config, params):
    """Creates a temp `Estimator` to grow the graph for the next iteration."""

    return tf.estimator.Estimator(
        model_fn=self._adanet_model_fn, config=config, params=params)

  def _execute_bookkeeping_phase(self, train_input_fn, iteration_number,
                                 train_hooks):
    """Run the AdaNet bookkeeping phase to prepare the next iteration.

    This method creates a TensorFlow graph up to three times:
      1. To evaluate all candidate ensembles to find the best one.
      2. To materialize reports and store them to disk (if report_materializer
         exists).
      3. To grow the TensorFlow graph and overwrite the model directory's
         checkpoint with the next iteration's ops.

    Args:
      train_input_fn: The input_fn used during training.
      iteration_number: Integer current iteration number.
      train_hooks: List of `SessionRunHook` passed for training.
    """

    next_iteration_number = iteration_number + 1
    logging.info("Preparing iteration %s:", next_iteration_number)

    if self._evaluator:
      evaluator_input_fn = self._evaluator.input_fn
    else:
      evaluator_input_fn = train_input_fn
    best_ensemble_index = self._execute_candidate_evaluation_phase(
        evaluator_input_fn, export_best_architecture=True)
    self._execute_report_materialization_phase(best_ensemble_index)
    self._execute_graph_growing_phase(train_input_fn, train_hooks)

    logging.info("Finished preparing iteration %s.", next_iteration_number)

  def _execute_candidate_evaluation_phase(self, evaluator_input_fn,
                                          export_best_architecture):
    """Evaluates and chooses the best ensemble for this iteration.

    Args:
      evaluator_input_fn: The input_fn for evaluation.
      export_best_architecture: Boolean whether to persist the best ensemble's
        architecture to the model_dir.

    Returns:
      Integer index of the best ensemble withing the list of candidate ensembles
      for the current iteration.
    """

    logging.info("Evaluating candidates...")
    config = self.config
    mode = tf.estimator.ModeKeys.EVAL
    with self._call_input_fn_in_new_graph(evaluator_input_fn, mode,
                                          config) as (features, labels,
                                                      input_hooks):
      current_iteration, _ = self._create_iteration(
          features, labels, mode, config, is_growing_phase=False)
      best_ensemble_index = self._get_best_ensemble_index(
          current_iteration, input_hooks)
      architecture = current_iteration.candidates[
          best_ensemble_index].ensemble_spec.architecture
    if export_best_architecture:
      iteration_number = self._latest_checkpoint_iteration_number()
      new_architecture_filename = self._architecture_filename(iteration_number)
      logging.info("Exporting best ensemble architecture to %s",
                   new_architecture_filename)
      self._save_architecture(new_architecture_filename, architecture)
    logging.info("Done evaluating candidates.")

    return best_ensemble_index

  def _execute_report_materialization_phase(self, best_ensemble_index):
    """Materializes and store subnetwork reports."""

    if not self._report_materializer:
      return

    logging.info("Materializing reports...")
    input_fn = self._report_materializer.input_fn
    mode = tf.estimator.ModeKeys.EVAL
    config = self.config
    with self._call_input_fn_in_new_graph(input_fn, mode,
                                          config) as (features, labels,
                                                      input_hooks):
      current_iteration, _ = self._create_iteration(
          features, labels, mode, config, is_growing_phase=False)
      self._materialize_report(current_iteration, input_hooks,
                               best_ensemble_index)
    logging.info("Done materializing reports.")

  def _execute_graph_growing_phase(self, train_input_fn, train_hooks):
    """Grows the tensorflow graph for the next iteration.

    Normally the MonitoredTrainingSession does not allow one to add new ops to
    the TensorFlow graph once training starts. To get around this limitation,
    create the graph for the next iteration and overwrite the model directory
    checkpoint with the expanded graph.

    Args:
      train_input_fn: The input_fn used during training.
      train_hooks: List of `SessionRunHook` passed for training.
    """

    logging.info("Adapting graph and incrementing iteration number...")
    config = self.config
    temp_model_dir = os.path.join(self.model_dir, "temp_model_dir")
    if not tf.io.gfile.exists(temp_model_dir):
      tf.io.gfile.makedirs(temp_model_dir)
    # Since deleting a model_dir can fail, we need each temporary directory to
    # be unique. So we use the UTC time when creating it.
    time_in_millis = int(time.time() * 1000)
    temp_model_sub_dir = os.path.join(temp_model_dir, str(time_in_millis))
    temp_run_config = config.replace(model_dir=temp_model_sub_dir)
    temp_estimator = self._create_temp_estimator(
        config=temp_run_config,
        params={
            "is_growing_phase": True,
            "is_inside_training_loop": True,
        })
    # Do not train with any saving_listeners since this is just a temporary
    # estimator.
    temp_estimator.train(
        input_fn=train_input_fn,
        max_steps=1,
        hooks=self._process_hooks_for_growing_phase(train_hooks),
        saving_listeners=None)

    _delete_directory(temp_model_dir)
    logging.info("Done adapting graph and incrementing iteration number.")

  def _architecture_filename(self, iteration_number):
    """Returns the filename of the given iteration's frozen graph."""

    frozen_checkpoint = os.path.join(self.model_dir, "architecture")
    return "{}-{}.json".format(frozen_checkpoint, iteration_number)

  def _get_best_ensemble_index(self, current_iteration, input_hooks):
    """Returns the best candidate ensemble's index in this iteration.

    Evaluates the ensembles using an `Evaluator` when provided. Otherwise,
    it returns the index of the best candidate as defined by the `_Iteration`.

    Args:
      current_iteration: Current `_Iteration`.
      input_hooks: List of SessionRunHooks to be included when running.

    Returns:
      Index of the best ensemble in the iteration's list of `_Candidates`.
    """

    # Skip the evaluation phase when there is only one candidate subnetwork.
    if len(current_iteration.candidates) == 1:
      logging.info("'%s' is the only ensemble",
                   current_iteration.candidates[0].ensemble_spec.name)
      return 0

    # The zero-th index candidate at iteration t>0 is always the
    # previous_ensemble.
    if current_iteration.number > 0 and self._force_grow and (len(
        current_iteration.candidates) == 2):
      logging.info("With `force_grow` enabled, '%s' is the only ensemble",
                   current_iteration.candidates[1].ensemble_spec.name)
      return 1

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    logging.info("Starting ensemble evaluation for iteration %s",
                 current_iteration.number)
    for hook in input_hooks:
      hook.begin()
    with tf_compat.v1.Session() as sess:
      init = tf.group(tf_compat.v1.global_variables_initializer(),
                      tf_compat.v1.local_variables_initializer(),
                      tf_compat.v1.tables_initializer())
      sess.run(init)
      coord = tf.train.Coordinator()
      for hook in input_hooks:
        hook.after_create_session(sess, coord)
      saver = tf_compat.v1.train.Saver(sharded=True)
      saver.restore(sess, latest_checkpoint)

      tf_compat.v1.train.start_queue_runners(sess=sess, coord=coord)
      ensemble_metrics = []
      for candidate in current_iteration.candidates:
        metrics = call_eval_metrics(candidate.ensemble_spec.eval_metrics)
        metrics["adanet_loss"] = tf_compat.v1.metrics.mean(
            candidate.ensemble_spec.adanet_loss)
        ensemble_metrics.append(metrics)
      if self._evaluator:
        metric_name = self._evaluator.metric_name
        metrics = self._evaluator.evaluate(sess, ensemble_metrics)
        objective_fn = self._evaluator.objective_fn
      else:
        metric_name = "adanet_loss"
        metrics = sess.run(
            [c.adanet_loss for c in current_iteration.candidates])
        objective_fn = np.nanargmin

      values = []
      for i in range(len(current_iteration.candidates)):
        ensemble_name = current_iteration.candidates[i].ensemble_spec.name
        values.append("{}/{} = {:.6f}".format(metric_name, ensemble_name,
                                              metrics[i]))
      logging.info("Computed ensemble metrics: %s", ", ".join(values))
      if self._force_grow and current_iteration.number > 0:
        logging.info(
            "The `force_grow` override is enabled, so the "
            "the performance of the previous ensemble will be ignored.")
        # NOTE: The zero-th index candidate at iteration t>0 is always the
        # previous_ensemble.
        metrics = metrics[1:]
        index = objective_fn(metrics) + 1
      else:
        index = objective_fn(metrics)
    logging.info("Finished ensemble evaluation for iteration %s",
                 current_iteration.number)
    logging.info("'%s' at index %s is the best ensemble",
                 current_iteration.candidates[index].ensemble_spec.name, index)
    return index

  def _materialize_report(self, current_iteration, input_hooks,
                          best_ensemble_index):
    """Generates reports as defined by `Builder`s.

    Materializes the Tensors and metrics defined in the `Builder`s'
    `build_subnetwork_report` method using `ReportMaterializer`, and stores
    them to disk using `_ReportAccessor`.

    Args:
      current_iteration: Current `_Iteration`.
      input_hooks: List of SessionRunHooks to be included when running.
      best_ensemble_index: Integer index of the best candidate ensemble.
    """

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    logging.info("Starting metric logging for iteration %s",
                 current_iteration.number)

    best_candidate = current_iteration.candidates[best_ensemble_index]
    best_architecture = best_candidate.ensemble_spec.architecture
    included_subnetwork_names = [
        name for i, name in best_architecture.subnetworks
        if i == current_iteration.number
    ]
    for hook in input_hooks:
      hook.begin()
    with tf_compat.v1.Session() as sess:
      init = tf.group(tf_compat.v1.global_variables_initializer(),
                      tf_compat.v1.local_variables_initializer(),
                      tf_compat.v1.tables_initializer())
      sess.run(init)
      coord = tf.train.Coordinator()
      for hook in input_hooks:
        hook.after_create_session(sess, coord)
      saver = tf_compat.v1.train.Saver(sharded=True)
      saver.restore(sess, latest_checkpoint)
      tf_compat.v1.train.start_queue_runners(sess=sess, coord=coord)
      materialized_reports = (
          self._report_materializer.materialize_subnetwork_reports(
              sess, current_iteration.number,
              current_iteration.subnetwork_reports, included_subnetwork_names))
      self._report_accessor.write_iteration_report(current_iteration.number,
                                                   materialized_reports)

    logging.info("Finished saving subnetwork reports for iteration %s",
                 current_iteration.number)

  def _process_hooks_for_growing_phase(self, hooks):
    """Processes hooks which will run during the graph growing phase.

    In particular the following things are done:
      - CheckpointSaverHooks are filtered out since they are not intended to
        run between training runs and will cause errors. We also reset the
        CheckpointSaverHooks' Saver between iterations, see b/122795064 for more
        details.
      - Decorate the remaining hooks with _GraphGrowingHookDecorator to only run
        the begin() and end() methods during the graph growing phase.

    Args:
      hooks: The list of `SessionRunHooks` to process.

    Returns:
      The processed hooks which should run during the growing phase.
    """

    processed_hooks = []
    for hook in hooks:
      # Reset CheckpointSaverHooks' Saver and filter out.
      if isinstance(hook, tf_compat.CheckpointSaverHook):
        hook._saver = None  # pylint: disable=protected-access
        continue
      # Do not decorate the _OverwriteCheckpointHook since it should always
      # run during the graph growing phase.
      if not isinstance(hook, _OverwriteCheckpointHook):
        hook = _GraphGrowingHookDecorator(hook)
      processed_hooks.append(hook)
    return processed_hooks

  def _training_chief_hooks(self, current_iteration, training):
    """Returns chief-only training hooks to be run this iteration.

    Args:
      current_iteration: Current `_Iteration`.
      training: Whether in training mode.

    Returns:
      A list of `SessionRunHook` instances.
    """

    if not training:
      return []

    training_hooks = []
    for summary in current_iteration.summaries:
      output_dir = self.model_dir
      if summary.scope:
        output_dir = os.path.join(output_dir, summary.namespace, summary.scope)
      summary_saver_hook = tf_compat.SummarySaverHook(
          save_steps=self.config.save_summary_steps,
          output_dir=output_dir,
          summary_op=summary.merge_all())
      training_hooks.append(summary_saver_hook)
    training_hooks += list(
        current_iteration.estimator_spec.training_chief_hooks)
    return training_hooks

  def _training_hooks(self, current_iteration, training,
                      iteration_number_tensor, previous_iteration_vars,
                      is_growing_phase):
    """Returns training hooks to be run on all workers and chief this iteration.

    Args:
      current_iteration: Current `_Iteration`.
      training: Whether in training mode.
      iteration_number_tensor: An int tensor of the current AdaNet iteraiton.
      previous_iteration_vars: The variables of the previous iteration to be
        restored by the _OverwriteCheckpointHook. If empty, no
        _OverwriteCheckpointHook will be created.
      is_growing_phase: Whether we are in the AdaNet graph growing phase.

    Returns:
      A list of `SessionRunHook` instances.
    """

    if not training:
      return []

    def after_fn():
      self._iteration_ended = True

    training_hooks = list(current_iteration.estimator_spec.training_hooks) + [
        _StopAfterTrainingHook(current_iteration, after_fn=after_fn)
    ]

    if is_growing_phase:
      training_hooks.append(
          _OverwriteCheckpointHook(current_iteration, iteration_number_tensor,
                                   previous_iteration_vars, self.config))
    return training_hooks

  def _evaluation_hooks(self, current_iteration, training, evaluation_name):
    """Returns evaluation hooks for this iteration.

    Args:
      current_iteration: Current `_Iteration`.
      training: Whether in training mode.
      evaluation_name: String name to append to the eval directory.

    Returns:
      A list of `SessionRunHook` instances.
    """

    if training:
      return []
    evaluation_hooks = []
    for subnetwork_spec in current_iteration.subnetwork_specs:
      evaluation_hooks.append(
          self._create_eval_metric_saver_hook(
              subnetwork_spec.eval_metrics,
              subnetwork_spec.name,
              kind="subnetwork",
              evaluation_name=evaluation_name))
    for candidate in current_iteration.candidates:
      evaluation_hooks.append(
          self._create_eval_metric_saver_hook(
              candidate.ensemble_spec.eval_metrics,
              candidate.ensemble_spec.name,
              kind="ensemble",
              evaluation_name=evaluation_name))
    return evaluation_hooks

  def _create_eval_metric_saver_hook(self, eval_metrics, name, kind,
                                     evaluation_name):
    eval_subdir = "eval"
    if evaluation_name:
      eval_subdir = "eval_{}".format(evaluation_name)
    return _EvalMetricSaverHook(
        name=name,
        kind=kind,
        eval_metrics=eval_metrics,
        output_dir=os.path.join(self.model_dir, kind, name, eval_subdir))

  def _save_architecture(self, filename, architecture):
    """Persists the ensemble's architecture in a serialized format.

    Writes to a text file with one subnetwork's iteration number and name
    per line.

    Args:
      filename: String filename to persist the ensemble architecture.
      architecture: Target `_Architecture` instance.
    """

    # Make directories since model_dir may not have been created yet.
    tf.io.gfile.makedirs(os.path.dirname(filename))
    with tf.io.gfile.GFile(filename, "w") as record_file:
      record_file.write(
          architecture.serialize(self._latest_checkpoint_global_step().item()))

  def _read_architecture(self, filename):
    """Reads an ensemble architecture from disk.

    Assumes the file was written with `_save_architecture`.

    Args:
      filename: String filename where features were recorded.

    Returns:
      An `_Architecture` instance.

    Raises:
      OSError: When file not found at `filename`.
    """

    if not tf.io.gfile.exists(filename):
      raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    with tf.io.gfile.GFile(filename, "rb") as gfile:
      return _Architecture.deserialize(gfile.read().decode())

  def _find_ensemble_candidate(self, ensemble_candidate_name,
                               ensemble_candidates):
    """Returns the ensemble candidate with the given name."""

    for ensemble_candidate in ensemble_candidates:
      if ensemble_candidate.name == ensemble_candidate_name:
        return ensemble_candidate
    raise ValueError(
        "Could not find a matching ensemble candidate with name '{}'. "
        "Are you sure the `adanet.ensemble.Strategy` is deterministic?".format(
            ensemble_candidate_name))

  # TODO: Refactor architecture building logic to its own module.
  def _architecture_ensemble_spec(self, architecture, iteration_number,
                                  features, mode, labels,
                                  previous_ensemble_spec, config):
    """Returns an `_EnsembleSpec` with the given architecture.

    Creates the ensemble architecture by calling `generate_subnetworks` on
    `self._subnetwork_generator` and only calling `build_subnetwork` on
    `Builders` included in the architecture. Once their ops are created, their
    variables are restored from the checkpoint.

    Args:
      architecture: An `_Architecture` instance.
      iteration_number: Integer current iteration number.
      features: Dictionary of `Tensor` objects keyed by feature name.
      mode: Defines whether this is training, evaluation or prediction. See
        `ModeKeys`.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head). Can be `None`.
      previous_ensemble_spec: The `_EnsembleSpec` for the previous iteration.
        Will be `None` for the first iteration.
      config: The current `tf.estimator.RunConfig`.

    Returns:
      An `EnsembleSpec` instance for the given architecture.

    Raises:
      ValueError: If a subnetwork from `architecture` is not found in the
        generated candidate `Builders` of the specified iteration.
    """

    previous_ensemble = None
    if previous_ensemble_spec:
      previous_ensemble = previous_ensemble_spec.ensemble
    current_iteration = None
    for t, names in architecture.subnetworks_grouped_by_iteration:
      if t != iteration_number:
        continue
      previous_ensemble_reports, all_reports = [], []
      if self._report_materializer:
        previous_ensemble_reports, all_reports = (
            self._collate_subnetwork_reports(iteration_number))
      generated_subnetwork_builders = (
          self._call_generate_candidates(
              previous_ensemble=previous_ensemble,
              iteration_number=iteration_number,
              previous_ensemble_reports=previous_ensemble_reports,
              all_reports=all_reports,
              config=config))
      subnetwork_builder_names = {
          b.name: b for b in generated_subnetwork_builders
      }
      rebuild_subnetwork_builders = []
      for name in names:
        if name not in subnetwork_builder_names:
          raise ValueError(
              "Required subnetwork builder is missing for iteration {}: {}"
              .format(iteration_number, name))
        rebuild_subnetwork_builders.append(subnetwork_builder_names[name])
      previous_ensemble_summary = None
      previous_ensemble_subnetwork_builders = None
      if previous_ensemble_spec:
        # Always skip summaries when rebuilding previous architecture,
        # since they are not useful.
        previous_ensemble_summary = self._summary_maker(
            namespace="ensemble",
            scope=previous_ensemble_spec.name,
            skip_summary=True)
        previous_ensemble_subnetwork_builders = (
            previous_ensemble_spec.subnetwork_builders)
      ensemble_candidates = []
      for ensemble_strategy in self._ensemble_strategies:
        ensemble_candidates += ensemble_strategy.generate_ensemble_candidates(
            rebuild_subnetwork_builders, previous_ensemble_subnetwork_builders)
      ensemble_candidate = self._find_ensemble_candidate(
          architecture.ensemble_candidate_name, ensemble_candidates)
      current_iteration = self._iteration_builder.build_iteration(
          base_global_step=architecture.global_step,
          iteration_number=iteration_number,
          ensemble_candidates=[ensemble_candidate],
          subnetwork_builders=rebuild_subnetwork_builders,
          features=features,
          labels=labels,
          mode=mode,
          config=config,
          previous_ensemble_summary=previous_ensemble_summary,
          previous_ensemble_spec=previous_ensemble_spec,
          rebuilding=True,
          rebuilding_ensembler_name=architecture.ensembler_name)
      max_candidates = 2 if previous_ensemble_spec else 1
      assert len(current_iteration.candidates) == max_candidates
      previous_ensemble_spec = current_iteration.candidates[-1].ensemble_spec
      previous_ensemble = previous_ensemble_spec.ensemble
    return previous_ensemble_spec

  def _collate_subnetwork_reports(self, iteration_number):
    """Prepares subnetwork.Reports to be passed to Generator.

    Reads subnetwork.MaterializedReports from past iterations,
    collates those that were included in previous_ensemble into
    previous_ensemble_reports as a List of subnetwork.MaterializedReports,
    and collates all reports from previous iterations into all_reports as
    another List of subnetwork.MaterializedReports.

    Args:
      iteration_number: Python integer AdaNet iteration number, starting from 0.

    Returns:
      (previous_ensemble_reports: List<subnetwork.MaterializedReport>,
       materialized_reports: List<MaterializedReport>)
    """

    materialized_reports_all = (self._report_accessor.read_iteration_reports())
    previous_ensemble_reports = []
    all_reports = []

    # Since the number of iteration reports changes after the
    # MATERIALIZE_REPORT phase, we need to make sure that we always pass the
    # same reports to the Generator in the same iteration,
    # otherwise the graph that is built in the FREEZE_ENSEMBLE phase would be
    # different from the graph built in the training phase.

    # Iteration 0 should have 0 iteration reports passed to the
    #   Generator, since there are no previous iterations.
    # Iteration 1 should have 1 list of reports for Builders
    #   generated in iteration 0.
    # Iteration 2 should have 2 lists of reports -- one for iteration 0,
    #   one for iteration 1. Note that the list of reports for iteration >= 1
    #   should contain "previous_ensemble", in addition to the
    #   Builders at the start of that iteration.
    # Iteration t should have t lists of reports.

    for i, iteration_reports in enumerate(materialized_reports_all):

      # This ensures that the FREEZE_ENSEMBLE phase does not pass the reports
      # generated in the previous phase of the same iteration to the
      # Generator when building the graph.
      if i >= iteration_number:
        break

      chosen_subnetworks_in_this_iteration = [
          subnetwork_report for subnetwork_report in iteration_reports
          if subnetwork_report.included_in_final_ensemble
      ]
      previous_ensemble_reports += chosen_subnetworks_in_this_iteration
      all_reports.extend(iteration_reports)

    return previous_ensemble_reports, all_reports

  def _train_op(self, iteration_estimator_spec, is_growing_phase):
    """Returns the iteration train op or tf.no_op if growing the graph."""

    train_op = iteration_estimator_spec.train_op
    if is_growing_phase:
      train_op = tf_compat.v1.train.get_global_step().assign_add(1)
      # NOTE: some version of TensorFlow check that train_op is an Op or Tensor
      # and crash if train_op is a Variable.
      train_op = tf.identity(train_op)
    return train_op

  def _create_estimator_spec(self, current_iteration, mode,
                             iteration_number_tensor, previous_iteration_vars,
                             is_growing_phase, evaluation_name):
    """Creates the EstimatorSpec which will be returned by _adanet_model_fn."""

    training = mode == tf.estimator.ModeKeys.TRAIN
    iteration_estimator_spec = current_iteration.estimator_spec
    training_chief_hooks = self._training_chief_hooks(current_iteration,
                                                      training)
    training_hooks = self._training_hooks(current_iteration, training,
                                          iteration_number_tensor,
                                          previous_iteration_vars,
                                          is_growing_phase)
    if is_growing_phase:
      training_chief_hooks = self._process_hooks_for_growing_phase(
          training_chief_hooks)
      training_hooks = self._process_hooks_for_growing_phase(training_hooks)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=iteration_estimator_spec.predictions,
        loss=iteration_estimator_spec.loss,
        train_op=self._train_op(iteration_estimator_spec, is_growing_phase),
        eval_metric_ops=iteration_estimator_spec.eval_metric_ops,
        training_chief_hooks=training_chief_hooks,
        training_hooks=training_hooks,
        evaluation_hooks=self._evaluation_hooks(current_iteration, training,
                                                evaluation_name),
        scaffold=tf_compat.v1.train.Scaffold(summary_op=tf.constant("")),
        export_outputs=iteration_estimator_spec.export_outputs)

  def _call_generate_candidates(self, previous_ensemble, iteration_number,
                                previous_ensemble_reports, all_reports, config):
    defined_args = inspect.getargspec(
        self._subnetwork_generator.generate_candidates).args
    generate_args = dict(
        previous_ensemble=previous_ensemble,
        iteration_number=iteration_number,
        previous_ensemble_reports=previous_ensemble_reports,
        all_reports=all_reports)
    if "config" in defined_args:
      generate_args["config"] = config
    return self._subnetwork_generator.generate_candidates(**generate_args)

  def _create_iteration(self,
                        features,
                        labels,
                        mode,
                        config,
                        is_growing_phase,
                        evaluation_checkpoint_path=None,
                        best_ensemble_index_override=None):
    """Constructs the TF ops and variables for the current iteration.

    Args:
      features: Dictionary of `Tensor` objects keyed by feature name.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head). Can be `None`.
      mode: Defines whether this is training, evaluation or prediction. See
        `ModeKeys`.
      config: The current `tf.estimator.RunConfig`.
      is_growing_phase: Whether we are in the AdaNet graph growing phase.
      evaluation_checkpoint_path: Path of the evaluation checkpoint to use. When
        `None`, this method uses the latest checkpoint instead.
      best_ensemble_index_override: Integer index to identify the latest
        iteration's best ensemble candidate instead of computing the best
        ensemble index dynamically conditional on the ensemble AdaNet losses.

    Returns:
      A two-tuple of the current `_Iteration`, and list of variables from
        the previous iteration for restoring during the graph growing phase.
    """

    iteration_number = self._latest_checkpoint_iteration_number()

    # Use the evaluation checkpoint path to get both the iteration number and
    # variable values to avoid any race conditions between the first and second
    # checkpoint reads.
    if mode == tf.estimator.ModeKeys.EVAL and evaluation_checkpoint_path:
      iteration_number = tf.train.load_variable(evaluation_checkpoint_path,
                                                self._Keys.CURRENT_ITERATION)

    if is_growing_phase:
      assert mode == tf.estimator.ModeKeys.TRAIN
      assert config.is_chief
      iteration_number += 1

    # Only record summaries when training.
    skip_summaries = (mode != tf.estimator.ModeKeys.TRAIN or is_growing_phase)
    base_global_step = 0
    with tf_compat.v1.variable_scope("adanet"):
      previous_ensemble_spec = None
      previous_ensemble = None
      previous_ensemble_summary = None
      previous_ensemble_subnetwork_builders = None
      architecture = None
      for i in range(iteration_number):
        architecture_filename = self._architecture_filename(i)
        if not tf.io.gfile.exists(architecture_filename):
          continue
        architecture = self._read_architecture(architecture_filename)
        logging.info(
            "Importing architecture from %s: [%s].", architecture_filename,
            ", ".join(
                sorted([
                    "'{}:{}'".format(t, n)
                    for t, n in architecture.subnetworks_grouped_by_iteration
                ])))
        base_global_step = architecture.global_step
        previous_ensemble_spec = self._architecture_ensemble_spec(
            architecture, i, features, mode, labels, previous_ensemble_spec,
            config)
        previous_ensemble = previous_ensemble_spec.ensemble
        previous_ensemble_summary = self._summary_maker(
            namespace="ensemble",
            scope=previous_ensemble_spec.name,
            skip_summary=skip_summaries)
        previous_ensemble_subnetwork_builders = (
            previous_ensemble_spec.subnetwork_builders)
      previous_iteration_vars = None
      if is_growing_phase:
        # Keep track of the previous iteration variables so we can restore them
        # from the previous checkpoint after growing the graph. After this line,
        # any variables created will not have a matching one in the checkpoint
        # until it gets overwritten.
        # Note: It's not possible to just create a tf.train.Saver here since
        # this code is also run on TPU, which does not support creating Savers
        # inside model_fn.
        previous_iteration_vars = (
            tf_compat.v1.get_collection(tf_compat.v1.GraphKeys.GLOBAL_VARIABLES)
            + tf_compat.v1.get_collection(
                tf_compat.v1.GraphKeys.SAVEABLE_OBJECTS))
      previous_ensemble_reports, all_reports = [], []
      if self._report_materializer:
        previous_ensemble_reports, all_reports = (
            self._collate_subnetwork_reports(iteration_number))

      subnetwork_builders = self._call_generate_candidates(
          previous_ensemble=previous_ensemble,
          iteration_number=iteration_number,
          previous_ensemble_reports=previous_ensemble_reports,
          all_reports=all_reports,
          config=config)
      ensemble_candidates = []
      for ensemble_strategy in self._ensemble_strategies:
        ensemble_candidates += ensemble_strategy.generate_ensemble_candidates(
            subnetwork_builders, previous_ensemble_subnetwork_builders)
      current_iteration = self._iteration_builder.build_iteration(
          base_global_step=base_global_step,
          iteration_number=iteration_number,
          ensemble_candidates=ensemble_candidates,
          subnetwork_builders=subnetwork_builders,
          features=features,
          labels=labels,
          mode=mode,
          config=config,
          previous_ensemble_summary=previous_ensemble_summary,
          previous_ensemble_spec=previous_ensemble_spec,
          best_ensemble_index_override=best_ensemble_index_override)

    return current_iteration, previous_iteration_vars

  def _adanet_model_fn(self, features, labels, mode, params, config):
    """AdaNet model_fn.

    Args:
      features: Dictionary of `Tensor` objects keyed by feature name.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head). Can be `None`.
      mode: Defines whether this is training, evaluation or prediction. See
        `ModeKeys`.
      params: A dict of parameters.
      config: The current `tf.estimator.RunConfig`.

    Returns:
      A `EstimatorSpec` instance.

    Raises:
      UserWarning: When calling model_fn directly in TRAIN mode.
    """

    # Unpack params.
    is_growing_phase = params.get("is_growing_phase", False)
    is_inside_training_loop = params.get("is_inside_training_loop", False)
    evaluation_checkpoint_path = params.get("evaluation_checkpoint_path", None)
    evaluation_name = params.get("evaluation_name", None)
    best_ensemble_index = params.get("best_ensemble_index", None)

    training = mode == tf.estimator.ModeKeys.TRAIN
    if training and not is_inside_training_loop:
      raise UserWarning(
          "The adanet.Estimator's model_fn should not be called directly in "
          "TRAIN mode, because its behavior is undefined outside the context "
          "of its `train` method. If you are trying to add custom metrics "
          "with `tf.contrib.estimator.add_metrics`, pass the `metric_fn` to "
          "this `Estimator's` constructor instead.")

    current_iteration, previous_iteration_vars = self._create_iteration(
        features,
        labels,
        mode,
        config,
        is_growing_phase,
        evaluation_checkpoint_path=evaluation_checkpoint_path,
        best_ensemble_index_override=best_ensemble_index)

    # Variable which allows us to read the current iteration from a checkpoint.
    # This must be created here so it is available when calling
    # _execute_bookkeeping_phase after the first iteration.
    iteration_number_tensor = tf_compat.v1.get_variable(
        self._Keys.CURRENT_ITERATION,
        shape=[],
        dtype=tf.int64,
        initializer=tf_compat.v1.zeros_initializer(),
        trainable=False)

    return self._create_estimator_spec(
        current_iteration,
        mode,
        iteration_number_tensor,
        previous_iteration_vars,
        is_growing_phase,
        evaluation_name=evaluation_name)
