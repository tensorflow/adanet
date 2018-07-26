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

import errno
import os
import time

from adanet.candidate import _CandidateBuilder
from adanet.ensemble import _EnsembleBuilder
from adanet.ensemble import MixtureWeightType
from adanet.freezer import _EnsembleFreezer
from adanet.input_utils import make_placeholder_input_fn
from adanet.iteration import _IterationBuilder
from adanet.report_accessor import _ReportAccessor
from adanet.summary import _ScopedSummary
from adanet.timer import _CountDownTimer
import numpy as np
import six
import tensorflow as tf


class _StopAfterTrainingHook(tf.train.SessionRunHook):
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

    del run_context  # Unused
    return tf.train.SessionRunArgs(self._iteration.is_over)

  def after_run(self, run_context, run_values):
    """See `SessionRunHook`."""

    is_over = run_values.results
    if not is_over:
      return
    run_context.request_stop()
    self._after_fn()


class _EvalMetricSaverHook(tf.train.SessionRunHook):
  """A hook for writing evaluation metrics as summaries to disk."""

  def __init__(self, name, eval_metric_ops, output_dir):
    """Initializes a `_EvalMetricSaverHook` instance.

    Args:
      name: String name of candidate owner of these metrics.
      eval_metric_ops: Dict of metric results keyed by name. The values of the
        dict are the results of calling a metric function, namely a
        `(metric_tensor, update_op)` tuple. `metric_tensor` should be evaluated
        without any impact on state (typically is a pure computation based on
        variables.). For example, it should not trigger the `update_op` or
        require any input fetching.
      output_dir: Directory for writing evaluation summaries.

    Returns:
      An `_EvalMetricSaverHook` instance.
    """

    self._name = name
    self._eval_metric_ops = eval_metric_ops
    self._output_dir = output_dir

  def before_run(self, run_context):
    """See `SessionRunHook`."""

    del run_context  # Unused
    return tf.train.SessionRunArgs(self._eval_metric_ops)

  def _dict_to_str(self, dictionary):
    """Get a `str` representation of a `dict`.

    Args:
      dictionary: The `dict` to be represented as `str`.

    Returns:
      A `str` representing the `dictionary`.
    """
    return ", ".join("%s = %s" % (k, v) for k, v in sorted(dictionary.items()))

  def end(self, session):
    """See `SessionRunHook`."""

    # Forked from tensorflow/python/estimator/estimator.py function called
    # _write_dict_to_summary.
    eval_dict = {}
    for key, metric in self._eval_metric_ops.items():
      eval_dict[key] = metric[0]
    current_global_step = tf.train.get_global_step()

    eval_dict, current_global_step = session.run((eval_dict,
                                                  current_global_step))

    tf.logging.info("Saving candidate '%s' dict for global step %d: %s",
                    self._name, current_global_step,
                    self._dict_to_str(eval_dict))
    summary_writer = tf.summary.FileWriterCache.get(self._output_dir)
    summary_proto = tf.summary.Summary()
    for key in eval_dict:
      value = eval_dict[key]
      if isinstance(value, np.float32) or isinstance(value, float):
        summary_proto.value.add(tag=key, simple_value=float(value))
      elif isinstance(value, six.binary_type):
        summ = tf.summary.Summary.FromString(value)
        for i, _ in enumerate(summ.value):
          summ.value[i].tag = "%s/%d" % (key, i)
        summary_proto.value.extend(summ.value)
      else:
        tf.logging.warn(
            "Skipping summary for %s, must be a float, np.float32, "
            "or a serialized string of Summary.", key)
    summary_writer.add_summary(summary_proto, current_global_step)
    summary_writer.flush()


class Estimator(tf.estimator.Estimator):
  """The AdaNet algorithm implemented with the `tf.estimator.Estimator` API.

  AdaNet is as defined in the paper: https://arxiv.org/abs/1607.01097.

  The AdaNet algorithm uses a base learning algorithm to iteratively generate a
  set of candidate base learners to add to an ensemble that minimizes the loss
  function defined in Equation (4). At the end of each iteration, the best
  ensemble is chosen based on its complexity-regularized train loss. New
  base learners are allowed to use any base learner weights within the previous
  iteration's ensemble in order to improve upon them. If the complexity-
  regularized loss of the new ensemble, as defined in Equation (4), is less than
  that of the previous iteration's ensemble, the AdaNet algorithm continues onto
  the next iteration.

  AdaNet attempts to minimize the following loss function to learn the mixture
  weights 'w' of each base learner 'h' in the ensemble with differentiable
  convex non-increasing surrogate loss function Phi:

  Equation (4):

    F(w) = 1/m * sum_{i = 1 to m}(Phi(1 - y_i * sum_{j = 1 to N}(w_j * h_j)))
           + sum_{j = 1 to N}(Gamma_j * |w_j|)

    where Gamma_j = lambda * r_j + beta, with lambda >= 0 and beta >= 0.

  This implementation trains candidate subnetworks in parallel using a single
  graph per iteration. At the end of each iteration, the estimator saves the
  sub-graph of the best subnetwork ensemble and its weights as a separate
  checkpoint. At the beginning of the next iteration, the estimator imports
  the previous iteration's frozen graph and adds ops for the next candidates
  as part of a new graph and session. This allows the estimator have the
  performance of Tensorflow's static graph constraint (minus the performance
  hit of reconstructing a graph between iterations), while having the
  flexibility of having a dynamic graph.

  NOTE: Subclassing `tf.estimator.Estimator` is only necessary to work with
  `tf.estimator.train_and_evaluate` which asserts that the estimator argument is
  a `tf.estimator.Estimator` subclass. However, all training is delegated to a
  separate `tf.estimator.Estimator` instance. It is responsible for supporting
  both local and distributed training. As such, the AdaNet `Estimator` is only
  responsible for bookkeeping across iterations.
  """

  class _Keys(object):
    CURRENT_ITERATION = "current_iteration"
    EVALUATE_ENSEMBLES = "evaluate_ensembles"
    MATERIALIZE_REPORT = "materialize_report"
    FREEZE_ENSEMBLE = "freeze_ensemble"
    FROZEN_ENSEMBLE_NAME = "previous_ensemble"
    INCREMENT_ITERATION = "increment_iteration"
    BASE_LEARNER_BUILDER_GENERATOR = "base_learner_builder_generator"

  def __init__(self,
               head,
               base_learner_builder_generator,
               max_iteration_steps,
               mixture_weight_type=MixtureWeightType.MATRIX,
               mixture_weight_initializer=None,
               warm_start_mixture_weights=False,
               adanet_lambda=0.,
               adanet_beta=0.,
               evaluator=None,
               report_materializer=None,
               use_bias=True,
               replicate_ensemble_in_training=False,
               adanet_loss_decay=.9,
               worker_wait_timeout_secs=7200,
               model_dir=None,
               report_dir=None,
               config=None):
    """Initializes an `Estimator`.

    Args:
      head: A `tf.contrib.estimator.Head` instance for computing loss and
        evaluation metrics for every candidate.
      base_learner_builder_generator: The `adanet.BaseLearnerBuilderGenerator`
        which defines the candidate base learners to train and evaluate at every
        AdaNet iteration.
      max_iteration_steps: Total number of steps for which to train candidates
        per iteration. If `OutOfRange` or `StopIteration` occurs in the middle,
        training stops before `max_iteration_steps` steps.
      mixture_weight_type: The `adanet.MixtureWeightType` defining which mixture
        weight type to learn in the linear combination of base learner outputs.

        A `SCALAR` mixture weight is a rank 0 tensor. It performs an element-
        wise multiplication with its base learner's logits. This mixture weight
        is the simplest to learn, the quickest to train, and most likely to
        generalize well.

        A `VECTOR` mixture weight is a tensor of shape [k] where k is the
        ensemble's logits dimension as defined by `head`. It is similar to
        `SCALAR` in that it performs an element-wise multiplication with its
        base learner's logits, but is more flexible in learning a base
        learner's preferences per class.

        A `MATRIX` mixture weight is a tensor of shape [a, b] where a is the
        number of outputs from the base learner's `last_layer` and b is the
        number of outputs from the ensemble's `logits`. This weight
        matrix-multiplies the base learner's `last_layer`. This mixture weight
        offers the most flexibility and expressivity, allowing base learners to
        have outputs of different dimensionalities. However, it also has the
        most trainable parameters (a*b), and is therefore the most sensitive to
        learning rates and regularization.
      mixture_weight_initializer: The initializer for mixture_weights. When
        `None`, the default is different according to `mixture_weight_type`:
        * `SCALAR`: Initializes to 1/N where N is the number of base learners
          in the ensemble giving a uniform average.
        * `VECTOR`: Initializes each entry to 1/N where N is the number of base
          learners in the ensemble giving a uniform average.
        * `MATRIX`: Uses `tf.glorot_uniform_initializer`.
      warm_start_mixture_weights: Whether, at the beginning of an iteration, to
        initialize the mixture weights of the base learners from the previous
        ensemble to their learned value at the previous iteration, as opposed to
        retraining them from scratch. Takes precedence over the value for
        `mixture_weight_initializer` for base learners from previous iterations.
      adanet_lambda: Float multiplier 'lambda' for applying L1 regularization to
        base learners' mixture weights 'w' in the ensemble proportional to their
        complexity. See Equation (4) in the AdaNet paper.
      adanet_beta: Float L1 regularization multiplier 'beta' to apply equally to
        all base learners' weights 'w' in the ensemble regardless of their
        complexity. See Equation (4) in the AdaNet paper.
      evaluator: An `Evaluator` for comparing `Ensemble` instances in evaluation
        mode using the training set, or a holdout set. When `None`, they are
        compared using a moving average of their `Ensemble`'s AdaNet loss during
        training.
      report_materializer: A `ReportMaterializer` for materializing the
        `BaseLearnerReport`s obtained from the `BaseLearnerBuilder`s' in each
        AdaNet iteration in to `MaterializedBaseLearnerReport`s. When `None`,
        does not materialize any `BaseLearnerReport`s.
      use_bias: Whether to add a bias term to the ensemble's logits. Adding a
        bias allows the ensemble to learn a shift in the data, often leading to
        more stable training and better predictions.
      replicate_ensemble_in_training: Whether to freeze a copy of the ensembled
        base learners' subgraphs in training mode in addition to prediction
        mode. A copy of the base learners' subgraphs is always saved in
        prediction mode so that at prediction time, the ensemble and composing
        base learners are all in prediction mode. This argument only affects the
        outputs of the frozen base learners in the ensemble. When `False` and
        during candidate training, the frozen base learners in the ensemble are
        in prediction mode, so training-only ops like dropout are not applied to
        them. When `True` and training the candidates, the frozen base learners
        will be in training mode as well, so they will apply training-only ops
        like dropout. However when `True`, this doubles the amount of disk space
        required to store the frozen ensembles, and increases the preparation
        stage between boosting iterations. This argument is useful for
        regularizing learning mixture weights, or for making training-only side
        inputs available in subsequent iterations. For most use-cases, this
        should be `False`.
      adanet_loss_decay: Float decay for the exponential-moving-average of the
        AdaNet objective throughout training. This moving average is a data-
        driven way tracking the best candidate with only the training set.
      worker_wait_timeout_secs: Float number of seconds for workers to wait for
        chief to prepare the next iteration during distributed training. This is
        needed to prevent workers waiting indefinitely for a chief that may have
        crashed or been turned down. When the timeout is exceeded, the worker
        exits the train loop. In situations where the chief job is much slower
        than the worker jobs, this timeout should be increased.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      report_dir: Directory where the `MaterializedBaseLearnerReport`s
        materialized by `report_materializer` would be saved.
        If `report_materializer` is None, this will not save
        anything. If `None` or empty string, defaults to "<model_dir>/report".
      config: `RunConfig` object to configure the runtime settings.

    Returns:
      An `Estimator` instance.

    Raises:
      ValueError: If `base_learner_builder_generator` is `None`.
      ValueError: If `max_iteration_steps` is <= 0.
    """

    # TODO: Add argument to specify how many frozen graph
    # checkpoints to keep.

    if base_learner_builder_generator is None:
      raise ValueError("base_learner_builder_generator can't be None.")
    if max_iteration_steps <= 0.:
      raise ValueError("max_iteration_steps must be > 0.")

    self._adanet_loss_decay = adanet_loss_decay

    # Overwrite superclass's assert that members are not overwritten in order
    # to overwrite public methods. Note that we are doing something that is not
    # explicitly supported by the Estimator API and may break in the future.
    tf.estimator.Estimator._assert_members_are_not_overridden = staticmethod(
        lambda _: None)

    self._ensemble_builder = _EnsembleBuilder(
        head=head,
        mixture_weight_type=mixture_weight_type,
        mixture_weight_initializer=mixture_weight_initializer,
        warm_start_mixture_weights=warm_start_mixture_weights,
        adanet_lambda=adanet_lambda,
        adanet_beta=adanet_beta,
        use_bias=use_bias)
    candidate_builder = _CandidateBuilder(
        max_steps=max_iteration_steps,
        adanet_loss_decay=self._adanet_loss_decay)
    self._iteration_builder = _IterationBuilder(candidate_builder,
                                                self._ensemble_builder)
    self._freezer = _EnsembleFreezer()
    self._evaluation_checkpoint_path = None
    self._evaluator = evaluator
    self._report_materializer = report_materializer

    self._replicate_ensemble_in_training = replicate_ensemble_in_training
    self._worker_wait_timeout_secs = worker_wait_timeout_secs

    self._evaluation_name = None

    # This `Estimator` is responsible for bookkeeping across iterations, and
    # for training the base learners in both a local and distributed setting.
    # Subclassing improves future-proofing against new private methods being
    # added to `tf.estimator.Estimator` that are expected to be callable by
    # external functions, such as in b/110435640.
    super(Estimator, self).__init__(
        model_fn=self._model_fn,
        params={
            self._Keys.BASE_LEARNER_BUILDER_GENERATOR:
                base_learner_builder_generator,
        },
        config=config,
        model_dir=model_dir)

    # This is defined after base Estimator's init so that report_accessor can
    # use the same temporary model_dir as the underlying Estimator even if
    # model_dir is not provided.
    report_dir = report_dir or os.path.join(self._model_dir, "report")
    self._report_accessor = _ReportAccessor(report_dir)

  def _latest_checkpoint_iteration_number(self):
    """Returns the iteration number from the latest checkpoint."""

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if latest_checkpoint is None:
      return 0
    return tf.contrib.framework.load_variable(latest_checkpoint,
                                              self._Keys.CURRENT_ITERATION)

  def _latest_checkpoint_global_step(self):
    """Returns the global step from the latest checkpoint."""

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if latest_checkpoint is None:
      return 0
    return tf.contrib.framework.load_variable(latest_checkpoint,
                                              tf.GraphKeys.GLOBAL_STEP)

  def train(self,
            input_fn,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None):
    """See `tf.estimator.Estimator` train."""

    if (steps is not None) and (max_steps is not None):
      raise ValueError("Can not provide both steps and max_steps.")
    if steps is not None and steps <= 0:
      raise ValueError("Must specify steps > 0, given: {}".format(steps))

    if steps is not None:
      max_steps = self._latest_checkpoint_global_step() + steps

    # Each iteration of this AdaNet loop represents an `_Iteration`. The
    # current iteration number is stored as a variable in the checkpoint so
    # that training can be stopped and started at anytime.
    while True:
      current_iteration = self._latest_checkpoint_iteration_number()
      tf.logging.info("Beginning training AdaNet iteration %s",
                      current_iteration)
      self._iteration_ended = False
      result = super(Estimator, self).train(
          input_fn=input_fn,
          hooks=hooks,
          max_steps=max_steps,
          saving_listeners=saving_listeners)

      # If training ended because the maximum number of training steps occurred,
      # exit training.
      if self._latest_checkpoint_global_step() >= max_steps:
        return result

      # If training ended for any reason other than the iteration ending,
      # exit training.
      if not self._iteration_ended:
        return result

      # The chief prepares the next AdaNet iteration, and increments the
      # iteration number by 1.
      if self.config.is_chief:
        # As the chief, store the train hooks and make a placeholder input_fn in
        # order to use them when preparing the next iteration.
        self._train_hooks = hooks
        self._placeholder_input_fn = make_placeholder_input_fn(input_fn)
        self._prepare_next_iteration()

      # This inner loop serves mainly for synchronizing the workers with the
      # chief during distributed training. Workers that finish training early
      # wait for the chief to prepare the next iteration and increment the
      # iteration number. Workers that are slow to finish training quickly move
      # onto the next iteration. And workers that go offline and return online
      # after training ended terminate gracefully.
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
          tf.logging.error(
              "Chief job did not prepare next iteration after %s secs. It may "
              "have been preempted, been turned down, or crashed. This worker "
              "is now exiting training.", self._worker_wait_timeout_secs)
          return result
        tf.logging.info("Waiting for chief to finish")
        time.sleep(5)

      tf.logging.info("Finished training Adanet iteration %s",
                      current_iteration)

      # Stagger starting workers to prevent training instability.
      if not self.config.is_chief:
        task_id = self.config.task_id or 0
        # Wait 5 secs more for each new worker up to 60 secs.
        delay_secs = min(60, task_id * 5)
        tf.logging.info("Waiting %d secs before starting training.", delay_secs)
        time.sleep(delay_secs)

  def evaluate(self,
               input_fn,
               steps=None,
               hooks=None,
               checkpoint_path=None,
               name=None):
    """See `tf.estimator.Estimator` evaluate."""

    if not checkpoint_path:
      checkpoint_path = tf.train.latest_checkpoint(self.model_dir)

    # Ensure that the read to get the iteration number and read to restore
    # variable values come from the same checkpoint during evaluation.
    self._evaluation_checkpoint_path = checkpoint_path
    self._evaluation_name = name
    result = super(Estimator, self).evaluate(
        input_fn,
        steps=steps,
        hooks=hooks,
        checkpoint_path=checkpoint_path,
        name=name)
    self._evaluation_checkpoint_path = None
    return result

  def _call_adanet_model_fn(self, input_fn, mode, params):
    """Calls model_fn with the given mode and parameters."""

    with tf.Graph().as_default():
      # Create global step before calling model_fn as does superclass.
      tf.train.get_or_create_global_step()
      features, labels = input_fn()
      self._model_fn(features, labels, mode, params)

  def _prepare_next_iteration(self):
    """Prepares the next iteration.

    This method calls model_fn up to four times:
      1. To evaluate all candidate ensembles to find the best one.
      2. To materialize reports and store them to disk (if report_materializer
         exists).
      3. To freeze the best ensemble's subgraph.
      4. To overwrite the model directory's checkpoint with the next iteration's
         ops.
    """

    # First, evaluate and choose the best ensemble for this iteration.
    params = self.params.copy()
    params[self._Keys.EVALUATE_ENSEMBLES] = True
    if self._evaluator:
      input_fn = self._evaluator.input_fn
    else:
      input_fn = self._placeholder_input_fn
    self._call_adanet_model_fn(input_fn, tf.estimator.ModeKeys.EVAL, params)

    # Then, if report_materializer exists, materialize and store the base
    # learner reports.
    if self._report_materializer:
      params = self.params.copy()
      params[self._Keys.MATERIALIZE_REPORT] = True
      self._call_adanet_model_fn(self._report_materializer.input_fn,
                                 tf.estimator.ModeKeys.EVAL, params)

    # Then freeze the best ensemble's graph in predict mode.
    params = self.params.copy()
    params[self._Keys.FREEZE_ENSEMBLE] = True
    self._call_adanet_model_fn(self._placeholder_input_fn,
                               tf.estimator.ModeKeys.PREDICT, params)
    if self._replicate_ensemble_in_training:
      self._call_adanet_model_fn(self._placeholder_input_fn,
                                 tf.estimator.ModeKeys.TRAIN, params)
    self._best_ensemble_index = None

    # Finally, create the graph for the next iteration and overwrite the model
    # directory checkpoint with the expanded graph.
    params = self.params.copy()
    params[self._Keys.INCREMENT_ITERATION] = True
    self._call_adanet_model_fn(self._placeholder_input_fn,
                               tf.estimator.ModeKeys.TRAIN, params)

  def _frozen_graph_filename(self, iteration_number, training):
    """Returns the filename of the given iteration's frozen graph."""

    frozen_checkpoint = os.path.join(self.model_dir, "frozen/ensemble")
    mode = "-train" if self._replicate_ensemble_in_training and training else ""
    return "{}-{}{}.meta".format(frozen_checkpoint, iteration_number, mode)

  def _overwrite_checkpoint(self, iteration_number_tensor, iteration_number):
    """Overwrites the latest checkpoint with the current graph.

    Before overwriting the checkpoint, it assigns the iteration number to the
    variable that stores that information in the checkpoint.

    Args:
      iteration_number_tensor: Int variable `Tensor` storing the current
        iteration number.
      iteration_number: Int number of the current iteration.
    """

    checkpoint_state = tf.train.get_checkpoint_state(self.model_dir)
    latest_checkpoint = checkpoint_state.model_checkpoint_path
    if not latest_checkpoint:
      return

    # Run train hook 'begin' methods which can add ops to the graph, so that
    # they are still present in the overwritten checkpoint.
    if self._train_hooks:
      for hook in self._train_hooks:
        hook.begin()

    global_step_tensor = tf.train.get_global_step()
    global_step = tf.contrib.framework.load_variable(latest_checkpoint,
                                                     tf.GraphKeys.GLOBAL_STEP)

    checkpoint_path = os.path.join(self.model_dir, "increment.ckpt")
    with tf.Session() as sess:
      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer(), tf.tables_initializer())
      sess.run(init)
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      control_deps = [
          tf.assign(global_step_tensor, global_step),
          tf.assign(iteration_number_tensor, iteration_number),
      ]
      with tf.control_dependencies(control_deps):
        saver = tf.train.Saver(
            sharded=True, max_to_keep=self.config.keep_checkpoint_max)
        saver.recover_last_checkpoints(
            checkpoint_state.all_model_checkpoint_paths)
        saver.save(sess, checkpoint_path, global_step=iteration_number)
      if self._train_hooks:
        for hook in self._train_hooks:
          hook.end(sess)

  def _freeze_ensemble(self, filename, current_iteration, features):
    """Freezes the given ensemble for the current iteration.

    Args:
      filename: String destination path for the frozen ensemble.
      current_iteration: Current `_Iteration`.
      features: Dictionary of `Tensor` objects keyed by feature name.

    Returns:
      A `MetaGraphDef` proto of the frozen ensemble.
    """

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    with tf.Session() as sess:
      saver = tf.train.Saver(sharded=True)
      saver.restore(sess, latest_checkpoint)
      best_candidate_index = self._best_ensemble_index
      best_candidate = current_iteration.candidates[best_candidate_index]
      return self._freezer.freeze_ensemble(
          sess=sess,
          filename=filename,
          weighted_base_learners=best_candidate.ensemble.weighted_base_learners,
          bias=best_candidate.ensemble.bias,
          features=features)

  def _get_best_ensemble_index(self, current_iteration):
    """Returns the best candidate ensemble's index in this iteration.

    Evaluates the ensembles using an `Evaluator` when provided. Otherwise,
    it returns the index of the best candidate as defined by the `_Iteration`.

    Args:
      current_iteration: Current `_Iteration`.

    Returns:
      Index of the best ensemble in the iteration's list of `_Candidates`.
    """

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    tf.logging.info("Starting ensemble evaluation for iteration %s",
                    current_iteration.number)
    with tf.Session() as sess:
      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer(), tf.tables_initializer())
      sess.run(init)
      saver = tf.train.Saver(sharded=True)
      saver.restore(sess, latest_checkpoint)
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      if self._evaluator:
        new_learners = [c.ensemble for c in current_iteration.candidates]
        index = self._evaluator.best_ensemble_index(sess, new_learners)
      else:
        index = sess.run(current_iteration.best_candidate_index)
    best_candidate = current_iteration.candidates[index]
    tf.logging.info("Finished ensemble evaluation for iteration %s",
                    current_iteration.number)
    tf.logging.info("The best ensemble is '%s' at index %s",
                    best_candidate.ensemble.name, index)
    return index

  def _materialize_report(self, current_iteration):
    """Generates reports as defined by `BaseLearnerBuilder`s.

    Materializes the Tensors and metrics defined in the `BaseLearnerBuilder`s'
    `build_base_learner_report` method using `ReportMaterializer`, and stores
    them to disk using `_ReportAccessor`.

    Args:
      current_iteration: Current `_Iteration`.
    """

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    tf.logging.info("Starting metric logging for iteration %s",
                    current_iteration.number)

    included_base_learner_indices = [self._best_ensemble_index]
    with tf.Session() as sess:
      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer(), tf.tables_initializer())
      sess.run(init)
      saver = tf.train.Saver(sharded=True)
      saver.restore(sess, latest_checkpoint)
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      materialized_base_learner_reports = (
          self._report_materializer.materialize_base_learner_reports(
              sess, current_iteration.base_learner_reports,
              included_base_learner_indices))
      self._report_accessor.write_iteration_report(
          current_iteration.number, materialized_base_learner_reports)

    tf.logging.info("Finished saving base learner reports for iteration %s",
                    current_iteration.number)

  def _training_hooks(self, current_iteration, training):
    """Returns training hooks for this iteration.

    Args:
      current_iteration: Current `_Iteration`.
      training: Whether in training mode.

    Returns:
      A list of `tf.train.SessionRunHook` instances.
    """

    if not training:
      return []

    def after_fn():
      self._iteration_ended = True

    training_hooks = [
        _StopAfterTrainingHook(current_iteration, after_fn=after_fn)
    ]

    for summary in current_iteration.summaries:
      output_dir = self.model_dir
      if summary.scope:
        output_dir = os.path.join(output_dir, "candidate", summary.scope)
      summary_saver_hook = tf.train.SummarySaverHook(
          save_steps=self.config.save_summary_steps,
          output_dir=output_dir,
          summary_op=summary.merge_all())
      training_hooks.append(summary_saver_hook)
    return training_hooks

  def _evaluation_hooks(self, current_iteration):
    """Returns evaluation hooks for this iteration.

    Args:
      current_iteration: Current `_Iteration`.

    Returns:
      A list of `tf.train.SessionRunHook` instances.
    """

    evaluation_hooks = []
    for candidate in current_iteration.candidates:
      eval_subdir = "eval"
      if self._evaluation_name:
        eval_subdir = "eval_{}".format(self._evaluation_name)
      eval_metric_hook = _EvalMetricSaverHook(
          name=candidate.ensemble.name,
          eval_metric_ops=candidate.ensemble.eval_metric_ops,
          output_dir=os.path.join(self.model_dir, "candidate",
                                  candidate.ensemble.name, eval_subdir))
      evaluation_hooks.append(eval_metric_hook)
    return evaluation_hooks

  def _record_features(self, filename, features):
    """Records features to be kept in the ensemble's frozen graph.

    Attempting to import a graph_def with an input_map containing additional
    features raises an error. This method can be used in combination with
    `_filter_recorded_features` to prevent this from happening.

    Args:
      filename: String filename to persist recorded features.
      features: Dictionary of `Tensor` objects keyed by feature name.
    """

    if not self.config.is_chief:
      # Only the chief should record features.
      return

    # Make directories since model_dir may not have been created yet.
    tf.gfile.MakeDirs(self.model_dir)
    with tf.gfile.GFile(filename, "w") as record_file:
      record_file.write(os.linesep.join(features.keys()))

  def _filter_recorded_features(self, filename, features):
    """Filters features that are not in the frozen graph.

    Attempting to import a graph_def with an input_map containing additional
    features raises an error. This method can be used in combination with
    `_record_features` to prevent this from happening.

    Args:
      filename: String filename where features were recorded.
      features: Dictionary of `Tensor` objects keyed by feature name.

    Returns:
      A copy of `features` containing only entries matching those recorded by
      `_record_features` at `filename`.

    Raises:
      OSError: When file not found at `filename`.
    """

    if not tf.gfile.Exists(filename):
      raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    recorded_feature_names = set()
    with tf.gfile.GFile(filename, "r") as record_file:
      for line in record_file:
        feature_name = line.rstrip()
        recorded_feature_names.add(feature_name)
    extra_features = recorded_feature_names - set(features.keys())
    if extra_features:
      tf.logging.warning(
          "Features dict contains features absent from frozen graph: [{}].".
          format(", ".join(sorted(["'{}'".format(f) for f in extra_features]))))
    missing_features = set(features.keys()) - recorded_feature_names
    if missing_features:
      tf.logging.warning(
          "Features dict is missing features present in frozen graph: [{}].".
          format(", ".join(
              sorted(["'{}'".format(f) for f in missing_features]))))
    filtered_feature_names = recorded_feature_names & set(features.keys())
    return {key: features[key] for key in filtered_feature_names}

  def _model_fn(self, features, labels, mode, params):
    """AdaNet model_fn.

    This model_fn is expected to be called four times per iteration. The first
    call is performed in order to build and train an iteration. Once that
    iteration is over, the next two calls are freeze its best ensemble for
    training and evaluation. The final call is responsible for loading the
    frozen graph, to create new ops for the next iteration, and to overwrite the
    latest checkpoint with its graph and variables, so that first call of the
    next iteration has the right ops in the checkpoint.

    Args:
      features: Dictionary of `Tensor` objects keyed by feature name.
      labels: `Tensor` of labels.
      mode: Defines whether this is training, evaluation or prediction.
        See `ModeKeys`.
      params: A dict of parameters.
        The following hyperparameters are expected:
        * freeze_ensemble: Whether to freeze the latest checkpoint's
            best ensemble to a separate checkpoint for the following
            iteration to use.
        * increment_iteration: Whether to overwrite the current checkpoint with
            the next iteration's graph and initialized weights.

    Returns:
      A `EstimatorSpec` instance.
    """

    # Wrap features so that their ops always have the same names for when
    # freezing and loading ensembles.
    features = self._freezer.wrapped_features(features)

    iteration_number = self._latest_checkpoint_iteration_number()

    training = mode == tf.estimator.ModeKeys.TRAIN

    filtered_features = features
    record_filename = os.path.join(self.model_dir, "features")
    if iteration_number == 0 and training:
      self._record_features(record_filename, features)
    else:
      filtered_features = self._filter_recorded_features(
          record_filename, features)

    # Use the evaluation checkpoint path to get both the iteration number and
    # variable values to avoid any race conditions between the first and second
    # checkpoint reads.
    if mode == tf.estimator.ModeKeys.EVAL and self._evaluation_checkpoint_path:
      iteration_number = tf.contrib.framework.load_variable(
          self._evaluation_checkpoint_path, self._Keys.CURRENT_ITERATION)

    if self._Keys.INCREMENT_ITERATION in params:
      iteration_number += 1

    ensemble = (None, None)
    frozen_graph_filename = self._frozen_graph_filename(iteration_number - 1,
                                                        training)
    if tf.gfile.Exists(frozen_graph_filename):
      tf.logging.info(
          "Importing frozen ensemble from %s with features: [%s].",
          frozen_graph_filename, ", ".join(
              sorted(["'{}'".format(f) for f in filtered_features])))
      ensemble = self._freezer.load_frozen_ensemble(
          filename=frozen_graph_filename, features=filtered_features)

    builder_generator = params[self._Keys.BASE_LEARNER_BUILDER_GENERATOR]

    skip_summaries = mode == tf.estimator.ModeKeys.PREDICT
    previous_ensemble_summary = _ScopedSummary(self._Keys.FROZEN_ENSEMBLE_NAME,
                                               skip_summaries)
    with tf.variable_scope("adanet"):
      previous_weighted_base_learners, bias = ensemble
      previous_ensemble = None
      if previous_weighted_base_learners:
        with tf.variable_scope(self._Keys.FROZEN_ENSEMBLE_NAME):
          previous_ensemble = self._ensemble_builder.build_ensemble(
              name=self._Keys.FROZEN_ENSEMBLE_NAME,
              weighted_base_learners=previous_weighted_base_learners,
              summary=previous_ensemble_summary,
              bias=bias,
              features=features,
              mode=mode,
              labels=labels)
      base_learner_builders = builder_generator.generate_candidates(
          previous_ensemble=previous_ensemble)
      current_iteration = self._iteration_builder.build_iteration(
          iteration_number=iteration_number,
          base_learner_builders=base_learner_builders,
          features=features,
          labels=labels,
          mode=mode,
          previous_ensemble_summary=previous_ensemble_summary,
          previous_ensemble=previous_ensemble)

    # Variable which allows us to read the current iteration from a checkpoint.
    iteration_number_tensor = tf.get_variable(
        self._Keys.CURRENT_ITERATION,
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES])

    adanet_summary = _ScopedSummary("global", skip_summaries)
    adanet_summary.scalar("iteration/adanet/iteration", iteration_number_tensor)
    if current_iteration.estimator_spec.loss is not None:
      adanet_summary.scalar("loss", current_iteration.estimator_spec.loss)
      adanet_summary.scalar("loss/adanet/adanet_weighted_ensemble",
                            current_iteration.estimator_spec.loss)

    iteraton_estimator_spec = current_iteration.estimator_spec
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=iteraton_estimator_spec.predictions,
        loss=iteraton_estimator_spec.loss,
        train_op=iteraton_estimator_spec.train_op,
        eval_metric_ops=iteraton_estimator_spec.eval_metric_ops,
        training_hooks=self._training_hooks(current_iteration, is_training),
        evaluation_hooks=self._evaluation_hooks(current_iteration),
        scaffold=tf.train.Scaffold(summary_op=adanet_summary.merge_all()),
        export_outputs=iteraton_estimator_spec.export_outputs)

    if self._Keys.EVALUATE_ENSEMBLES in params:
      self._best_ensemble_index = self._get_best_ensemble_index(
          current_iteration)
    elif self._Keys.MATERIALIZE_REPORT in params:
      assert self._best_ensemble_index is not None
      self._materialize_report(current_iteration)
    elif self._Keys.FREEZE_ENSEMBLE in params:
      assert self._best_ensemble_index is not None
      new_frozen_graph_filename = self._frozen_graph_filename(
          iteration_number, training)
      tf.logging.info("Freezing best ensemble to %s", new_frozen_graph_filename)
      self._freeze_ensemble(
          filename=new_frozen_graph_filename,
          current_iteration=current_iteration,
          features=features)
    elif self._Keys.INCREMENT_ITERATION in params:
      latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
      tf.logging.info(
          "Overwriting checkpoint with new graph for iteration %s to %s",
          iteration_number, latest_checkpoint)
      self._overwrite_checkpoint(iteration_number_tensor, iteration_number)

    return estimator_spec
