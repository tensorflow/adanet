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
import os
import time

from adanet.core.candidate import _CandidateBuilder
from adanet.core.ensemble import _EnsembleBuilder
from adanet.core.ensemble import MixtureWeightType
from adanet.core.iteration import _IterationBuilder
from adanet.core.report_accessor import _ReportAccessor
from adanet.core.summary import _ScopedSummary
from adanet.core.timer import _CountDownTimer
import numpy as np
import six
import tensorflow as tf

from tensorflow.python.ops import resources


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
    return tf.train.SessionRunArgs(self._iteration.is_over_fn())

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
      if isinstance(value, (np.float32, float)):
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
  # pyformat: disable
  r"""The AdaNet algorithm implemented as a :class:`tf.estimator.Estimator`.

  AdaNet is as defined in the paper: https://arxiv.org/abs/1607.01097.

  The AdaNet algorithm uses a weak learning algorithm to iteratively generate a
  set of candidate subnetworks that attempt to minimize the loss function
  defined in Equation (4) as part of an ensemble. At the end of each iteration,
  the best candidate is chosen based on its ensemble's complexity-regularized
  train loss. New subnetworks are allowed to use any subnetwork weights within
  the previous iteration's ensemble in order to improve upon them. If the
  complexity-regularized loss of the new ensemble, as defined in Equation (4),
  is less than that of the previous iteration's ensemble, the AdaNet algorithm
  continues onto the next iteration.

  AdaNet attempts to minimize the following loss function to learn the mixture
  weights 'w' of each subnetwork 'h' in the ensemble with differentiable
  convex non-increasing surrogate loss function Phi:

  Equation (4):

  .. math::

      F(w) = \frac{1}{m} \sum_{i=1}^{m} \Phi \left(\sum_{j=1}^{N}w_jh_j(x_i),
      y_i \right) + \sum_{j=1}^{N} \left(\lambda r(h_j) + \beta \right) |w_j|

  with :math:`\lambda >= 0` and :math:`\beta >= 0`.

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
      middle, training stops before `max_iteration_steps` steps.
    mixture_weight_type: The :class:`adanet.MixtureWeightType` defining which
      mixture weight type to learn in the linear combination of subnetwork
      outputs:
        - :class:`SCALAR`: creates a rank 0 tensor mixture weight . It performs
          an element- wise multiplication with its subnetwork's logits. This
          mixture weight is the simplest to learn, the quickest to train, and
          most likely to generalize well.
        - :class:`VECTOR`:  creates a tensor with shape [k] where k is the
          ensemble's logits dimension as defined by `head`. It is similar to
          `SCALAR` in that it performs an element-wise multiplication with its
          subnetwork's logits, but is more flexible in learning a subnetworks's
          preferences per class.
        - :class:`MATRIX`: creates a tensor of shape [a, b] where a is the
          number of outputs from the subnetwork's `last_layer` and b is the
          number of outputs from the ensemble's `logits`. This weight
          matrix-multiplies the subnetwork's `last_layer`. This mixture weight
          offers the most flexibility and expressivity, allowing subnetworks to
          have outputs of different dimensionalities. However, it also has the
          most trainable parameters (a*b), and is therefore the most sensitive
          to learning rates and regularization.
    mixture_weight_initializer: The initializer for mixture_weights. When
      `None`, the default is different according to `mixture_weight_type`:
        - :class:`SCALAR`: initializes to 1/N where N is the number of
          subnetworks in the ensemble giving a uniform average.
        - :class:`VECTOR`: initializes each entry to 1/N where N is the number
          of subnetworks in the ensemble giving a uniform average.
        - :class:`MATRIX`: uses :meth:`tf.zeros_initializer`.
    warm_start_mixture_weights: Whether, at the beginning of an iteration, to
      initialize the mixture weights of the subnetworks from the previous
      ensemble to their learned value at the previous iteration, as opposed to
      retraining them from scratch. Takes precedence over the value for
      `mixture_weight_initializer` for subnetworks from previous iterations.
    adanet_lambda: Float multiplier 'lambda' for applying L1 regularization to
      subnetworks' mixture weights 'w' in the ensemble proportional to their
      complexity. See Equation (4) in the AdaNet paper.
    adanet_beta: Float L1 regularization multiplier 'beta' to apply equally to
      all subnetworks' weights 'w' in the ensemble regardless of their
      complexity. See Equation (4) in the AdaNet paper.
    evaluator: An :class:`adanet.Evaluator` for candidate selection after all
      subnetworks are done training. When `None`, candidate selection uses a
      moving average of their :class:`adanet.Ensemble` AdaNet loss during
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
    use_bias: Whether to add a bias term to the ensemble's logits. Adding a bias
      allows the ensemble to learn a shift in the data, often leading to more
      stable training and better predictions.
    metric_fn: A function for adding custom evaluation metrics, which should
      obey the following signature:
        - `Args`:
          Can only have the following three arguments in any order:
          - `predictions`: Predictions `Tensor` or dict of `Tensor` created by
            given `head`.
          - `features`: Input `dict` of `Tensor` objects created by `input_fn`
            which is given to `estimator.evaluate` as an argument.
          - `labels`:  Labels `Tensor` or dict of `Tensor` (for multi-head)
            created by `input_fn` which is given to `estimator.evaluate` as an
            argument.
        - `Returns`: Dict of metric results keyed by name. Final metrics are a
          union of this and `head's` existing metrics. If there is a name
          conflict between this and `head`s existing metrics, this will override
          the existing one. The values of the dict are the results of calling a
          metric function, namely a `(metric_tensor, update_op)` tuple.
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
    worker_wait_timeout_secs: Float number of seconds for workers to wait for
      chief to prepare the next iteration during distributed training. This is
      needed to prevent workers waiting indefinitely for a chief that may have
      crashed or been turned down. When the timeout is exceeded, the worker
      exits the train loop. In situations where the chief job is much slower
      than the worker jobs, this timeout should be increased.
    model_dir: Directory to save model parameters, graph and etc. This can also
      be used to load checkpoints from the directory into a estimator to
      continue training a previously saved model.
    report_dir: Directory where the `adanet.subnetwork.MaterializedReport`s
      materialized by `report_materializer` would be saved. If
      `report_materializer` is None, this will not save anything. If `None` or
      empty string, defaults to "<model_dir>/report".
    config: `RunConfig` object to configure the runtime settings.
    **kwargs: Extra keyword args passed to the parent.

  Returns:
    An `Estimator` instance.

  Raises:
    ValueError: If `subnetwork_generator` is `None`.
    ValueError: If `max_iteration_steps` is <= 0.
  """
  # pyformat: enable

  class _Keys(object):
    CURRENT_ITERATION = "current_iteration"
    EVALUATE_ENSEMBLES = "evaluate_ensembles"
    MATERIALIZE_REPORT = "materialize_report"
    INCREMENT_ITERATION = "increment_iteration"
    PREVIOUS_ENSEMBLE_ARCHITECTURE = "previous_ensemble_architecture"
    SUBNETWORK_GENERATOR = "subnetwork_generator"

  def __init__(self,
               head,
               subnetwork_generator,
               max_iteration_steps,
               mixture_weight_type=MixtureWeightType.SCALAR,
               mixture_weight_initializer=None,
               warm_start_mixture_weights=False,
               adanet_lambda=0.,
               adanet_beta=0.,
               evaluator=None,
               report_materializer=None,
               use_bias=False,
               metric_fn=None,
               force_grow=False,
               replicate_ensemble_in_training=False,
               adanet_loss_decay=.9,
               worker_wait_timeout_secs=7200,
               model_dir=None,
               report_dir=None,
               config=None,
               **kwargs):
    # TODO: Add argument to specify how many frozen graph
    # checkpoints to keep.

    if subnetwork_generator is None:
      raise ValueError("subnetwork_generator can't be None.")
    if max_iteration_steps <= 0.:
      raise ValueError("max_iteration_steps must be > 0.")

    self._subnetwork_generator = subnetwork_generator

    self._adanet_loss_decay = adanet_loss_decay

    # Overwrite superclass's assert that members are not overwritten in order
    # to overwrite public methods. Note that we are doing something that is not
    # explicitly supported by the Estimator API and may break in the future.
    tf.estimator.Estimator._assert_members_are_not_overridden = staticmethod(  # pylint: disable=protected-access
        lambda _: None)

    self._evaluation_checkpoint_path = None
    self._evaluator = evaluator
    self._report_materializer = report_materializer

    self._force_grow = force_grow
    self._worker_wait_timeout_secs = worker_wait_timeout_secs

    self._evaluation_name = None

    self._inside_adanet_training_loop = False

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

    # These are defined after base Estimator's init so that they can
    # use the same temporary model_dir as the underlying Estimator even if
    # model_dir is not provided.
    self._ensemble_builder = _EnsembleBuilder(
        head=head,
        mixture_weight_type=mixture_weight_type,
        mixture_weight_initializer=mixture_weight_initializer,
        warm_start_mixture_weights=warm_start_mixture_weights,
        checkpoint_dir=self._model_dir,
        adanet_lambda=adanet_lambda,
        adanet_beta=adanet_beta,
        use_bias=use_bias,
        metric_fn=metric_fn)
    candidate_builder = _CandidateBuilder(
        max_steps=max_iteration_steps,
        adanet_loss_decay=self._adanet_loss_decay)
    self._iteration_builder = _IterationBuilder(candidate_builder,
                                                self._ensemble_builder,
                                                replicate_ensemble_in_training)
    report_dir = report_dir or os.path.join(self._model_dir, "report")
    self._report_accessor = _ReportAccessor(report_dir)

  def _latest_checkpoint_iteration_number(self):
    """Returns the iteration number from the latest checkpoint."""

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if latest_checkpoint is None:
      return 0
    return tf.contrib.framework.load_variable(latest_checkpoint,
                                              self._Keys.CURRENT_ITERATION)

  def _latest_checkpoint_architecture(self):
    """Returns the iteration number from the latest checkpoint."""

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if latest_checkpoint is None:
      return ""
    return tf.contrib.framework.load_variable(
        latest_checkpoint, self._Keys.PREVIOUS_ENSEMBLE_ARCHITECTURE)

  def _latest_checkpoint_global_step(self):
    """Returns the global step from the latest checkpoint."""

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    if latest_checkpoint is None:
      return 0
    return tf.contrib.framework.load_variable(latest_checkpoint,
                                              tf.GraphKeys.GLOBAL_STEP)

  @contextlib.contextmanager
  def _train_loop_context(self):
    """Tracks where the context is within the AdaNet train loop."""

    self._inside_adanet_training_loop = True
    yield
    self._inside_adanet_training_loop = False

  def train(self,
            input_fn,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None):
    if (steps is not None) and (max_steps is not None):
      raise ValueError("Can not provide both steps and max_steps.")
    if steps is not None and steps <= 0:
      raise ValueError("Must specify steps > 0, given: {}".format(steps))

    if steps is not None:
      max_steps = self._latest_checkpoint_global_step() + steps

    # Each iteration of this AdaNet loop represents an `_Iteration`. The
    # current iteration number is stored as a variable in the checkpoint so
    # that training can be stopped and started at anytime.
    with self._train_loop_context():
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

        tf.logging.info("Finished training Adanet iteration %s",
                        current_iteration)

        # If training ended because the maximum number of training steps
        # occurred, exit training.
        if self._latest_checkpoint_global_step() >= max_steps:
          return result

        # If training ended for any reason other than the iteration ending,
        # exit training.
        if not self._iteration_ended:
          return result

        tf.logging.info("Beginning bookkeeping phase for iteration %s",
                        current_iteration)

        # The chief prepares the next AdaNet iteration, and increments the
        # iteration number by 1.
        if self.config.is_chief:
          # As the chief, store the train hooks and make a placeholder input_fn
          # in order to use them when preparing the next iteration.
          self._train_hooks = hooks or ()
          self._prepare_next_iteration(input_fn)

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
            tf.logging.error(
                "Chief job did not prepare next iteration after %s secs. It "
                "may have been preempted, been turned down, or crashed. This "
                "worker is now exiting training.",
                self._worker_wait_timeout_secs)
            return result
          tf.logging.info("Waiting for chief to finish")
          time.sleep(5)

        # Stagger starting workers to prevent training instability.
        if not self.config.is_chief:
          task_id = self.config.task_id or 0
          # Wait 5 secs more for each new worker up to 60 secs.
          delay_secs = min(60, task_id * 5)
          tf.logging.info("Waiting %d secs before starting training.",
                          delay_secs)
          time.sleep(delay_secs)

        tf.logging.info("Finished bookkeeping phase for iteration %s",
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
      tf.set_random_seed(self.config.tf_random_seed)
      # Create global step before calling model_fn as does superclass.
      tf.train.get_or_create_global_step()
      features, labels = input_fn()
      self._adanet_model_fn(features, labels, mode, params)

  def _prepare_next_iteration(self, train_input_fn):
    """Prepares the next iteration.

    This method calls model_fn up to four times:
      1. To evaluate all candidate ensembles to find the best one.
      2. To materialize reports and store them to disk (if report_materializer
         exists).
      3. To overwrite the model directory's checkpoint with the next iteration's
         ops.

    Args:
      train_input_fn: The input_fn used during training.
    """

    # First, evaluate and choose the best ensemble for this iteration.
    params = self.params.copy()
    params[self._Keys.EVALUATE_ENSEMBLES] = True
    if self._evaluator:
      evaluator_input_fn = self._evaluator.input_fn
    else:
      evaluator_input_fn = train_input_fn
    self._call_adanet_model_fn(evaluator_input_fn, tf.estimator.ModeKeys.EVAL,
                               params)

    # Then materialize and store the subnetwork reports.
    if self._report_materializer:
      params = self.params.copy()
      params[self._Keys.MATERIALIZE_REPORT] = True
      self._call_adanet_model_fn(self._report_materializer.input_fn,
                                 tf.estimator.ModeKeys.EVAL, params)

    self._best_ensemble_index = None

    # Finally, create the graph for the next iteration and overwrite the model
    # directory checkpoint with the expanded graph.
    params = self.params.copy()
    params[self._Keys.INCREMENT_ITERATION] = True
    self._call_adanet_model_fn(train_input_fn, tf.estimator.ModeKeys.TRAIN,
                               params)

  def _architecture_filename(self, iteration_number):
    """Returns the filename of the given iteration's frozen graph."""

    frozen_checkpoint = os.path.join(self.model_dir, "architecture")
    return "{}-{}.txt".format(frozen_checkpoint, iteration_number)

  def _overwrite_checkpoint(self, current_iteration, iteration_number_tensor):
    """Overwrites the latest checkpoint with the current graph.

    This is necessary for two reasons:
     1. To add variables to the checkpoint that were newly created for the
     next iteration. Otherwise Estimator will raise an exception for having a
     checkpoint missing variables.
     2. To increment the current iteration number so that workers know when to
     begin training the next iteration.

    Args:
      current_iteration: Current `_Iteration` object.
      iteration_number_tensor: Int variable `Tensor` storing the current
        iteration number.
    """

    checkpoint_state = tf.train.get_checkpoint_state(self.model_dir)
    latest_checkpoint = checkpoint_state.model_checkpoint_path
    if not latest_checkpoint:
      return

    # Run train hook 'begin' methods which can add ops to the graph, so that
    # they are still present in the overwritten checkpoint.
    train_hooks = tuple(self._train_hooks) or ()
    for candidate in current_iteration.candidates:
      if not candidate.ensemble_spec.subnetwork_train_op:
        assert not candidate.ensemble_spec.ensemble_train_op
        continue
      train_hooks += candidate.ensemble_spec.subnetwork_train_op.chief_hooks
      train_hooks += candidate.ensemble_spec.subnetwork_train_op.hooks
      train_hooks += candidate.ensemble_spec.ensemble_train_op.chief_hooks
      train_hooks += candidate.ensemble_spec.ensemble_train_op.hooks
    for hook in train_hooks:
      hook.begin()

    global_step_tensor = tf.train.get_global_step()
    global_step = tf.contrib.framework.load_variable(latest_checkpoint,
                                                     tf.GraphKeys.GLOBAL_STEP)

    checkpoint_path = os.path.join(self.model_dir, "increment.ckpt")
    with tf.Session(target=self.config.master) as sess:
      init = tf.group(
          tf.global_variables_initializer(), tf.local_variables_initializer(),
          tf.tables_initializer(),
          resources.initialize_resources(resources.shared_resources()))
      sess.run(init)
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      control_deps = [
          tf.assign(global_step_tensor, global_step),
          tf.assign(iteration_number_tensor, current_iteration.number),
      ]
      with tf.control_dependencies(control_deps):
        saver = tf.train.Saver(
            sharded=True, max_to_keep=self.config.keep_checkpoint_max)
        saver.recover_last_checkpoints(
            checkpoint_state.all_model_checkpoint_paths)
        saver.save(sess, checkpoint_path, global_step=current_iteration.number)
      for hook in train_hooks:
        hook.end(sess)

  def _get_best_ensemble_index(self, current_iteration):
    """Returns the best candidate ensemble's index in this iteration.

    Evaluates the ensembles using an `Evaluator` when provided. Otherwise,
    it returns the index of the best candidate as defined by the `_Iteration`.

    Args:
      current_iteration: Current `_Iteration`.

    Returns:
      Index of the best ensemble in the iteration's list of `_Candidates`.
    """

    # Skip the evaluation phase when there is only one candidate subnetwork.
    if len(current_iteration.candidates) == 1:
      tf.logging.info(
          "As the only candidate, '%s' is moving onto the next iteration.",
          current_iteration.candidates[0].ensemble_spec.name)
      return 0

    # The zero-th index candidate at iteration t>0 is always the
    # previous_ensemble.
    if current_iteration.number > 0 and self._force_grow and (len(
        current_iteration.candidates) == 2):
      tf.logging.info(
          "As the only candidate with `force_grow` enabled, '%s' is moving"
          "onto the next iteration.",
          current_iteration.candidates[1].ensemble_spec.name)
      return 1

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
        adanet_losses = [
            c.ensemble_spec.adanet_loss for c in current_iteration.candidates
        ]
        adanet_losses = self._evaluator.evaluate_adanet_losses(
            sess, adanet_losses)
      else:
        adanet_losses = sess.run(
            [c.adanet_loss for c in current_iteration.candidates])
      values = []
      for i in range(len(current_iteration.candidates)):
        metric_name = "adanet_loss"
        ensemble_name = current_iteration.candidates[i].ensemble_spec.name
        values.append("{}/{} = {:.6f}".format(metric_name, ensemble_name,
                                              adanet_losses[i]))
      tf.logging.info("Computed ensemble metrics: %s", ", ".join(values))
      if self._force_grow and current_iteration.number > 0:
        tf.logging.info(
            "The `force_grow` override is enabled, so the "
            "the performance of the previous ensemble will be ignored.")
        # NOTE: The zero-th index candidate at iteration t>0 is always the
        # previous_ensemble.
        adanet_losses = adanet_losses[1:]
        index = np.argmin(adanet_losses) + 1
      else:
        index = np.argmin(adanet_losses)
    tf.logging.info("Finished ensemble evaluation for iteration %s",
                    current_iteration.number)
    tf.logging.info("'%s' at index %s is moving onto the next iteration",
                    current_iteration.candidates[index].ensemble_spec.name,
                    index)
    return index

  def _materialize_report(self, current_iteration):
    """Generates reports as defined by `Builder`s.

    Materializes the Tensors and metrics defined in the `Builder`s'
    `build_subnetwork_report` method using `ReportMaterializer`, and stores
    them to disk using `_ReportAccessor`.

    Args:
      current_iteration: Current `_Iteration`.
    """

    latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
    tf.logging.info("Starting metric logging for iteration %s",
                    current_iteration.number)

    assert self._best_ensemble_index is not None
    best_candidate = current_iteration.candidates[self._best_ensemble_index]
    best_ensemble = best_candidate.ensemble_spec.ensemble
    best_name = best_ensemble.weighted_subnetworks[-1].name
    included_subnetwork_names = [best_name]
    with tf.Session() as sess:
      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer(), tf.tables_initializer())
      sess.run(init)
      saver = tf.train.Saver(sharded=True)
      saver.restore(sess, latest_checkpoint)
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      materialized_reports = (
          self._report_materializer.materialize_subnetwork_reports(
              sess, current_iteration.number,
              current_iteration.subnetwork_reports, included_subnetwork_names))
      self._report_accessor.write_iteration_report(current_iteration.number,
                                                   materialized_reports)

    tf.logging.info("Finished saving subnetwork reports for iteration %s",
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

    training_hooks = list(current_iteration.estimator_spec.training_hooks) + [
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

  def _evaluation_hooks(self, current_iteration, training):
    """Returns evaluation hooks for this iteration.

    Args:
      current_iteration: Current `_Iteration`.
      training: Whether in training mode.

    Returns:
      A list of `tf.train.SessionRunHook` instances.
    """

    if training:
      return []
    evaluation_hooks = []
    for candidate in current_iteration.candidates:
      eval_subdir = "eval"
      if self._evaluation_name:
        eval_subdir = "eval_{}".format(self._evaluation_name)
      eval_metric_hook = _EvalMetricSaverHook(
          name=candidate.ensemble_spec.name,
          eval_metric_ops=candidate.ensemble_spec.eval_metric_ops,
          output_dir=os.path.join(self.model_dir, "candidate",
                                  candidate.ensemble_spec.name, eval_subdir))
      evaluation_hooks.append(eval_metric_hook)
    return evaluation_hooks

  def _save_architecture(self, filename, ensemble):
    """Persists the ensemble's architecture in a serialized format.

    Writes to a text file with one subnetwork's iteration number and name
    per line.

    Args:
      filename: String filename to persist the ensemble architecture.
      ensemble: Target `adanet.Ensemble` instance.
    """

    architecture = [
        "{}:{}".format(w.iteration_number, w.name)
        for w in ensemble.weighted_subnetworks
    ]
    # Make directories since model_dir may not have been created yet.
    tf.gfile.MakeDirs(os.path.dirname(filename))
    with tf.gfile.GFile(filename, "w") as record_file:
      record_file.write(os.linesep.join(architecture))

  def _read_architecture(self, filename):
    """Reads an ensemble architecture from disk.

    Assumes the file was written with `_save_architecture`.

    Args:
      filename: String filename where features were recorded.

    Returns:
      A list of <iteration_number>:<subnetwork name> strings.

    Raises:
      OSError: When file not found at `filename`.
    """

    if not tf.gfile.Exists(filename):
      raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    architecture = []
    with tf.gfile.GFile(filename, "r") as record_file:
      for line in record_file:
        feature_name = line.rstrip()
        architecture.append(feature_name)
    return architecture

  # TODO: Refactor architecture building logic to its own module.
  def _architecture_ensemble_spec(self, architecture, features, mode, labels):
    """Returns an `_EnsembleSpec` with the given architecture.

    Creates the ensemble architecture by calling `generate_subnetworks` on
    `self._subnetwork_generator` and only calling `build_subnetwork` on
    `Builders` included in the architecture. Once their ops are created, their
    variables are restored from the checkpoint.

    Args:
      architecture: A list of <iteration_number>:<subnetwork name> strings.
      features: Dictionary of `Tensor` objects keyed by feature name.
      mode: Defines whether this is training, evaluation or prediction. See
        `ModeKeys`.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head). Can be `None`.

    Returns:
      An `EnsembleSpec` instance for the given architecture.

    Raises:
      ValueError: If a subnetwork from `architecture` is not found in the
        generated candidate `Builders` of the specified iteration.
    """

    previous_ensemble_spec = None
    previous_ensemble = None
    for serialized_subnetwork in architecture:
      serialized_iteration_number, name = serialized_subnetwork.split(":")
      rebuild_iteration_number = int(serialized_iteration_number)
      previous_ensemble_reports, all_reports = [], []
      if self._report_materializer:
        previous_ensemble_reports, all_reports = (
            self._collate_subnetwork_reports(rebuild_iteration_number))
      generated_subnetwork_builders = (
          self._subnetwork_generator.generate_candidates(
              previous_ensemble=previous_ensemble,
              iteration_number=rebuild_iteration_number,
              previous_ensemble_reports=previous_ensemble_reports,
              all_reports=all_reports))
      rebuild_subnetwork_builder = None
      for builder in generated_subnetwork_builders:
        if builder.name == name:
          rebuild_subnetwork_builder = builder
          break
      if rebuild_subnetwork_builder is None:
        raise ValueError("Required subnetwork name is missing from "
                         "generated candidates: {}".format(name))

      previous_ensemble_summary = None
      if previous_ensemble_spec:
        # Always skip summaries when rebuilding previous architecture,
        # since they are not useful.
        previous_ensemble_summary = _ScopedSummary(
            previous_ensemble_spec.name, skip_summary=True)

      current_iteration = self._iteration_builder.build_iteration(
          iteration_number=rebuild_iteration_number,
          subnetwork_builders=[rebuild_subnetwork_builder],
          features=features,
          labels=labels,
          mode=mode,
          previous_ensemble_summary=previous_ensemble_summary,
          previous_ensemble_spec=previous_ensemble_spec,
          rebuilding=True)
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

      # Assumes that only one subnetwork is added to the ensemble in
      # each iteration.
      chosen_subnetwork_in_this_iteration = [
          subnetwork_report for subnetwork_report in iteration_reports
          if subnetwork_report.included_in_final_ensemble
      ][0]
      previous_ensemble_reports.append(chosen_subnetwork_in_this_iteration)

      all_reports.extend(iteration_reports)

    return previous_ensemble_reports, all_reports

  def _adanet_model_fn(self, features, labels, mode, params):
    """AdaNet model_fn.

    This model_fn is called at least three times per iteration:
     1. The first call generates, builds, and trains the candidate subnetworks
     to ensemble in this iteration.
     2. Once training is over, bookkeeping begins. The next call is to evaluate
     the best candidate ensembles according to the AdaNet objective.
     2.b. Optionally, when a report materializer is provided, another call
     creates the graph for producing subnetwork reports for the next iteration
     and other AdaNet runs.
     3. The final call is responsible for rebuilding the ensemble architecture
     from t-1 by regenerating the best builders and warm-starting their weights,
     adding ops and initialing the weights for the next candidate subnetworks,
     and overwriting the latest checkpoint with its graph and variables, so that
     first call of the next iteration has the right variables in the checkpoint.

    Args:
      features: Dictionary of `Tensor` objects keyed by feature name.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head). Can be `None`.
      mode: Defines whether this is training, evaluation or prediction. See
        `ModeKeys`.
      params: A dict of parameters.

    Returns:
      A `EstimatorSpec` instance.

    Raises:
      UserWarning: When calling model_fn directly in TRAIN mode.
    """

    training = mode == tf.estimator.ModeKeys.TRAIN
    if training and not self._inside_adanet_training_loop:
      raise UserWarning(
          "The adanet.Estimator's model_fn should not be called directly in "
          "TRAIN mode, because its behavior is undefined outside the context "
          "of its `train` method. If you are trying to add custom metrics "
          "with `tf.contrib.estimator.add_metrics`, pass the `metric_fn` to "
          "this `Estimator's` constructor instead.")

    iteration_number = self._latest_checkpoint_iteration_number()

    # Use the evaluation checkpoint path to get both the iteration number and
    # variable values to avoid any race conditions between the first and second
    # checkpoint reads.
    if mode == tf.estimator.ModeKeys.EVAL and self._evaluation_checkpoint_path:
      iteration_number = tf.contrib.framework.load_variable(
          self._evaluation_checkpoint_path, self._Keys.CURRENT_ITERATION)

    if self._Keys.INCREMENT_ITERATION in params:
      iteration_number += 1

    architecture_filename = self._architecture_filename(iteration_number - 1)
    architecture = []
    if tf.gfile.Exists(architecture_filename):
      architecture = self._read_architecture(architecture_filename)
      tf.logging.info(
          "Importing architecture from %s: [%s].", architecture_filename,
          ", ".join(sorted(["'{}'".format(f) for f in architecture])))

    skip_summaries = mode == tf.estimator.ModeKeys.PREDICT
    with tf.variable_scope("adanet"):
      previous_ensemble_spec = None
      previous_ensemble = None
      previous_ensemble_summary = None
      if architecture:
        previous_ensemble_spec = self._architecture_ensemble_spec(
            architecture, features, mode, labels)
        previous_ensemble = previous_ensemble_spec.ensemble
        previous_ensemble_summary = _ScopedSummary(
            previous_ensemble_spec.name, skip_summary=skip_summaries)
      if self._Keys.INCREMENT_ITERATION in params:
        latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        tf.train.warm_start(latest_checkpoint, vars_to_warm_start=[".*"])
      previous_ensemble_reports, all_reports = [], []
      if self._report_materializer:
        previous_ensemble_reports, all_reports = (
            self._collate_subnetwork_reports(iteration_number))
      subnetwork_builders = self._subnetwork_generator.generate_candidates(
          previous_ensemble=previous_ensemble,
          iteration_number=iteration_number,
          previous_ensemble_reports=previous_ensemble_reports,
          all_reports=all_reports)
      current_iteration = self._iteration_builder.build_iteration(
          iteration_number=iteration_number,
          subnetwork_builders=subnetwork_builders,
          features=features,
          labels=labels,
          mode=mode,
          previous_ensemble_summary=previous_ensemble_summary,
          previous_ensemble_spec=previous_ensemble_spec)

    # Variable which allows us to read the current iteration from a checkpoint.
    iteration_number_tensor = tf.get_variable(
        self._Keys.CURRENT_ITERATION,
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False)

    adanet_summary = _ScopedSummary("global", skip_summaries)
    adanet_summary.scalar("iteration/adanet/iteration", iteration_number_tensor)
    adanet_summary.scalar("iteration_step/adanet/iteration_step",
                          current_iteration.step)
    if current_iteration.estimator_spec.loss is not None:
      adanet_summary.scalar("loss", current_iteration.estimator_spec.loss)
      adanet_summary.scalar("loss/adanet/adanet_weighted_ensemble",
                            current_iteration.estimator_spec.loss)

    iteration_estimator_spec = current_iteration.estimator_spec
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=iteration_estimator_spec.predictions,
        loss=iteration_estimator_spec.loss,
        train_op=iteration_estimator_spec.train_op,
        eval_metric_ops=iteration_estimator_spec.eval_metric_ops,
        training_chief_hooks=iteration_estimator_spec.training_chief_hooks,
        training_hooks=self._training_hooks(current_iteration, training),
        evaluation_hooks=self._evaluation_hooks(current_iteration, training),
        scaffold=tf.train.Scaffold(summary_op=adanet_summary.merge_all()),
        export_outputs=iteration_estimator_spec.export_outputs)

    if self._Keys.EVALUATE_ENSEMBLES in params:
      assert self.config.is_chief
      self._best_ensemble_index = self._get_best_ensemble_index(
          current_iteration)
      ensemble = current_iteration.candidates[
          self._best_ensemble_index].ensemble_spec.ensemble
      new_architecture_filename = self._architecture_filename(iteration_number)
      self._save_architecture(new_architecture_filename, ensemble)
    elif self._Keys.MATERIALIZE_REPORT in params:
      assert self.config.is_chief
      assert self._best_ensemble_index is not None
      self._materialize_report(current_iteration)
    elif self._Keys.INCREMENT_ITERATION in params:
      assert self.config.is_chief
      latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
      tf.logging.info(
          "Overwriting checkpoint with new graph for iteration %s to %s",
          iteration_number, latest_checkpoint)
      self._overwrite_checkpoint(current_iteration, iteration_number_tensor)

    return estimator_spec
