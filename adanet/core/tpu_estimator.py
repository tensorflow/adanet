"""An AdaNet estimator implementation which can run on TPU.

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
import functools

from adanet.core.estimator import Estimator
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.framework import ops  # pylint: disable=g-direct-tensorflow-import


# TODO: Move hooks to their own module.
class _StepCounterHook(tf.train.SessionRunHook):
  """Hook that counts steps per second.

  TODO: Remove once Estimator uses summaries v2 by default.

  """

  def __init__(self,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):

    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError(
          "exactly one of every_n_steps and every_n_secs should be provided.")
    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    assert output_dir
    self._summary_writer = summary_writer
    self._output_dir = output_dir
    self._last_global_step = None
    self._steps_per_run = 1

  def begin(self):
    if self._summary_writer is None and self._output_dir:
      self._summary_writer = tf.summary.FileWriter(
          self._output_dir,
          session=ops.get_default_session(),
          filename_suffix=".step")
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use StepCounterHook.")
    self._summary_tag = tf.train.get_global_step().op.name + "/sec"

  def after_create_session(self, session, coord):
    del coord
    # Reset any stale state in case we're recovering from a previous error.
    session.run(tf.contrib.summary.summary_writer_initializer_op())
    self._last_global_step = None
    self._timer.reset()

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
    steps_per_sec = elapsed_steps / elapsed_time
    if self._summary_writer is not None:
      summary = tf.Summary(value=[
          tf.Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec)
      ])
      self._summary_writer.add_summary(summary, global_step)

  def after_run(self, run_context, run_values):
    stale_global_step = run_values.results
    if self._timer.should_trigger_for_step(stale_global_step +
                                           self._steps_per_run):
      # Get the real value after train op.
      global_step = run_context.session.run(self._global_step_tensor)
      if self._timer.should_trigger_for_step(global_step):
        elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
            global_step)
        if elapsed_time is not None:
          with ops.default_session(run_context.session):
            self._log_and_record(elapsed_steps, elapsed_time, global_step)

    self._last_global_step = stale_global_step

  def end(self, session):
    if self._summary_writer is not None:
      with ops.default_session(session):
        self._summary_writer.flush()


class TPUEstimator(Estimator, tf.contrib.tpu.TPUEstimator):
  """An adanet.Estimator capable of running on TPU.

  If running on TPU, all summary calls are rewired to be no-ops during training.

  WARNING: this API is highly experimental, unstable, and can change  without
  warning.
  """

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
               worker_wait_timeout_secs=7200,
               model_dir=None,
               report_dir=None,
               config=None,
               use_tpu=True,
               train_batch_size=None,
               eval_batch_size=None,
               **kwargs):
    self._use_tpu = use_tpu
    if not self._use_tpu:
      tf.logging.warning(
          "This adanet.TPUEstimator is meant to be used for running on TPU. "
          "If you want to run on CPU/GPU, use adanet.Estimator instead.")
    # TODO: Figure out why self.config.log_step_count_steps is
    # always None with TPUEstimator.
    self._log_step_count_steps = config.log_step_count_steps or 100
    self._train_batch_size = train_batch_size or 0
    self._eval_batch_size = eval_batch_size or train_batch_size or 0

    super(TPUEstimator, self).__init__(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        ensemblers=ensemblers,
        ensemble_strategies=ensemble_strategies,
        evaluator=evaluator,
        report_materializer=report_materializer,
        metric_fn=metric_fn,
        force_grow=force_grow,
        replicate_ensemble_in_training=replicate_ensemble_in_training,
        adanet_loss_decay=adanet_loss_decay,
        worker_wait_timeout_secs=worker_wait_timeout_secs,
        model_dir=model_dir,
        report_dir=report_dir,
        config=config if config else tf.contrib.tpu.RunConfig(),
        use_tpu=use_tpu,
        eval_on_tpu=use_tpu,
        export_to_tpu=False,
        train_batch_size=self._train_batch_size,
        eval_batch_size=self._eval_batch_size,
        **kwargs)

  def predict(self,
              input_fn,
              predict_keys=None,
              hooks=None,
              checkpoint_path=None,
              yield_single_examples=True):
    # TODO: Required to support predict on CPU for TPUEstimator.
    # This is the recommended method from TensorFlow TPUEstimator docs:
    # https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimator#current_limitations
    tf.logging.warning(
        "The adanet.TPUEstimator does not support predicting on TPU. "
        "Instead, all predictions are run on CPU.")
    tpu_estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=self._adanet_model_fn,
        model_dir=self.model_dir,
        config=self.config,
        params=self.params,
        use_tpu=False)
    return tpu_estimator.predict(
        input_fn,
        predict_keys=predict_keys,
        hooks=hooks,
        checkpoint_path=checkpoint_path,
        yield_single_examples=yield_single_examples)

  def _create_temp_estimator(self, temp_model_dir):
    """See the `Estimator` base class for details."""

    config = self.config
    temp_run_config = tf.contrib.tpu.RunConfig(
        model_dir=temp_model_dir,
        tpu_config=config.tpu_config,
        evaluation_master=config.evaluation_master,
        master=config.master,
        cluster=config.cluster,
        tf_random_seed=config.tf_random_seed,
        save_summary_steps=config.save_summary_steps,
        save_checkpoints_steps=config.save_checkpoints_steps,
        save_checkpoints_secs=config.save_checkpoints_secs,
        session_config=config.session_config,
        keep_checkpoint_max=config.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=config.keep_checkpoint_every_n_hours,
        log_step_count_steps=config.log_step_count_steps,
        device_fn=config.device_fn,
        protocol=config.protocol)
    return tf.contrib.tpu.TPUEstimator(
        model_fn=self._adanet_model_fn,
        params={},
        config=temp_run_config,
        model_dir=temp_model_dir,
        use_tpu=self._use_tpu,
        eval_on_tpu=self._use_tpu,
        export_to_tpu=False,
        train_batch_size=self._train_batch_size,
        eval_batch_size=self._eval_batch_size)

  def _call_adanet_model_fn(self, input_fn, mode):
    """See the `Estimator` base class for details."""

    # Fakes TPU shard context before calling through to the parent to suppress
    # warnings by CrossShardOptimizer when running on TPU. Warnings are issued
    # when `_adanet_model_fn` is called directly on CPU during the bookkeeping
    # phase. Since we rebuild the graph each time `_adanet_model_fn` is called,
    # this has no adverse effects.
    with tpu_function.tpu_shard_context(0):
      # Bind params to input_fn since the parent's input_fn is not expected to
      # have any arguments.
      input_fn = functools.partial(input_fn, self.params)  # A deep copy.
      super(TPUEstimator, self)._call_adanet_model_fn(input_fn, mode)

  def _create_estimator_spec(self, current_iteration, mode,
                             iteration_number_tensor, previous_iteration_vars):
    """See the `Estimator` base class for details."""

    if not self._use_tpu:
      return super(TPUEstimator, self)._create_estimator_spec(
          current_iteration, mode, iteration_number_tensor,
          previous_iteration_vars)

    training = mode == tf.estimator.ModeKeys.TRAIN
    iteration_estimator_spec = current_iteration.estimator_spec
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions=iteration_estimator_spec.predictions,
        loss=iteration_estimator_spec.loss,
        train_op=self._train_op(iteration_estimator_spec),
        host_call=self._create_host_call(current_iteration, training),
        eval_metrics=iteration_estimator_spec.eval_metrics,
        export_outputs=iteration_estimator_spec.export_outputs,
        # Return a constant summary_op, otherwise `Estimator` creates summary
        # ops that do not work on TPU.
        scaffold_fn=lambda: tf.train.Scaffold(summary_op=tf.constant("")),
        training_hooks=self._decorate_hooks(
            self._training_hooks(current_iteration, training,
                                 iteration_number_tensor,
                                 previous_iteration_vars)),
        evaluation_hooks=self._evaluation_hooks(current_iteration, training))

  def _training_hooks(self, current_iteration, training,
                      iteration_number_tensor, previous_iteration_vars):
    """See the `Estimator` base class for details."""

    training_hooks = super(TPUEstimator, self)._training_hooks(
        current_iteration, training, iteration_number_tensor,
        previous_iteration_vars)
    if self._use_tpu:
      # Remove summary hooks on TPU since summaries are saved via host_call.
      training_hooks = [
          hook for hook in training_hooks
          if not isinstance(hook, tf.train.SummarySaverHook)
      ]
      training_hooks.append(
          _StepCounterHook(
              every_n_steps=self._log_step_count_steps,
              output_dir=self.model_dir))
    return training_hooks

  def _create_host_call(self, current_iteration, training):
    """Construct a host_call writing scalar summaries.

    Args:
      current_iteration: The current `_Iteration`.
      training: Boolean indicating whether in training mode.

    Returns:
      (fn, args) Pair to be called by TPUEstimator as the host_call.
    """

    summary_kwargs = collections.OrderedDict()
    gs_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
    summary_kwargs["global_step"] = gs_t

    i = 0
    summary_fns = []
    for summary in current_iteration.summaries:
      for summary_fn, tensor in summary.lazy_fns():
        summary_fns.append(summary_fn)
        summary_kwargs["summary_{}".format(i)] = tensor
        i += 1

    def _host_call_fn(**kwargs):
      """Training host call.

      Creates summaries for training metrics.

      Args:
        **kwargs: Dict of {str: Tensor} , with `Tensor` of shape `[batch]`. Must
          contain key "global_step" with value of current global_step Tensor.

      Returns:
        List of summary ops to run on the CPU host.
      """

      gs = tf.to_int64(kwargs.pop("global_step")[0])
      if not training:
        return [tf.no_op()]
      with tf.contrib.summary.record_summaries_every_n_global_steps(
          n=self.config.save_summary_steps, global_step=gs):
        for i, summary_fn in enumerate(summary_fns):
          tensor = kwargs.pop("summary_{}".format(i))
          summary_fn(tensor, step=gs)
      return tf.contrib.summary.all_summary_ops()

    return _host_call_fn, summary_kwargs
