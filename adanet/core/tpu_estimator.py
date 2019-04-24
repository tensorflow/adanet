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
from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_function


class TPUEstimator(Estimator, tf.contrib.tpu.TPUEstimator):
  """An :class:`adanet.Estimator` capable of training and evaluating on TPU.

  Note: Unless :code:`use_tpu=False`, training will run on TPU. However,
  certain parts of AdaNet training loop, such as report materialization and best
  candidate selection still occurr on CPU. Furthermore, inference also occurs on
  CPU.

  Args:
    head: See :class:`adanet.Estimator`.
    subnetwork_generator: See :class:`adanet.Estimator`.
    max_iteration_steps: See :class:`adanet.Estimator`.
    ensemblers: See :class:`adanet.Estimator`.
    ensemble_strategies: See :class:`adanet.Estimator`.
    evaluator: See :class:`adanet.Estimator`.
    report_materializer: See :class:`adanet.Estimator`.
    metric_fn: See :class:`adanet.Estimator`.
    force_grow: See :class:`adanet.Estimator`.
    replicate_ensemble_in_training: See :class:`adanet.Estimator`.
    adanet_loss_decay: See :class:`adanet.Estimator`.
    report_dir: See :class:`adanet.Estimator`.
    config: See :class:`adanet.Estimator`.
    use_tpu: Boolean to enable *both* training and evaluating on TPU. Defaults
      to :code:`True` and is only provided to allow debugging models on CPU/GPU.
      Use :class:`adanet.Estimator` instead if you do not plan to run on TPU.
    train_batch_size: See :class:`tf.contrib.tpu.TPUEstimator`.
    eval_batch_size: See :class:`tf.contrib.tpu.TPUEstimator`.
    debug: See :class:`adanet.Estimator`.
    **kwargs: Extra keyword args passed to the parent.
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
               model_dir=None,
               report_dir=None,
               config=None,
               use_tpu=True,
               train_batch_size=None,
               eval_batch_size=None,
               debug=False,
               **kwargs):

    self._use_tpu = use_tpu
    if not self._use_tpu:
      tf.logging.warning(
          "This adanet.TPUEstimator is meant to be used for running on TPU. "
          "If you want to run on CPU/GPU, use adanet.Estimator instead.")

    # TPUEstimator modifies config under the hood. We keep track of it here so
    # we can use it during the bookkeeping phase and when predict() is called.
    self._original_config = config or tf.contrib.RunConfig()
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
        model_dir=model_dir,
        report_dir=report_dir,
        config=self._original_config,
        use_tpu=use_tpu,
        eval_on_tpu=use_tpu,
        export_to_tpu=False,
        train_batch_size=self._train_batch_size,
        eval_batch_size=self._eval_batch_size,
        **kwargs)

  # Yields predictions on CPU even when use_tpu=True.
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
        config=self._original_config,
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

    temp_run_config_kwargs = dict(
        model_dir=temp_model_dir,
        tpu_config=self._original_config.tpu_config,
        evaluation_master=self._original_config.evaluation_master,
        master=self._original_config.master,
        cluster=self._original_config.cluster,
        tf_random_seed=self._original_config.tf_random_seed,
        session_config=self._original_config.session_config,
        device_fn=self._original_config.device_fn)
    if LooseVersion(tf.VERSION) >= LooseVersion("1.11.0"):
      temp_run_config_kwargs["protocol"] = self._original_config.protocol
    return tf.contrib.tpu.TPUEstimator(
        model_fn=self._adanet_model_fn,
        params={},
        config=tf.contrib.tpu.RunConfig(**temp_run_config_kwargs),
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
      return super(TPUEstimator,
                   self)._create_estimator_spec(current_iteration, mode,
                                                iteration_number_tensor,
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

    training_hooks = super(TPUEstimator,
                           self)._training_hooks(current_iteration, training,
                                                 iteration_number_tensor,
                                                 previous_iteration_vars)
    if self._use_tpu:
      # Remove summary hooks on TPU since summaries are saved via host_call.
      training_hooks = [
          hook for hook in training_hooks
          if not isinstance(hook, tf.train.SummarySaverHook)
      ]

    return training_hooks

  def _create_host_call(self, current_iteration, training):
    """Construct a host_call writing scalar summaries.

    Args:
      current_iteration: The current `_Iteration`.
      training: Boolean indicating whether in training mode.

    Returns:
      (fn, args) Pair to be called by TPUEstimator as the host_call.
    """

    # Collect and flatten summary functions and arguments.
    summary_kwargs = collections.OrderedDict()
    gs_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
    summary_kwargs["global_step"] = gs_t

    summary_fns = collections.defaultdict(list)
    for i, summary in enumerate(current_iteration.summaries):
      for j, (summary_fn, tensor) in enumerate(summary.summary_tuples()):
        summary_fns[i].append(summary_fn)
        summary_kwargs["summary_{}_{}".format(i, j)] = tensor

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

      for i, summary in enumerate(current_iteration.summaries):
        with tf.contrib.summary.create_file_writer(summary.logdir).as_default():
          with tf.contrib.summary.record_summaries_every_n_global_steps(
              n=self.config.save_summary_steps, global_step=gs):
            for j, summary_fn in enumerate(summary_fns[i]):
              tensor = kwargs["summary_{}_{}".format(i, j)]
              summary_fn(tensor, step=gs)
        summary.clear_summary_tuples()
      return tf.contrib.summary.all_summary_ops()

    return _host_call_fn, summary_kwargs
