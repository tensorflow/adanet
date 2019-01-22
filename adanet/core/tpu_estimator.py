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

from adanet.core.ensemble_builder import MixtureWeightType
from adanet.core.estimator import _StepCounterHook
from adanet.core.estimator import Estimator
import six
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_function


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
               use_tpu=True,
               train_batch_size=None,
               eval_batch_size=None):
    if not use_tpu:
      tf.logging.warning(
          "This adanet.TPUEstimator is meant to be used for running on TPU. "
          "If you want to run on CPU/GPU, use adanet.Estimator instead.")

    # TODO: Figure out why self.config.log_step_count_steps is
    # always None with TPUEstimator.
    self._log_step_count_steps = config.log_step_count_steps or 100

    super(TPUEstimator, self).__init__(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        mixture_weight_type=mixture_weight_type,
        mixture_weight_initializer=mixture_weight_initializer,
        warm_start_mixture_weights=warm_start_mixture_weights,
        adanet_lambda=adanet_lambda,
        adanet_beta=adanet_beta,
        evaluator=evaluator,
        report_materializer=report_materializer,
        use_bias=use_bias,
        metric_fn=metric_fn,
        force_grow=force_grow,
        replicate_ensemble_in_training=replicate_ensemble_in_training,
        adanet_loss_decay=adanet_loss_decay,
        worker_wait_timeout_secs=worker_wait_timeout_secs,
        model_dir=model_dir,
        report_dir=report_dir,
        config=config if config else tf.contrib.tpu.RunConfig(),
        use_tpu=use_tpu,
        eval_on_tpu=False,
        export_to_tpu=False,
        train_batch_size=train_batch_size or 0,
        eval_batch_size=eval_batch_size or train_batch_size or 0)

  def predict(self,
              input_fn,
              predict_keys=None,
              hooks=None,
              checkpoint_path=None,
              yield_single_examples=True):
    # TODO: Required to support predict on CPU for TPUEstiamtor.
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

  def _call_adanet_model_fn(self, input_fn, mode, params):
    """See the `Estimator` base class for details."""

    # Fakes TPU shard context before calling through to the parent to supress
    # warnings by CrossShardOptimizer when running on TPU. Warnings are issued
    # when `_adanet_model_fn` is called directly on CPU during the bookkeeping
    # phase. Since we rebuild the graph each time `_adanet_model_fn` is called,
    # this has no adverse effects.
    with tpu_function.tpu_shard_context(0):
      # Bind params to input_fn since the parent's input_fn is not expected to
      # have any arguments.
      input_fn = functools.partial(input_fn, params)
      super(TPUEstimator, self)._call_adanet_model_fn(input_fn, mode, params)

  def _adanet_model_fn(self, features, labels, mode, params):
    """See the `Estimator` base class for details."""

    estimator_spec = super(TPUEstimator, self)._adanet_model_fn(
        features, labels, mode, params)
    current_iteration = params["current_iteration"]
    training = mode == tf.estimator.ModeKeys.TRAIN
    if "use_tpu" in params and training:
      kwargs = {
          key: value
          for key, value in six.iteritems(estimator_spec._asdict())
          if key not in ("eval_metric_ops", "scaffold", "training_chief_hooks")
      }
      # Return a constant summary_op, otherwise `Estimator` creates summary ops
      # that do not work on TPU.
      kwargs["scaffold_fn"] = (
          lambda: tf.train.Scaffold(summary_op=tf.constant("")))
      kwargs["host_call"] = self._create_host_call(current_iteration, training)
      kwargs["training_hooks"] += (_StepCounterHook(
          every_n_steps=self._log_step_count_steps, output_dir=self.model_dir),)
      estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(**kwargs)
    return estimator_spec

  def _chief_training_hooks(self, current_iteration, training):
    """Overrides parent's method."""
    return []

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
