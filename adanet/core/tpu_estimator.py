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

import contextlib
import functools

from adanet.core.ensemble import MixtureWeightType
from adanet.core.estimator import Estimator
import six
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python import summary


# TODO: support summaries on TPU during training.
@contextlib.contextmanager
def _rewire_summaries():
  """Rewire Tensorflow summaries to be no-ops when running on TPU.

  Summaries are not currently supported on TPU.

  Yields:
    Context where summary functions are rewired to be no-ops when on TPU.
  """

  if tpu_function.get_tpu_context().number_of_shards == 0:
    yield
    return

  tf.logging.log_first_n(
      tf.logging.WARN,
      "Converting summaries to no-ops on TPU since they are not supported.", 1)
  old_summary_audio = summary.audio
  old_summary_histogram = summary.histogram
  old_summary_image = summary.image
  old_summary_scalar = summary.scalar
  old_summary_tensor_summary = summary.tensor_summary
  old_summary_text = summary.text

  def _no_op(*args, **kwargs):
    del args, kwargs  # Unused
    return tf.constant("", name="summary_no_op")

  # Monkey-patch global attributes.
  summary.audio = _no_op
  summary.histogram = _no_op
  summary.image = _no_op
  summary.scalar = _no_op
  summary.tensor_summary = _no_op
  summary.text = _no_op

  tf.summary.audio = _no_op
  tf.summary.histogram = _no_op
  tf.summary.image = _no_op
  tf.summary.scalar = _no_op
  tf.summary.tensor_summary = _no_op
  tf.summary.text = _no_op

  try:
    yield
  finally:
    # Revert monkey-patches.
    summary.audio = old_summary_audio
    summary.histogram = old_summary_histogram
    summary.image = old_summary_image
    summary.scalar = old_summary_scalar
    summary.tensor_summary = old_summary_tensor_summary
    summary.text = old_summary_text

    tf.summary.audio = old_summary_audio
    tf.summary.histogram = old_summary_histogram
    tf.summary.image = old_summary_image
    tf.summary.scalar = old_summary_scalar
    tf.summary.tensor_summary = old_summary_tensor_summary
    tf.summary.text = old_summary_text


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

  def train(self,
            input_fn,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None):
    # Rewire summaries to be no-ops when running on TPU.
    # TODO: Rewire predict and eval when TPU support is added.
    with _rewire_summaries():
      return super(TPUEstimator, self).train(
          input_fn=input_fn,
          hooks=hooks,
          max_steps=max_steps,
          saving_listeners=saving_listeners)

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
    if "use_tpu" in params and mode == tf.estimator.ModeKeys.TRAIN:
      kwargs = {
          key: value
          for key, value in six.iteritems(estimator_spec._asdict())
          if key not in ("eval_metric_ops", "scaffold", "training_chief_hooks")
      }
      estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(**kwargs)
    return estimator_spec
