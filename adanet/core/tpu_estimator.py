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

import functools

from adanet.core.ensemble import MixtureWeightType
from adanet.core.estimator import Estimator
from adanet.core.summary import _ScopedSummary
import six
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python import summary


# TODO: support summaries on TPU during training.
def _rewire_summaries():
  """Rewire Tensorflow summaries to be no-ops when running on TPU.

  Summaries are not currently supported on TPU.
  """

  def tpu_no_op(fn):

    def _fn(*args, **kwargs):
      if tpu_function.get_tpu_context().number_of_shards:
        return None
      return fn(*args, **kwargs)

    return _fn

  summary.audio = tpu_no_op(summary.audio)
  summary.histogram = tpu_no_op(summary.histogram)
  summary.image = tpu_no_op(summary.image)
  summary.scalar = tpu_no_op(summary.scalar)
  summary.tensor_summary = tpu_no_op(summary.tensor_summary)
  summary.text = tpu_no_op(summary.text)

  tf.summary.audio = tpu_no_op(tf.summary.audio)
  tf.summary.histogram = tpu_no_op(tf.summary.histogram)
  tf.summary.image = tpu_no_op(tf.summary.image)
  tf.summary.scalar = tpu_no_op(tf.summary.scalar)
  tf.summary.tensor_summary = tpu_no_op(tf.summary.tensor_summary)
  tf.summary.text = tpu_no_op(tf.summary.text)

  _ScopedSummary.audio = tpu_no_op(_ScopedSummary.audio)
  _ScopedSummary.histogram = tpu_no_op(_ScopedSummary.histogram)
  _ScopedSummary.image = tpu_no_op(_ScopedSummary.image)
  _ScopedSummary.scalar = tpu_no_op(_ScopedSummary.scalar)


# Rewire summaries to be no-ops when running on TPU.
_rewire_summaries()


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
               batch_size=None):
    if not use_tpu:
      tf.logging.warning(
          'This adanet.TPUEstimator is meant to be used for running on TPU. '
          'If you want to run on CPU/GPU, use adanet.Estimator instead.')

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
        train_batch_size=batch_size or 0)

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
    if 'use_tpu' in params and mode == tf.estimator.ModeKeys.TRAIN:
      kwargs = {
          key: value
          for key, value in six.iteritems(estimator_spec._asdict())
          if key not in ('eval_metric_ops', 'scaffold', 'training_chief_hooks')
      }
      estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(**kwargs)
    return estimator_spec
