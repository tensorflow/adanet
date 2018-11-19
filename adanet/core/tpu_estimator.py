"""An AdaNet estimator implementation which can run on TPUs.

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

from adanet.core.ensemble import MixtureWeightType
from adanet.core.estimator import Estimator
import six
import tensorflow as tf


# TODO: No TPU support is currently implemented. This Estimator
# will simply run on CPU/GPU if used.
class TPUEstimator(Estimator, tf.contrib.tpu.TPUEstimator):
  """An adanet.Estimator capable of running on TPUs.

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
               train_batch_size=None):
    """Initializes a `TPUEstimator`. See base classes for details."""

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
        use_tpu=False,
        eval_on_tpu=False,
        export_to_tpu=False,
        train_batch_size=train_batch_size if train_batch_size else 0)

  def _adanet_model_fn(self, features, labels, mode, params):
    """See the `Estimator` base class for details."""

    estimator_spec = super(TPUEstimator, self)._adanet_model_fn(
        features, labels, mode, params)
    if mode != tf.estimator.ModeKeys.TRAIN:
      return estimator_spec
    else:
      kwargs = {
          key: value
          for key, value in six.iteritems(estimator_spec._asdict())
          if key not in ('eval_metric_ops', 'scaffold', 'training_chief_hooks')
      }
      # TODO: wrapping eval_metric_ops and scaffold in lambdas is a hack
      # to get TPUEstimator to work when use_tpu=False. This will not work when
      # we actually start using TPUs.
      eval_metrics = (lambda: estimator_spec.eval_metric_ops, ())
      scaffold_fn = lambda: estimator_spec.scaffold
      return tf.contrib.tpu.TPUEstimatorSpec(
          eval_metrics=eval_metrics, scaffold_fn=scaffold_fn, **kwargs)
