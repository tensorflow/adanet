"""An estimator that learns to ensemble.

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

from adanet.core import Estimator
from adanet.core.subnetwork import Builder
from adanet.core.subnetwork import Generator
from adanet.core.subnetwork import Subnetwork
from adanet.core.subnetwork import TrainOpSpec
import tensorflow as tf

from tensorflow.python.estimator.canned import prediction_keys


def _default_logits(estimator_spec):
  if isinstance(estimator_spec.predictions, dict):
    pred_keys = prediction_keys.PredictionKeys
    if pred_keys.LOGITS in estimator_spec.predictions:
      return estimator_spec.predictions[pred_keys.LOGITS]
    if pred_keys.PREDICTIONS in estimator_spec.predictions:
      return estimator_spec.predictions[pred_keys.PREDICTIONS]
  return estimator_spec.predictions


class _BuilderFromEstimator(Builder):
  """An `adanet.Builder` from a :class:`tf.estimator.Estimator`."""

  def __init__(self, estimator, index, logits_fn):
    self._estimator = estimator
    self._index = index
    self._logits_fn = logits_fn

  @property
  def name(self):
    return "{class_name}{index}".format(
        class_name=self._estimator.__class__.__name__, index=self._index)

  def build_subnetwork(self, features, labels, logits_dimension, training,
                       iteration_step, summary, previous_ensemble):
    model_fn = self._estimator.model_fn

    # We don't need an EVAL mode since AdaNet takes care of evaluation for us.
    mode = tf.estimator.ModeKeys.PREDICT
    if training:
      mode = tf.estimator.ModeKeys.TRAIN
    estimator_spec = model_fn(
        features=features,
        labels=labels,
        mode=mode,
        config=self._estimator.config)
    logits = self._logits_fn(estimator_spec=estimator_spec)

    self._subnetwork_train_op = TrainOpSpec(
        estimator_spec.train_op,
        chief_hooks=estimator_spec.training_chief_hooks,
        hooks=estimator_spec.training_hooks)

    # TODO: Replace with variance complexity measure.
    complexity = tf.constant(0.)
    return Subnetwork(
        logits=logits,
        last_layer=logits,
        persisted_tensors={},
        complexity=complexity)

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    return self._subnetwork_train_op

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    return tf.no_op()


class _GeneratorFromCandidatePool(Generator):
  """An `adanet.Generator` from a pool of `Estimator` and `Model` instances."""

  def __init__(self, candidate_pool, logits_fn):
    self._candidate_pool = candidate_pool
    self._logits_fn = logits_fn

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    builders = []
    for i, candidate in enumerate(self._candidate_pool):
      if isinstance(candidate, tf.estimator.Estimator):
        builders.append(
            _BuilderFromEstimator(
                candidate, index=i, logits_fn=self._logits_fn))
        continue
    return builders


class AutoEnsembleEstimator(Estimator):
  # pyformat: disable
  """A :class:`tf.estimator.Estimator` that learns to ensemble models.

  Specifically, it learns to ensemble models from a candidate pool using the
  Adanet algorithm.

  .. code-block:: python

      # A simple example of learning to ensemble linear and neural network
      # models.

      import adanet
      import tensorflow as tf

      feature_columns = ...

      head = tf.contrib.estimator.multi_class_head(n_classes=3)

      # Learn to ensemble linear and DNN models.
      estimator = adanet.AutoEnsembleEstimator(
          head=head,
          candidate_pool=[
              tf.estimator.LinearEstimator(
                  head=head,
                  feature_columns=feature_columns,
                  optimizer=tf.train.FtrlOptimizer(...)),
              tf.estimator.DNNEstimator(
                  head=head,
                  feature_columns=feature_columns,
                  optimizer=tf.train.ProximalAdagradOptimizer(...),
                  hidden_units=[1000, 500, 100])],
          max_iteration_steps=50)

      # Input builders
      def input_fn_train:
        # Returns tf.data.Dataset of (x, y) tuple where y represents label's
        # class index.
        pass
      def input_fn_eval:
        # Returns tf.data.Dataset of (x, y) tuple where y represents label's
        # class index.
        pass
      def input_fn_predict:
        # Returns tf.data.Dataset of (x, None) tuple.
        pass
      estimator.train(input_fn=input_fn_train, steps=100)
      metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
      predictions = estimator.predict(input_fn=input_fn_predict)

  Args:
    head: A :class:`tf.contrib.estimator.Head` instance for computing loss and
      evaluation metrics for every candidate.
    candidate_pool: List of :class:`tf.estimator.Estimator` objects that are
      candidates to ensemble at each iteration. The order does not directly
      affect which candidates will be included in the final ensemble.
    max_iteration_steps: Total number of steps for which to train candidates per
      iteration. If `OutOfRange` or `StopIteration` occurs in the middle,
      training stops before `max_iteration_steps` steps.
    logits_fn: A function for fetching the subnetwork logits from a
      :class:`tf.estimator.EstimatorSpec`, which should obey the following
      signature:
        - `Args`: Can only have following argument:
          - estimator_spec: The candidate's :class:`tf.estimator.EstimatorSpec`.
        - `Returns`: Logits :class:`tf.Tensor` or dict of string to logits
          :class:`tf.Tensor` (for multi-head) for the candidate subnetwork
          extracted from the given `estimator_spec`. When `None`, it will
          default to returning `estimator_spec.predictions` when they are a
          :class:`tf.Tensor` or the :class:`tf.Tensor` for the key 'logits' when
          they are a dict of string to :class:`tf.Tensor`.
    adanet_lambda: See :class:`adanet.Estimator`.
    evaluator:  See :class:`adanet.Estimator`.
    metric_fn:  See :class:`adanet.Estimator`.
    force_grow:  See :class:`adanet.Estimator`.
    adanet_loss_decay: See :class:`adanet.Estimator`.
    worker_wait_timeout_secs: See :class:`adanet.Estimator`.
    model_dir: See :class:`adanet.Estimator`.
    config: See :class:`adanet.Estimator`.

  Returns:
    An :class:`adanet.AutoEnsembleEstimator` instance.

  Raises:
    ValueError: If any of the candidates in `candidate_pool` are not
      :class:`tf.estimator.Estimator` instances.
  """
  # pyformat: enable

  def __init__(self,
               head,
               candidate_pool,
               max_iteration_steps,
               logits_fn=None,
               adanet_lambda=0.,
               evaluator=None,
               metric_fn=None,
               force_grow=False,
               adanet_loss_decay=.9,
               worker_wait_timeout_secs=7200,
               model_dir=None,
               config=None):
    for candidate in candidate_pool:
      if isinstance(candidate, tf.estimator.Estimator):
        continue
      raise ValueError("Elements in candidate_pool must have type "
                       "tf.estimator.Estimator but got {}".format(
                           candidate.__class__))
    if logits_fn is None:
      logits_fn = _default_logits
    subnetwork_generator = _GeneratorFromCandidatePool(candidate_pool,
                                                       logits_fn)
    super(AutoEnsembleEstimator, self).__init__(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        adanet_lambda=adanet_lambda,
        evaluator=evaluator,
        metric_fn=metric_fn,
        force_grow=force_grow,
        adanet_loss_decay=adanet_loss_decay,
        worker_wait_timeout_secs=worker_wait_timeout_secs,
        model_dir=model_dir,
        config=config)
