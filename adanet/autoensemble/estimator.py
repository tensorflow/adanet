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
import tensorflow as tf


class _BuilderFromEstimator(Builder):
  """An `adanet.Builder` from a `tf.estimator.Estimator`."""

  def __init__(self, estimator, iteration_number, index, logits_key):
    self._estimator = estimator
    self._iteration_number = iteration_number
    self._index = index
    self._logits_key = logits_key

  @property
  def name(self):
    """See `adanet.subnetwork.Builder`."""

    return "t{t}_{class_name}{index}".format(
        t=self._iteration_number,
        class_name=self._estimator.__class__.__name__,
        index=self._index)

  def build_subnetwork(self, features, labels, logits_dimension, training,
                       iteration_step, summary, previous_ensemble):
    """See `adanet.subnetwork.Builder`."""

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

    self._subnetwork_train_op = estimator_spec.train_op

    # TODO: Replace with variance complexity measure.
    complexity = tf.constant(0.)
    if isinstance(estimator_spec.predictions, dict):
      logits = estimator_spec.predictions[self._logits_key]
    else:
      logits = estimator_spec.predictions
    return Subnetwork(
        logits=logits,
        last_layer=logits,
        persisted_tensors={},
        complexity=complexity)

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    """See `adanet.subnetwork.Builder`."""

    return self._subnetwork_train_op

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    """See `adanet.subnetwork.Builder`."""

    return tf.no_op()


class _GeneratorFromCandidatePool(Generator):
  """An `adanet.Generator` from a pool of `Estimator` and `Model` instances."""

  def __init__(self, candidate_pool, logits_key):
    self._candidate_pool = candidate_pool
    self._logits_key = logits_key

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    """See `adanet.subnetwork.Generator`."""

    builders = []
    for i, candidate in enumerate(self._candidate_pool):
      if isinstance(candidate, tf.estimator.Estimator):
        builders.append(
            _BuilderFromEstimator(
                candidate,
                iteration_number,
                index=i,
                logits_key=self._logits_key))
        continue
    return builders


class AutoEnsembleEstimator(Estimator):
  """An Estimator that learns to ensemble models.

  This `Estimator` learns to automatically ensemble models from a candidate
  pool using the Adanet algorithm.

  Example:

  ```python
  numeric_feature = numeric_column(...)
  categorical_column_a = categorical_column_with_hash_bucket(...)
  categorical_column_b = categorical_column_with_hash_bucket(...)

  categorical_feature_a_x_categorical_feature_b = crossed_column(...)
  categorical_feature_a_emb = embedding_column(
      categorical_column=categorical_feature_a, ...)
  categorical_feature_b_emb = embedding_column(
      categorical_column=categorical_feature_b, ...)

  head = tf.contrib.estimator.multi_label_head(n_classes=3)

  # Learn to ensemble linear and DNN models.
  estimator = adanet.AutoEnsembleEstimator(
      head=head,
      candidate_pool=[
          tf.estimator.LinearEstimator(
              head=head,
              feature_columns=[categorical_feature_a_x_categorical_feature_b],
              optimizer=tf.train.FtrlOptimizer(...)),
          tf.estimator.DNNEstimator(
              head=head,
              feature_columns=[
                  categorical_feature_a_emb, categorical_feature_b_emb,
                  numeric_feature],
              optimizer=tf.train.ProximalAdagradOptimizer(...),
              hidden_units=[1000, 500, 100])],
      max_iteration_steps=50)

  # Input builders
  def input_fn_train:
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
    # index.
    pass
  def input_fn_eval:
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class
    # index.
    pass
  def input_fn_predict:
    # Returns tf.data.Dataset of (x, None) tuple.
    pass
  estimator.train(input_fn=input_fn_train, steps=100)
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```
  """

  def __init__(self,
               head,
               candidate_pool,
               max_iteration_steps,
               logits_key="logits",
               adanet_lambda=0.,
               evaluator=None,
               force_grow=False,
               adanet_loss_decay=.9,
               worker_wait_timeout_secs=7200,
               train_batch_size=None,
               model_dir=None,
               config=None):
    """Initializes an `AutoEnsembleEstimator`.

    Args:
      head: A `tf.contrib.estimator.Head` instance for computing loss and
        evaluation metrics for every candidate.
      candidate_pool: List of `tf.estimator.Estimator` objects that are
        candidates to ensemble at each iteration. The order does not directly
        affect which candidates will be included in the final ensemble.
      max_iteration_steps: Total number of steps for which to train candidates
        per iteration. If `OutOfRange` or `StopIteration` occurs in the middle,
        training stops before `max_iteration_steps` steps.
      logits_key: String key for accessing `tf.estimator.Estimator` logits which
        are present in its `model_fn`'s `EstimatorSpec` predictions dictionary.
      adanet_lambda: See `adanet.Estimator`.
      evaluator:  See `adanet.Estimator`.
      force_grow:  See `adanet.Estimator`.
      adanet_loss_decay: See `adanet.Estimator`.
      worker_wait_timeout_secs: See `adanet.Estimator`.
      train_batch_size:  See `adanet.Estimator`.
      model_dir: See `adanet.Estimator`.
      config: See `adanet.Estimator`.

    Returns:
      An `AutoEnsembleEstimator` instance.

    Raises:
      ValueError: If any of the candidates in `candidate_pool` are not
        `tf.estimator.Estimator` instances.
    """

    for candidate in candidate_pool:
      if isinstance(candidate, tf.estimator.Estimator):
        continue
      raise ValueError("Elements in candidate_pool must have type "
                       "tf.estimator.Estimator but got {}".format(
                           candidate.__class__))
    subnetwork_generator = _GeneratorFromCandidatePool(candidate_pool,
                                                       logits_key)
    super(AutoEnsembleEstimator, self).__init__(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        adanet_lambda=adanet_lambda,
        evaluator=evaluator,
        force_grow=force_grow,
        adanet_loss_decay=adanet_loss_decay,
        worker_wait_timeout_secs=worker_wait_timeout_secs,
        model_dir=model_dir,
        config=config)
