"""Common utilities for AutoEnsemblers.

Copyright 2019 The AdaNet Authors. All Rights Reserved.

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
import inspect

from adanet import subnetwork as subnetwork_lib
from adanet import tf_compat

import tensorflow as tf
tf = tf.compat.v2


def _default_logits(estimator_spec):
  from tensorflow.python.estimator.canned import prediction_keys  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

  if isinstance(estimator_spec.predictions, dict):
    pred_keys = prediction_keys.PredictionKeys
    if pred_keys.LOGITS in estimator_spec.predictions:
      return estimator_spec.predictions[pred_keys.LOGITS]
    if pred_keys.PREDICTIONS in estimator_spec.predictions:
      return estimator_spec.predictions[pred_keys.PREDICTIONS]
  return estimator_spec.predictions


class _SecondaryTrainOpRunnerHook(tf_compat.SessionRunHook):
  """A hook for running a train op separate from the main session run call."""

  def __init__(self, train_op):
    """Initializes a `_SecondaryTrainOpRunnerHook`.

    Args:
      train_op: The secondary train op to execute before runs.
    """

    self._train_op = train_op

  def before_run(self, run_context):
    run_context.session.run(self._train_op)


class AutoEnsembleSubestimator(  # pylint: disable=g-classes-have-attributes
    collections.namedtuple("AutoEnsembleSubestimator",
                           ["estimator", "train_input_fn"])):
  """A subestimator to train and consider for ensembling.

  Args:
    estimator: A `tf.estimator.Estimator` or `tf.estimator.tpu.TPUEstimator`
      instance to consider for ensembling.
    train_input_fn: A function that provides input data for training as
      minibatches. It can be used to implement ensemble methods like bootstrap
      aggregating (a.k.a bagging) where each subnetwork trains on different
      slices of the training data. The function should construct and return one
      of the following:
       * A `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
         `(features, labels)` with same constraints as below. NOTE: A Dataset
           must return *at least* two batches before hitting the end-of-input,
           otherwise all of training terminates.
         TODO: Figure out how to handle single-batch datasets.
       * A tuple `(features, labels)`: Where `features` is a `tf.Tensor` or a
         dictionary of string feature name to `Tensor` and `labels` is a
         `Tensor` or a dictionary of string label name to `Tensor`. Both
         `features` and `labels` are consumed by `estimator#model_fn`. They
         should satisfy the expectation of `estimator#model_fn` from inputs.

  Returns:
    An `AutoEnsembleSubestimator` instance to be auto-ensembled.
  """

  # pylint: enable=g-classes-have-attributes

  def __new__(cls, estimator, train_input_fn=None):
    return super(AutoEnsembleSubestimator, cls).__new__(cls, estimator,
                                                        train_input_fn)


class _BuilderFromSubestimator(subnetwork_lib.Builder):
  """An `adanet.Builder` from a :class:`tf.estimator.Estimator`."""

  def __init__(self, name, subestimator, logits_fn, last_layer_fn, config):
    self._name = name
    self._subestimator = subestimator
    self._logits_fn = logits_fn
    self._last_layer_fn = last_layer_fn
    self._config = config

  @property
  def name(self):
    return self._name

  def _call_model_fn(self, subestimator, features, labels, mode, summary):
    with summary.current_scope():
      model_fn = subestimator.estimator.model_fn
      estimator_spec = model_fn(
          features=features, labels=labels, mode=mode, config=self._config)
      logits = self._logits_fn(estimator_spec=estimator_spec)
      last_layer = logits
      if self._last_layer_fn:
        last_layer = self._last_layer_fn(estimator_spec=estimator_spec)

      if estimator_spec.scaffold and estimator_spec.scaffold.local_init_op:
        local_init_op = estimator_spec.scaffold.local_init_op
      else:
        local_init_op = None

      train_op = subnetwork_lib.TrainOpSpec(
          estimator_spec.train_op,
          chief_hooks=estimator_spec.training_chief_hooks,
          hooks=estimator_spec.training_hooks)
    return logits, last_layer, train_op, local_init_op

  def build_subnetwork(self,
                       features,
                       labels,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble,
                       config=None):
    # We don't need an EVAL mode since AdaNet takes care of evaluation for us.
    mode = tf.estimator.ModeKeys.PREDICT
    if training:
      mode = tf.estimator.ModeKeys.TRAIN

    # Call in template to ensure that variables are created once and reused.
    call_model_fn_template = tf.compat.v1.make_template("model_fn",
                                                        self._call_model_fn)
    subestimator_features, subestimator_labels = features, labels
    local_init_ops = []
    subestimator = self._subestimator(config)
    if training and subestimator.train_input_fn:
      # TODO: Consider tensorflow_estimator/python/estimator/util.py.
      inputs = subestimator.train_input_fn()
      if isinstance(inputs, (tf_compat.DatasetV1, tf_compat.DatasetV2)):
        subestimator_features, subestimator_labels = (
            tf_compat.make_one_shot_iterator(inputs).get_next())
      else:
        subestimator_features, subestimator_labels = inputs

      # Construct subnetwork graph first because of dependencies on scope.
      _, _, bagging_train_op_spec, sub_local_init_op = call_model_fn_template(
          subestimator, subestimator_features, subestimator_labels, mode,
          summary)
      # Graph for ensemble learning gets model_fn_1 for scope.
      logits, last_layer, _, ensemble_local_init_op = call_model_fn_template(
          subestimator, features, labels, mode, summary)

      if sub_local_init_op:
        local_init_ops.append(sub_local_init_op)
      if ensemble_local_init_op:
        local_init_ops.append(ensemble_local_init_op)

      # Run train op in a hook so that exceptions can be intercepted by the
      # AdaNet framework instead of the Estimator's monitored training session.
      hooks = bagging_train_op_spec.hooks + (_SecondaryTrainOpRunnerHook(
          bagging_train_op_spec.train_op),)
      train_op_spec = subnetwork_lib.TrainOpSpec(
          train_op=tf.no_op(),
          chief_hooks=bagging_train_op_spec.chief_hooks,
          hooks=hooks)
    else:
      logits, last_layer, train_op_spec, local_init_op = call_model_fn_template(
          subestimator, features, labels, mode, summary)
      if local_init_op:
        local_init_ops.append(local_init_op)

    # TODO: Replace with variance complexity measure.
    complexity = tf.constant(0.)
    return subnetwork_lib.Subnetwork(
        logits=logits,
        last_layer=last_layer,
        shared={"train_op": train_op_spec},
        complexity=complexity,
        local_init_ops=local_init_ops)

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    return subnetwork.shared["train_op"]


def _convert_to_subestimator(candidate):
  """Converts a candidate to an AutoEnsembleSubestimator."""

  if callable(candidate):
    return candidate
  if isinstance(candidate, AutoEnsembleSubestimator):
    return lambda config: candidate

  from tensorflow_estimator.python.estimator import estimator as estimator_lib  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  if isinstance(candidate,
                (estimator_lib.Estimator, estimator_lib.EstimatorV2)):
    return lambda config: AutoEnsembleSubestimator(candidate)
  raise ValueError(
      "subestimator in candidate_pool must have type tf.estimator.Estimator or "
      "adanet.AutoEnsembleSubestimator but got {}".format(candidate.__class__))


class _GeneratorFromCandidatePool(subnetwork_lib.Generator):
  """An `adanet.Generator` from a pool of `Estimator` and `Model` instances."""

  def __init__(self, candidate_pool, logits_fn, last_layer_fn):
    self._candidate_pool = candidate_pool
    if logits_fn is None:
      logits_fn = _default_logits
    self._logits_fn = logits_fn
    self._last_layer_fn = last_layer_fn

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports, config):
    assert config
    builders = []
    candidate_pool = self._maybe_call_candidate_pool(config, iteration_number)

    if isinstance(candidate_pool, dict):
      for name in sorted(candidate_pool):
        builders.append(
            _BuilderFromSubestimator(
                name,
                _convert_to_subestimator(candidate_pool[name]),
                logits_fn=self._logits_fn,
                last_layer_fn=self._last_layer_fn,
                config=config))
      return builders

    for i, estimator in enumerate(candidate_pool):
      name = "{class_name}{index}".format(
          class_name=estimator.__class__.__name__, index=i)
      builders.append(
          _BuilderFromSubestimator(
              name,
              _convert_to_subestimator(estimator),
              logits_fn=self._logits_fn,
              last_layer_fn=self._last_layer_fn,
              config=config))
    return builders

  def _maybe_call_candidate_pool(self, config, iteration_number):
    if callable(self._candidate_pool):
      # candidate_pool can be a function.
      candidate_pool_args = inspect.getargs(self._candidate_pool.__code__).args
      if "iteration_number" in candidate_pool_args:
        # TODO: Make the "config" argument optional using introspection.
        return self._candidate_pool(
            config=config, iteration_number=iteration_number)
      else:
        return self._candidate_pool(config=config)

    return self._candidate_pool
