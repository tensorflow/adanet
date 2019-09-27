"""An AdaNet ensemble definition in Tensorflow using a single graph.

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
import contextlib
import copy
import functools
import inspect
import os

from absl import logging
from adanet import tf_compat
from adanet.core.architecture import _Architecture
from adanet.core.eval_metrics import _EnsembleMetrics
from adanet.core.eval_metrics import _SubnetworkMetrics
from adanet.core.summary import monkey_patched_summaries
from adanet.ensemble import ComplexityRegularized
from adanet.subnetwork import TrainOpSpec
import tensorflow as tf_v1
import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.training import training as train
from tensorflow.python.training import training_util
# pylint: enable=g-direct-tensorflow-import

_VALID_METRIC_FN_ARGS = {"features", "labels", "predictions"}


class _EnsembleSpec(
    collections.namedtuple("_EnsembleSpec", [
        "name",
        "ensemble",
        "architecture",
        "subnetwork_builders",
        "subnetwork_specs",
        "predictions",
        "step",
        "loss",
        "adanet_loss",
        "train_op",
        "eval_metrics",
        "export_outputs",
    ])):
  """Ensemble training and evaluation `Tensors` and `Ops`.

  Args:
    name: String name of this ensemble. Should be unique in the graph.
    ensemble: The `adanet.ensemble.Ensemble` of interest.
    architecture: The `_Architecture` that represents this ensemble.
    subnetwork_builders: The Iterable of candidate subnetworks for the current
      iteration.
    predictions: Predictions `Tensor` or dict of `Tensor`.
    step: `tf.Variable` step counter representing the number of steps this
      ensemble trained for. Resets at every AdaNet iteration.
    loss: Loss `Tensor` as defined by the surrogate loss function Phi in
      Equations (4), (5), and (6). Must be either scalar, or with shape `[1]`.
    adanet_loss: Loss `Tensor` as defined by F(w) in Equation (4). Must be
      either scalar, or with shape `[1]`. The AdaNet algorithm aims to minimize
      this objective which balances training loss with the total complexity of
      the subnetworks in the ensemble.
    train_op: Candidate ensemble's mixture weights `TrainOpSpec`.
    eval_metrics: Tuple of (metric_fn, tensors) where metric_fn(tensors) returns
      the dict of eval metrics keyed by name. The values of the dict are the
      results of calling a metric function, namely a `(metric_tensor,
      update_op)` tuple. `metric_tensor` should be evaluated without any impact
      on state (typically is a pure computation based on variables.). For
      example, it should not trigger the `update_op` or require any input
      fetching.
    export_outputs: Describes the output signatures to be exported to
      `SavedModel` and used during serving. See `tf.estimator.EstimatorSpec`.
    subnetwork_specs: Iterable of `_SubnetworkSpecs` for this iteration.

  Returns:
    An `EnsembleSpec` object.
  """

  def __new__(cls,
              name,
              ensemble,
              architecture,
              subnetwork_builders,
              predictions,
              step,
              loss=None,
              adanet_loss=None,
              train_op=None,
              eval_metrics=None,
              export_outputs=None,
              subnetwork_specs=None):
    if subnetwork_specs is None:
      subnetwork_specs = []
    return super(_EnsembleSpec, cls).__new__(
        cls,
        name=name,
        ensemble=ensemble,
        architecture=architecture,
        subnetwork_builders=subnetwork_builders,
        subnetwork_specs=subnetwork_specs,
        predictions=predictions,
        step=step,
        loss=loss,
        adanet_loss=adanet_loss,
        train_op=train_op,
        eval_metrics=eval_metrics,
        export_outputs=export_outputs)


def _verify_metric_fn_args(metric_fn):
  if not metric_fn:
    return
  args = set(inspect.getargspec(metric_fn).args)
  invalid_args = list(args - _VALID_METRIC_FN_ARGS)
  if invalid_args:
    raise ValueError("metric_fn (%s) has following not expected args: %s" %
                     (metric_fn, invalid_args))


def _get_value(target, key):
  if isinstance(target, dict):
    return target[key]
  return target


def _to_train_op_spec(train_op):
  if isinstance(train_op, TrainOpSpec):
    return train_op
  return TrainOpSpec(train_op)


@contextlib.contextmanager
def _monkey_patch_context(iteration_step_scope, scoped_summary, trainable_vars):
  """Monkey-patches global attributes with subnetwork-specifics ones."""

  old_get_global_step_fn = tf_compat.v1.train.get_global_step
  old_get_or_create_global_step_fn = tf_compat.v1.train.get_or_create_global_step
  old_trainable_vars = tf_compat.v1.trainable_variables()

  def iteration_step(graph=None):
    graph = graph or tf_compat.v1.get_default_graph()
    with graph.as_default() as g, g.name_scope(None):
      with tf_compat.v1.variable_scope(
          iteration_step_scope, reuse=tf_compat.v1.AUTO_REUSE):
        return tf_compat.v1.get_variable(
            "iteration_step",
            shape=[],
            initializer=tf_compat.v1.zeros_initializer(),
            trainable=False,
            dtype=tf.int64)

  # monkey-patch global attributes.
  setattr(tf_compat.v1.train, "get_global_step", iteration_step)
  setattr(tf_compat.v1.train, "get_or_create_global_step", iteration_step)
  setattr(tf_v1.train, "get_global_step", iteration_step)
  setattr(tf_v1.train, "get_or_create_global_step", iteration_step)
  setattr(tf.train, "get_global_step", iteration_step)
  setattr(tf.train, "get_or_create_global_step", iteration_step)
  setattr(train, "get_global_step", iteration_step)
  setattr(training_util, "get_global_step", iteration_step)
  setattr(train, "get_or_create_global_step", iteration_step)
  setattr(training_util, "get_or_create_global_step", iteration_step)
  # The TPUEmbedding uses dummy variables to coordinate sending and receiving
  # gradients. If no gradients are computed on these dummy variables, the
  # TPUEmbedding will throw an error.
  embedding_variables = tf_compat.v1.get_collection(
      "tpu_embedding_dummy_table_variables")
  _set_trainable_variables(trainable_vars + embedding_variables)

  try:
    with monkey_patched_summaries(scoped_summary):
      yield
  finally:
    # Revert monkey-patches.
    new_trainable_vars = _new_trainable_variables(trainable_vars)
    _set_trainable_variables(old_trainable_vars + new_trainable_vars)
    setattr(training_util, "get_or_create_global_step",
            old_get_or_create_global_step_fn)
    setattr(train, "get_or_create_global_step",
            old_get_or_create_global_step_fn)
    setattr(training_util, "get_global_step", old_get_global_step_fn)
    setattr(train, "get_global_step", old_get_global_step_fn)
    setattr(tf.train, "get_or_create_global_step",
            old_get_or_create_global_step_fn)
    setattr(tf.train, "get_global_step", old_get_global_step_fn)
    setattr(tf_v1.train, "get_or_create_global_step",
            old_get_or_create_global_step_fn)
    setattr(tf_v1.train, "get_global_step", old_get_global_step_fn)
    setattr(tf_compat.v1.train, "get_or_create_global_step",
            old_get_or_create_global_step_fn)
    setattr(tf_compat.v1.train, "get_global_step", old_get_global_step_fn)


def _clear_trainable_variables():
  del tf_compat.v1.get_collection_ref(
      tf_compat.v1.GraphKeys.TRAINABLE_VARIABLES)[:]


def _set_trainable_variables(var_list):
  _clear_trainable_variables()
  for var in var_list:
    assert isinstance(var, tf.Variable)
    tf_compat.v1.add_to_collections(tf_compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                    var)


def _new_trainable_variables(old_vars):
  # Assumes that new trainable variables are always appended to the collection.
  return tf_compat.v1.trainable_variables()[len(old_vars):]


class _EnsembleBuilder(object):
  """Builds `_EnsembleSpec` instances.

  Args:
    head: A `tf.contrib.estimator.Head` instance.
    metric_fn: A function which should obey the following signature:
      - Args: can only have following three arguments in any order:
        * predictions: Predictions `Tensor` or dict of `Tensor` created by given
          `Head`.
        * features: Input `dict` of `Tensor` objects created by `input_fn` which
          is given to `estimator.evaluate` as an argument.
        * labels:  Labels `Tensor` or dict of `Tensor` (for multi-head) created
          by `input_fn` which is given to `estimator.evaluate` as an argument.
      - Returns: Dict of metric results keyed by name. Final metrics are a union
        of this and `Head's` existing metrics. If there is a name conflict
        between this and `estimator`s existing metrics, this will override the
        existing one. The values of the dict are the results of calling a metric
        function, namely a `(metric_tensor, update_op)` tuple.
    use_tpu: Whether AdaNet is running on TPU.
    export_subnetwork_logits: Include subnetwork logits in exports.
    export_subnetwork_last_layer: Include subnetwork last layer in exports.

  Returns:
    An `_EnsembleBuilder` instance.
  """

  _SUBNETWORK_LOGITS_EXPORT_SIGNATURE = "subnetwork_logits"
  _SUBNETWORK_LAST_LAYER_EXPORT_SIGNATURE = "subnetwork_last_layer"

  def __init__(self,
               head,
               metric_fn=None,
               use_tpu=False,
               export_subnetwork_logits=False,
               export_subnetwork_last_layer=False):
    _verify_metric_fn_args(metric_fn)

    self._head = head
    self._metric_fn = metric_fn
    self._use_tpu = use_tpu
    self._export_subnetwork_logits = export_subnetwork_logits
    self._export_subnetwork_last_layer = export_subnetwork_last_layer

  def build_ensemble_spec(self,
                          name,
                          candidate,
                          ensembler,
                          subnetwork_specs,
                          summary,
                          features,
                          mode,
                          iteration_number,
                          labels=None,
                          my_ensemble_index=None,
                          previous_ensemble_spec=None):
    """Builds an `_EnsembleSpec` with the given `adanet.ensemble.Candidate`.

    Args:
      name: The string name of the ensemble. Typically the name of the builder
        that returned the given `Subnetwork`.
      candidate: The `adanet.ensemble.Candidate` for this spec.
      ensembler: The :class:`adanet.ensemble.Ensembler` to use to ensemble a
        group of subnetworks.
      subnetwork_specs: Iterable of `_SubnetworkSpecs` for this iteration.
      summary: A `_ScopedSummary` instance for recording ensemble summaries.
      features: Input `dict` of `Tensor` objects.
      mode: Estimator `ModeKeys` indicating training, evaluation, or inference.
      iteration_number: Integer current iteration number.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head).
      my_ensemble_index: An integer holding the index of the ensemble in the
        candidates list of AdaNet.
      previous_ensemble_spec: Link the rest of the `_EnsembleSpec` from
        iteration t-1. Used for creating the subnetwork train_op.

    Returns:
      An `_EnsembleSpec` instance.
    """

    with tf_compat.v1.variable_scope("ensemble_{}".format(name)):
      step = tf_compat.v1.get_variable(
          "step",
          shape=[],
          initializer=tf_compat.v1.zeros_initializer(),
          trainable=False,
          dtype=tf.int64)
      # Convert to tensor so that users cannot mutate it.
      step_tensor = tf.convert_to_tensor(value=step)
      with summary.current_scope():
        summary.scalar("iteration_step/adanet/iteration_step", step_tensor)
      replay_indices = []
      if previous_ensemble_spec:
        replay_indices = copy.copy(
            previous_ensemble_spec.architecture.replay_indices)
      if my_ensemble_index is not None:
        replay_indices.append(my_ensemble_index)

      architecture = _Architecture(candidate.name, ensembler.name,
                                   replay_indices=replay_indices)
      previous_subnetworks = []
      previous_subnetwork_specs = []
      subnetwork_builders = []
      previous_ensemble = None
      if previous_ensemble_spec:
        previous_ensemble = previous_ensemble_spec.ensemble
        previous_architecture = previous_ensemble_spec.architecture
        keep_indices = range(len(previous_ensemble.subnetworks))
        if len(candidate.subnetwork_builders) == 1 and previous_ensemble:
          # Prune previous ensemble according to the subnetwork.Builder for
          # backwards compatibility.
          subnetwork_builder = candidate.subnetwork_builders[0]
          prune_previous_ensemble = getattr(subnetwork_builder,
                                            "prune_previous_ensemble", None)
          if callable(prune_previous_ensemble):
            logging.warn(
                "Using an `adanet.subnetwork.Builder#prune_previous_ensemble` "
                "is deprecated. Please use a custom `adanet.ensemble.Strategy` "
                "instead.")
            keep_indices = prune_previous_ensemble(previous_ensemble)
        for i, builder in enumerate(previous_ensemble_spec.subnetwork_builders):
          if i not in keep_indices:
            continue
          if builder not in candidate.previous_ensemble_subnetwork_builders:
            continue
          previous_subnetworks.append(previous_ensemble.subnetworks[i])
          previous_subnetwork_specs.append(
              previous_ensemble_spec.subnetwork_specs[i])
          subnetwork_builders.append(builder)
          architecture.add_subnetwork(*previous_architecture.subnetworks[i])
      for builder in candidate.subnetwork_builders:
        architecture.add_subnetwork(iteration_number, builder.name)
        subnetwork_builders.append(builder)
      subnetwork_map = {s.builder.name: s for s in subnetwork_specs}
      used_subnetwork_specs = [
          subnetwork_map[s.name] for s in candidate.subnetwork_builders
      ]
      subnetworks = [s.subnetwork for s in used_subnetwork_specs]
      ensemble_scope = tf_compat.v1.get_variable_scope()
      before_var_list = tf_compat.v1.trainable_variables()
      with summary.current_scope(), _monkey_patch_context(
          iteration_step_scope=ensemble_scope,
          scoped_summary=summary,
          trainable_vars=[]):
        ensemble = ensembler.build_ensemble(
            subnetworks,
            previous_ensemble_subnetworks=previous_subnetworks,
            features=features,
            labels=labels,
            logits_dimension=self._head.logits_dimension,
            training=mode == tf.estimator.ModeKeys.TRAIN,
            iteration_step=step_tensor,
            summary=summary,
            previous_ensemble=previous_ensemble)
      ensemble_var_list = _new_trainable_variables(before_var_list)

      estimator_spec = _create_estimator_spec(self._head, features, labels,
                                              mode, ensemble.logits,
                                              self._use_tpu)

      ensemble_loss = estimator_spec.loss
      adanet_loss = None
      if mode != tf.estimator.ModeKeys.PREDICT:
        adanet_loss = estimator_spec.loss
        # Add ensembler specific loss
        if isinstance(ensemble, ComplexityRegularized):
          adanet_loss += ensemble.complexity_regularization

      predictions = estimator_spec.predictions
      export_outputs = estimator_spec.export_outputs

      if self._export_subnetwork_logits and export_outputs and subnetwork_map:
        first_subnetwork_logits = list(
            subnetwork_map.values())[0].subnetwork.logits
        if isinstance(first_subnetwork_logits, dict):
          for head_name in first_subnetwork_logits.keys():
            subnetwork_logits = {
                subnetwork_name: subnetwork_spec.subnetwork.logits[head_name]
                for subnetwork_name, subnetwork_spec in subnetwork_map.items()
            }
            export_outputs.update({
                "{}_{}".format(
                    _EnsembleBuilder._SUBNETWORK_LOGITS_EXPORT_SIGNATURE,
                    head_name):
                    tf.estimator.export.PredictOutput(subnetwork_logits)
            })
        else:
          subnetwork_logits = {
              subnetwork_name: subnetwork_spec.subnetwork.logits
              for subnetwork_name, subnetwork_spec in subnetwork_map.items()
          }
          export_outputs.update({
              _EnsembleBuilder._SUBNETWORK_LOGITS_EXPORT_SIGNATURE:
                  tf.estimator.export.PredictOutput(subnetwork_logits)
          })

      if (self._export_subnetwork_last_layer and export_outputs and
          subnetwork_map and
          list(subnetwork_map.values())[0].subnetwork.last_layer is not None):
        first_subnetwork_last_layer = list(
            subnetwork_map.values())[0].subnetwork.last_layer
        if isinstance(first_subnetwork_last_layer, dict):
          for head_name in first_subnetwork_last_layer.keys():
            subnetwork_last_layer = {
                subnetwork_name:
                subnetwork_spec.subnetwork.last_layer[head_name]
                for subnetwork_name, subnetwork_spec in subnetwork_map.items()
            }
            export_outputs.update({
                "{}_{}".format(
                    _EnsembleBuilder._SUBNETWORK_LAST_LAYER_EXPORT_SIGNATURE,
                    head_name):
                    tf.estimator.export.PredictOutput(subnetwork_last_layer)
            })
        else:
          subnetwork_last_layer = {
              subnetwork_name: subnetwork_spec.subnetwork.last_layer
              for subnetwork_name, subnetwork_spec in subnetwork_map.items()
          }
          export_outputs.update({
              _EnsembleBuilder._SUBNETWORK_LAST_LAYER_EXPORT_SIGNATURE:
                  tf.estimator.export.PredictOutput(subnetwork_last_layer)
          })

      if ensemble.predictions and predictions:
        predictions.update(ensemble.predictions)
      if ensemble.predictions and export_outputs:
        export_outputs.update({
            k: tf.estimator.export.PredictOutput(v)
            for k, v in ensemble.predictions.items()
        })

      ensemble_metrics = _EnsembleMetrics(use_tpu=self._use_tpu)
      if mode == tf.estimator.ModeKeys.EVAL:
        ensemble_metrics.create_eval_metrics(
            features=features,
            labels=labels,
            estimator_spec=estimator_spec,
            metric_fn=self._metric_fn,
            architecture=architecture)

      if mode == tf.estimator.ModeKeys.TRAIN:
        with summary.current_scope():
          summary.scalar("loss", estimator_spec.loss)

      # Create train ops for training subnetworks and ensembles.
      train_op = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        # Note that these mixture weights are on top of the last_layer of the
        # subnetwork constructed in TRAIN mode, which means that dropout is
        # still applied when the mixture weights are being trained.
        ensemble_scope = tf_compat.v1.get_variable_scope()
        with tf_compat.v1.variable_scope("train_mixture_weights"):
          with summary.current_scope(), _monkey_patch_context(
              iteration_step_scope=ensemble_scope,
              scoped_summary=summary,
              trainable_vars=ensemble_var_list):
            # For backwards compatibility.
            subnetwork_builder = candidate.subnetwork_builders[0]
            old_train_op_fn = getattr(subnetwork_builder,
                                      "build_mixture_weights_train_op", None)
            if callable(old_train_op_fn):
              logging.warn(
                  "The `build_mixture_weights_train_op` method is deprecated. "
                  "Please use the `Ensembler#build_train_op` instead.")
              train_op = _to_train_op_spec(
                  subnetwork_builder.build_mixture_weights_train_op(
                      loss=adanet_loss,
                      var_list=ensemble_var_list,
                      logits=ensemble.logits,
                      labels=labels,
                      iteration_step=step_tensor,
                      summary=summary))
            else:
              train_op = _to_train_op_spec(
                  ensembler.build_train_op(
                      ensemble=ensemble,
                      loss=adanet_loss,
                      var_list=ensemble_var_list,
                      labels=labels,
                      iteration_step=step_tensor,
                      summary=summary,
                      previous_ensemble=previous_ensemble))
    return _EnsembleSpec(
        name=name,
        architecture=architecture,
        subnetwork_builders=subnetwork_builders,
        subnetwork_specs=previous_subnetwork_specs + used_subnetwork_specs,
        ensemble=ensemble,
        predictions=predictions,
        step=step,
        loss=ensemble_loss,
        adanet_loss=adanet_loss,
        train_op=train_op,
        eval_metrics=ensemble_metrics.eval_metrics_tuple(),
        export_outputs=export_outputs)


def _create_estimator_spec(head, features, labels, mode, logits, use_tpu):
  """Creates the head's EstimatorSpec or TPUEstimatorSpec on TPU."""

  if use_tpu:
    create_spec_fn = head._create_tpu_estimator_spec  # pylint: disable=protected-access
  else:
    create_spec_fn = head.create_estimator_spec
  return create_spec_fn(
      features=features,
      labels=labels,
      mode=mode,
      logits=logits,
      train_op_fn=lambda _: tf.no_op())


class _SubnetworkSpec(
    collections.namedtuple("_SubnetworkSpec", [
        "name",
        "subnetwork",
        "builder",
        "predictions",
        "step",
        "loss",
        "train_op",
        "eval_metrics",
        "asset_dir",
    ])):
  """Subnetwork training and evaluation `Tensors` and `Ops`.

  Args:
    name: String name of this subnetwork. Should be unique in the graph.
    subnetwork: The `adanet.subnetwork.Subnetwork` for this spec.
    builder: The `adanet.subnetwork.Builder` that produced `subnetwork`.
    predictions: Predictions `Tensor` or dict of `Tensor`.
    step: `tf.Variable` step counter representing the number of steps this
      subnetwork trained for. Resets at every AdaNet iteration.
    loss: Loss `Tensor` as computed by the `Head`. Must be either scalar, or
      with shape `[1]`.
    train_op: Candidate subnetwork's `TrainOpSpec`.
    eval_metrics: Tuple of (metric_fn, tensors) where metric_fn(tensors) returns
      the dict of eval metrics keyed by name. The values of the dict are the
      results of calling a metric function, namely a `(metric_tensor,
      update_op)` tuple. `metric_tensor` should be evaluated without any impact
      on state (typically is a pure computation based on variables.). For
      example, it should not trigger the `update_op` or require any input
      fetching.
    asset_dir: Checkpoint directory for the sub-estimators.

  Returns:
    A `_SubnetworkSpec` object.
  """

  def __new__(cls,
              name,
              subnetwork,
              builder,
              predictions,
              step,
              loss=None,
              train_op=None,
              eval_metrics=None,
              asset_dir=None):
    return super(_SubnetworkSpec, cls).__new__(
        cls,
        name=name,
        subnetwork=subnetwork,
        builder=builder,
        predictions=predictions,
        step=step,
        loss=loss,
        train_op=train_op,
        eval_metrics=eval_metrics,
        asset_dir=asset_dir)


class _SubnetworkManager(object):
  """Builds `_SubnetworkSpec` instances.

  This class manages an `adanet.subnetwork.Builder`, creates its subnetwork and
  train ops, and returns a `_SubnetworkSpec` that holds them.

  Args:
    head: A `tf.contrib.estimator.Head` instance.
    metric_fn: A function which should obey the following signature:
      - Args: can only have following three arguments in any order:
        * predictions: Predictions `Tensor` or dict of `Tensor` created by given
          `Head`.
        * features: Input `dict` of `Tensor` objects created by `input_fn` which
          is given to `estimator.evaluate` as an argument.
        * labels:  Labels `Tensor` or dict of `Tensor` (for multi-head) created
          by `input_fn` which is given to `estimator.evaluate` as an argument.
      - Returns: Dict of metric results keyed by name. Final metrics are a union
        of this and `Head's` existing metrics. If there is a name conflict
        between this and `estimator`s existing metrics, this will override the
        existing one. The values of the dict are the results of calling a metric
        function, namely a `(metric_tensor, update_op)` tuple.
    use_tpu: Whether AdaNet is running on TPU.

  Returns:
    An `_SubnetworkManager` instance.

  Raises:
    ValueError: If `max_steps` is <= 0.
  """

  def __init__(self, head, metric_fn=None, use_tpu=False):
    _verify_metric_fn_args(metric_fn)
    self._head = head
    self._metric_fn = metric_fn
    self._use_tpu = use_tpu

  def build_subnetwork_spec(self,
                            name,
                            subnetwork_builder,
                            summary,
                            features,
                            mode,
                            labels=None,
                            previous_ensemble=None,
                            config=None):
    """Builds a `_SubnetworkSpec` from the given `adanet.subnetwork.Builder`.

    Args:
      name: String name of the subnetwork.
      subnetwork_builder: A `adanet.Builder` instance which defines how to train
        the subnetwork and ensemble mixture weights.
      summary: A `_ScopedSummary` instance for recording ensemble summaries.
      features: Input `dict` of `Tensor` objects.
      mode: Estimator's `ModeKeys`.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head). Can be `None`.
      previous_ensemble: The previous `Ensemble` from iteration t-1. Used for
        creating the subnetwork train_op.
      config: The `tf.estimator.RunConfig` to use this iteration.

    Returns:
      An new `EnsembleSpec` instance with the `Subnetwork` appended.
    """

    before_var_list = tf_compat.v1.trainable_variables()
    with tf_compat.v1.variable_scope("subnetwork_{}".format(name)):
      step = tf_compat.v1.get_variable(
          "step",
          shape=[],
          initializer=tf_compat.v1.zeros_initializer(),
          trainable=False,
          dtype=tf.int64)

      # Convert to tensor so that users cannot mutate it.
      step_tensor = tf.convert_to_tensor(value=step)
      with summary.current_scope():
        summary.scalar("iteration_step/adanet/iteration_step", step_tensor)
      if config:
        subnetwork_config = config.replace(
            model_dir=os.path.join(config.model_dir, "assets", name))
      else:
        subnetwork_config = tf.estimator.RunConfig()

      build_subnetwork = functools.partial(
          subnetwork_builder.build_subnetwork,
          features=features,
          logits_dimension=self._head.logits_dimension,
          training=mode == tf.estimator.ModeKeys.TRAIN,
          iteration_step=step_tensor,
          summary=summary,
          previous_ensemble=previous_ensemble)
      # Check which args are in the implemented build_subnetwork method
      # signature for backwards compatibility.
      defined_args = inspect.getargspec(
          subnetwork_builder.build_subnetwork).args
      if "labels" in defined_args:
        build_subnetwork = functools.partial(build_subnetwork, labels=labels)
      if "config" in defined_args:
        build_subnetwork = functools.partial(
            build_subnetwork, config=subnetwork_config)
      subnetwork_scope = tf_compat.v1.get_variable_scope()
      with summary.current_scope(), _monkey_patch_context(
          iteration_step_scope=subnetwork_scope,
          scoped_summary=summary,
          trainable_vars=[]):
        subnetwork = build_subnetwork()
      subnetwork_var_list = _new_trainable_variables(before_var_list)

      estimator_spec = _create_estimator_spec(self._head, features, labels,
                                              mode, subnetwork.logits,
                                              self._use_tpu)

      subnetwork_metrics = _SubnetworkMetrics(self._use_tpu)
      if mode == tf.estimator.ModeKeys.EVAL:
        subnetwork_metrics.create_eval_metrics(
            features=features,
            labels=labels,
            estimator_spec=estimator_spec,
            metric_fn=self._metric_fn)

      if mode == tf.estimator.ModeKeys.TRAIN:
        with summary.current_scope():
          summary.scalar("loss", estimator_spec.loss)

      # Create train ops for training subnetworks and ensembles.
      train_op = None
      if mode == tf.estimator.ModeKeys.TRAIN and subnetwork_builder:
        with summary.current_scope(), _monkey_patch_context(
            iteration_step_scope=subnetwork_scope,
            scoped_summary=summary,
            trainable_vars=subnetwork_var_list):
          train_op = _to_train_op_spec(
              subnetwork_builder.build_subnetwork_train_op(
                  subnetwork=subnetwork,
                  loss=estimator_spec.loss,
                  var_list=subnetwork_var_list,
                  labels=labels,
                  iteration_step=step_tensor,
                  summary=summary,
                  previous_ensemble=previous_ensemble))
    return _SubnetworkSpec(
        name=name,
        subnetwork=subnetwork,
        builder=subnetwork_builder,
        predictions=estimator_spec.predictions,
        loss=estimator_spec.loss,
        step=step,
        train_op=train_op,
        eval_metrics=subnetwork_metrics.eval_metrics_tuple(),
        asset_dir=subnetwork_config.model_dir)
