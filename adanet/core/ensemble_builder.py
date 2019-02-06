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
import functools
import inspect

from adanet.core import dict_utils
from adanet.core.architecture import _Architecture
from adanet.core.ensemble import ComplexityRegularized
from adanet.core.subnetwork import TrainOpSpec
from adanet.core.summary import monkey_patched_summaries
import six
import tensorflow as tf

from tensorflow.python.training import training_util  # pylint: disable=g-direct-tensorflow-import

_VALID_METRIC_FN_ARGS = {"features", "labels", "predictions"}

_LABELS_KEY = "__labels__"
_FEATURES_KEY = "__features__"
_PREDICTIONS_KEY = "__predictions__"
_KWARGS_KEY = "__kwargs__"
_PREFIXES = (_LABELS_KEY, _FEATURES_KEY, _PREDICTIONS_KEY, _KWARGS_KEY)


# TODO: create an eval metrics object to encapulate the metric
# tuples.
class _EnsembleSpec(
    collections.namedtuple("_EnsembleSpec", [
        "name",
        "ensemble",
        "architecture",
        "subnetwork_builders",
        "predictions",
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

  Returns:
    An `EnsembleSpec` object.
  """

  def __new__(cls,
              name,
              ensemble,
              architecture,
              subnetwork_builders,
              predictions,
              loss=None,
              adanet_loss=None,
              train_op=None,
              eval_metrics=None,
              export_outputs=None):
    return super(_EnsembleSpec, cls).__new__(
        cls,
        name=name,
        ensemble=ensemble,
        architecture=architecture,
        subnetwork_builders=subnetwork_builders,
        predictions=predictions,
        loss=loss,
        adanet_loss=adanet_loss,
        train_op=train_op,
        eval_metrics=eval_metrics,
        export_outputs=export_outputs)


def _architecture_as_metric(architecture):
  """Returns a representation of the ensemble's architecture as a tf.metric."""

  def _architecture_metric_fn(**kwargs):
    """Manually creates the tf.metric with a serialized tf.Summary proto."""

    del kwargs  # Unused.

    # TODO: Should architecture.subnetworks be sorted by iteration
    # number first? Or perhaps, to make this more general, to have one line for
    # each iteration, with "|" as a delimiter if there are multiple subnetworks
    # in one iteration? Something like:
    # 0 linear
    # 1 dnn_width_32_depth_1 | dnn_width_64_depth_1
    # 2
    # 3 dnn_with_32_depth_2
    architecture_ = " | ".join([name for _, name in architecture.subnetworks])
    architecture_ = "| {} |".format(architecture_)
    summary_metadata = tf.SummaryMetadata(
        plugin_data=tf.SummaryMetadata.PluginData(plugin_name="text"))
    summary_proto = tf.summary.Summary()
    summary_proto.value.add(
        metadata=summary_metadata,
        tag="architecture/adanet",
        tensor=tf.make_tensor_proto(architecture_, dtype=tf.string))
    architecture_summary = tf.convert_to_tensor(
        summary_proto.SerializeToString(), name="architecture")
    return {"architecture/adanet/ensembles": (architecture_summary, tf.no_op())}

  return _architecture_metric_fn


def _verify_metric_fn_args(metric_fn):
  if not metric_fn:
    return
  args = set(inspect.getargspec(metric_fn).args)
  invalid_args = list(args - _VALID_METRIC_FN_ARGS)
  if invalid_args:
    raise ValueError("metric_fn (%s) has following not expected args: %s" %
                     (metric_fn, invalid_args))


def _reflective_call(fn, **kwargs):
  """Extracts fn's required args from **kwargs and calls fn with them."""

  argspec = inspect.getargspec(fn)
  args = {k: v for k, v in six.iteritems(kwargs) if k in argspec.args}
  if argspec.keywords:
    args.update(kwargs[_KWARGS_KEY])
  return fn(**args)


def _reconstruct_tuple_keys(tensors):
  """Reconstructs tuple keys from flat strings if tensors is a dict."""

  if not isinstance(tensors, dict):
    return tensors

  result = {}
  for key, value in six.iteritems(tensors):
    parts = key.split("|")
    if len(parts) > 1:
      result[tuple(parts)] = value
    else:
      result[key] = value
  return result


def _create_metric_fn(metric_fn):
  """Wraps the metric_fn to scope its returned metrics by group_name."""

  def _wrapped_metric_fn(**kwargs):
    """The wrapping function to be returned."""

    if not metric_fn:
      return {}

    kwargs = dict_utils.unflatten_dict(kwargs, prefixes=_PREFIXES)
    kwargs = {k: _reconstruct_tuple_keys(v) for k, v in six.iteritems(kwargs)}
    kwargs_ = {}
    for key, value in six.iteritems(kwargs):
      if key in _PREFIXES and key != _KWARGS_KEY:
        kwargs_[key.replace("_", "")] = value
      else:
        kwargs_[key] = value
    kwargs = kwargs_

    metrics = _reflective_call(metric_fn, **kwargs)
    wrapped_metrics = {}
    # Hooks on TPU cannot depend on any graph Tensors. Instead the metric values
    # are stored in Variables that are later read from the evaluation hooks.
    for i, key in enumerate(sorted(metrics)):
      tensor, op = metrics[key]
      # `key` cannot be in the var name since it can contain illegal characters.
      var = tf.get_variable(
          "metric_{}".format(i),
          shape=tensor.shape,
          dtype=tensor.dtype,
          trainable=False,
          initializer=tf.zeros_initializer(),
          collections=[tf.GraphKeys.LOCAL_VARIABLES])
      if isinstance(op, tf.Operation):
        with tf.control_dependencies([op]):
          op = tf.assign(var, tensor)
      metric = (var, tf.assign(var, op))
      wrapped_metrics[key] = metric
    return wrapped_metrics

  return tf.make_template("metric_fn_template", _wrapped_metric_fn)


def _prefix(tensors, flat_key, default_key):
  """Prefixes tensors by either flat_key or default_key.

  If tensors is a dict each tensor is rekeyed as group_name/flat_key/key. If
  tensors is a single Tensor, it is keyed by group_name/default_key.

  Args:
    tensors: A Tensor or dictionary of Tensors.
    flat_key: The key to use in the prefix if tensors is a dictionary.
    default_key: The default key to use if tensors is a single Tensor.

  Returns:
    A dictionary of tensors prefixed by group_name and key. If tensors is a
    single Tensor, the returned dictionary will only have one element.
  """
  prefix = default_key
  if isinstance(tensors, dict):
    prefix = flat_key
    tensors_ = {}
    for key in six.iterkeys(tensors):
      # multi_head uses tuples of strings as the key.
      if isinstance(key, tuple):
        tensors_["|".join(key)] = tensors[key]
      else:
        tensors_[key] = tensors[key]
    tensors = tensors_

  tensors = {prefix: tensors}
  tensors = dict_utils.flatten_dict(tensors)
  return tensors


def _create_metrics(features, labels, estimator_spec, metric_fn, params):
  """Creates eval metric functions and tensors for the given group."""

  metric_fns = []
  tensors = {}

  # If estimator_spec is not a TPUEstimatorSpec we create dummy eval_metric_fn
  # and tensors.
  if isinstance(estimator_spec, tf.estimator.EstimatorSpec):
    spec_metric_fn = lambda: estimator_spec.eval_metric_ops
    spec_tensors = {}
  else:
    spec_metric_fn, spec_tensors = estimator_spec.eval_metrics
  metric_fns.append(_create_metric_fn(spec_metric_fn))
  for key, value in six.iteritems(spec_tensors):
    tensors["{}/{}".format(_KWARGS_KEY, key)] = value

  loss_fn = lambda loss: {"loss": tf.metrics.mean(loss)}
  metric_fns.append(_create_metric_fn(loss_fn))
  # All tensors outfed from the TPU must be batch-major.
  batch_size = params.get("batch_size", 1) if params else 1
  tensors["loss"] = tf.ones((batch_size, 1)) * estimator_spec.loss

  # TODO: (Optimization): features and labels are shared between all
  # group metrics so they should only be outfed once. However, this makes the
  # kwarg parsing harder.
  tensors.update(_prefix(features, _FEATURES_KEY, "features"))
  tensors.update(_prefix(labels, _LABELS_KEY, "labels"))
  tensors.update(
      _prefix(estimator_spec.predictions, _PREDICTIONS_KEY, "predictions"))

  # NOTE: the user supplied metrics_fn must be added last. This is because we
  # want user metrics to override AdaNet's metrics.
  metric_fns.append(_create_metric_fn(metric_fn))

  return metric_fns, tensors


def _create_eval_metrics_tuple(metric_fns, metric_tensors):

  def _eval_metrics_fn(**kwargs):
    eval_metric_ops = {}
    for metric_fn in metric_fns:
      eval_metric_ops.update(metric_fn(**kwargs))
    return eval_metric_ops

  return _eval_metrics_fn, metric_tensors


def _get_value(target, key):
  if isinstance(target, dict):
    return target[key]
  return target


def _to_train_op_spec(train_op):
  if isinstance(train_op, TrainOpSpec):
    return train_op
  return TrainOpSpec(train_op)


@contextlib.contextmanager
def _monkey_patch_context(iteration_step_scope, scoped_summary):
  """Monkey-patches global attributes with subnetwork-specifics ones."""

  old_get_global_step_fn = tf.train.get_global_step
  old_get_or_create_global_step_fn = tf.train.get_or_create_global_step

  def iteration_step(graph=None):
    del graph
    with tf.variable_scope(iteration_step_scope, reuse=tf.AUTO_REUSE):
      return tf.get_variable(
          "iteration_step",
          shape=[],
          initializer=tf.zeros_initializer(),
          trainable=False,
          dtype=tf.int64)

  # monkey-patch global attributes.
  tf.train.get_global_step = iteration_step
  tf.train.get_or_create_global_step = iteration_step
  training_util.get_global_step = iteration_step
  training_util.get_or_create_global_step = iteration_step

  try:
    with monkey_patched_summaries(scoped_summary):
      yield
  finally:
    # Revert monkey-patches.
    training_util.get_or_create_global_step = old_get_or_create_global_step_fn
    training_util.get_global_step = old_get_global_step_fn
    tf.train.get_or_create_global_step = old_get_or_create_global_step_fn
    tf.train.get_global_step = old_get_global_step_fn


def _clear_trainable_variables():
  del tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)[:]


def _set_trainable_variables(var_list):
  _clear_trainable_variables()
  for var in var_list:
    assert isinstance(var, tf.Variable)
    tf.add_to_collections(tf.GraphKeys.TRAINABLE_VARIABLES, var)


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

  Returns:
    An `_EnsembleBuilder` instance.
  """

  def __init__(self, head, metric_fn=None, use_tpu=False):
    _verify_metric_fn_args(metric_fn)

    self._head = head
    self._metric_fn = metric_fn
    self._use_tpu = use_tpu

  def build_ensemble_spec(self,
                          name,
                          candidate,
                          ensembler,
                          subnetwork_specs,
                          summary,
                          features,
                          mode,
                          iteration_step,
                          iteration_number,
                          labels=None,
                          previous_ensemble_spec=None,
                          params=None):
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
      iteration_step: Integer `Tensor` representing the step since the beginning
        of the current iteration, as opposed to the global step.
      iteration_number: Integer current iteration number.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head).
      previous_ensemble_spec: Link the rest of the `_EnsembleSpec` from
        iteration t-1. Used for creating the subnetwork train_op.
      params: The params passed to model_fn.

    Returns:
      An `_EnsembleSpec` instance.
    """

    with tf.variable_scope("ensemble_{}".format(name)):
      architecture = _Architecture()
      previous_subnetworks = []
      subnetwork_builders = []
      previous_ensemble = None
      if previous_ensemble_spec:
        previous_ensemble = previous_ensemble_spec.ensemble
        previous_architecture = previous_ensemble_spec.architecture
        keep_indices = range(len(previous_ensemble.subnetworks))
        if len(candidate.subnetwork_builders) == 1 and previous_ensemble:
          # Prune previous ensemble according to the subnetwork.Builder for
          # backwards compatibility.
          tf.logging.warn(
              "Using an `adanet.subnetwork.Builder#prune_previous_ensemble` "
              "is deprecated. Please use a custom `adanet.ensemble.Strategy` "
              "instead.")
          subnetwork_builder = candidate.subnetwork_builders[0]
          keep_indices = subnetwork_builder.prune_previous_ensemble(
              previous_ensemble)
        for i, builder in enumerate(previous_ensemble_spec.subnetwork_builders):
          if i not in keep_indices:
            continue
          if builder not in candidate.previous_ensemble_subnetwork_builders:
            continue
          previous_subnetworks.append(previous_ensemble.subnetworks[i])
          subnetwork_builders.append(builder)
          architecture.add_subnetwork(*previous_architecture.subnetworks[i])
      for builder in candidate.subnetwork_builders:
        architecture.add_subnetwork(iteration_number, builder.name)
        subnetwork_builders.append(builder)
      subnetwork_map = {s.builder.name: s.subnetwork for s in subnetwork_specs}
      subnetworks = [
          subnetwork_map[s.name] for s in candidate.subnetwork_builders
      ]
      ensemble_scope = tf.get_variable_scope()
      # TODO: Restore variables after call.
      _clear_trainable_variables()
      with summary.current_scope(), _monkey_patch_context(
          iteration_step_scope=ensemble_scope, scoped_summary=summary):
        ensemble = ensembler.build_ensemble(
            subnetworks,
            previous_ensemble_subnetworks=previous_subnetworks,
            features=features,
            labels=labels,
            logits_dimension=self._head.logits_dimension,
            training=mode == tf.estimator.ModeKeys.TRAIN,
            iteration_step=iteration_step,
            summary=summary,
            previous_ensemble=previous_ensemble)
      ensemble_var_list = tf.trainable_variables()

      estimator_spec = _create_estimator_spec(
          self._head, features, labels, mode, ensemble.logits, self._use_tpu)

      ensemble_loss = estimator_spec.loss
      adanet_loss = None
      if mode != tf.estimator.ModeKeys.PREDICT:
        # TODO: Support any kind of Ensemble. Use a moving average of
        # their train loss for the 'adanet_loss'.
        if not isinstance(ensemble, ComplexityRegularized):
          raise ValueError(
              "Only ComplexityRegularized ensembles are supported.")
        adanet_loss = estimator_spec.loss + ensemble.complexity_regularization

      metric_fns = []
      metric_tensors = {}
      if mode == tf.estimator.ModeKeys.EVAL:
        fns, tensors = _create_metrics(
            features=features,
            labels=labels,
            estimator_spec=estimator_spec,
            metric_fn=self._metric_fn,
            params=params)
        metric_fns.extend(fns)
        metric_tensors.update(tensors)
        metric_fns.append(_architecture_as_metric(architecture))

      if mode == tf.estimator.ModeKeys.TRAIN:
        with summary.current_scope():
          summary.scalar("loss", estimator_spec.loss)

      # Create train ops for training subnetworks and ensembles.
      train_op = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        # Note that these mixture weights are on top of the last_layer of the
        # subnetwork constructed in TRAIN mode, which means that dropout is
        # still applied when the mixture weights are being trained.
        _set_trainable_variables(ensemble_var_list)
        ensemble_scope = tf.get_variable_scope()
        with tf.variable_scope("train_mixture_weights"):
          with summary.current_scope(), _monkey_patch_context(
              iteration_step_scope=ensemble_scope, scoped_summary=summary):
            # For backwards compatibility.
            subnetwork_builder = candidate.subnetwork_builders[0]
            old_train_op_fn = getattr(subnetwork_builder,
                                      "build_mixture_weights_train_op", None)
            if callable(old_train_op_fn):
              tf.logging.warn(
                  "The `build_mixture_weights_train_op` method is deprecated. "
                  "Please use the `Ensembler#build_train_op` instead.")
              train_op = _to_train_op_spec(
                  subnetwork_builder.build_mixture_weights_train_op(
                      loss=adanet_loss,
                      var_list=ensemble_var_list,
                      logits=ensemble.logits,
                      labels=labels,
                      iteration_step=iteration_step,
                      summary=summary))
            else:
              train_op = _to_train_op_spec(
                  ensembler.build_train_op(
                      ensemble=ensemble,
                      loss=adanet_loss,
                      var_list=ensemble_var_list,
                      labels=labels,
                      iteration_step=iteration_step,
                      summary=summary,
                      previous_ensemble=previous_ensemble))
    return _EnsembleSpec(
        name=name,
        architecture=architecture,
        subnetwork_builders=subnetwork_builders,
        ensemble=ensemble,
        predictions=estimator_spec.predictions,
        loss=ensemble_loss,
        adanet_loss=adanet_loss,
        train_op=train_op,
        eval_metrics=_create_eval_metrics_tuple(metric_fns, metric_tensors),
        export_outputs=estimator_spec.export_outputs)


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
        "loss",
        "train_op",
        "eval_metrics",
    ])):
  """Subnetwork training and evaluation `Tensors` and `Ops`.

  Args:
    name: String name of this subnetwork. Should be unique in the graph.
    subnetwork: The `adanet.subnetwork.Subnetwork` for this spec.
    builder: The `adanet.subnetwork.Builder` that produced `subnetwork`.
    predictions: Predictions `Tensor` or dict of `Tensor`.
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

  Returns:
    A `_SubnetworkSpec` object.
  """

  def __new__(cls,
              name,
              subnetwork,
              builder,
              predictions,
              loss=None,
              train_op=None,
              eval_metrics=None):
    return super(_SubnetworkSpec, cls).__new__(
        cls,
        name=name,
        subnetwork=subnetwork,
        builder=builder,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metrics=eval_metrics)


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
  """

  def __init__(self, head, metric_fn=None, use_tpu=False):
    _verify_metric_fn_args(metric_fn)
    self._head = head
    self._metric_fn = metric_fn
    self._use_tpu = use_tpu

  def build_subnetwork_spec(self,
                            name,
                            subnetwork_builder,
                            iteration_step,
                            summary,
                            features,
                            mode,
                            labels=None,
                            previous_ensemble=None,
                            params=None):
    """Builds a `_SubnetworkSpec` from the given `adanet.subnetwork.Builder`.

    Args:
      name: String name of the subnetwork.
      subnetwork_builder: A `adanet.Builder` instance which defines how to train
        the subnetwork and ensemble mixture weights.
      iteration_step: Integer `Tensor` representing the step since the beginning
        of the current iteration, as opposed to the global step.
      summary: A `_ScopedSummary` instance for recording ensemble summaries.
      features: Input `dict` of `Tensor` objects.
      mode: Estimator's `ModeKeys`.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head). Can be `None`.
      previous_ensemble: The previous `Ensemble` from iteration t-1. Used for
        creating the subnetwork train_op.
      params: The params passed to model_fn.

    Returns:
      An new `EnsembleSpec` instance with the `Subnetwork` appended.
    """

    with tf.variable_scope("subnetwork_{}".format(name)):
      build_subnetwork = functools.partial(
          subnetwork_builder.build_subnetwork,
          features=features,
          logits_dimension=self._head.logits_dimension,
          training=mode == tf.estimator.ModeKeys.TRAIN,
          iteration_step=iteration_step,
          summary=summary,
          previous_ensemble=previous_ensemble)
      # Check which args are in the implemented build_subnetwork method
      # signature for backwards compatibility.
      defined_args = inspect.getargspec(
          subnetwork_builder.build_subnetwork).args
      if "labels" in defined_args:
        build_subnetwork = functools.partial(build_subnetwork, labels=labels)
      subnetwork_scope = tf.get_variable_scope()
      tf.logging.info("Building subnetwork '%s'", subnetwork_builder.name)
      # TODO: Restore variables after call.
      _clear_trainable_variables()
      with summary.current_scope(), _monkey_patch_context(
          iteration_step_scope=subnetwork_scope, scoped_summary=summary):
        subnetwork = build_subnetwork()
      subnetwork_var_list = tf.trainable_variables()

      estimator_spec = _create_estimator_spec(
          self._head, features, labels, mode, subnetwork.logits, self._use_tpu)

      metric_fns = []
      metric_tensors = {}
      if mode == tf.estimator.ModeKeys.EVAL:
        fns, tensors = _create_metrics(
            features=features,
            labels=labels,
            estimator_spec=estimator_spec,
            metric_fn=self._metric_fn,
            params=params)
        metric_fns.extend(fns)
        metric_tensors.update(tensors)

      if mode == tf.estimator.ModeKeys.TRAIN:
        with summary.current_scope():
          summary.scalar("loss", estimator_spec.loss)

      # Create train ops for training subnetworks and ensembles.
      train_op = None
      if mode == tf.estimator.ModeKeys.TRAIN and subnetwork_builder:
        _set_trainable_variables(subnetwork_var_list)
        with summary.current_scope(), _monkey_patch_context(
            iteration_step_scope=subnetwork_scope, scoped_summary=summary):
          train_op = _to_train_op_spec(
              subnetwork_builder.build_subnetwork_train_op(
                  subnetwork=subnetwork,
                  loss=estimator_spec.loss,
                  var_list=subnetwork_var_list,
                  labels=labels,
                  iteration_step=iteration_step,
                  summary=summary,
                  previous_ensemble=previous_ensemble))
    return _SubnetworkSpec(
        name=name,
        subnetwork=subnetwork,
        builder=subnetwork_builder,
        predictions=estimator_spec.predictions,
        loss=estimator_spec.loss,
        train_op=train_op,
        eval_metrics=_create_eval_metrics_tuple(metric_fns, metric_tensors))
