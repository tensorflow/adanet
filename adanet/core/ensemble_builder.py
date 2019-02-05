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
import itertools

from adanet.core import dict_utils
from adanet.core.architecture import _Architecture
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


class WeightedSubnetwork(
    collections.namedtuple(
        "WeightedSubnetwork",
        ["name", "iteration_number", "weight", "logits", "subnetwork"])):
  """An AdaNet weighted subnetwork.

  A weighted subnetwork is a weight 'w' applied to a subnetwork's last layer
  'u'. The results is the weighted subnetwork's logits, regularized by its
  complexity.

  Args:
    name: String name of `subnetwork` as defined by its
      :class:`adanet.subnetwork.Builder`.
    iteration_number: Integer iteration when the subnetwork was created.
    weight: The weight :class:`tf.Tensor` or dict of string to weight
      :class:`tf.Tensor` (for multi-head) to apply to this subnetwork. The
      AdaNet paper refers to this weight as 'w' in Equations (4), (5), and (6).
    logits: The output :class:`tf.Tensor` or dict of string to weight
      :class:`tf.Tensor` (for multi-head) after the matrix multiplication of
      `weight` and the subnetwork's :meth:`last_layer`. The output's shape is
        [batch_size, logits_dimension]. It is equivalent to a linear logits
        layer in a neural network.
    subnetwork: The :class:`adanet.subnetwork.Subnetwork` to weight.

  Returns:
    An :class:`adanet.WeightedSubnetwork` object.
  """

  def __new__(cls,
              name="",
              iteration_number=0,
              weight=None,
              logits=None,
              subnetwork=None):

    return super(WeightedSubnetwork, cls).__new__(
        cls,
        name=name,
        iteration_number=iteration_number,
        weight=weight,
        logits=logits,
        subnetwork=subnetwork)


class Ensemble(
    collections.namedtuple("Ensemble",
                           ["weighted_subnetworks", "bias", "logits"])):
  """An AdaNet ensemble.

  An ensemble is a collection of subnetworks which forms a neural network
  through the weighted sum of their outputs. It is represented by 'f' throughout
  the AdaNet paper. Its component subnetworks' weights are complexity
  regularized (Gamma) as defined in Equation (4).

  Args:
    weighted_subnetworks: List of :class:`adanet.WeightedSubnetwork` instances
      that form this ensemble. Ordered from first to most recent.
    bias: Bias term :class:`tf.Tensor` or dict of string to bias term
      :class:`tf.Tensor` (for multi-head) for the ensemble's logits.
    logits: Logits :class:`tf.Tensor` or dict of string to logits
      :class:`tf.Tensor` (for multi-head). The result of the function 'f' as
      defined in Section 5.1 which is the sum of the logits of all
      :class:`adanet.WeightedSubnetwork` instances in ensemble.

  Returns:
    An :class:`adanet.Ensemble` instance.
  """

  def __new__(cls, weighted_subnetworks, bias, logits):
    # TODO: Make weighted_subnetworks property a tuple so that
    # `Ensemble` is immutable.
    return super(Ensemble, cls).__new__(
        cls,
        weighted_subnetworks=weighted_subnetworks,
        bias=bias,
        logits=logits)


# TODO: create an eval metrics object to encapulate the metric
# tuples.
class _EnsembleSpec(
    collections.namedtuple("_EnsembleSpec", [
        "name",
        "ensemble",
        "architecture",
        "predictions",
        "loss",
        "adanet_loss",
        "subnetwork_train_op",
        "ensemble_train_op",
        "eval_metrics",
        "export_outputs",
    ])):
  """A collections of a ensemble training and evaluation `Tensors`."""

  def __new__(cls,
              name,
              ensemble,
              architecture,
              predictions,
              loss=None,
              adanet_loss=None,
              subnetwork_train_op=None,
              ensemble_train_op=None,
              eval_metrics=None,
              export_outputs=None):
    """Creates an `EnsembleSpec` instance.

    Args:
      name: String name of this ensemble. Should be unique in the graph.
      ensemble: The `Ensemble` of interest.
      architecture: The `_Architecture` representation of the ensemble.
      predictions: Predictions `Tensor` or dict of `Tensor`.
      loss: Loss `Tensor` as defined by the surrogate loss function Phi in
        Equations (4), (5), and (6). Must be either scalar, or with shape `[1]`.
      adanet_loss: Loss `Tensor` as defined by F(w) in Equation (4). Must be
        either scalar, or with shape `[1]`. The AdaNet algorithm aims to
        minimize this objective which balances training loss with the total
        complexity of the subnetworks in the ensemble.
      subnetwork_train_op: Candidate subnetwork's `TrainOpSpec`.
      ensemble_train_op: Candidate ensemble's mixture weights `TrainOpSpec`.
      eval_metrics: Tuple of (metric_fn, tensors) where metric_fn(tensors)
        returns the dict of eval metrics keyed by name. The values of the
        dict are the results of calling a metric function, namely a
        `(metric_tensor, update_op)` tuple. `metric_tensor` should be evaluated
        without any impact on state (typically is a pure computation based on
        variables.). For example, it should not trigger the `update_op` or
        require any input fetching.
      export_outputs: Describes the output signatures to be exported to
        `SavedModel` and used during serving. See `tf.estimator.EstimatorSpec`.

    Returns:
      An `EnsembleSpec` object.
    """

    # TODO: Make weighted_subnetworks property a tuple so that
    # `Ensemble` is immutable.
    return super(_EnsembleSpec, cls).__new__(
        cls,
        name=name,
        ensemble=ensemble,
        architecture=architecture,
        predictions=predictions,
        loss=loss,
        adanet_loss=adanet_loss,
        subnetwork_train_op=subnetwork_train_op,
        ensemble_train_op=ensemble_train_op,
        eval_metrics=eval_metrics,
        export_outputs=export_outputs)


class MixtureWeightType(object):
  """Mixture weight types available for learning subnetwork contributions.

  The following mixture weight types are defined:

  * `SCALAR`: Produces a rank 0 `Tensor` mixture weight.
  * `VECTOR`: Produces a rank 1 `Tensor` mixture weight.
  * `MATRIX`: Produces a rank 2 `Tensor` mixture weight.
  """

  SCALAR = "scalar"
  VECTOR = "vector"
  MATRIX = "matrix"


def _architecture_as_metric(architecture):
  """Returns a representation of the ensemble's architecture as a tf.metric."""

  def _architecture_metric_fn(**kwargs):
    """Manually creates the tf.metric with a serialized tf.Summary proto."""

    del kwargs  # Unused.

    architecture_ = " | ".join(
        itertools.chain(*[names for _, names in architecture.subnetworks]))
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


def _create_scoped_metric_fn(metric_fn, group_name):
  """Wraps the metric_fn to scope its returned metrics by group_name."""

  def _scoped_metric_fn(**kwargs):
    """The wrapping function to be returned."""

    if not metric_fn:
      return {}

    kwargs = dict_utils.unflatten_dict(
        kwargs, prefixes=[group_name])[group_name]
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
    rescoped_metrics = {}
    # Hooks on TPU cannot depend on any graph Tensors. Instead the metric values
    # are stored in Variables that are later read from the evaluation hooks.
    for i, key in enumerate(sorted(metrics)):
      tensor, op = metrics[key]
      var = tf.get_variable(
          "metric_{}".format(i),
          shape=tensor.shape,
          dtype=tensor.dtype,
          trainable=False,
          initializer=tf.zeros_initializer(),
          collections=[tf.GraphKeys.LOCAL_VARIABLES])
      metric = (var, tf.assign(var, op))
      rescoped_metrics["{}/adanet/{}".format(key, group_name)] = metric
    return rescoped_metrics

  return tf.make_template("metric_fn_template", _scoped_metric_fn)


def _prefix(tensors, group_name, flat_key, default_key):
  """Prefixes tensors by group_name and either flat_key or default_key.

  If tensors is a dict each tensor is rekeyed as group_name/flat_key/key. If
  tensors is a single Tensor, it is keyed by group_name/default_key.

  Args:
    tensors: A Tensor or dictionary of Tensors.
    group_name: The group name to use in the prefix.
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
  tensors = dict_utils.flatten_dict({group_name: tensors})
  return tensors


def _create_group_metrics(group_name, features, labels, estimator_spec,
                          metric_fn, params):
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
  metric_fns.append(_create_scoped_metric_fn(spec_metric_fn, group_name))
  for key, value in six.iteritems(spec_tensors):
    tensors["{}/{}/{}".format(group_name, _KWARGS_KEY, key)] = value

  loss_fn = lambda loss: {"loss": tf.metrics.mean(loss)}
  metric_fns.append(_create_scoped_metric_fn(loss_fn, group_name))
  # All tensors outfed from the TPU must be batch-major.
  batch_size = params.get("batch_size", 1) if params else 1
  tensors["{}/loss".format(group_name)] = tf.ones(
      (batch_size, 1)) * estimator_spec.loss

  # TODO: (Optimization): features and labels are shared between all
  # group metrics so they should only be outfed once. However, this makes the
  # kwarg parsing harder.
  tensors.update(_prefix(features, group_name, _FEATURES_KEY, "features"))
  tensors.update(_prefix(labels, group_name, _LABELS_KEY, "labels"))
  tensors.update(
      _prefix(estimator_spec.predictions, group_name, _PREDICTIONS_KEY,
              "predictions"))

  # NOTE: the user supplied metrics_fn must be added last. This is because we
  # want user metrics to override AdaNet's metrics.
  metric_fns.append(_create_scoped_metric_fn(metric_fn, group_name))

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
def _subnetwork_context(iteration_step_scope, scoped_summary):
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


def _get_ensemble_var_list(weighted_subnetworks):
  var_list = []
  for weighted_subnetwork in weighted_subnetworks:
    weight = weighted_subnetwork.weight
    if isinstance(weight, dict):
      var_list.extend(weight.values())
    else:
      var_list.append(weight)
  return var_list


class _EnsembleBuilder(object):
  """Builds `Ensemble` instances."""

  def __init__(self,
               head,
               mixture_weight_type,
               mixture_weight_initializer=None,
               warm_start_mixture_weights=False,
               checkpoint_dir=None,
               adanet_lambda=0.,
               adanet_beta=0.,
               use_bias=True,
               metric_fn=None,
               use_tpu=False):
    """Returns an initialized `_EnsembleBuilder`.

    Args:
      head: A `tf.contrib.estimator.Head` instance.
      mixture_weight_type: The `adanet.MixtureWeightType` defining which mixture
        weight type to learn.
      mixture_weight_initializer: The initializer for mixture_weights. When
        `None`, the default is different according to `mixture_weight_type`.
        `SCALAR` initializes to 1/N where N is the number of subnetworks in the
        ensemble giving a uniform average. `VECTOR` initializes each entry to
        1/N where N is the number of subnetworks in the ensemble giving a
        uniform average. `MATRIX` uses `tf.zeros_initializer`.
      warm_start_mixture_weights: Whether, at the beginning of an iteration, to
        initialize the mixture weights of the subnetworks from the previous
        ensemble to their learned value at the previous iteration, as opposed to
        retraining them from scratch. Takes precedence over the value for
        `mixture_weight_initializer` for subnetworks from previous iterations.
      checkpoint_dir: The checkpoint_dir to use for warm-starting mixture
        weights and bias at the logit layer. Ignored if
        warm_start_mixture_weights is False.
      adanet_lambda: Float multiplier 'lambda' for applying L1 regularization to
        subnetworks' mixture weights 'w' in the ensemble proportional to their
        complexity. See Equation (4) in the AdaNet paper.
      adanet_beta: Float L1 regularization multiplier 'beta' to apply equally to
        all subnetworks' weights 'w' in the ensemble regardless of their
        complexity. See Equation (4) in the AdaNet paper.
      use_bias: Whether to add a bias term to the ensemble's logits.
      metric_fn: A function which should obey the following signature:
        - Args: can only have following three arguments in any order:
          * predictions: Predictions `Tensor` or dict of `Tensor` created by
            given `Head`.
          * features: Input `dict` of `Tensor` objects created by `input_fn`
            which is given to `estimator.evaluate` as an argument.
          * labels:  Labels `Tensor` or dict of `Tensor` (for multi-head)
            created by `input_fn` which is given to `estimator.evaluate` as an
            argument.
        - Returns: Dict of metric results keyed by name. Final metrics are a
          union of this and `Head's` existing metrics. If there is a name
          conflict between this and `estimator`s existing metrics, this will
          override the existing one. The values of the dict are the results of
          calling a metric function, namely a `(metric_tensor, update_op)`
          tuple.
     use_tpu: Whether AdaNet is running on TPU.

    Returns:
      An `_EnsembleBuilder` instance.

    Raises:
      ValueError: if warm_start_mixture_weights is True but checkpoint_dir is
      None.
      ValueError: if metric_fn is invalid.
    """

    if warm_start_mixture_weights:
      if checkpoint_dir is None:
        raise ValueError("checkpoint_dir cannot be None when "
                         "warm_start_mixture_weights is True.")

    _verify_metric_fn_args(metric_fn)

    self._use_tpu = use_tpu
    self._head = head
    self._mixture_weight_type = mixture_weight_type
    self._mixture_weight_initializer = mixture_weight_initializer
    self._warm_start_mixture_weights = warm_start_mixture_weights
    self._checkpoint_dir = checkpoint_dir
    self._adanet_lambda = adanet_lambda
    self._adanet_beta = adanet_beta
    self._use_bias = use_bias
    self._metric_fn = metric_fn

  def append_new_subnetwork(self,
                            ensemble_name,
                            ensemble_spec,
                            subnetwork_builder,
                            iteration_number,
                            iteration_step,
                            summary,
                            features,
                            mode,
                            labels=None,
                            params=None):
    """Adds a `Subnetwork` to an `_EnsembleSpec`.

    For iteration t > 0, the ensemble is built given the `Ensemble` for t-1 and
    the new subnetwork to train as part of the ensemble. The `Ensemble` at
    iteration 0 is comprised of just the subnetwork.

    The subnetwork is first given a weight 'w' in a `WeightedSubnetwork`
    which determines its contribution to the ensemble. The subnetwork's
    complexity L1-regularizes this weight.

    Args:
      ensemble_name: String name of the ensemble.
      ensemble_spec: The recipient `_EnsembleSpec` for the `Subnetwork`.
      subnetwork_builder: A `adanet.Builder` instance which defines how to train
        the subnetwork and ensemble mixture weights.
      iteration_number: Integer current iteration number.
      iteration_step: Integer `Tensor` representing the step since the beginning
        of the current iteration, as opposed to the global step.
      summary: A `_ScopedSummary` instance for recording ensemble summaries.
      features: Input `dict` of `Tensor` objects.
      mode: Estimator's `ModeKeys`.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head). Can be `None`.
      params: The params passed to model_fn.

    Returns:
      An new `EnsembleSpec` instance with the `Subnetwork` appended.
    """

    with tf.variable_scope("ensemble_{}".format(ensemble_name)):
      weighted_subnetworks = []
      subnetwork_index = 0
      num_subnetworks = 1
      ensemble = None
      architecture = _Architecture()
      if ensemble_spec:
        ensemble = ensemble_spec.ensemble
        previous_subnetworks = [
            ensemble.weighted_subnetworks[index]
            for index in subnetwork_builder.prune_previous_ensemble(ensemble)
        ]
        num_subnetworks += len(previous_subnetworks)
        for weighted_subnetwork in previous_subnetworks:
          weight_initializer = None
          if self._warm_start_mixture_weights:
            weight_initializer = tf.contrib.framework.load_variable(
                self._checkpoint_dir, weighted_subnetwork.weight.op.name)
          with tf.variable_scope(
              "weighted_subnetwork_{}".format(subnetwork_index)):
            weighted_subnetworks.append(
                self._build_weighted_subnetwork(
                    weighted_subnetwork.name,
                    weighted_subnetwork.iteration_number,
                    weighted_subnetwork.subnetwork,
                    num_subnetworks,
                    weight_initializer=weight_initializer))
          architecture.add_subnetwork(weighted_subnetwork.iteration_number,
                                      weighted_subnetwork.name)
          subnetwork_index += 1

      ensemble_scope = tf.get_variable_scope()

      with tf.variable_scope("weighted_subnetwork_{}".format(subnetwork_index)):
        with tf.variable_scope("subnetwork"):
          _clear_trainable_variables()
          build_subnetwork = functools.partial(
              subnetwork_builder.build_subnetwork,
              features=features,
              logits_dimension=self._head.logits_dimension,
              training=mode == tf.estimator.ModeKeys.TRAIN,
              iteration_step=iteration_step,
              summary=summary,
              previous_ensemble=ensemble)
          # Check which args are in the implemented build_subnetwork method
          # signature for backwards compatibility.
          defined_args = inspect.getargspec(
              subnetwork_builder.build_subnetwork).args
          if "labels" in defined_args:
            build_subnetwork = functools.partial(
                build_subnetwork, labels=labels)
          with summary.current_scope(), _subnetwork_context(
              iteration_step_scope=ensemble_scope, scoped_summary=summary):
            tf.logging.info("Building subnetwork '%s'", subnetwork_builder.name)
            subnetwork = build_subnetwork()
          var_list = tf.trainable_variables()
        weighted_subnetworks.append(
            self._build_weighted_subnetwork(subnetwork_builder.name,
                                            iteration_number, subnetwork,
                                            num_subnetworks))
        architecture.add_subnetwork(iteration_number, subnetwork_builder.name)
      if ensemble:
        if len(previous_subnetworks) == len(ensemble.weighted_subnetworks):
          bias = self._create_bias_term(
              weighted_subnetworks, prior=ensemble.bias)
        else:
          bias = self._create_bias_term(weighted_subnetworks)
          tf.logging.info(
              "Builder '%s' is using a subset of the subnetworks "
              "from the previous ensemble, so its ensemble's bias "
              "term will not be warm started with the previous "
              "ensemble's bias.", subnetwork_builder.name)
      else:
        bias = self._create_bias_term(weighted_subnetworks)

      return self._build_ensemble_spec(
          name=ensemble_name,
          weighted_subnetworks=weighted_subnetworks,
          architecture=architecture,
          summary=summary,
          bias=bias,
          features=features,
          mode=mode,
          iteration_step=iteration_step,
          labels=labels,
          subnetwork_builder=subnetwork_builder,
          var_list=var_list,
          previous_ensemble_spec=ensemble_spec,
          params=params)

  def _build_ensemble_spec(self,
                           name,
                           weighted_subnetworks,
                           architecture,
                           summary,
                           bias,
                           features,
                           mode,
                           iteration_step,
                           labels=None,
                           subnetwork_builder=None,
                           var_list=None,
                           previous_ensemble_spec=None,
                           params=None):
    """Builds an `_EnsembleSpec` with the given `WeightedSubnetwork`s.

    Args:
      name: The string name of the ensemble. Typically the name of the builder
        that returned the given `Subnetwork`.
      weighted_subnetworks: List of `WeightedSubnetwork` instances that form
        this ensemble. Ordered from first to most recent.
      architecture: The `_Architecture` representation of the ensemble.
      summary: A `_ScopedSummary` instance for recording ensemble summaries.
      bias: Bias term `Tensor` or dict of string to `Tensor` (for multi-head)
        for the AdaNet-weighted ensemble logits.
      features: Input `dict` of `Tensor` objects.
      mode: Estimator `ModeKeys` indicating training, evaluation, or inference.
      iteration_step: Integer `Tensor` representing the step since the beginning
        of the current iteration, as opposed to the global step.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head).
      subnetwork_builder: A `adanet.Builder` instance which defines how to train
        the subnetwork and ensemble mixture weights.
      var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.
      previous_ensemble_spec: Link the rest of the `_EnsembleSpec` from
        iteration t-1. Used for creating the subnetwork train_op.
      params: The params passed to model_fn.

    Returns:
      An `_EnsembleSpec` instance.
    """

    ensemble_logits, ensemble_complexity_regularization = (
        self._adanet_weighted_ensemble_logits(weighted_subnetworks, bias,
                                              summary))

    # The AdaNet-weighted ensemble.
    adanet_weighted_ensemble_spec = self._create_estimator_spec(
        features, labels, mode, ensemble_logits)

    # A baseline ensemble: the uniform-average of subnetwork outputs.
    # It is practically free to compute, requiring no additional training, and
    # tends to generalize very well. However the AdaNet-weighted ensemble
    # should perform at least as well given the correct hyperparameters.
    uniform_average_ensemble_spec = self._create_estimator_spec(
        features, labels, mode,
        self._uniform_average_ensemble_logits(weighted_subnetworks))

    # The subnetwork.
    new_subnetwork = weighted_subnetworks[-1].subnetwork
    subnetwork_spec = self._create_estimator_spec(features, labels, mode,
                                                  new_subnetwork.logits)

    ensemble_loss = adanet_weighted_ensemble_spec.loss
    adanet_loss = None
    if mode != tf.estimator.ModeKeys.PREDICT:
      adanet_loss = ensemble_loss
      if isinstance(ensemble_complexity_regularization, dict):
        for key in sorted(ensemble_complexity_regularization):
          adanet_loss += ensemble_complexity_regularization[key]
      else:
        adanet_loss += ensemble_complexity_regularization

    metric_fns = []
    metric_tensors = {}
    if mode == tf.estimator.ModeKeys.EVAL:
      fns, tensors = _create_group_metrics(
          features=features,
          labels=labels,
          group_name="adanet_weighted_ensemble",
          estimator_spec=adanet_weighted_ensemble_spec,
          metric_fn=self._metric_fn,
          params=params)
      metric_fns.extend(fns)
      metric_tensors.update(tensors)
      fns, tensors = _create_group_metrics(
          features=features,
          labels=labels,
          group_name="uniform_average_ensemble",
          estimator_spec=uniform_average_ensemble_spec,
          metric_fn=self._metric_fn,
          params=params)
      metric_fns.extend(fns)
      metric_tensors.update(tensors)
      fns, tensors = _create_group_metrics(
          features=features,
          labels=labels,
          group_name="subnetwork",
          estimator_spec=subnetwork_spec,
          metric_fn=self._metric_fn,
          params=params)
      metric_fns.extend(fns)
      metric_tensors.update(tensors)
      metric_fns.append(_architecture_as_metric(architecture))

    if mode == tf.estimator.ModeKeys.TRAIN:
      with summary.current_scope():
        summary.scalar("loss/adanet/adanet_weighted_ensemble",
                       adanet_weighted_ensemble_spec.loss)
        summary.scalar("loss/adanet/subnetwork", subnetwork_spec.loss)
        summary.scalar("loss/adanet/uniform_average_ensemble",
                       uniform_average_ensemble_spec.loss)

    # Create train ops for training subnetworks and learning mixture weights.
    subnetwork_train_op = None
    ensemble_train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN and subnetwork_builder:
      ensemble_scope = tf.get_variable_scope()
      _set_trainable_variables(var_list)
      with tf.variable_scope("train_subnetwork"):
        previous_ensemble = None
        if previous_ensemble_spec:
          previous_ensemble = previous_ensemble_spec.ensemble

        with summary.current_scope(), _subnetwork_context(
            iteration_step_scope=ensemble_scope, scoped_summary=summary):
          subnetwork_train_op = _to_train_op_spec(
              subnetwork_builder.build_subnetwork_train_op(
                  subnetwork=new_subnetwork,
                  loss=subnetwork_spec.loss,
                  var_list=var_list,
                  labels=labels,
                  iteration_step=iteration_step,
                  summary=summary,
                  previous_ensemble=previous_ensemble))
      # Note that these mixture weights are on top of the last_layer of the
      # subnetwork constructed in TRAIN mode, which means that dropout is
      # still applied when the mixture weights are being trained.
      ensemble_var_list = _get_ensemble_var_list(weighted_subnetworks)
      if self._use_bias:
        ensemble_var_list.insert(0, bias)
      _set_trainable_variables(ensemble_var_list)
      ensemble_scope = tf.get_variable_scope()
      with tf.variable_scope("train_mixture_weights"):
        with summary.current_scope(), _subnetwork_context(
            iteration_step_scope=ensemble_scope, scoped_summary=summary):
          ensemble_train_op = _to_train_op_spec(
              subnetwork_builder.build_mixture_weights_train_op(
                  loss=adanet_loss,
                  var_list=ensemble_var_list,
                  logits=ensemble_logits,
                  labels=labels,
                  iteration_step=iteration_step,
                  summary=summary))

    return _EnsembleSpec(
        name=name,
        ensemble=Ensemble(
            weighted_subnetworks=weighted_subnetworks,
            bias=bias,
            logits=ensemble_logits,
        ),
        architecture=architecture,
        predictions=adanet_weighted_ensemble_spec.predictions,
        loss=ensemble_loss,
        adanet_loss=adanet_loss,
        subnetwork_train_op=subnetwork_train_op,
        ensemble_train_op=ensemble_train_op,
        eval_metrics=_create_eval_metrics_tuple(metric_fns, metric_tensors),
        export_outputs=adanet_weighted_ensemble_spec.export_outputs)

  def _create_estimator_spec(self, features, labels, mode, logits):
    """Creates the head's EstimatorSpec or TPUEstimatorSpec on TPU."""

    if self._use_tpu:
      create_spec_fn = self._head._create_tpu_estimator_spec  # pylint: disable=protected-access
    else:
      create_spec_fn = self._head.create_estimator_spec
    return create_spec_fn(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits,
        train_op_fn=lambda _: tf.no_op())

  def _complexity_regularization(self, weight_l1_norm, complexity):
    """For a subnetwork, computes: (lambda * r(h) + beta) * |w|."""

    if self._adanet_lambda == 0. and self._adanet_beta == 0.:
      return tf.constant(0., name="zero")
    return tf.scalar_mul(self._adanet_gamma(complexity), weight_l1_norm)

  def _adanet_gamma(self, complexity):
    """For a subnetwork, computes: lambda * r(h) + beta."""

    if self._adanet_lambda == 0.:
      return self._adanet_beta
    return tf.scalar_mul(self._adanet_lambda,
                         tf.to_float(complexity)) + self._adanet_beta

  def _select_mixture_weight_initializer(self, num_subnetworks):
    if self._mixture_weight_initializer:
      return self._mixture_weight_initializer
    if (self._mixture_weight_type == MixtureWeightType.SCALAR or
        self._mixture_weight_type == MixtureWeightType.VECTOR):
      return tf.constant_initializer(1. / num_subnetworks)
    return tf.zeros_initializer()

  def _build_weighted_subnetwork(self,
                                 name,
                                 iteration_number,
                                 subnetwork,
                                 num_subnetworks,
                                 weight_initializer=None):
    """Builds an `WeightedSubnetwork`.

    Args:
      name: String name of `subnetwork`.
      iteration_number: Integer iteration when the subnetwork was created.
      subnetwork: The `Subnetwork` to weight.
      num_subnetworks: The number of subnetworks in the ensemble.
      weight_initializer: Initializer for the weight variable.

    Returns:
      A `WeightedSubnetwork` instance.

    Raises:
      ValueError: When the subnetwork's last layer and logits dimension do
        not match and requiring a SCALAR or VECTOR mixture weight.
    """

    if isinstance(subnetwork.last_layer, dict):
      logits, weight = {}, {}
      for key in sorted(subnetwork.last_layer):
        logits[key], weight[key] = self._build_weighted_subnetwork_helper(
            subnetwork, num_subnetworks, weight_initializer, key)
    else:
      logits, weight = self._build_weighted_subnetwork_helper(
          subnetwork, num_subnetworks, weight_initializer)

    return WeightedSubnetwork(
        name=name,
        iteration_number=iteration_number,
        subnetwork=subnetwork,
        logits=logits,
        weight=weight)

  def _build_weighted_subnetwork_helper(self,
                                        subnetwork,
                                        num_subnetworks,
                                        weight_initializer=None,
                                        key=None):
    """Returns the logits and weight of the `WeightedSubnetwork` for key."""

    # Treat subnetworks as if their weights are frozen, and ensure that
    # mixture weight gradients do not propagate through.
    last_layer = _get_value(subnetwork.last_layer, key)
    logits = _get_value(subnetwork.logits, key)
    weight_shape = None
    last_layer_size = last_layer.get_shape().as_list()[-1]
    logits_size = logits.get_shape().as_list()[-1]
    batch_size = tf.shape(last_layer)[0]

    if weight_initializer is None:
      weight_initializer = self._select_mixture_weight_initializer(
          num_subnetworks)
      if self._mixture_weight_type == MixtureWeightType.SCALAR:
        weight_shape = []
      if self._mixture_weight_type == MixtureWeightType.VECTOR:
        weight_shape = [logits_size]
      if self._mixture_weight_type == MixtureWeightType.MATRIX:
        weight_shape = [last_layer_size, logits_size]

    with tf.variable_scope("{}logits".format(key + "_" if key else "")):
      # Mark as not trainable to not add to the TRAINABLE_VARIABLES
      # collection. Training is handled explicitly with var_lists.
      weight = tf.get_variable(
          name="mixture_weight",
          shape=weight_shape,
          initializer=weight_initializer,
          trainable=False)
      if self._mixture_weight_type == MixtureWeightType.MATRIX:
        # TODO: Add Unit tests for the ndims == 3 path.
        ndims = len(last_layer.get_shape().as_list())
        if ndims > 3:
          raise NotImplementedError(
              "Last Layer with more than 3 dimensions are not supported with "
              "matrix mixture weights.")
        # This is reshaping [batch_size, timesteps, emb_dim ] to
        # [batch_size x timesteps, emb_dim] for matrix multiplication
        # and reshaping back.
        if ndims == 3:
          tf.logging.info("Rank 3 tensors like [batch_size, timesteps, d]  are "
                          "reshaped to rank 2 [ batch_size x timesteps, d] for "
                          "the weight matrix multiplication, and are reshaped "
                          "to their original shape afterwards.")
          last_layer = tf.reshape(last_layer, [-1, last_layer_size])
        logits = tf.matmul(last_layer, weight)
        if ndims == 3:
          logits = tf.reshape(logits, [batch_size, -1, logits_size])
      else:
        logits = tf.multiply(_get_value(subnetwork.logits, key), weight)
    return logits, weight

  def _create_bias_term(self, weighted_subnetworks, prior=None):
    """Returns a bias term vector.

    If `use_bias` is set, then it returns a trainable bias variable initialized
    to zero, or warm-started with the given prior. Otherwise it returns
    a zero constant bias.

    Args:
      weighted_subnetworks: List of `WeightedSubnetwork` instances that form
        this ensemble. Ordered from first to most recent.
      prior: Prior bias term `Tensor` of dict of string to `Tensor` (for multi-
        head) for warm-starting the bias term variable.

    Returns:
      A bias term `Tensor` or dict of string to bias term `Tensor` (for multi-
        head).
    """

    if not isinstance(weighted_subnetworks[0].subnetwork.logits, dict):
      return self._create_bias_term_helper(weighted_subnetworks, prior)
    bias_terms = {}
    for key in sorted(weighted_subnetworks[0].subnetwork.logits):
      bias_terms[key] = self._create_bias_term_helper(weighted_subnetworks,
                                                      prior, key)
    return bias_terms

  def _create_bias_term_helper(self, weighted_subnetworks, prior, key=None):
    """Returns a bias term for weights with the given key."""

    shape = None
    if prior is None:
      prior = tf.zeros_initializer()
      logits = _get_value(weighted_subnetworks[0].subnetwork.logits, key)
      logits_dimension = logits.get_shape().as_list()[-1]
      shape = logits_dimension
    else:
      prior = tf.contrib.framework.load_variable(self._checkpoint_dir,
                                                 _get_value(prior, key).op.name)
    # Mark as not trainable to not add to the TRAINABLE_VARIABLES collection.
    # Training is handled explicitly with var_lists.
    return tf.get_variable(
        name="{}bias".format(key + "_" if key else ""),
        shape=shape,
        initializer=prior,
        trainable=False)

  def _adanet_weighted_ensemble_logits(self, weighted_subnetworks, bias,
                                       summary):
    """Computes the AdaNet weighted ensemble logits.

    If `use_bias` is set, then it returns a trainable bias variable initialized
    to zero, or warm-started with the given prior. Otherwise it returns
    a zero constant bias.

    Args:
      weighted_subnetworks: List of `WeightedSubnetwork` instances that form
        this ensemble. Ordered from first to most recent.
      bias: Bias term `Tensor` or dict of string to `Tensor` (for multi-head)
        for the AdaNet-weighted ensemble logits.
      summary: A `_ScopedSummary` instance for recording ensemble summaries.

    Returns:
      A two-tuple of:
       1. Ensemble logits `Tensor` or dict of string to logits `Tensor` (for
         multi-head).
       2. Ensemble complexity regularization
    """

    if not isinstance(weighted_subnetworks[0].subnetwork.logits, dict):
      return self._adanet_weighted_ensemble_logits_helper(
          weighted_subnetworks, bias, summary)
    logits, ensemble_complexity_regularization = {}, {}
    for key in sorted(weighted_subnetworks[0].subnetwork.logits):
      logits[key], ensemble_complexity_regularization[key] = (
          self._adanet_weighted_ensemble_logits_helper(weighted_subnetworks,
                                                       bias, summary, key))
    return logits, ensemble_complexity_regularization

  def _adanet_weighted_ensemble_logits_helper(self,
                                              weighted_subnetworks,
                                              bias,
                                              summary,
                                              key=None):
    """Returns the AdaNet ensemble logits and regularization term for key."""

    subnetwork_logits = []
    ensemble_complexity_regularization = 0
    total_weight_l1_norms = 0
    weights = []
    for weighted_subnetwork in weighted_subnetworks:
      weight_l1_norm = tf.norm(
          _get_value(weighted_subnetwork.weight, key), ord=1)
      total_weight_l1_norms += weight_l1_norm
      ensemble_complexity_regularization += self._complexity_regularization(
          weight_l1_norm, weighted_subnetwork.subnetwork.complexity)
      subnetwork_logits.append(_get_value(weighted_subnetwork.logits, key))
      weights.append(weight_l1_norm)

    with tf.variable_scope("{}logits".format(key + "_" if key else "")):
      ensemble_logits = _get_value(bias, key)
      for logits in subnetwork_logits:
        ensemble_logits = tf.add(ensemble_logits, logits)

    with summary.current_scope():
      summary.scalar(
          "complexity_regularization/adanet/adanet_weighted_ensemble",
          ensemble_complexity_regularization)
      summary.histogram("mixture_weights/adanet/adanet_weighted_ensemble",
                        weights)
      for iteration, weight in enumerate(weights):
        scope = "adanet/adanet_weighted_ensemble/subnetwork_{}".format(
            iteration)
        summary.scalar("mixture_weight_norms/{}".format(scope), weight)
        fraction = weight / total_weight_l1_norms
        summary.scalar("mixture_weight_fractions/{}".format(scope), fraction)
    return ensemble_logits, ensemble_complexity_regularization

  def _uniform_average_ensemble_logits(self, weighted_subnetworks):
    """Computes the uniform average ensemble logits.

    Args:
      weighted_subnetworks: List of `WeightedSubnetwork` instances that form
        this ensemble. Ordered from first to most recent.

    Returns:
      Ensemble logits `Tensor` or dict of string to logits `Tensor` (for
         multi-head).
    """

    if not isinstance(weighted_subnetworks[0].subnetwork.logits, dict):
      return self._uniform_average_ensemble_logits_helper(weighted_subnetworks)
    logits = {}
    for key in sorted(weighted_subnetworks[0].subnetwork.logits):
      logits[key] = self._uniform_average_ensemble_logits_helper(
          weighted_subnetworks, key)
    return logits

  def _uniform_average_ensemble_logits_helper(self,
                                              weighted_subnetworks,
                                              key=None):
    """Returns logits for the baseline ensemble for the given key."""

    return tf.add_n([
        _get_value(wwl.subnetwork.logits, key) for wwl in weighted_subnetworks
    ]) / len(weighted_subnetworks)
