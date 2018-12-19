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

from adanet.core.subnetwork import TrainOpSpec
import tensorflow as tf

from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.training import training_util

_VALID_METRIC_FN_ARGS = set(["features", "labels", "predictions"])


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
      [batch_size, logits_dimension]. It is equivalent to a linear logits layer
      in a neural network.
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


class _EnsembleSpec(
    collections.namedtuple("_EnsembleSpec", [
        "name",
        "ensemble",
        "predictions",
        "loss",
        "adanet_loss",
        "subnetwork_train_op",
        "ensemble_train_op",
        "eval_metric_ops",
        "export_outputs",
    ])):
  """A collections of a ensemble training and evaluation `Tensors`."""

  def __new__(cls,
              name,
              ensemble,
              predictions,
              loss=None,
              adanet_loss=None,
              subnetwork_train_op=None,
              ensemble_train_op=None,
              eval_metric_ops=None,
              export_outputs=None):
    """Creates an `EnsembleSpec` instance.

    Args:
      name: String name of this ensemble. Should be unique in the graph.
      ensemble: The `Ensemble` of interest.
      predictions: Predictions `Tensor` or dict of `Tensor`.
      loss: Loss `Tensor` as defined by the surrogate loss function Phi in
        Equations (4), (5), and (6). Must be either scalar, or with shape `[1]`.
      adanet_loss: Loss `Tensor` as defined by F(w) in Equation (4). Must be
        either scalar, or with shape `[1]`. The AdaNet algorithm aims to
        minimize this objective which balances training loss with the total
        complexity of the subnetworks in the ensemble.
      subnetwork_train_op: Candidate subnetwork's `TrainOpSpec`.
      ensemble_train_op: Candidate ensemble's mixture weights `TrainOpSpec`.
      eval_metric_ops: Dict of metric results keyed by name. The values of the
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
        predictions=predictions,
        loss=loss,
        adanet_loss=adanet_loss,
        subnetwork_train_op=subnetwork_train_op,
        ensemble_train_op=ensemble_train_op,
        eval_metric_ops=eval_metric_ops,
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


def _architecture_as_metric(weighted_subnetworks):
  """Returns a representation of the ensemble's architecture as a tf.metric."""

  joined_names = " | ".join([w.name for w in weighted_subnetworks])
  architecture = tf.convert_to_tensor(
      "| {} |".format(joined_names), name="architecture")
  architecture_summary = tf.summary.text("architecture/adanet", architecture)
  return (architecture_summary, tf.no_op())


def _call_metric_fn(metric_fn, features, labels, predictions):
  """Calls metric fn with proper arguments."""

  if not metric_fn:
    return {}

  metric_fn_args = inspect.getargspec(metric_fn).args
  kwargs = {}
  if "features" in metric_fn_args:
    kwargs["features"] = features
  if "labels" in metric_fn_args:
    kwargs["labels"] = labels
  if "predictions" in metric_fn_args:
    kwargs["predictions"] = predictions
  return metric_fn(**kwargs)


def _verify_metric_fn_args(metric_fn):
  if not metric_fn:
    return
  args = set(inspect.getargspec(metric_fn).args)
  invalid_args = list(args - _VALID_METRIC_FN_ARGS)
  if invalid_args:
    raise ValueError("metric_fn (%s) has following not expected args: %s" %
                     (metric_fn, invalid_args))


def _add_eval_metric_ops(eval_metric_ops, group_name, estimator_spec,
                         metric_fn):
  """Adds eval metric ops to the given dictionary for the given group name."""

  eval_metric_ops["loss/adanet/{}".format(group_name)] = tf.metrics.mean(
      estimator_spec.loss)
  metric_ops = estimator_spec.eval_metric_ops
  for metric in sorted(metric_ops):
    eval_metric_ops["{metric}/adanet/{group_name}".format(
        metric=metric, group_name=group_name)] = metric_ops[metric]
  metric_ops = metric_fn(predictions=estimator_spec.predictions)
  for metric in sorted(metric_ops):
    eval_metric_ops["{metric}/adanet/{group_name}".format(
        metric=metric, group_name=group_name)] = metric_ops[metric]


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
  old_summary_scalar = summary_lib.scalar
  old_summary_image = summary_lib.image
  old_summary_histogram = summary_lib.histogram
  old_summary_audio = summary_lib.audio

  def iteration_step(graph=None):
    del graph
    with tf.variable_scope(iteration_step_scope, reuse=tf.AUTO_REUSE):
      return tf.get_variable(
          "iteration_step",
          shape=[],
          initializer=tf.zeros_initializer(),
          trainable=False,
          dtype=tf.int64)

  # Monkey-patch global attributes.
  tf.summary.scalar = scoped_summary.scalar
  tf.summary.image = scoped_summary.image
  tf.summary.histogram = scoped_summary.histogram
  tf.summary.audio = scoped_summary.audio
  summary_lib.scalar = scoped_summary.scalar
  summary_lib.image = scoped_summary.image
  summary_lib.histogram = scoped_summary.histogram
  summary_lib.audio = scoped_summary.audio
  tf.train.get_global_step = iteration_step
  tf.train.get_or_create_global_step = iteration_step
  training_util.get_global_step = iteration_step
  training_util.get_or_create_global_step = iteration_step

  try:
    yield
  finally:
    # Revert monkey-patches.
    training_util.get_or_create_global_step = old_get_or_create_global_step_fn
    training_util.get_global_step = old_get_global_step_fn
    tf.train.get_or_create_global_step = old_get_or_create_global_step_fn
    tf.train.get_global_step = old_get_global_step_fn
    summary_lib.audio = old_summary_audio
    summary_lib.histogram = old_summary_histogram
    summary_lib.image = old_summary_image
    summary_lib.scalar = old_summary_scalar
    tf.summary.audio = old_summary_audio
    tf.summary.histogram = old_summary_histogram
    tf.summary.image = old_summary_image
    tf.summary.scalar = old_summary_scalar


def _clear_trainable_variables():
  del tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)[:]


def _set_trainable_variables(var_list):
  _clear_trainable_variables()
  for var in var_list:
    tf.add_to_collections(tf.GraphKeys.TRAINABLE_VARIABLES, var)


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
               metric_fn=None):
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
                            labels=None):
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

    Returns:
      An new `EnsembleSpec` instance with the `Subnetwork` appended.
    """

    with tf.variable_scope("ensemble_{}".format(ensemble_name)):
      weighted_subnetworks = []
      subnetwork_index = 0
      num_subnetworks = 1
      ensemble = None
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
          summary=summary,
          bias=bias,
          features=features,
          mode=mode,
          iteration_step=iteration_step,
          labels=labels,
          subnetwork_builder=subnetwork_builder,
          var_list=var_list,
          previous_ensemble_spec=ensemble_spec)

  def _build_ensemble_spec(self,
                           name,
                           weighted_subnetworks,
                           summary,
                           bias,
                           features,
                           mode,
                           iteration_step,
                           labels=None,
                           subnetwork_builder=None,
                           var_list=None,
                           previous_ensemble_spec=None):
    """Builds an `_EnsembleSpec` with the given `WeightedSubnetwork`s.

    Args:
      name: The string name of the ensemble. Typically the name of the builder
        that returned the given `Subnetwork`.
      weighted_subnetworks: List of `WeightedSubnetwork` instances that form
        this ensemble. Ordered from first to most recent.
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

    Returns:
      An `_EnsembleSpec` instance.
    """

    ensemble_logits, ensemble_complexity_regularization = (
        self._adanet_weighted_ensemble_logits(weighted_subnetworks, bias,
                                              summary))

    # The AdaNet-weighted ensemble.
    adanet_weighted_ensemble_spec = self._head.create_estimator_spec(
        features=features,
        mode=mode,
        logits=ensemble_logits,
        labels=labels,
        train_op_fn=lambda _: tf.no_op())

    # A baseline ensemble: the uniform-average of subnetwork outputs.
    # It is practically free to compute, requiring no additional training, and
    # tends to generalize very well. However the AdaNet-weighted ensemble
    # should perform at least as well given the correct hyperparameters.
    uniform_average_ensemble_spec = self._head.create_estimator_spec(
        features=features,
        mode=mode,
        logits=self._uniform_average_ensemble_logits(weighted_subnetworks),
        labels=labels,
        train_op_fn=lambda _: tf.no_op())

    # The subnetwork.
    new_subnetwork = weighted_subnetworks[-1].subnetwork
    subnetwork_spec = self._head.create_estimator_spec(
        features=features,
        mode=mode,
        logits=new_subnetwork.logits,
        labels=labels,
        train_op_fn=lambda _: tf.no_op())

    ensemble_loss = adanet_weighted_ensemble_spec.loss
    adanet_loss = None
    eval_metric_ops = {}
    if mode != tf.estimator.ModeKeys.PREDICT:
      adanet_loss = ensemble_loss
      if isinstance(ensemble_complexity_regularization, dict):
        for key in sorted(ensemble_complexity_regularization):
          adanet_loss += ensemble_complexity_regularization[key]
      else:
        adanet_loss += ensemble_complexity_regularization

    if mode == tf.estimator.ModeKeys.EVAL:
      metric_fn = functools.partial(
          _call_metric_fn,
          metric_fn=self._metric_fn,
          features=features,
          labels=labels)
      _add_eval_metric_ops(
          eval_metric_ops=eval_metric_ops,
          group_name="adanet_weighted_ensemble",
          estimator_spec=adanet_weighted_ensemble_spec,
          metric_fn=metric_fn)
      _add_eval_metric_ops(
          eval_metric_ops=eval_metric_ops,
          group_name="uniform_average_ensemble",
          estimator_spec=uniform_average_ensemble_spec,
          metric_fn=metric_fn)
      _add_eval_metric_ops(
          eval_metric_ops=eval_metric_ops,
          group_name="subnetwork",
          estimator_spec=subnetwork_spec,
          metric_fn=metric_fn)
      eval_metric_ops["architecture/adanet/ensembles"] = (
          _architecture_as_metric(weighted_subnetworks))

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
      ensemble_var_list = [w.weight for w in weighted_subnetworks]
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
        predictions=adanet_weighted_ensemble_spec.predictions,
        loss=ensemble_loss,
        adanet_loss=adanet_loss,
        subnetwork_train_op=subnetwork_train_op,
        ensemble_train_op=ensemble_train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=adanet_weighted_ensemble_spec.export_outputs)

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
