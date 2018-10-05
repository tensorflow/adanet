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

import tensorflow as tf


class WeightedSubnetwork(
    collections.namedtuple("WeightedSubnetwork",
                           ["name", "weight", "logits", "subnetwork"])):
  """An AdaNet weighted subnetwork.

  A weighted subnetwork is a weight 'w' applied to a subnetwork's last layer
  'u'. The results is the weighted subnetwork's logits, regularized by its
  complexity.
  """

  def __new__(cls, name, weight, logits, subnetwork):
    """Creates a `WeightedSubnetwork` instance.

    Args:
      name: The string `tf.constant` name of `subnetwork`.
      weight: The weight `Tensor` to apply to this subnetwork. The AdaNet paper
        refers to this weight as 'w' in Equations (4), (5), and (6).
      logits: The output `Tensor` after the matrix multiplication of `weight`
        and the subnetwork's `last_layer`. The weight's shape is [batch_size,
        logits_dimension]. It is equivalent to a linear logits layer in a neural
        network.
      subnetwork: The `adanet.Subnetwork` to weight.

    Returns:
      A `WeightedSubnetwork` object.
    """

    return super(WeightedSubnetwork, cls).__new__(
        cls, name=name, weight=weight, logits=logits, subnetwork=subnetwork)


class Ensemble(
    collections.namedtuple("Ensemble", [
        "name",
        "weighted_subnetworks",
        "bias",
        "logits",
        "predictions",
        "loss",
        "adanet_loss",
        "complexity_regularized_loss",
        "train_op",
        "complexity_regularization",
        "eval_metric_ops",
        "export_outputs",
    ])):
  """An AdaNet ensemble.

  An ensemble is a collection of subnetworks which forms a neural network
  through the weighted sum of their outputs. It is represented by 'f' throughout
  the AdaNet paper. Its component subnetworks' weights are complexity
  regularized (Gamma) as defined in Equation (4).

  # TODO: Remove fields related to training and evaluation.
  """

  def __new__(cls,
              name,
              weighted_subnetworks,
              bias,
              logits,
              predictions,
              loss=None,
              adanet_loss=None,
              complexity_regularized_loss=None,
              train_op=None,
              complexity_regularization=None,
              eval_metric_ops=None,
              export_outputs=None):
    """Creates an `Ensemble` instance.

    Args:
      name: String name of this ensemble. Should be unique in the graph.
      weighted_subnetworks: List of `WeightedSubnetwork` instances that form
        this ensemble. Ordered from first to most recent.
      bias: `Tensor` bias vector for the ensemble logits.
      logits: Logits `Tensor`. The result of the function 'f' as defined in
        Section 5.1 which is the sum of the logits of all `WeightedSubnetwork`
        instances in ensemble.
      predictions: Predictions `Tensor` or dict of `Tensor`.
      loss: Loss `Tensor` as defined by the surrogate loss function Phi in
        Equations (4), (5), and (6). Must be either scalar, or with shape `[1]`.
      adanet_loss: Loss `Tensor` as defined by F(w) in Equation (4). Must be
        either scalar, or with shape `[1]`. The AdaNet algorithm aims to
        minimize this objective which balances training loss with the total
        complexity of the subnetworks in the ensemble.
      complexity_regularized_loss: Loss `Tensor` as defined by F(w,u) in
        Equation (5). Must be either scalar, or with shape `[1]`.
      train_op: Op for the training step.
      complexity_regularization: Complexity regularization `Tensor` of the
        weighted-L1 penalty regularization term in F(w) in Equation (4).
      eval_metric_ops: Dict of metric results keyed by name. The values of the
        dict are the results of calling a metric function, namely a
        `(metric_tensor, update_op)` tuple. `metric_tensor` should be evaluated
        without any impact on state (typically is a pure computation based on
        variables.). For example, it should not trigger the `update_op` or
        require any input fetching.
      export_outputs: Describes the output signatures to be exported to
        `SavedModel` and used during serving. See `tf.estimator.EstimatorSpec`.

    Returns:
      An `Ensemble` object.
    """

    # TODO: Make weighted_subnetworks property a tuple so that
    # `Ensemble` is immutable.
    return super(Ensemble, cls).__new__(
        cls,
        name=name,
        weighted_subnetworks=weighted_subnetworks,
        bias=bias,
        logits=logits,
        predictions=predictions,
        loss=loss,
        adanet_loss=adanet_loss,
        complexity_regularized_loss=complexity_regularized_loss,
        train_op=train_op,
        complexity_regularization=complexity_regularization,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs)


class MixtureWeightType(object):
  """Mixture weight types available for learning subnetwork contributions.

  The following mixture weight types are defined:

  * `SCALAR`: A rank 0 `Tensor` mixture weight.
  * `VECTOR`: A rank 1 `Tensor` mixture weight.
  * `MATRIX`: A rank 2 `Tensor` mixture weight.
  """

  SCALAR = "scalar"
  VECTOR = "vector"
  MATRIX = "matrix"


def _architecture_as_metric(weighted_subnetworks):
  """Returns a representation of the ensemble's architecture as a tf.metric."""

  joined_names = " | ".join([
      str(tf.contrib.util.constant_value(w.name)) for w in weighted_subnetworks
  ])
  architecture = tf.convert_to_tensor(
      "| {} |".format(joined_names), name="architecture")
  architecture_summary = tf.summary.text("architecture/adanet", architecture)
  return (architecture_summary, tf.no_op())


class _EnsembleBuilder(object):
  """Builds `Ensemble` instances."""

  def __init__(self,
               head,
               mixture_weight_type,
               mixture_weight_initializer=None,
               warm_start_mixture_weights=False,
               adanet_lambda=0.,
               adanet_beta=0.,
               use_bias=True):
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
      adanet_lambda: Float multiplier 'lambda' for applying L1 regularization to
        subnetworks' mixture weights 'w' in the ensemble proportional to their
        complexity. See Equation (4) in the AdaNet paper.
      adanet_beta: Float L1 regularization multiplier 'beta' to apply equally to
        all subnetworks' weights 'w' in the ensemble regardless of their
        complexity. See Equation (4) in the AdaNet paper.
      use_bias: Whether to add a bias term to the ensemble's logits.

    Returns:
      An `_EnsembleBuilder` instance.
    """

    self._head = head
    self._mixture_weight_type = mixture_weight_type
    self._mixture_weight_initializer = mixture_weight_initializer
    self._warm_start_mixture_weights = warm_start_mixture_weights
    self._adanet_lambda = adanet_lambda
    self._adanet_beta = adanet_beta
    self._use_bias = use_bias

  def append_new_subnetwork(self,
                            ensemble,
                            subnetwork_builder,
                            iteration_step,
                            summary,
                            features,
                            mode,
                            labels=None):
    """Adds a `Subnetwork` to an `Ensemble` from iteration t-1 for iteration t.

    For iteration t > 0, the ensemble is built given the `Ensemble` for t-1 and
    the new subnetwork to train as part of the ensemble. The `Ensemble` at
    iteration 0 is comprised of just the subnetwork.

    The subnetwork is first given a weight 'w' in a `WeightedSubnetwork`
    which determines its contribution to the ensemble. The subnetwork's
    complexity L1-regularizes this weight.

    Args:
      ensemble: The recipient `Ensemble` for the `Subnetwork`.
      subnetwork_builder: A `adanet.Builder` instance which defines how to train
        the subnetwork and ensemble mixture weights.
      iteration_step: Integer `Tensor` representing the step since the beginning
        of the current iteration, as opposed to the global step.
      summary: A `_ScopedSummary` instance for recording ensemble summaries.
      features: Input `dict` of `Tensor` objects.
      mode: Estimator's `ModeKeys`.
      labels: Labels `Tensor`, or `dict` of same. Can be None during inference.

    Returns:
      An new `Ensemble` instance with the `Subnetwork` appended.
    """

    with tf.variable_scope("ensemble_{}".format(subnetwork_builder.name)):
      weighted_subnetworks = []
      iteration = 0
      num_subnetworks = 1
      if ensemble:
        num_subnetworks += len(ensemble.weighted_subnetworks)
        for weighted_subnetwork in ensemble.weighted_subnetworks:
          weight_initializer = None
          if self._warm_start_mixture_weights:
            weight_initializer = weighted_subnetwork.weight
          with tf.variable_scope("weighted_subnetwork_{}".format(iteration)):
            weighted_subnetworks.append(
                self._build_weighted_subnetwork(
                    weighted_subnetwork.name,
                    weighted_subnetwork.subnetwork,
                    self._head.logits_dimension,
                    num_subnetworks,
                    weight_initializer=weight_initializer))
          iteration += 1
        bias = self._create_bias(
            self._head.logits_dimension, prior=ensemble.bias)
      else:
        bias = self._create_bias(self._head.logits_dimension)

      with tf.variable_scope("weighted_subnetwork_{}".format(iteration)):
        with tf.variable_scope("subnetwork"):
          trainable_vars_before = tf.trainable_variables()
          subnetwork = subnetwork_builder.build_subnetwork(
              features=features,
              logits_dimension=self._head.logits_dimension,
              training=mode == tf.estimator.ModeKeys.TRAIN,
              iteration_step=iteration_step,
              summary=summary,
              previous_ensemble=ensemble)
          trainable_vars_after = tf.trainable_variables()
          var_list = list(
              set(trainable_vars_after) - set(trainable_vars_before))
        weighted_subnetworks.append(
            self._build_weighted_subnetwork(
                tf.constant(subnetwork_builder.name, name="name"), subnetwork,
                self._head.logits_dimension, num_subnetworks))

      return self.build_ensemble(
          name=subnetwork_builder.name,
          weighted_subnetworks=weighted_subnetworks,
          summary=summary,
          bias=bias,
          features=features,
          mode=mode,
          iteration_step=iteration_step,
          labels=labels,
          subnetwork_builder=subnetwork_builder,
          var_list=var_list,
          previous_ensemble=ensemble)

  def build_ensemble(self,
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
                     previous_ensemble=None):
    """Builds an `Ensemble` with the given `WeightedSubnetwork`s.

    Args:
      name: The string name of the ensemble. Typically the name of the builder
        that returned the given `Subnetwork`.
      weighted_subnetworks: List of `WeightedSubnetwork` instances that form
        this ensemble. Ordered from first to most recent.
      summary: A `_ScopedSummary` instance for recording ensemble summaries.
      bias: `Tensor` bias vector for the ensemble logits.
      features: Input `dict` of `Tensor` objects.
      mode: Estimator `ModeKeys` indicating training, evaluation, or inference.
      iteration_step: Integer `Tensor` representing the step since the beginning
        of the current iteration, as opposed to the global step.
      labels: Labels `Tensor`, or `dict` of same. Can be None during inference.
      subnetwork_builder: A `adanet.Builder` instance which defines how to train
        the subnetwork and ensemble mixture weights.
      var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.
      previous_ensemble: Link the rest of the `Ensemble` from iteration t-1.
        Used for creating the subnetwork train_op.

    Returns:
      An `Ensemble` instance.
    """

    subnetwork_logits = []
    ensemble_complexity_regularization = 0
    total_weight_l1_norms = 0
    weights = []
    for weighted_subnetwork in weighted_subnetworks:
      weight_l1_norm = tf.norm(weighted_subnetwork.weight, ord=1)
      total_weight_l1_norms += weight_l1_norm
      ensemble_complexity_regularization += self._complexity_regularization(
          weight_l1_norm, weighted_subnetwork.subnetwork.complexity)
      subnetwork_logits.append(weighted_subnetwork.logits)
      weights.append(weight_l1_norm)

    with tf.variable_scope("logits"):
      ensemble_logits = bias
      for logits in subnetwork_logits:
        ensemble_logits = tf.add(ensemble_logits, logits)

    with tf.name_scope(""):
      summary.histogram("mixture_weights/adanet/adanet_weighted_ensemble",
                        weights)
      for iteration, weight in enumerate(weights):
        scope = "adanet/adanet_weighted_ensemble/subnetwork_{}".format(
            iteration)
        summary.scalar("mixture_weight_norms/{}".format(scope), weight)
        fraction = weight / total_weight_l1_norms
        summary.scalar("mixture_weight_fractions/{}".format(scope), fraction)

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
    uniform_average_ensemble_logits = tf.add_n([
        wwl.subnetwork.logits for wwl in weighted_subnetworks
    ]) / len(weighted_subnetworks)
    uniform_average_ensemble_spec = self._head.create_estimator_spec(
        features=features,
        mode=mode,
        logits=uniform_average_ensemble_logits,
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
      adanet_loss = ensemble_loss + ensemble_complexity_regularization
      eval_metric_ops["loss/adanet/adanet_weighted_ensemble"] = tf.metrics.mean(
          ensemble_loss)
      for metric, ops in adanet_weighted_ensemble_spec.eval_metric_ops.items():
        eval_metric_ops["{}/adanet/adanet_weighted_ensemble".format(
            metric)] = ops
      avg_metric_ops = uniform_average_ensemble_spec.eval_metric_ops
      eval_metric_ops["loss/adanet/uniform_average_ensemble"] = tf.metrics.mean(
          uniform_average_ensemble_spec.loss)
      for metric, ops in avg_metric_ops.items():
        eval_metric_ops["{}/adanet/uniform_average_ensemble".format(
            metric)] = ops
      eval_metric_ops["loss/adanet/subnetwork"] = tf.metrics.mean(
          subnetwork_spec.loss)
      for metric, ops in subnetwork_spec.eval_metric_ops.items():
        eval_metric_ops["{}/adanet/subnetwork".format(metric)] = ops
      eval_metric_ops["architecture/adanet/ensembles"] = (
          _architecture_as_metric(weighted_subnetworks))
      with tf.name_scope(""):
        summary.scalar("loss/adanet/adanet_weighted_ensemble",
                       adanet_weighted_ensemble_spec.loss)
        summary.scalar("loss/adanet/subnetwork", subnetwork_spec.loss)
        summary.scalar("loss/adanet/uniform_average_ensemble",
                       uniform_average_ensemble_spec.loss)

    # TODO: Merge AdaNet loss and complexity_regularized_loss.
    complexity_regularized_loss = adanet_loss

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN and subnetwork_builder:
      with tf.variable_scope("train_subnetwork"):
        subnetwork_train_op = (
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
      with tf.variable_scope("train_mixture_weights"):
        ensemble_train_op = subnetwork_builder.build_mixture_weights_train_op(
            loss=adanet_loss,
            var_list=ensemble_var_list,
            logits=ensemble_logits,
            labels=labels,
            iteration_step=iteration_step,
            summary=summary)
      train_op = tf.group(subnetwork_train_op, ensemble_train_op)

    return Ensemble(
        name=name,
        weighted_subnetworks=weighted_subnetworks,
        bias=bias,
        logits=ensemble_logits,
        predictions=adanet_weighted_ensemble_spec.predictions,
        loss=ensemble_loss,
        adanet_loss=adanet_loss,
        complexity_regularized_loss=complexity_regularized_loss,
        train_op=train_op,
        complexity_regularization=ensemble_complexity_regularization,
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
                                 subnetwork,
                                 logits_dimension,
                                 num_subnetworks,
                                 weight_initializer=None):
    """Builds an `WeightedSubnetwork`.

    Args:
      name: The string `tf.constant` name of `subnetwork`.
      subnetwork: The `Subnetwork` to weight.
      logits_dimension: The number of outputs from the logits.
      num_subnetworks: The number of subnetworks in the ensemble.
      weight_initializer: Initializer for the weight variable. Can be a
        `Constant` prior weight to use for warm-starting.

    Returns:
      A `WeightedSubnetwork` instance.

    Raises:
      ValueError: When the subnetwork's last layer and logits dimension do
        not match and requiring a SCALAR or VECTOR mixture weight.
    """

    # Treat subnetworks as if their weights are frozen, and ensure that
    # mixture weight gradients do not propagate through.
    last_layer = tf.stop_gradient(subnetwork.last_layer)

    weight_shape = None
    static_shape = last_layer.get_shape().as_list()
    last_layer_size = static_shape[-1]
    ndims = len(static_shape)
    batch_size = tf.shape(last_layer)[0]

    if weight_initializer is None:
      weight_initializer = self._select_mixture_weight_initializer(
          num_subnetworks)
      if self._mixture_weight_type == MixtureWeightType.SCALAR:
        weight_shape = []
      if self._mixture_weight_type == MixtureWeightType.VECTOR:
        weight_shape = [logits_dimension]
      if self._mixture_weight_type == MixtureWeightType.MATRIX:
        weight_shape = [last_layer_size, logits_dimension]

    with tf.variable_scope("logits"):
      weight = tf.get_variable(
          name="mixture_weight",
          shape=weight_shape,
          initializer=weight_initializer)
      if self._mixture_weight_type == MixtureWeightType.MATRIX:
        # TODO: Add Unit tests for the ndims == 3 path.
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
          logits = tf.reshape(logits, [batch_size, -1, logits_dimension])
      else:
        logits = tf.multiply(subnetwork.logits, weight)

    return WeightedSubnetwork(
        name=name, subnetwork=subnetwork, logits=logits, weight=weight)

  def _create_bias(self, logits_dimension, prior=None):
    """Returns a bias term vector.

    If `use_bias` is set, then it returns a trainable bias variable initialized
    to zero, or warm-started with the given prior. Otherwise it returns
    a zero constant bias.

    Args:
      logits_dimension: The number of outputs from the logits.
      prior: Prior for the bias variable for warm-starting.

    Returns:
      A bias term `Tensor`.
    """

    if not self._use_bias:
      if prior is not None:
        return prior
      return tf.constant(0., name="zero_bias")
    shape = None
    if prior is None:
      prior = tf.zeros_initializer()
      shape = logits_dimension
    return tf.get_variable(name="bias", shape=shape, initializer=prior)
