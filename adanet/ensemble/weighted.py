# Copyright 2019 The AdaNet Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adanet implementation for weighted Subnetwork and Ensemblers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
from adanet import tf_compat
from adanet.ensemble.ensembler import Ensemble
from adanet.ensemble.ensembler import Ensembler
import tensorflow as tf


def _lookup_if_dict(target, key):
  if isinstance(target, dict):
    return target[key]
  return target


class WeightedSubnetwork(
    collections.namedtuple(
        "WeightedSubnetwork",
        ["name", "iteration_number", "weight", "logits", "subnetwork"])):
  # pyformat: disable
  """An AdaNet weighted subnetwork.

  A weighted subnetwork is a weight applied to a subnetwork's last layer
  or logits (depending on the mixture weights type).

  Args:
    name: String name of :code:`subnetwork` as defined by its
      :class:`adanet.subnetwork.Builder`.
    iteration_number: Integer iteration when the subnetwork was created.
    weight: The weight :class:`tf.Tensor` or dict of string to weight
      :class:`tf.Tensor` (for multi-head) to apply to this subnetwork. The
      AdaNet paper refers to this weight as :math:`w` in Equations (4), (5),
      and (6).
    logits: The output :class:`tf.Tensor` or dict of string to weight
      :class:`tf.Tensor` (for multi-head) after the matrix multiplication of
      :code:`weight` and the subnetwork's :code:`last_layer`. The output's shape
      is [batch_size, logits_dimension]. It is equivalent to a linear logits
      layer in a neural network.
    subnetwork: The :class:`adanet.subnetwork.Subnetwork` to weight.

  Returns:
    An :class:`adanet.ensemble.WeightedSubnetwork` object.
  """
  # pyformat: enable

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


class ComplexityRegularized(
    collections.namedtuple("ComplexityRegularized", [
        "weighted_subnetworks", "bias", "logits", "subnetworks",
        "complexity_regularization"
    ]), Ensemble):
  r"""An AdaNet ensemble where subnetworks are regularized by model complexity.

  Hence an ensemble is a collection of subnetworks which forms a neural network
  through the weighted sum of their outputs:

  .. math::

      F(x) = \sum_{i=1}^{N}w_ih_i(x) + b

  Args:
    weighted_subnetworks: List of :class:`adanet.ensemble.WeightedSubnetwork`
      instances that form this ensemble. Ordered from first to most recent.
    bias: Bias term :class:`tf.Tensor` or dict of string to bias term
      :class:`tf.Tensor` (for multi-head) for the ensemble's logits.
    logits: Logits :class:`tf.Tensor` or dict of string to logits
      :class:`tf.Tensor` (for multi-head). The result of the function *f* as
      defined in Section 5.1 which is the sum of the logits of all
      :class:`adanet.WeightedSubnetwork` instances in ensemble.
    subnetworks: List of :class:`adanet.subnetwork.Subnetwork` instances that
      form this ensemble. This is kept together with weighted_subnetworks for
      legacy reasons.
    complexity_regularization: Regularization to be added in the Adanet loss.

  Returns:
    An :class:`adanet.ensemble.Weighted` instance.
  """

  def __new__(cls,
              weighted_subnetworks,
              bias,
              logits,
              subnetworks=None,
              complexity_regularization=None):
    return super(ComplexityRegularized, cls).__new__(
        cls,
        weighted_subnetworks=list(weighted_subnetworks),
        bias=bias,
        logits=logits,
        subnetworks=list(subnetworks or []),
        complexity_regularization=complexity_regularization)


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


class ComplexityRegularizedEnsembler(Ensembler):
  # pyformat: disable
  r"""The AdaNet algorithm implemented as an :class:`adanet.ensemble.Ensembler`.

  The AdaNet algorithm was introduced in the [Cortes et al. ICML 2017] paper:
  https://arxiv.org/abs/1607.01097.

  The AdaNet algorithm uses a weak learning algorithm to iteratively generate a
  set of candidate subnetworks that attempt to minimize the loss function
  defined in Equation (4) as part of an ensemble. At the end of each iteration,
  the best candidate is chosen based on its ensemble's complexity-regularized
  train loss. New subnetworks are allowed to use any subnetwork weights within
  the previous iteration's ensemble in order to improve upon them. If the
  complexity-regularized loss of the new ensemble, as defined in Equation (4),
  is less than that of the previous iteration's ensemble, the AdaNet algorithm
  continues onto the next iteration.

  AdaNet attempts to minimize the following loss function to learn the mixture
  weights :math:`w` of each subnetwork :math:`h` in the ensemble with
  differentiable convex non-increasing surrogate loss function :math:`\Phi`:

  Equation (4):

  .. math::

      F(w) = \frac{1}{m} \sum_{i=1}^{m} \Phi \left(\sum_{j=1}^{N}w_jh_j(x_i),
      y_i \right) + \sum_{j=1}^{N} \left(\lambda r(h_j) + \beta \right) |w_j|

  with :math:`\lambda >= 0` and :math:`\beta >= 0`.

  Args:
    optimizer: A :class:`tf.train.Optimizer` instance to be used for building
      the train op. If left as None, :meth:`tf.no_op` is returned as train op.
    mixture_weight_type: The :class:`adanet.ensemble.MixtureWeightType` defining
      which mixture weight type to learn on top of the subnetworks' logits.
    mixture_weight_initializer: The initializer for mixture_weights. When
      :code:`None`, the default is different according to
      :code:`mixture_weight_type`:

        - :code:`SCALAR` initializes to :math:`1/N` where :math:`N` is the
          number of subnetworks in the ensemble giving a uniform average.
        - :code:`VECTOR` initializes each entry to :math:`1/N` where :math:`N`
          is the number of subnetworks in the ensemble giving a uniform average.
        - :code:`MATRIX` uses :meth:`tf.zeros_initializer`.
    warm_start_mixture_weights: Whether, at the beginning of an iteration, to
      initialize the mixture weights of the subnetworks from the previous
      ensemble to their learned value at the previous iteration, as opposed to
      retraining them from scratch. Takes precedence over the value for
      :code:`mixture_weight_initializer` for subnetworks from previous
      iterations.
    model_dir: The model dir to use for warm-starting mixture weights and bias
      at the logit layer. Ignored if :code:`warm_start_mixture_weights` is
      :code:`False`.
    adanet_lambda: Float multiplier :math:`\lambda` for applying :math:`L1`
      regularization to subnetworks' mixture weights :math:`w` in the ensemble
      proportional to their complexity. See Equation (4) in the AdaNet paper.
    adanet_beta: Float :math:`L1` regularization multiplier :math:`\beta` to apply
      equally to all subnetworks' weights :math:`w` in the ensemble regardless of
      their complexity. See Equation (4) in the AdaNet paper.
    use_bias: Whether to add a bias term to the ensemble's logits.

  Returns:
    An `adanet.ensemble.ComplexityRegularizedEnsembler` instance.

  Raises:
    ValueError: if :code:`warm_start_mixture_weights` is :code:`True` but
    :code:`model_dir` is :code:`None`.
  """
  # pyformat: enable

  def __init__(self,
               optimizer=None,
               mixture_weight_type=MixtureWeightType.SCALAR,
               mixture_weight_initializer=None,
               warm_start_mixture_weights=False,
               model_dir=None,
               adanet_lambda=0.,
               adanet_beta=0.,
               use_bias=False):
    if warm_start_mixture_weights:
      if model_dir is None:
        raise ValueError("model_dir cannot be None when "
                         "warm_start_mixture_weights is True.")

    self._optimizer = optimizer
    self._mixture_weight_type = mixture_weight_type
    self._mixture_weight_initializer = mixture_weight_initializer
    self._warm_start_mixture_weights = warm_start_mixture_weights
    self._model_dir = model_dir
    self._adanet_lambda = adanet_lambda
    self._adanet_beta = adanet_beta
    self._use_bias = use_bias

  @property
  def name(self):
    return "complexity_regularized"

  def build_ensemble(self, subnetworks, previous_ensemble_subnetworks, features,
                     labels, logits_dimension, training, iteration_step,
                     summary, previous_ensemble):
    del features, labels, logits_dimension, training, iteration_step  # unused
    weighted_subnetworks = []
    subnetwork_index = 0
    num_subnetworks = len(subnetworks)

    if previous_ensemble_subnetworks and previous_ensemble:
      num_subnetworks += len(previous_ensemble_subnetworks)
      for weighted_subnetwork in previous_ensemble.weighted_subnetworks:
        if weighted_subnetwork.subnetwork not in previous_ensemble_subnetworks:
          # Pruned.
          continue
        weight_initializer = None
        if self._warm_start_mixture_weights:
          if isinstance(weighted_subnetwork.subnetwork.last_layer, dict):
            weight_initializer = {
                key: self._load_variable_from_model_dir(
                    weighted_subnetwork.weight[key])
                for key in sorted(weighted_subnetwork.subnetwork.last_layer)
            }
          else:
            weight_initializer = self._load_variable_from_model_dir(
                weighted_subnetwork.weight)
        with tf_compat.v1.variable_scope(
            "weighted_subnetwork_{}".format(subnetwork_index)):
          weighted_subnetworks.append(
              self._build_weighted_subnetwork(
                  weighted_subnetwork.subnetwork,
                  num_subnetworks,
                  weight_initializer=weight_initializer))
        subnetwork_index += 1

    for subnetwork in subnetworks:
      with tf_compat.v1.variable_scope(
          "weighted_subnetwork_{}".format(subnetwork_index)):
        weighted_subnetworks.append(
            self._build_weighted_subnetwork(subnetwork, num_subnetworks))
      subnetwork_index += 1

    if previous_ensemble:
      if len(
          previous_ensemble.subnetworks) == len(previous_ensemble_subnetworks):
        bias = self._create_bias_term(
            weighted_subnetworks, prior=previous_ensemble.bias)
      else:
        bias = self._create_bias_term(weighted_subnetworks)
        logging.info("Builders using a pruned set of the subnetworks "
                     "from the previous ensemble, so its ensemble's bias "
                     "term will not be warm started with the previous "
                     "ensemble's bias.")
    else:
      bias = self._create_bias_term(weighted_subnetworks)

    logits = self._create_ensemble_logits(weighted_subnetworks, bias, summary)
    complexity_regularization = 0
    if isinstance(logits, dict):
      for key in sorted(logits):
        complexity_regularization += self._compute_complexity_regularization(
            weighted_subnetworks, summary, key)
    else:
      complexity_regularization = self._compute_complexity_regularization(
          weighted_subnetworks, summary)

    return ComplexityRegularized(
        weighted_subnetworks=weighted_subnetworks,
        bias=bias,
        subnetworks=[ws.subnetwork for ws in weighted_subnetworks],
        logits=logits,
        complexity_regularization=complexity_regularization)

  def _load_variable_from_model_dir(self, var):
    return tf.train.load_variable(self._model_dir, tf_compat.tensor_name(var))

  def _compute_adanet_gamma(self, complexity):
    """For a subnetwork, computes: lambda * r(h) + beta."""

    if self._adanet_lambda == 0.:
      return self._adanet_beta
    return tf.scalar_mul(self._adanet_lambda,
                         tf.cast(complexity,
                                 dtype=tf.float32)) + self._adanet_beta

  def _select_mixture_weight_initializer(self, num_subnetworks):
    if self._mixture_weight_initializer:
      return self._mixture_weight_initializer
    if (self._mixture_weight_type == MixtureWeightType.SCALAR or
        self._mixture_weight_type == MixtureWeightType.VECTOR):
      return tf_compat.v1.constant_initializer(1. / num_subnetworks)
    return tf_compat.v1.zeros_initializer()

  def _build_weighted_subnetwork(self,
                                 subnetwork,
                                 num_subnetworks,
                                 weight_initializer=None):
    """Builds an `adanet.ensemble.WeightedSubnetwork`.

    Args:
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
      for i, key in enumerate(sorted(subnetwork.last_layer)):
        logits[key], weight[key] = self._build_weighted_subnetwork_helper(
            subnetwork, num_subnetworks,
            _lookup_if_dict(weight_initializer, key), key, i)
    else:
      logits, weight = self._build_weighted_subnetwork_helper(
          subnetwork, num_subnetworks, weight_initializer)

    return WeightedSubnetwork(
        subnetwork=subnetwork, logits=logits, weight=weight)

  def _build_weighted_subnetwork_helper(self,
                                        subnetwork,
                                        num_subnetworks,
                                        weight_initializer=None,
                                        key=None,
                                        index=None):
    """Returns the logits and weight of the `WeightedSubnetwork` for key."""

    # Treat subnetworks as if their weights are frozen, and ensure that
    # mixture weight gradients do not propagate through.
    last_layer = _lookup_if_dict(subnetwork.last_layer, key)
    logits = _lookup_if_dict(subnetwork.logits, key)
    weight_shape = None
    last_layer_size = last_layer.get_shape().as_list()[-1]
    logits_size = logits.get_shape().as_list()[-1]
    batch_size = tf.shape(input=last_layer)[0]

    if weight_initializer is None:
      weight_initializer = self._select_mixture_weight_initializer(
          num_subnetworks)
      if self._mixture_weight_type == MixtureWeightType.SCALAR:
        weight_shape = []
      if self._mixture_weight_type == MixtureWeightType.VECTOR:
        weight_shape = [logits_size]
      if self._mixture_weight_type == MixtureWeightType.MATRIX:
        weight_shape = [last_layer_size, logits_size]

    with tf_compat.v1.variable_scope(
        "logits_{}".format(index) if index else "logits"):
      weight = tf_compat.v1.get_variable(
          name="mixture_weight",
          shape=weight_shape,
          initializer=weight_initializer)
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
          logging.info("Rank 3 tensors like [batch_size, timesteps, d]  are "
                       "reshaped to rank 2 [ batch_size x timesteps, d] for "
                       "the weight matrix multiplication, and are reshaped "
                       "to their original shape afterwards.")
          last_layer = tf.reshape(last_layer, [-1, last_layer_size])
        logits = tf.matmul(last_layer, weight)
        if ndims == 3:
          logits = tf.reshape(logits, [batch_size, -1, logits_size])
      else:
        logits = tf.multiply(logits, weight)
    return logits, weight

  def _create_bias_term(self, weighted_subnetworks, prior=None):
    """Returns a bias term vector.

    If `use_bias` is set, then it returns a trainable bias variable initialized
    to zero, or warm-started with the given prior. Otherwise it returns
    a non-trainable zero variable.

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
    for i, key in enumerate(sorted(weighted_subnetworks[0].subnetwork.logits)):
      bias_terms[key] = self._create_bias_term_helper(weighted_subnetworks,
                                                      prior, key, i)
    return bias_terms

  def _create_bias_term_helper(self,
                               weighted_subnetworks,
                               prior,
                               key=None,
                               index=None):
    """Returns a bias term for weights with the given key."""

    shape = None
    if prior is None or not self._warm_start_mixture_weights:
      prior = tf_compat.v1.zeros_initializer()
      logits = _lookup_if_dict(weighted_subnetworks[0].subnetwork.logits, key)
      dims = logits.shape.as_list()
      shape = dims[-1] if len(dims) > 1 else 1
    else:
      prior = self._load_variable_from_model_dir(_lookup_if_dict(prior, key))
    return tf_compat.v1.get_variable(
        name="bias_{}".format(index) if index else "bias",
        shape=shape,
        initializer=prior,
        trainable=self._use_bias)

  def _create_ensemble_logits(self, weighted_subnetworks, bias, summary):
    """Computes the AdaNet weighted ensemble logits.

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
      return self._create_ensemble_logits_helper(weighted_subnetworks, bias,
                                                 summary)
    logits_dict = weighted_subnetworks[0].subnetwork.logits
    return {
        key: self._create_ensemble_logits_helper(
            weighted_subnetworks, bias, summary, key=key, index=i)
        for i, key in enumerate(sorted(logits_dict))
    }

  def _create_ensemble_logits_helper(self,
                                     weighted_subnetworks,
                                     bias,
                                     summary,
                                     key=None,
                                     index=None):
    """Returns the AdaNet ensemble logits and regularization term for key."""

    subnetwork_logits = []
    for weighted_subnetwork in weighted_subnetworks:
      subnetwork_logits.append(_lookup_if_dict(weighted_subnetwork.logits, key))
    with tf_compat.v1.variable_scope(
        "logits_{}".format(index) if index else "logits"):
      ensemble_logits = _lookup_if_dict(bias, key)
      for logits in subnetwork_logits:
        ensemble_logits = tf.add(ensemble_logits, logits)
    return ensemble_logits

  def _compute_complexity_regularization(self,
                                         weighted_subnetworks,
                                         summary,
                                         key=None):
    """Returns the AdaNet regularization term contribution for a key."""

    ensemble_complexity_regularization = 0
    total_weight_l1_norms = 0
    weights = []
    for weighted_subnetwork in weighted_subnetworks:
      weight_l1_norm = tf.norm(
          tensor=_lookup_if_dict(weighted_subnetwork.weight, key), ord=1)
      total_weight_l1_norms += weight_l1_norm
      ensemble_complexity_regularization += (
          self._compute_complexity_regularization_helper(
              weight_l1_norm, weighted_subnetwork.subnetwork.complexity))
      weights.append(weight_l1_norm)

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
    return ensemble_complexity_regularization

  def _compute_complexity_regularization_helper(self, weight_l1_norm,
                                                complexity):
    """For a subnetwork, computes: (lambda * r(h) + beta) * |w|."""

    # Note: Unsafe comparison against float zero.
    if self._adanet_lambda == 0. and self._adanet_beta == 0.:
      return tf.constant(0., name="zero")
    return tf.scalar_mul(self._compute_adanet_gamma(complexity), weight_l1_norm)

  def build_train_op(self, ensemble, loss, var_list, labels, iteration_step,
                     summary, previous_ensemble):
    del labels, iteration_step, summary, previous_ensemble  # unused
    if self._optimizer is None:
      return tf.no_op()

    # The AdaNet Estimator is responsible for incrementing the global step.
    return self._optimizer.minimize(
        loss=loss + ensemble.complexity_regularization, var_list=var_list)
