"""A simple dense neural network search space.

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

import functools

import adanet
from six.moves import range
import tensorflow as tf

_NUM_LAYERS_KEY = "num_layers"


class _SimpleDNNBuilder(adanet.subnetwork.Builder):
  """Builds a DNN subnetwork for AdaNet."""

  def __init__(self, feature_columns, optimizer, layer_size, num_layers,
               learn_mixture_weights, dropout, seed):
    """Initializes a `_DNNBuilder`.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      optimizer: An `Optimizer` instance for training both the subnetwork and
        the mixture weights.
      layer_size: The number of nodes to output at each hidden layer.
      num_layers: The number of hidden layers.
      learn_mixture_weights: Whether to solve a learning problem to find the
        best mixture weights, or use their default value according to the
        mixture weight type. When `False`, the subnetworks will return a no_op
        for the mixture weight train op.
      dropout: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
        10% of input units.
      seed: A random seed.

    Returns:
      An instance of `_DNNBuilder`.
    """

    self._feature_columns = feature_columns
    self._optimizer = optimizer
    self._layer_size = layer_size
    self._num_layers = num_layers
    self._learn_mixture_weights = learn_mixture_weights
    self._dropout = dropout
    self._seed = seed

  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    """See `adanet.subnetwork.Builder`."""

    input_layer = tf.feature_column.input_layer(
        features=features, feature_columns=self._feature_columns)
    last_layer = input_layer
    for _ in range(self._num_layers):
      last_layer = tf.layers.dense(
          last_layer,
          units=self._layer_size,
          activation=tf.nn.relu,
          kernel_initializer=tf.glorot_uniform_initializer(seed=self._seed))
      last_layer = tf.layers.dropout(
          last_layer, rate=self._dropout, seed=self._seed, training=training)
    logits = tf.layers.dense(
        last_layer,
        units=logits_dimension,
        kernel_initializer=tf.glorot_uniform_initializer(seed=self._seed))

    # Approximate the Rademacher complexity of this subnetwork as the square-
    # root of its depth.
    complexity = tf.sqrt(tf.to_float(self._num_layers))

    with tf.name_scope(""):
      summary.scalar("complexity", complexity)
      summary.scalar("num_layers", self._num_layers)

    persisted_tensors = {_NUM_LAYERS_KEY: tf.constant(self._num_layers)}
    return adanet.Subnetwork(
        last_layer=last_layer,
        logits=logits,
        complexity=complexity,
        persisted_tensors=persisted_tensors)

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    """See `adanet.subnetwork.Builder`."""

    # NOTE: The `adanet.Estimator` increments the global step.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      return self._optimizer.minimize(loss=loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    """See `adanet.subnetwork.Builder`."""

    if not self._learn_mixture_weights:
      return tf.no_op("mixture_weights_train_op")

    # NOTE: The `adanet.Estimator` increments the global step.
    return self._optimizer.minimize(loss=loss, var_list=var_list)

  @property
  def name(self):
    """See `adanet.subnetwork.Builder`."""

    if self._num_layers == 0:
      # A DNN with no hidden layers is a linear model.
      return "linear"
    return "{}_layer_dnn".format(self._num_layers)


class Generator(adanet.subnetwork.Generator):
  """Generates a two DNN subnetworks at each iteration.

  The first DNN has an identical shape to the most recently added subnetwork
  in `previous_ensemble`. The second has the same shape plus one more dense
  layer on top. This is similar to the adaptive network presented in Figure 2 of
  [Cortes et al. ICML 2017](https://arxiv.org/abs/1607.01097), without the
  connections to hidden layers of networks from previous iterations.
  """

  def __init__(self,
               feature_columns,
               optimizer,
               layer_size=32,
               initial_num_layers=0,
               learn_mixture_weights=False,
               dropout=0.,
               seed=None):
    """Initializes a DNN `Generator`.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        DNN models. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      optimizer: An `Optimizer` instance for training both the subnetwork and
        the mixture weights.
      layer_size: Number of nodes in each hidden layer of the subnetwork
        candidates. Note that this parameter is ignored in a DNN with no hidden
        layers.
      initial_num_layers: Minimum number of layers for each DNN subnetwork. At
        iteration 0, the subnetworks will be `initial_num_layers` deep.
        Subnetworks at subsequent iterations will be at least as deep.
      learn_mixture_weights: Whether to solve a learning problem to find the
        best mixture weights, or use their default value according to the
        mixture weight type. When `False`, the subnetworks will return a no_op
        for the mixture weight train op.
      dropout: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
        10% of input units.
      seed: A random seed.

    Returns:
      An instance of `Generator`.

    Raises:
      ValueError: If feature_columns is empty.
      ValueError: If layer_size < 1.
      ValueError: If initial_num_layers < 0.
    """

    if not feature_columns:
      raise ValueError("feature_columns must not be empty")

    if layer_size < 1:
      raise ValueError("layer_size must be >= 1")

    if initial_num_layers < 0:
      raise ValueError("initial_num_layers must be >= 0")

    self._initial_num_layers = initial_num_layers
    self._dnn_builder_fn = functools.partial(
        _SimpleDNNBuilder,
        feature_columns=feature_columns,
        optimizer=optimizer,
        layer_size=layer_size,
        learn_mixture_weights=learn_mixture_weights,
        dropout=dropout,
        seed=seed)

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    """See `adanet.subnetwork.Generator`."""

    num_layers = self._initial_num_layers
    if previous_ensemble:
      num_layers = tf.contrib.util.constant_value(
          previous_ensemble.weighted_subnetworks[-1].subnetwork
          .persisted_tensors[_NUM_LAYERS_KEY])
    return [
        self._dnn_builder_fn(num_layers=num_layers),
        self._dnn_builder_fn(num_layers=num_layers + 1),
    ]
