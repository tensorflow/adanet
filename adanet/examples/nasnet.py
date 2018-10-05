"""A NASNet-A generator for AdaNet.

NASNet-A paper [Zoph et al., 2017]: https://arxiv.org/abs/1707.07012.

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
import tensorflow as tf

from research.slim.nets import nets_factory
from research.slim.nets.nasnet import nasnet


def _default_config_for_model_name(model_name):
  """Returns the default hparams config for each supported model."""

  if model_name == "nasnet_cifar":
    return nasnet.cifar_config()
  if model_name == "nasnet_mobile":
    return nasnet.mobile_imagenet_config()
  if model_name == "nasnet_large":
    return nasnet.large_imagenet_config()
  raise ValueError("Unsupported model name: {}".format(model_name))


class _NASNet(adanet.subnetwork.Builder):
  """Builds a NASNet-A subnetwork for AdaNet."""

  def __init__(self, optimizer_fn, learning_rate_schedule_fn,
               initial_learning_rate, model_name, weight_decay, clip_gradients,
               config):
    """Initializes a `_NASNet` instance.

    Args:
      optimizer_fn: Function that accepts a float 'learning_rate' argument and
        returns an `Optimizer` instance which may have a custom learning rate
        schedule applied.
      learning_rate_schedule_fn: Function that accepts a float 'learning_rate'
        argument and returns a learning rate `Tensor` with schedule applied.
      initial_learning_rate: Float learning rate to use when there are no
        previous_ensemble. When available, the learning rate of the previous
        best subnetwork is used instead.
      model_name: String model name defining the high level topology of the
        NASNet-A. One of `nasnet_cifar`, `nasnet_mobile`, or `nasnet_large`.
      weight_decay: The l2 coefficient for the model weights.
      clip_gradients: Float cut-off for clipping gradients.
      config: Configuration `tf.Hparams` object for the given `model_name`, as
        defined in //models/research/slim/nets/nasnet/nasnet.py. See
        `default_config_for_model_name` for default configuration.

    Returns:
      An instance of `_NASNet`.
    """

    self._optimizer_fn = optimizer_fn
    self._learning_rate_schedule_fn = learning_rate_schedule_fn
    self._initial_learning_rate = initial_learning_rate
    self._weight_decay = weight_decay
    self._model_name = model_name
    self._clip_gradients = clip_gradients
    self._config = config

  def build_subnetwork(self, features, logits_dimension, training,
                       iteration_step, summary, previous_ensemble):
    """See `adanet.subnetwork.Builder`."""

    # Required for scoping weight decay.
    self._name_scope = tf.get_default_graph().get_name_scope()

    if len(features) != 1:
      raise ValueError(
          "Features dict must only contain a single image tensor; got {}".
          format(features))
    if logits_dimension == 1:
      raise ValueError("Only multi-class classification is supported")

    images = tf.to_float(list(features.values())[0])

    nasnet_fn = nets_factory.get_network_fn(
        self._model_name,
        num_classes=logits_dimension,
        weight_decay=self._weight_decay,
        is_training=training)
    logits, end_points = nasnet_fn(
        images, config=self._config, current_step=iteration_step)

    persisted_tensors = {}
    if "AuxLogits" in end_points:
      persisted_tensors["aux_logits"] = end_points["AuxLogits"]
    return adanet.Subnetwork(
        last_layer=logits,
        logits=logits,
        complexity=1,
        persisted_tensors=persisted_tensors)

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    """See `adanet.subnetwork.Builder`."""

    del loss  # Loss for training is defined below.

    learning_rate = self._initial_learning_rate
    if self._learning_rate_schedule_fn:
      learning_rate = self._learning_rate_schedule_fn(
          learning_rate=learning_rate, global_step=iteration_step)

    # The AdaNet Estimator is responsible for incrementing the global step.
    optimizer = self._optimizer_fn(learning_rate=learning_rate)
    with tf.name_scope(""):
      summary.scalar("learning_rate/adanet/subnetwork", learning_rate)

    onehot_labels = tf.one_hot(
        tf.reshape(labels, [-1]), subnetwork.logits.shape[-1], dtype=tf.int32)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=subnetwork.logits,
        weights=1.0,
        label_smoothing=0.1)

    # TODO: Figure out how to handle auxiliary heads in a more
    # flexible way than hard-coding the loss for classification.
    if "aux_logits" in subnetwork.persisted_tensors:
      loss += tf.losses.softmax_cross_entropy(
          onehot_labels=onehot_labels,
          logits=subnetwork.persisted_tensors["aux_logits"],
          weights=0.4,
          label_smoothing=0.1,
          scope="aux_loss")

    # Add weight decay.
    loss += tf.losses.get_regularization_loss(scope=self._name_scope)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      if self._clip_gradients > 0:
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(
            optimizer, self._clip_gradients)
      return optimizer.minimize(loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    """See `adanet.subnetwork.Builder`."""

    # Use default mixture weight initialization.
    return tf.no_op("mixture_weights_train_op")

  @property
  def name(self):
    """See `adanet.subnetwork.Builder`."""

    return "{}_NF_{}_NC_{}".format(
        self._model_name, self._config.num_conv_filters,
        self._config.num_cells // (self._config.num_reduction_layers + 1))


class Generator(adanet.subnetwork.Generator):
  """Generates NASNet-A subnetworks with the configured structure."""

  def __init__(self,
               optimizer_fn,
               initial_learning_rate,
               config,
               model_name="nasnet_cifar",
               learning_rate_schedule_fn=None,
               weight_decay=5e-4,
               clip_gradients=5.):
    """Initializes a `Generator`.

    NOTE: Currently only multi-class classification is supported.

    Args:
      optimizer_fn: Function that accepts a float 'learning_rate' argument and
        returns an `Optimizer` instance which may have a custom learning rate
        schedule applied.
      initial_learning_rate: Float learning rate to use when there are no
        previous_ensemble. When available, the learning rate of the previous
        best subnetwork is used instead.
      config: Configuration `tf.contrib.training.HParams` object for the given
        `model_name`, as defined in models/research/slim/nets/nasnet/nasnet.py.
        A partially defined `HParams` instance will override the default values
        with its defined key-value pairs. An empty `HParams` instance will use
        the default configuration.
      model_name: String model name defining the high level topology of the
        NASNet-A. One of `nasnet_cifar`, `nasnet_mobile`, or `nasnet_large`.
      learning_rate_schedule_fn: Function that accepts a float 'learning_rate'
        argument and returns a learning rate `Tensor` with schedule applied.
      weight_decay: The l2 coefficient for the model weights.
      clip_gradients: Float cut-off for clipping gradients.

    Returns:
      An instance of `Generator`.

    Raises:
      ValueError: If model_name is unsupported.
    """

    default_config = _default_config_for_model_name(model_name)
    default_config.override_from_dict(config.values())
    config = default_config

    tf.logging.info("Using configuration: {}".format(config))

    self._builder_fn = functools.partial(
        _NASNet,
        optimizer_fn=optimizer_fn,
        initial_learning_rate=initial_learning_rate,
        learning_rate_schedule_fn=learning_rate_schedule_fn,
        model_name=model_name,
        weight_decay=weight_decay,
        clip_gradients=clip_gradients,
        config=config)

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    """See `adanet.subnetwork.Generator`."""

    return [self._builder_fn()]
