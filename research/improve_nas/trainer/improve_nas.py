# Lint as: python3
"""Defines NASNet subnetwork and subnetwork generators.

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

import functools

import adanet
import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from adanet.research.improve_nas.trainer import nasnet
  from adanet.research.improve_nas.trainer import subnetwork_utils
except ImportError as e:
  from trainer import nasnet
  from trainer import subnetwork_utils
# pylint: enable=g-import-not-at-top


_PREVIOUS_NUM_CELLS = "num_cells"
_PREVIOUS_CONV_FILTERS = "num_conv_filters"


class KnowledgeDistillation(object):
  """Controls what type of knowledge distillation is used.

  In knowledge distillation we want the new subnetwork to learn from the logits
  of previous ensemble or previous subnetwork.

  The following distillations are defined:

  * `ADAPTIVE`: Distill previous ensemble. Inspired by Distilling the Knowledge
        in a Neural Network [Hinton at al., 2015]
        (https://arxiv.org/abs/1503.02531).
  * `BORN_AGAIN`: Distill previous subnetwork. Introduced in Born Again Networks
        [Furlanello et al., 2018](https://arxiv.org/abs/1805.04770).
  * `NONE`: Do not use knowledge distillation.
  """

  ADAPTIVE = "adaptive"
  BORN_AGAIN = "born_again"
  NONE = "none"


class Builder(adanet.subnetwork.Builder):
  """Builds a NASNet subnetwork for AdaNet."""

  def __init__(self, feature_columns, optimizer_fn, checkpoint_dir, hparams,
               seed):
    """Initializes a `Builder`.

    Args:
      feature_columns: The input feature columns of the problem.
      optimizer_fn: Function that accepts a float 'learning_rate' argument and
        returns an `Optimizer` instance and learning rate `Tensor` which may
        have a custom learning rate schedule applied.
      checkpoint_dir: Checkpoint directory.
      hparams: A `HParams` instance.
      seed: A Python integer. Used to create random seeds. See
        tf.set_random_seed for behavior.

    Returns:
      An instance of `Subnetwork`.
    """

    self._feature_columns = feature_columns
    self._optimizer_fn = optimizer_fn
    self._checkpoint_dir = checkpoint_dir
    self._hparams = hparams

    self._aux_head_weight = hparams.aux_head_weight
    self._learn_mixture_weights = hparams.learn_mixture_weights
    self._initial_learning_rate = hparams.initial_learning_rate
    self._knowledge_distillation = hparams.knowledge_distillation
    self._label_smoothing = hparams.label_smoothing
    self._model_version = hparams.model_version
    self._weight_decay = hparams.weight_decay
    # `num_cells` and `num_conv_filters` are not directly used here. They are
    # passed inside hparams to build_nasnet function. They are just saved in
    # `shared`.
    self._num_cells = hparams.num_cells
    self._num_conv_filters = hparams.num_conv_filters
    self._seed = seed

  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    """See `adanet.subnetwork.Builder`."""

    # Prepare the input.
    assert len(self._feature_columns) == 1, "Got feature columns: {}".format(
        self._feature_columns)
    images = tf.to_float(features[self._feature_columns[0].name])
    self._name_scope = tf.get_default_graph().get_name_scope()

    seed = self._seed
    if seed is not None and previous_ensemble:
      # Deterministically change the seed for different iterations so that
      # subnetworks are not correlated.
      seed += len(previous_ensemble.weighted_subnetworks)

    arg_scope = nasnet.nasnet_cifar_arg_scope(weight_decay=self._weight_decay)

    with tf.contrib.slim.arg_scope(arg_scope):
      build_fn = nasnet.build_nasnet_cifar
      logits, end_points = build_fn(
          images,
          num_classes=logits_dimension,
          is_training=training,
          config=self._hparams)
    last_layer = end_points["global_pool"]

    subnetwork_shared_data = {
        _PREVIOUS_NUM_CELLS: tf.constant(self._num_cells),
        _PREVIOUS_CONV_FILTERS: tf.constant(self._num_conv_filters)
    }

    return adanet.Subnetwork(
        last_layer=last_layer,
        logits=logits,
        complexity=1,
        shared=subnetwork_shared_data)

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    """See `adanet.subnetwork.Builder`."""

    del loss  # Loss for training is defined below.

    # The AdaNet Estimator is responsible for incrementing the global step.
    optimizer, learning_rate = self._optimizer_fn(
        learning_rate=self._initial_learning_rate)
    with tf.name_scope(""):
      summary.scalar("learning_rate/adanet/subnetwork", learning_rate)

    onehot_labels = tf.one_hot(
        tf.reshape(labels, [-1]), subnetwork.logits.shape[-1], dtype=tf.int32)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=subnetwork.logits,
        weights=1.0,
        label_smoothing=self._label_smoothing)

    # Add knowledge ditillation loss.
    if previous_ensemble:
      if self._knowledge_distillation == KnowledgeDistillation.ADAPTIVE:
        loss += tf.losses.softmax_cross_entropy(
            onehot_labels=tf.nn.softmax(previous_ensemble.logits),
            logits=subnetwork.logits,
            weights=1.0,
            scope="loss_adaptive_kd")

      if self._knowledge_distillation == KnowledgeDistillation.BORN_AGAIN:
        loss += tf.losses.softmax_cross_entropy(
            onehot_labels=tf.nn.softmax(
                previous_ensemble.weighted_subnetworks[-1].logits),
            logits=subnetwork.logits,
            weights=1.0,
            scope="loss_born_again_kd")

    # Add weight decay.
    loss += tf.losses.get_regularization_loss(scope=self._name_scope)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      if self._hparams.clip_gradients > 0:
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(
            optimizer, self._hparams.clip_gradients)
      return optimizer.minimize(loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    """See `adanet.subnetwork.Builder`."""

    if not self._learn_mixture_weights:
      return tf.no_op("mixture_weights_train_op")

    # The AdaNet Estimator is responsible for incrementing the global step.
    optimizer, learning_rate = self._optimizer_fn(
        learning_rate=self._initial_learning_rate)
    summary.scalar("learning_rate/adanet/mixture_weights", learning_rate)
    return optimizer.minimize(loss=loss, var_list=var_list)

  @property
  def name(self):
    """Returns this subnetwork's name."""
    name = "NasNet_A_{}_{}".format(self._hparams.num_cells / 3,
                                   self._hparams.num_conv_filters * 24)
    if self._knowledge_distillation != KnowledgeDistillation.NONE:
      name += "_" + self._knowledge_distillation
    name += "_" + self._model_version
    return name


class Generator(adanet.subnetwork.Generator):
  """Generates a list of Builders."""

  def __init__(self,
               feature_columns,
               optimizer_fn,
               iteration_steps,
               checkpoint_dir,
               hparams,
               seed=None):
    """Initializes a `Generator`.

    Args:
      feature_columns: The input feature columns of the problem.
      optimizer_fn: Function that accepts a float 'learning_rate' argument and
        returns an `Optimizer` instance and learning rate `Tensor` which may
        have a custom learning rate schedule applied.
      iteration_steps: The number of train steps in per iteration. Required for
        ScheduleDropPath algorithm.
      checkpoint_dir: Checkpoint directory.
      hparams: Hyper-parameters.
      seed: A Python integer. Used to create random seeds. See
        tf.set_random_seed for behavior.

    Returns:
      An instance of `Generator`.

    Raises:
      ValueError: If num_cells is not divisible by 3.
    """

    if hparams.num_cells % 3 != 0:
      raise ValueError("num_cells must be a multiple of 3.")

    self._builder_fn = functools.partial(
        Builder,
        feature_columns=feature_columns,
        optimizer_fn=optimizer_fn,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        hparams=hparams)

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    """See `adanet.subnetwork.Generator`."""

    return [self._builder_fn()]


class DynamicGenerator(adanet.subnetwork.Generator):
  """Generates a list of `Builders`."""

  def __init__(self,
               feature_columns,
               optimizer_fn,
               iteration_steps,
               checkpoint_dir,
               hparams,
               seed=None):
    """Generator that gradually grows the architecture.

    In each iteration, we generate one deeper candidate and one wider candidate.

    Args:
      feature_columns: The input feature columns of the problem.
      optimizer_fn: Function that accepts a float 'learning_rate' argument and
        returns an `Optimizer` instance and learning rate `Tensor` which may
        have a custom learning rate schedule applied.
      iteration_steps: The number of train steps in per iteration. Required for
        ScheduleDropPath algorithm.
      checkpoint_dir: Checkpoint directory.
      hparams: Hyper-parameters.
      seed: A Python integer. Used to create random seeds. See
        tf.set_random_seed for behavior.

    Returns:
      An instance of `Generator`.

    Raises:
      ValueError: If num_cells is not divisible by 3.
    """

    if hparams.num_cells % 3 != 0:
      raise ValueError("num_cells must be a multiple of 3.")

    self._hparams = hparams
    self._builder_fn = functools.partial(
        Builder,
        feature_columns=feature_columns,
        optimizer_fn=optimizer_fn,
        checkpoint_dir=checkpoint_dir,
        seed=seed)

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    """See `adanet.subnetwork.Generator`."""

    num_cells = self._hparams.num_cells
    num_conv_filters = self._hparams.num_conv_filters
    # Get the architecture of the last subnetwork.
    if previous_ensemble:
      num_cells = int(
          subnetwork_utils.get_persisted_value_from_ensemble(
              previous_ensemble, _PREVIOUS_NUM_CELLS))

      num_conv_filters = int(
          subnetwork_utils.get_persisted_value_from_ensemble(
              previous_ensemble, _PREVIOUS_CONV_FILTERS))

    candidates = [
        self._builder_fn(
            hparams=subnetwork_utils.copy_update(
                self._hparams,
                num_cells=num_cells + 3,
                num_conv_filters=num_conv_filters)),
        self._builder_fn(
            hparams=subnetwork_utils.copy_update(
                self._hparams,
                num_cells=num_cells,
                num_conv_filters=num_conv_filters + 10)),
    ]
    return candidates
