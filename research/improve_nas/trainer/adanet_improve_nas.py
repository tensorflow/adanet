"""Defines adanet estimator builder.

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

import adanet
import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from adanet.research.improve_nas.trainer import improve_nas
  from adanet.research.improve_nas.trainer import optimizer
except ImportError as e:
  from trainer import improve_nas
  from trainer import optimizer
# pylint: enable=g-import-not-at-top


class GeneratorType(object):
  """Controls what generator is used."""
  DYNAMIC = "dynamic"
  SIMPLE = "simple"


class Builder(object):
  """An AdaNet estimator builder."""

  def estimator(self,
                data_provider,
                run_config,
                hparams,
                train_steps=None,
                seed=None):
    """Returns an AdaNet `Estimator` for train and evaluation.

    Args:
      data_provider: Data `Provider` for dataset to model.
      run_config: `RunConfig` object to configure the runtime settings.
      hparams: `HParams` instance defining custom hyperparameters.
      train_steps: number of train steps.
      seed: An integer seed if determinism is required.

    Returns:
      Returns an `Estimator`.
    """

    max_iteration_steps = int(train_steps / hparams.boosting_iterations)

    optimizer_fn = optimizer.fn_with_name(
        hparams.optimizer,
        learning_rate_schedule=hparams.learning_rate_schedule,
        cosine_decay_steps=max_iteration_steps)
    hparams.add_hparam("total_training_steps", max_iteration_steps)

    if hparams.generator == GeneratorType.SIMPLE:
      subnetwork_generator = improve_nas.Generator(
          feature_columns=data_provider.get_feature_columns(),
          optimizer_fn=optimizer_fn,
          iteration_steps=max_iteration_steps,
          checkpoint_dir=run_config.model_dir,
          hparams=hparams,
          seed=seed)
    elif hparams.generator == GeneratorType.DYNAMIC:
      subnetwork_generator = improve_nas.DynamicGenerator(
          feature_columns=data_provider.get_feature_columns(),
          optimizer_fn=optimizer_fn,
          iteration_steps=max_iteration_steps,
          checkpoint_dir=run_config.model_dir,
          hparams=hparams,
          seed=seed)
    else:
      raise ValueError("Invalid generator: `%s`" % hparams.generator)

    evaluator = adanet.Evaluator(
        input_fn=data_provider.get_input_fn(
            partition="train",
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=hparams.evaluator_batch_size),
        steps=hparams.evaluator_steps)

    return adanet.Estimator(
        head=data_provider.get_head(),
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        adanet_lambda=hparams.adanet_lambda,
        adanet_beta=hparams.adanet_beta,
        mixture_weight_type=hparams.mixture_weight_type,
        force_grow=hparams.force_grow,
        evaluator=evaluator,
        config=run_config,
        model_dir=run_config.model_dir)

  def hparams(self, default_batch_size, hparams_string):
    """Returns hyperparameters, including any flag value overrides.

    In order to allow for automated hyperparameter tuning, model hyperparameters
    are aggregated within a tf.HParams object.  In this case, here are the
    hyperparameters and their descriptions:
    - optimizer: Name of the optimizer to use. See `optimizers.fn_with_name`.
    - learning_rate_schedule: Learning rate schedule string.
    - initial_learning_rate: The initial learning rate to use during training.
    - num_cells: Number of cells in the model. Must be divisible by 3.
    - num_conv_filters: The initial number of convolutional filters. The final
        layer will have 24*num_conv_filters channels.
    - weight_decay: Float amount of weight decay to apply to train loss.
    - use_aux_head: Whether to create an auxiliary head for training. This adds
        some non-determinism to training.
    - knowledge_distillation: Whether subnetworks should learn from the
        logits of the 'previous ensemble'/'previous subnetwork' in addition to
        the labels to distill/transfer/compress the knowledge in a manner
        inspired by Born Again Networks [Furlanello et al., 2018]
        (https://arxiv.org/abs/1805.04770) and Distilling the Knowledge in
        a Neural Network [Hinton at al., 2015]
        (https://arxiv.org/abs/1503.02531).
    - model_version: See `improve_nas.ModelVersion`.
    - adanet_lambda: See `adanet.Estimator`.
    - adanet_beta: See `adanet.Estimator`.
    - generator: Type of generator. `simple` generator is just ensembling,
        `dynamic` generator gradually grows the network.
    - boosting_iterations: The number of boosting iterations to perform. The
      final ensemble will have at most this many subnetworks comprising it.
    - evaluator_batch_size: Batch size for the evaluator to use when comparing
        candidates.
    - evaluator_steps: Number of batches for the evaluator to use when
        comparing candidates.
    - learn_mixture_weights: Whether to learn adanet mixture weights.
    - mixture_weight_type: Type of mxture weights.
    - batch_size: Batch size for training.
    - force_grow: Force AdaNet to add a candidate in each itteration, even if it
        would decreases the performance of the ensemble.
    - label_smoothing: Strength of label smoothing that will be applied (even
        non true labels will have a non zero representation in one hot encoding
        when computing loss).
    - clip_gradients: Clip gradient to this value.
    - aux_head_weight: NASNet cell parameter. Weight of auxiliary loss.
    - stem_multiplier: NASNet cell parameter.
    - drop_path_keep_prob: NASNet cell parameter. Propability for drop_path
        regularization.
    - dense_dropout_keep_prob: NASNet cell parameter. Dropout keep probability.
    - filter_scaling_rate: NASNet cell parameter. Controls growth of number of
        filters.
    - num_reduction_layers: NASNet cell parameter. Number of reduction layers
        that will be added to the architecture.
    - data_format: NASNet cell parameter. Controls whether data is in channels
        last or channels first format.
    - skip_reduction_layer_input: NASNet cell parameter. Whether to skip
        reduction layer.
    - use_bounded_activation: NASNet cell parameter. Whether to use bounded
        activations.

    Args:
      default_batch_size: The default batch_size specified for training.
      hparams_string: If the hparams_string is given, then it will use any
        values specified in hparams to override any individually-set
        hyperparameter. This logic allows tuners to override hyperparameter
        settings to find optimal values.

    Returns:
      The hyperparameters as a tf.HParams object.
    """
    hparams = tf.contrib.training.HParams(
        # Nasnet config hparams (default cifar config)
        num_cells=3,
        num_conv_filters=10,
        aux_head_weight=0.4,
        stem_multiplier=3.0,
        drop_path_keep_prob=0.6,
        use_aux_head=True,
        dense_dropout_keep_prob=1.0,
        filter_scaling_rate=2.0,
        num_reduction_layers=2,
        data_format="NHWC",
        skip_reduction_layer_input=0,
        use_bounded_activation=False,
        # Other hparams
        clip_gradients=5,
        optimizer="momentum",
        learning_rate_schedule="cosine",
        initial_learning_rate=.025,
        weight_decay=5e-4,
        label_smoothing=0.1,
        knowledge_distillation=improve_nas.KnowledgeDistillation.ADAPTIVE,
        model_version="cifar",
        adanet_lambda=0.,
        adanet_beta=0.,
        generator=GeneratorType.SIMPLE,
        boosting_iterations=3,
        force_grow=True,
        evaluator_batch_size=-1,
        evaluator_steps=-1,
        batch_size=default_batch_size,
        learn_mixture_weights=False,
        mixture_weight_type=adanet.MixtureWeightType.SCALAR,
    )
    if hparams_string:
      hparams = hparams.parse(hparams_string)
    if hparams.evaluator_batch_size < 0:
      hparams.evaluator_batch_size = default_batch_size
    if hparams.evaluator_steps < 0:
      hparams.evaluator_steps = None
    return hparams
