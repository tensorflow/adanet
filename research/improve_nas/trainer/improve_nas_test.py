"""Tests for improve_nas.

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

import os
import shutil

from absl import flags
from absl.testing import parameterized
import adanet
from adanet.research.improve_nas.trainer import improve_nas
import numpy as np
import tensorflow as tf


_IMAGE_DIM = 32


class _FakeSummary(object):
  """A fake `Summary`."""

  def scalar(self, name, tensor):
    del name  # Unused
    del tensor  # Unused


def _optimizer(learning_rate):
  return tf.train.GradientDescentOptimizer(learning_rate), learning_rate


def _builder(snapshot=False,
             knowledge_distillation=improve_nas.KnowledgeDistillation.NONE,
             checkpoint_dir=None,
             use_aux_head=False,
             learn_mixture_weights=False,
             model_version="cifar"):
  hparams = tf.contrib.training.HParams(
      clip_gradients=5.,
      stem_multiplier=3.0,
      drop_path_keep_prob=0.6,
      num_cells=3,
      use_aux_head=use_aux_head,
      aux_head_weight=0.4,
      label_smoothing=0.1,
      num_conv_filters=4,
      dense_dropout_keep_prob=1.0,
      filter_scaling_rate=2.0,
      num_reduction_layers=2,
      data_format="NHWC",
      use_bounded_activation=False,
      skip_reduction_layer_input=0,
      initial_learning_rate=.01,
      complexity_decay_rate=0.9,
      weight_decay=.0001,
      knowledge_distillation=knowledge_distillation,
      snapshot=snapshot,
      learn_mixture_weights=learn_mixture_weights,
      mixture_weight_type=adanet.MixtureWeightType.SCALAR,
      model_version=model_version,
      total_training_steps=100)
  return improve_nas.Builder(
      [tf.feature_column.numeric_column(key="x", shape=[32, 32, 3])],
      seed=11,
      optimizer_fn=_optimizer,
      checkpoint_dir=checkpoint_dir,
      hparams=hparams)


def _subnetwork_generator(checkpoint_dir):
  hparams = tf.contrib.training.HParams(
      clip_gradients=5.,
      stem_multiplier=3.0,
      drop_path_keep_prob=0.6,
      num_cells=3,
      use_aux_head=False,
      aux_head_weight=0.4,
      label_smoothing=0.1,
      num_conv_filters=4,
      dense_dropout_keep_prob=1.0,
      filter_scaling_rate=2.0,
      complexity_decay_rate=0.9,
      num_reduction_layers=2,
      data_format="NHWC",
      skip_reduction_layer_input=0,
      initial_learning_rate=.01,
      use_bounded_activation=False,
      weight_decay=.0001,
      knowledge_distillation=improve_nas.KnowledgeDistillation.NONE,
      snapshot=False,
      learn_mixture_weights=False,
      mixture_weight_type=adanet.MixtureWeightType.SCALAR,
      model_version="cifar",
      total_training_steps=100)
  return improve_nas.Generator(
      [tf.feature_column.numeric_column(key="x", shape=[32, 32, 3])],
      seed=11,
      optimizer_fn=_optimizer,
      iteration_steps=3,
      checkpoint_dir=checkpoint_dir,
      hparams=hparams)


class ImproveNasBuilderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ImproveNasBuilderTest, self).setUp()
    # Setup and cleanup test directory.
    self.test_subdirectory = os.path.join(tf.flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.makedirs(self.test_subdirectory)

  def tearDown(self):
    super(ImproveNasBuilderTest, self).tearDown()
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  @parameterized.named_parameters({
      "testcase_name": "two_subnetworks_adaptive_knowledge_distillation_aux",
      "builder_params": [
          {
              "knowledge_distillation":
                  improve_nas.KnowledgeDistillation.ADAPTIVE,
              "use_aux_head": True,
          },
          {
              "knowledge_distillation":
                  improve_nas.KnowledgeDistillation.ADAPTIVE,
              "use_aux_head": True,
          },
      ],
      "want_name": "NasNet_A_1.0_96_adaptive_cifar",
  }, {
      "testcase_name": "two_subnetworks_born_again_knowledge_distillation_w",
      "builder_params": [
          {
              "knowledge_distillation":
                  improve_nas.KnowledgeDistillation.BORN_AGAIN,
              "use_aux_head":
                  True,
              "learn_mixture_weights": True,
          },
          {
              "knowledge_distillation":
                  improve_nas.KnowledgeDistillation.BORN_AGAIN,
              "use_aux_head":
                  True,
              "learn_mixture_weights": True,
          },
      ],
      "want_name": "NasNet_A_1.0_96_born_again_cifar",
  })
  def test_build_subnetwork(self, builder_params, want_name):
    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      data = np.concatenate([
          np.ones((1, _IMAGE_DIM, _IMAGE_DIM, 1)), 2. * np.ones(
              (1, _IMAGE_DIM, _IMAGE_DIM, 1))
      ])
      features = {"x": tf.constant(data)}
      labels = tf.constant([0, 1])
      training = True
      mode = tf.estimator.ModeKeys.TRAIN
      head = tf.contrib.estimator.binary_classification_head(
          loss_reduction=tf.losses.Reduction.SUM)
      ensemble = None
      name = None
      subnetwork = None
      builders = []
      for builder_param in builder_params:
        builders.append(
            _builder(checkpoint_dir=self.test_subdirectory, **builder_param))
      for idx, builder in enumerate(builders):
        name = builder.name
        # Pass the subnetworks of previous builders to the next builder.
        with tf.variable_scope("subnetwork_{}".format(idx)):
          subnetwork = builder.build_subnetwork(
              features=features,
              logits_dimension=head.logits_dimension,
              training=training,
              iteration_step=tf.train.get_or_create_global_step(),
              summary=_FakeSummary(),
              previous_ensemble=ensemble)
          logits = subnetwork.logits
          weighted_subnetworks = []
          if ensemble:
            logits += ensemble.logits
            weighted_subnetworks = ensemble.weighted_subnetworks
          ensemble = adanet.Ensemble(
              weighted_subnetworks=weighted_subnetworks + [
                  adanet.WeightedSubnetwork(
                      name=None,
                      logits=logits,
                      weight=None,
                      subnetwork=subnetwork)
              ],
              logits=logits,
              bias=0.)

      estimator_spec = head.create_estimator_spec(
          features=features,
          labels=labels,
          mode=mode,
          train_op_fn=lambda loss: tf.no_op(),
          logits=ensemble.logits)
      sess.run(tf.global_variables_initializer())
      train_op = builders[-1].build_subnetwork_train_op(
          subnetwork,
          estimator_spec.loss,
          var_list=None,
          labels=labels,
          iteration_step=tf.train.get_or_create_global_step(),
          summary=_FakeSummary(),
          previous_ensemble=ensemble)
      for _ in range(10):
        sess.run(train_op)
      self.assertEqual(want_name, name)
      self.assertGreater(sess.run(estimator_spec.loss), 0.0)


class QuetzalGeneratorTest(tf.test.TestCase):

  def test_candidate_generation(self):
    self.test_subdirectory = os.path.join(flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.mkdir(self.test_subdirectory)

    subnetwork_generator = _subnetwork_generator(self.test_subdirectory)
    subnetwork_builders = subnetwork_generator.generate_candidates(
        previous_ensemble=None,
        # The following arguments are unused by
        # quetzal.Generator.
        iteration_number=0,
        previous_ensemble_reports=[],
        all_reports=[])
    self.assertEqual(1, len(subnetwork_builders))


if __name__ == "__main__":
  tf.test.main()
