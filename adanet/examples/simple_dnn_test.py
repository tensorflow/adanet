"""Tests for a simple dense neural network search space.

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

from absl.testing import parameterized
import adanet
from adanet.examples import simple_dnn
import tensorflow as tf


class _FakeEnsemble(object):
  """A fake ensemble of one base learner."""

  def __init__(self, num_layers):
    persisted_tensors = {"num_layers": tf.constant(num_layers)}
    self._weighted_base_learners = [
        adanet.WeightedBaseLearner(
            weight=None,
            logits=None,
            base_learner=adanet.BaseLearner(
                last_layer=[1],
                logits=[1],
                complexity=1,
                persisted_tensors=persisted_tensors))
    ]

  @property
  def weighted_base_learners(self):
    return self._weighted_base_learners


class GeneratorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "defaults",
      "want_names": ["linear", "1_layer_dnn"],
      "want_base_learner_losses": [.871, .932],
      "want_mixture_weight_losses": [.871, .932],
      "want_complexities": [0., 1.],
  }, {
      "testcase_name": "learn_mixture_weights",
      "learn_mixture_weights": True,
      "want_names": ["linear", "1_layer_dnn"],
      "want_base_learner_losses": [.871, .932],
      "want_mixture_weight_losses": [.842, .892],
      "want_complexities": [0., 1.],
  }, {
      "testcase_name": "one_initial_num_layers",
      "initial_num_layers": 1,
      "want_names": ["1_layer_dnn", "2_layer_dnn"],
      "want_base_learner_losses": [.932, .660],
      "want_mixture_weight_losses": [.932, .660],
      "want_complexities": [1., 1.414],
  }, {
      "testcase_name": "previous_ensemble",
      "previous_ensemble": _FakeEnsemble(1),
      "want_names": ["1_layer_dnn", "2_layer_dnn"],
      "want_base_learner_losses": [.932, .660],
      "want_mixture_weight_losses": [.932, .660],
      "want_complexities": [1., 1.414],
  })
  def test_generate_candidates(self,
                               want_names,
                               want_base_learner_losses,
                               want_mixture_weight_losses,
                               want_complexities,
                               learn_mixture_weights=False,
                               initial_num_layers=0,
                               previous_ensemble=None):
    feature_columns = [tf.feature_column.numeric_column("x")]
    generator = simple_dnn.Generator(
        feature_columns=feature_columns,
        optimizer=tf.train.GradientDescentOptimizer(.1),
        layer_size=3,
        initial_num_layers=initial_num_layers,
        learn_mixture_weights=learn_mixture_weights,
        seed=42)
    with tf.Graph().as_default() as g:
      iteration_step = tf.train.create_global_step()
      features = {"x": [[1.], [2.]]}
      labels = tf.constant([[0.], [1.]])
      names = []
      base_learner_losses = []
      mixture_weight_losses = []
      complexities = []
      for builder in generator.generate_candidates(
          previous_ensemble,
          # The following arguments are not used by
          # simple_dnn.BaseLearnerBuilderGenerator's generate_candidates.
          iteration_number=0,
          previous_ensemble_reports=[],
          all_reports=[]):
        names.append(builder.name)

        # 1. Build base learner graph.
        base_learner = builder.build_base_learner(
            features,
            logits_dimension=1,
            training=True,
            iteration_step=iteration_step,
            summary=tf.summary,
            previous_ensemble=previous_ensemble)

        # 2. Build base learner train ops.
        base_learner_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=base_learner.logits, labels=labels))
        base_learner_train_op = builder.build_base_learner_train_op(
            base_learner,
            base_learner_loss,
            var_list=None,
            labels=labels,
            iteration_step=iteration_step,
            summary=tf.summary,
            previous_ensemble=None)

        # 3. Build mixture weight train ops.

        # Stop gradients since mixture weights should have not propagate
        # beyond top layer.
        base_learner_logits = tf.stop_gradient(base_learner.logits)

        # Mixture weight will initialize to a one-valued scalar.
        mixture_weight_logits = tf.layers.dense(
            base_learner_logits,
            units=1,
            use_bias=False,
            kernel_initializer=tf.ones_initializer())
        mixture_weight_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=mixture_weight_logits, labels=labels))
        mixture_weight_train_op = builder.build_mixture_weights_train_op(
            mixture_weight_loss,
            var_list=None,
            labels=labels,
            logits=mixture_weight_logits,
            iteration_step=iteration_step,
            summary=tf.summary)

        with self.test_session(graph=g) as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(base_learner_train_op)
          sess.run(mixture_weight_train_op)
          base_learner_losses.append(sess.run(base_learner_loss))
          mixture_weight_losses.append(sess.run(mixture_weight_loss))
          complexities.append(sess.run(base_learner.complexity))

    self.assertEqual(want_names, names)
    self.assertAllClose(
        want_base_learner_losses, base_learner_losses, atol=1e-3)
    self.assertAllClose(
        want_mixture_weight_losses, mixture_weight_losses, atol=1e-3)
    self.assertAllClose(want_complexities, complexities, atol=1e-3)

  @parameterized.named_parameters({
      "testcase_name": "empty_feature_columns",
      "feature_columns": [],
  }, {
      "testcase_name": "zero_layer_size",
      "feature_columns": [tf.feature_column.numeric_column("x")],
      "layer_size": 0,
  }, {
      "testcase_name": "negative_initial_num_layers",
      "feature_columns": [tf.feature_column.numeric_column("x")],
      "initial_num_layers": -1,
  })
  def test_constructor_errors(self,
                              feature_columns,
                              layer_size=3,
                              initial_num_layers=0):
    with self.assertRaises(ValueError):
      simple_dnn.Generator(
          feature_columns=feature_columns,
          optimizer=tf.train.GradientDescentOptimizer(.1),
          layer_size=layer_size,
          initial_num_layers=initial_num_layers)


if __name__ == "__main__":
  tf.test.main()
