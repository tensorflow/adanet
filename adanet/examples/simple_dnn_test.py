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

from absl.testing import parameterized
import adanet
from adanet.examples import simple_dnn
import tensorflow.compat.v2 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
# pylint: enable=g-direct-tensorflow-import


class _FakeEnsemble(object):
  """A fake ensemble of one subnetwork."""

  def __init__(self, num_layers):
    shared_tensors = {"num_layers": num_layers}
    self._weighted_subnetworks = [
        adanet.WeightedSubnetwork(
            name=None,
            iteration_number=None,
            weight=None,
            logits=None,
            subnetwork=adanet.Subnetwork(
                last_layer=[1],
                logits=[1],
                complexity=1,
                shared=shared_tensors))
    ]

  @property
  def weighted_subnetworks(self):
    return self._weighted_subnetworks


class GeneratorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "defaults",
      "want_names": ["linear", "1_layer_dnn"],
      "want_subnetwork_losses": [.871, .932],
      "want_mixture_weight_losses": [.871, .932],
      "want_complexities": [0., 1.],
  }, {
      "testcase_name": "learn_mixture_weights",
      "learn_mixture_weights": True,
      "want_names": ["linear", "1_layer_dnn"],
      "want_subnetwork_losses": [.871, .932],
      "want_mixture_weight_losses": [.842, .892],
      "want_complexities": [0., 1.],
  }, {
      "testcase_name": "one_initial_num_layers",
      "initial_num_layers": 1,
      "want_names": ["1_layer_dnn", "2_layer_dnn"],
      "want_subnetwork_losses": [.932, .660],
      "want_mixture_weight_losses": [.932, .660],
      "want_complexities": [1., 1.414],
  }, {
      "testcase_name": "previous_ensemble",
      "previous_ensemble": _FakeEnsemble(1),
      "want_names": ["1_layer_dnn", "2_layer_dnn"],
      "want_subnetwork_losses": [.932, .660],
      "want_mixture_weight_losses": [.932, .660],
      "want_complexities": [1., 1.414],
  })
  @test_util.run_in_graph_and_eager_modes
  def test_generate_candidates(self,
                               want_names,
                               want_subnetwork_losses,
                               want_mixture_weight_losses,
                               want_complexities,
                               learn_mixture_weights=False,
                               initial_num_layers=0,
                               previous_ensemble=None):
    feature_columns = [tf.feature_column.numeric_column("x")]
    generator = simple_dnn.Generator(
        feature_columns=feature_columns,
        optimizer=tf.compat.v1.train.GradientDescentOptimizer(.1),
        layer_size=3,
        initial_num_layers=initial_num_layers,
        learn_mixture_weights=learn_mixture_weights,
        seed=42)
    with context.graph_mode(), tf.Graph().as_default() as g:
      iteration_step = tf.compat.v1.train.create_global_step()
      features = {"x": [[1.], [2.]]}
      labels = tf.constant([[0.], [1.]])
      names = []
      subnetwork_losses = []
      mixture_weight_losses = []
      complexities = []
      for builder in generator.generate_candidates(
          previous_ensemble,
          # The following arguments are not used by
          # simple_dnn.BuilderGenerator's generate_candidates.
          iteration_number=0,
          previous_ensemble_reports=[],
          all_reports=[]):
        names.append(builder.name)

        # 1. Build subnetwork graph.
        subnetwork = builder.build_subnetwork(
            features,
            logits_dimension=1,
            training=True,
            iteration_step=iteration_step,
            summary=tf.summary,
            previous_ensemble=previous_ensemble)

        # 2. Build subnetwork train ops.
        subnetwork_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=subnetwork.logits, labels=labels))
        subnetwork_train_op = builder.build_subnetwork_train_op(
            subnetwork,
            subnetwork_loss,
            var_list=None,
            labels=labels,
            iteration_step=iteration_step,
            summary=tf.summary,
            previous_ensemble=None)

        # 3. Build mixture weight train ops.

        # Stop gradients since mixture weights should have not propagate
        # beyond top layer.
        subnetwork_logits = tf.stop_gradient(subnetwork.logits)

        # Mixture weight will initialize to a one-valued scalar.
        mixture_weight_logits = tf.compat.v1.layers.dense(
            subnetwork_logits,
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
          sess.run(tf.compat.v1.global_variables_initializer())
          sess.run(subnetwork_train_op)
          sess.run(mixture_weight_train_op)
          subnetwork_losses.append(sess.run(subnetwork_loss))
          mixture_weight_losses.append(sess.run(mixture_weight_loss))
          complexities.append(sess.run(subnetwork.complexity))

    self.assertEqual(want_names, names)
    self.assertAllClose(want_subnetwork_losses, subnetwork_losses, atol=1e-3)
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
          optimizer=tf.compat.v1.train.GradientDescentOptimizer(.1),
          layer_size=layer_size,
          initial_num_layers=initial_num_layers)


if __name__ == "__main__":
  tf.test.main()
