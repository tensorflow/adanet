"""Tests for NASNet-A search space.

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
from adanet.examples import nasnet
import tensorflow as tf


class GeneratorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "cifar",
      "model_name": "nasnet_cifar",
      "image_shape": [32, 32, 3],
      "want_names": ["nasnet_cifar_NF_32_NC_6"]
  }, {
      "testcase_name": "cifar_no_clip_gradients",
      "model_name": "nasnet_cifar",
      "image_shape": [32, 32, 3],
      "clip_gradients": 0.,
      "want_names": ["nasnet_cifar_NF_32_NC_6"]
  }, {
      "testcase_name": "cifar_no_aux_head",
      "model_name": "nasnet_cifar",
      "image_shape": [32, 32, 3],
      "use_aux_head": 0,
      "want_names": ["nasnet_cifar_NF_32_NC_6"]
  }, {
      "testcase_name": "mobile",
      "model_name": "nasnet_mobile",
      "image_shape": [224, 224, 3],
      "want_names": ["nasnet_mobile_NF_44_NC_4"]
  }, {
      "testcase_name": "large",
      "model_name": "nasnet_large",
      "image_shape": [224, 224, 3],
      "want_names": ["nasnet_large_NF_168_NC_6"]
  })
  def test_generate_candidates(self,
                               model_name,
                               image_shape,
                               want_names,
                               clip_gradients=5.,
                               use_aux_head=1):
    optimizer_fn = tf.train.GradientDescentOptimizer
    config = tf.contrib.training.HParams(use_aux_head=use_aux_head)
    generator = nasnet.Generator(
        optimizer_fn=optimizer_fn,
        initial_learning_rate=.1,
        config=config,
        model_name=model_name,
        clip_gradients=clip_gradients)
    with tf.Graph().as_default():
      iteration_step = tf.train.create_global_step()
      features = {"x": tf.random_normal(shape=[2] + image_shape)}
      labels = tf.constant([[0], [1]])
      names = []
      for builder in generator.generate_candidates(
          previous_ensemble=None,
          iteration_number=0,
          previous_ensemble_reports=[],
          all_reports=[]):
        names.append(builder.name)

        # 1. Build subnetwork graph.
        subnetwork = builder.build_subnetwork(
            features,
            logits_dimension=2,
            training=True,
            iteration_step=iteration_step,
            summary=tf.summary,
            previous_ensemble=None)

        # 2. Build subnetwork train ops.
        subnetwork_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=subnetwork.logits, labels=labels))
        subnetwork_train_op = builder.build_subnetwork_train_op(
            subnetwork,
            subnetwork_loss,
            var_list=None,
            labels=labels,
            iteration_step=iteration_step,
            summary=tf.summary,
            previous_ensemble=None)
        self.assertIsNotNone(subnetwork_train_op)

        # 3. Build mixture weight train ops.

        # Stop gradients since mixture weights should have not propagate
        # beyond top layer.
        subnetwork_logits = tf.stop_gradient(subnetwork.logits)

        # Mixture weight will initialize to a one-valued scalar.
        mixture_weight_logits = tf.layers.dense(
            subnetwork_logits,
            units=1,
            use_bias=False,
            kernel_initializer=tf.ones_initializer())
        mixture_weight_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=mixture_weight_logits, labels=labels))
        mixture_weight_train_op = builder.build_mixture_weights_train_op(
            mixture_weight_loss,
            var_list=None,
            labels=labels,
            logits=mixture_weight_logits,
            iteration_step=iteration_step,
            summary=tf.summary)
        self.assertIsNotNone(mixture_weight_train_op)
    self.assertEqual(want_names, names)

  def test_constructor_error_unsupported_model_name(self):
    with self.assertRaises(ValueError):
      optimizer_fn = tf.train.GradientDescentOptimizer
      nasnet.Generator(
          optimizer_fn=optimizer_fn,
          initial_learning_rate=.1,
          config=tf.contrib.training.HParams(),
          model_name="unsupported_model_name")

  @parameterized.named_parameters({
      "testcase_name": "single_logits_dimension",
      "logits_dimension": 1,
  }, {
      "testcase_name": "multiple_features",
      "features": {"x": [1], "y": [2]},
  })
  def test_build_subnetwork_errors(self, logits_dimension=2, features=None):
    if not features:
      features = {"x": tf.random_normal(shape=[2, 32, 32, 3])}
    optimizer_fn = tf.train.GradientDescentOptimizer
    generator = nasnet.Generator(
        optimizer_fn=optimizer_fn,
        initial_learning_rate=.1,
        config=tf.contrib.training.HParams(),
        model_name="nasnet_cifar")
    builder = generator.generate_candidates(
        previous_ensemble=None,
        iteration_number=0,
        previous_ensemble_reports=[],
        all_reports=[])[0]
    with self.assertRaises(ValueError):
      iteration_step = tf.train.create_global_step()
      builder.build_subnetwork(
          features,
          logits_dimension=logits_dimension,
          training=True,
          iteration_step=iteration_step,
          summary=tf.summary,
          previous_ensemble=None)


if __name__ == "__main__":
  tf.test.main()
