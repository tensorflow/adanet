"""Test AdaNet estimator single graph implementation for TF 2.

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

from absl import logging
from adanet import tf_compat
from adanet.core import testing_utils as tu
from adanet.core.estimator import Estimator
from adanet.core.report_materializer import ReportMaterializer
from adanet.subnetwork import Builder
from adanet.subnetwork import SimpleGenerator
from adanet.subnetwork import Subnetwork
import tensorflow as tf

from tensorflow_estimator.python.estimator.head import regression_head

logging.set_verbosity(logging.INFO)

XOR_FEATURES = [[1., 0.], [0., 0], [0., 1.], [1., 1.]]
XOR_LABELS = [[1.], [0.], [1.], [0.]]


class _SimpleBuilder(Builder):
  """A simple subnetwork builder that takes feature_columns."""

  def __init__(self, name, seed=42):
    self._name = name
    self._seed = seed

  @property
  def name(self):
    return self._name

  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    seed = self._seed
    if previous_ensemble:
      # Increment seed so different iterations don't learn the exact same thing.
      seed += 1

    with tf_compat.v1.variable_scope("simple"):
      input_layer = tf_compat.v1.feature_column.input_layer(
          features=features,
          feature_columns=tf.feature_column.numeric_column("x", 2))
      last_layer = input_layer

    with tf_compat.v1.variable_scope("logits"):
      logits = tf_compat.v1.layers.dense(
          last_layer,
          logits_dimension,
          kernel_initializer=tf_compat.v1.glorot_uniform_initializer(seed=seed))

    summary.scalar("scalar", 3)
    batch_size = features["x"].get_shape().as_list()[0]
    summary.image("image", tf.ones([batch_size, 3, 3, 1]))
    with tf_compat.v1.variable_scope("nested"):
      summary.scalar("scalar", 5)

    return Subnetwork(
        last_layer=last_layer,
        logits=logits,
        complexity=1,
        persisted_tensors={},
    )

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    optimizer = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=.001)
    return optimizer.minimize(loss, var_list=var_list)


class EstimatorSummaryWriterTest(tu.AdanetTestCase):
  """Test that Tensorboard summaries get written correctly."""

  @tf_compat.skip_for_tf1
  def test_summaries(self):
    """Tests that summaries are written to candidate directory."""

    run_config = tf.estimator.RunConfig(
        tf_random_seed=42,
        log_step_count_steps=2,
        save_summary_steps=2,
        model_dir=self.test_subdirectory)
    subnetwork_generator = SimpleGenerator([_SimpleBuilder("dnn")])
    report_materializer = ReportMaterializer(
        input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=1)
    estimator = Estimator(
        head=regression_head.RegressionHead(
            loss_reduction=tf_compat.SUM_OVER_BATCH_SIZE),
        subnetwork_generator=subnetwork_generator,
        report_materializer=report_materializer,
        max_iteration_steps=10,
        config=run_config)
    train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])
    estimator.train(input_fn=train_input_fn, max_steps=3)

    ensemble_loss = 1.52950
    self.assertAlmostEqual(
        ensemble_loss,
        tu.check_eventfile_for_keyword("loss", self.test_subdirectory),
        places=3)
    self.assertIsNotNone(
        tu.check_eventfile_for_keyword("global_step/sec",
                                       self.test_subdirectory))
    self.assertEqual(
        0.,
        tu.check_eventfile_for_keyword("iteration/adanet/iteration",
                                       self.test_subdirectory))

    subnetwork_subdir = os.path.join(self.test_subdirectory,
                                     "subnetwork/t0_dnn")
    self.assertAlmostEqual(
        3.,
        tu.check_eventfile_for_keyword("scalar", subnetwork_subdir),
        places=3)
    self.assertEqual((3, 3, 1),
                     tu.check_eventfile_for_keyword("image", subnetwork_subdir))
    self.assertAlmostEqual(
        5.,
        tu.check_eventfile_for_keyword("nested/scalar", subnetwork_subdir),
        places=3)

    ensemble_subdir = os.path.join(
        self.test_subdirectory, "ensemble/t0_dnn_grow_complexity_regularized")
    self.assertAlmostEqual(
        ensemble_loss,
        tu.check_eventfile_for_keyword(
            "adanet_loss/adanet/adanet_weighted_ensemble", ensemble_subdir),
        places=1)
    self.assertAlmostEqual(
        0.,
        tu.check_eventfile_for_keyword(
            "complexity_regularization/adanet/adanet_weighted_ensemble",
            ensemble_subdir),
        places=3)
    self.assertAlmostEqual(
        1.,
        tu.check_eventfile_for_keyword(
            "mixture_weight_norms/adanet/"
            "adanet_weighted_ensemble/subnetwork_0", ensemble_subdir),
        places=3)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
