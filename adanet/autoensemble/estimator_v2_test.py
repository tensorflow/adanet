"""Tests for AdaNet AutoEnsembleEstimator in TF 2.

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
import sys

from absl import flags
from absl.testing import parameterized
from adanet import tf_compat
from adanet.autoensemble.estimator import AutoEnsembleEstimator
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.estimator.export import export
from tensorflow_estimator.python.estimator.head import regression_head
# pylint: enable=g-direct-tensorflow-import


class AutoEnsembleEstimatorV2Test(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(AutoEnsembleEstimatorV2Test, self).setUp()
    # Setup and cleanup test directory.
    # Flags are not automatically parsed at this point.
    flags.FLAGS(sys.argv)
    self.test_subdirectory = os.path.join(flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.makedirs(self.test_subdirectory)

  def tearDown(self):
    super(AutoEnsembleEstimatorV2Test, self).tearDown()
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      {
          "testcase_name":
              "candidate_pool_lambda",
          "candidate_pool":
              lambda head, feature_columns, optimizer: lambda config: {
                  "dnn":
                      tf.estimator.DNNEstimator(
                          head=head,
                          feature_columns=feature_columns,
                          optimizer=optimizer,
                          hidden_units=[3],
                          config=config),
                  "linear":
                      tf.estimator.LinearEstimator(
                          head=head,
                          feature_columns=feature_columns,
                          optimizer=optimizer,
                          config=config),
              },
          "want_loss":
              .209,
      },)
  # pylint: enable=g-long-lambda
  @tf_compat.skip_for_tf1
  def test_auto_ensemble_estimator_lifecycle(self,
                                             candidate_pool,
                                             want_loss,
                                             max_train_steps=30):
    features = {"input_1": [[1., 0.]]}
    labels = [[1.]]

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    head = regression_head.RegressionHead()

    # Always create optimizers in a lambda to prevent error like:
    # `RuntimeError: Cannot set `iterations` to a new Variable after the
    # Optimizer weights have been created`
    optimizer = lambda: tf.keras.optimizers.SGD(lr=.01)
    feature_columns = [tf.feature_column.numeric_column("input_1", shape=[2])]

    def train_input_fn():
      input_features = {}
      for key, feature in features.items():
        input_features[key] = tf.constant(feature, name=key)
      input_labels = tf.constant(labels, name="labels")
      return input_features, input_labels

    def test_input_fn():
      dataset = tf.data.Dataset.from_tensors([tf.constant(features["input_1"])])
      input_features = tf.compat.v1.data.make_one_shot_iterator(
          dataset).get_next()
      return {"input_1": input_features}, None

    estimator = AutoEnsembleEstimator(
        head=head,
        candidate_pool=candidate_pool(head, feature_columns, optimizer),
        max_iteration_steps=10,
        force_grow=True,
        model_dir=self.test_subdirectory,
        config=run_config)

    # Train for three iterations.
    estimator.train(input_fn=train_input_fn, max_steps=max_train_steps)

    # Evaluate.
    eval_results = estimator.evaluate(input_fn=train_input_fn, steps=1)

    self.assertAllClose(max_train_steps, eval_results["global_step"])
    self.assertAllClose(want_loss, eval_results["loss"], atol=.3)

    # Predict.
    predictions = estimator.predict(input_fn=test_input_fn)
    for prediction in predictions:
      self.assertIsNotNone(prediction["predictions"])

    # Export SavedModel.
    def serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf.compat.v1.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      for key, value in features.items():
        features[key] = tf.constant(value)
      return export.SupervisedInputReceiver(
          features=features,
          labels=tf.constant(labels),
          receiver_tensors=serialized_example)

    export_dir_base = os.path.join(self.test_subdirectory, "export")
    estimator.export_saved_model(
        export_dir_base=export_dir_base,
        serving_input_receiver_fn=serving_input_fn)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
