"""Tests for AdaNet AutoEnsembleEstimator.

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

import os
import shutil

from absl.testing import parameterized
from adanet.autoensemble.estimator import AutoEnsembleEstimator
import tensorflow as tf

from tensorflow.python.estimator.export import export


class AutoEnsembleEstimatorTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    # Setup and cleanup test directory.
    self.test_subdirectory = os.path.join(tf.flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.mkdir(self.test_subdirectory)

  def tearDown(self):
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  def test_auto_ensemble_estimator_lifecycle(self):

    features = {"input_1": [[1., 0.]]}
    labels = [[1.]]

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    head = tf.contrib.estimator.binary_classification_head(
        loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=3.)
    feature_columns = [tf.feature_column.numeric_column("input_1", shape=[2])]

    def train_input_fn():
      input_features = {}
      for key, feature in features.items():
        input_features[key] = tf.constant(feature, name=key)
      input_labels = tf.constant(labels, name="labels")
      return input_features, input_labels

    def test_input_fn():
      input_features = tf.data.Dataset.from_tensors([
          tf.constant(features["input_1"])
      ]).make_one_shot_iterator().get_next()
      return {"input_1": input_features}, None

    estimator = AutoEnsembleEstimator(
        head=head,
        candidate_pool=[
            tf.contrib.estimator.LinearEstimator(
                head=head, feature_columns=feature_columns,
                optimizer=optimizer),
            tf.contrib.estimator.DNNEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=optimizer,
                hidden_units=[3]),
        ],
        max_iteration_steps=4,
        force_grow=True,
        model_dir=self.test_subdirectory,
        config=run_config)

    # Train.
    estimator.train(input_fn=train_input_fn, max_steps=12)

    # Evaluate.
    eval_results = estimator.evaluate(input_fn=train_input_fn, steps=3)
    self.assertAlmostEqual(0.0039954609, eval_results["loss"], places=3)
    self.assertAlmostEqual(1., eval_results["accuracy"])

    want_subnetworks = [
        "t0_DNNEstimator1", "t1_DNNEstimator1",
    ]
    for subnetwork in want_subnetworks:
      self.assertIn(subnetwork,
                    str(eval_results["architecture/adanet/ensembles"]))

    # Predict.
    predictions = estimator.predict(input_fn=test_input_fn)
    for prediction in predictions:
      self.assertIsNotNone(prediction["classes"])
      self.assertIsNotNone(prediction["probabilities"])

    # Export SavedModel.
    def serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      for key, value in features.items():
        features[key] = tf.constant(value)
      return export.SupervisedInputReceiver(
          features=features,
          labels=tf.constant(labels),
          receiver_tensors=serialized_example)

    export_dir_base = os.path.join(self.test_subdirectory, "export")
    tf.contrib.estimator.export_saved_model_for_mode(
        estimator,
        export_dir_base=export_dir_base,
        input_receiver_fn=serving_input_fn,
        mode=tf.estimator.ModeKeys.EVAL)


if __name__ == "__main__":
  tf.test.main()
