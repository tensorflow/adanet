# Lint as: python3
"""Tests for AutoEnsembleTPUEstimator.

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

import itertools
import os
import shutil
import sys

from absl import flags
from absl.testing import parameterized
from adanet import tf_compat
from adanet.autoensemble.estimator import AutoEnsembleTPUEstimator
import numpy as np
import tensorflow as tf

# pylint: disable=g-import-not-at-top
# pylint: disable=g-direct-tensorflow-import
try:
  from tensorflow_estimator.contrib.estimator.python.estimator import head as head_lib
except (AttributeError, ImportError):
  head_lib = None
from tensorflow_estimator.python.estimator.canned import dnn
# pylint: enable=g-direct-tensorflow-import
# pylint: enable=g-import-not-at-top


class _DNNTPUEstimator(tf.compat.v1.estimator.tpu.TPUEstimator):

  def __init__(self,
               head,
               hidden_units,
               feature_columns,
               optimizer,
               use_tpu,
               embedding_config_spec=None):
    config = tf.compat.v1.estimator.tpu.RunConfig(
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            per_host_input_for_training=tf.compat.v1.estimator.tpu
            .InputPipelineConfig.PER_HOST_V2))

    def model_fn(features, labels, mode=None, params=None, config=None):
      del params  # Unused.

      return dnn._dnn_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          hidden_units=hidden_units,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          config=config,
          use_tpu=use_tpu)

    super(_DNNTPUEstimator, self).__init__(
        model_fn=model_fn,
        config=config,
        train_batch_size=64,
        use_tpu=use_tpu,
        embedding_config_spec=embedding_config_spec)


class AutoEnsembleTPUEstimatorTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(AutoEnsembleTPUEstimatorTest, self).setUp()
    # Setup and cleanup test directory.
    # Flags are not automatically parsed at this point.
    flags.FLAGS(sys.argv)
    self.test_subdirectory = os.path.join(flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.makedirs(self.test_subdirectory)

  def tearDown(self):
    super(AutoEnsembleTPUEstimatorTest, self).tearDown()
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  @parameterized.named_parameters(
      {
          "testcase_name": "not_use_tpu",
          "use_tpu": False,
      },
  )
  # TODO: Support V2 head and optimizer in AdaNet TPU.
  @tf_compat.skip_for_tf2
  def test_auto_ensemble_estimator_lifecycle(self, use_tpu):
    head = head_lib.regression_head()
    feature_columns = [tf.feature_column.numeric_column("xor", shape=2)]

    def optimizer_fn():
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=.01)
      if use_tpu:
        optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)
      return optimizer

    candidate_pool = {
        "tpu_estimator_dnn":
            _DNNTPUEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=optimizer_fn,
                hidden_units=[3],
                use_tpu=True),
        "tpu_estimator_wider_dnn":
            _DNNTPUEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=optimizer_fn,
                hidden_units=[6],
                use_tpu=True),
        "estimator_dnn":
            tf.compat.v1.estimator.DNNEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=optimizer_fn,
                hidden_units=[3]),
        "estimator_linear":
            tf.compat.v1.estimator.LinearEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=optimizer_fn),
    }

    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        master="", tf_random_seed=42)
    estimator = AutoEnsembleTPUEstimator(
        head=head,
        candidate_pool=candidate_pool,
        max_iteration_steps=10,
        model_dir=self.test_subdirectory,
        config=run_config,
        use_tpu=use_tpu,
        # TODO: Support export_to_tpu for canned models.
        export_to_tpu=False,
        train_batch_size=4,
        force_grow=True)

    features = {"xor": [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]}
    labels = [[0.], [1.], [1.], [0.]]

    def train_input_fn(params):
      del params  # Unused.

      input_features = {}
      for key, feature in features.items():
        input_features[key] = tf.constant(feature, name=key)
      input_labels = tf.constant(labels, name="labels")
      return input_features, input_labels

    def test_input_fn(params):
      del params  # Unused.
      return tf.compat.v1.data.Dataset.from_tensor_slices(
          [features["xor"]]).map(lambda f: {"xor": f})

    # Train for three iterations.
    estimator.train(input_fn=train_input_fn, max_steps=30)

    # Evaluate.
    eval_results = estimator.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAllClose(30, eval_results["global_step"])
    self.assertAllClose(0.315863, eval_results["loss"], atol=.3)

    # Predict.
    predictions = estimator.predict(input_fn=test_input_fn)
    # We need to iterate over all the predictions before moving on, otherwise
    # the TPU will not be shut down.
    for prediction in predictions:
      self.assertIsNotNone(prediction["predictions"])

    # Export SavedModel.
    def serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf.compat.v1.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      for key in features:
        features[key] = tf.constant([[0., 0.], [0., 0.]])
      return tf.estimator.export.ServingInputReceiver(
          features=features, receiver_tensors=serialized_example)

    export_dir_base = os.path.join(self.test_subdirectory, "export")
    export_saved_model_fn = getattr(estimator, "export_saved_model", None)
    if not callable(export_saved_model_fn):
      export_saved_model_fn = estimator.export_savedmodel
    export_saved_model_fn(
        export_dir_base=export_dir_base,
        serving_input_receiver_fn=serving_input_fn)



if __name__ == "__main__":
  tf.test.main()
