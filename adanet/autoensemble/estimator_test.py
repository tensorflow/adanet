"""Tests for AdaNet AutoEnsembleEstimator in TF 1.

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

import contextlib
import os
import shutil
import sys

from absl import flags
from absl import logging
from absl.testing import parameterized
from adanet import tf_compat
from adanet.autoensemble.estimator import _GeneratorFromCandidatePool
from adanet.autoensemble.estimator import AutoEnsembleEstimator
from adanet.autoensemble.estimator import AutoEnsembleSubestimator
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.estimator.export import export
# pylint: enable=g-direct-tensorflow-import

logging.set_verbosity(logging.INFO)


# Ensures "local_init_op" is called.
class CheckLocalInitOpEstimator(tf.estimator.Estimator):

  def __init__(self):
    super(CheckLocalInitOpEstimator,
          self).__init__(model_fn=self._get_model_fn())

  def _get_model_fn(self):

    def _model_fn(features, labels, mode, params):

      del labels
      del params

      flag = tf.Variable(initial_value=False, collections=[])
      set_flag = tf.assign(flag, True)
      test_flag = tf.debugging.Assert(tf.equal(flag, True), [flag])

      scaffold = tf.train.Scaffold(
          local_init_op=tf.group(tf.train.Scaffold.default_local_init_op(),
                                 set_flag))

      # Note: Not consuming the feature stales the test input_fn.
      feature = next(iter(features.values()))
      with tf.control_dependencies([feature, test_flag]):
        batch_size = tf.shape(feature)[0]
        predictions = tf.zeros([batch_size, 1])

      return tf.estimator.EstimatorSpec(
          mode,
          loss=tf.constant(-1.0),
          train_op=test_flag,
          scaffold=scaffold,
          predictions=predictions)

    return _model_fn


class AutoEnsembleEstimatorTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(AutoEnsembleEstimatorTest, self).setUp()
    # Setup and cleanup test directory.
    # Flags are not automatically parsed at this point.
    flags.FLAGS(sys.argv)
    self.test_subdirectory = os.path.join(flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.makedirs(self.test_subdirectory)

  def tearDown(self):
    super(AutoEnsembleEstimatorTest, self).tearDown()
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      {
          "testcase_name":
              "dict_candidate_pool",
          "candidate_pool":
              lambda head, feature_columns, optimizer: {
                  "dnn":
                      tf.estimator.DNNEstimator(
                          head=head,
                          feature_columns=feature_columns,
                          optimizer=optimizer,
                          hidden_units=[3]),
                  "linear":
                      tf.estimator.LinearEstimator(
                          head=head,
                          feature_columns=feature_columns,
                          optimizer=optimizer),
              },
          "want_loss":
              .209,
      },
      {
          "testcase_name":
              "list_candidate_pool",
          "candidate_pool":
              lambda head, feature_columns, optimizer: [
                  tf.estimator.DNNEstimator(
                      head=head,
                      feature_columns=feature_columns,
                      optimizer=optimizer,
                      hidden_units=[3]),
                  tf.estimator.LinearEstimator(
                      head=head,
                      feature_columns=feature_columns,
                      optimizer=optimizer),
              ],
          "want_loss":
              .209,
      },
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
      },
      {
          "testcase_name":
              "bagging",
          "candidate_pool":
              lambda head, feature_columns, optimizer: {
                  "same_train_data":
                      AutoEnsembleSubestimator(
                          tf.estimator.LinearEstimator(
                              head=head,
                              feature_columns=feature_columns,
                              optimizer=optimizer)),
                  "different_train_data":
                      AutoEnsembleSubestimator(
                          tf.estimator.DNNEstimator(
                              head=head,
                              feature_columns=feature_columns,
                              optimizer=optimizer,
                              hidden_units=[3]),
                          train_input_fn=lambda: ({
                              "input_1": tf.constant([[0., 1.]])
                          }, tf.constant([[1.]]))),
              },
          # TODO: Figure out why this test's loss changes with every
          # change to the TensorFlow graph.
          "want_loss":
              0.2,
      },
      {
          "testcase_name":
              "bagging_out_of_range_error",
          "max_train_steps":
              15,
          "candidate_pool":
              lambda head, feature_columns, optimizer: {
                  "same_train_data":
                      AutoEnsembleSubestimator(
                          tf.estimator.LinearEstimator(
                              head=head,
                              feature_columns=feature_columns,
                              optimizer=optimizer)),
                  "different_train_data":
                      AutoEnsembleSubestimator(
                          tf.estimator.DNNEstimator(
                              head=head,
                              feature_columns=feature_columns,
                              optimizer=optimizer,
                              hidden_units=[3]),
                          # TODO: Dataset must have at least 2 batches,
                          # otherwise all of training terminates.
                          train_input_fn=lambda: tf.data.Dataset.
                          from_tensor_slices(({
                              "input_1": [[0., 1.], [0., 1.]]
                          }, [[1.], [1.]])).batch(1),
                      ),
              },
          # TODO: Figure out why this test's loss changes with every
          # change to the TensorFlow graph.
          "want_loss":
              0.3,
      },
      {
          "testcase_name":
              "check_local_init_op",
          "max_train_steps":
              10,
          "candidate_pool":
              lambda head, feature_columns, optimizer: {
                  "expect_local_init_op":
                      AutoEnsembleSubestimator(CheckLocalInitOpEstimator()),
              },
          "want_loss":
              1.0,
      },
  )
  # pylint: enable=g-long-lambda
  @tf_compat.skip_for_tf2
  def test_auto_ensemble_estimator_lifecycle(self,
                                             candidate_pool,
                                             want_loss,
                                             max_train_steps=30):
    features = {"input_1": [[1., 0.]]}
    labels = [[1.]]

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    head = tf.contrib.estimator.regression_head(
        loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01)
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
      serialized_example = tf.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      for key, value in features.items():
        features[key] = tf.constant(value)
      return export.SupervisedInputReceiver(
          features=features,
          labels=tf.constant(labels),
          receiver_tensors=serialized_example)

    export_dir_base = os.path.join(self.test_subdirectory, "export")
    export_saved_model_fn = getattr(estimator, "export_saved_model", None)
    if not callable(export_saved_model_fn):
      export_saved_model_fn = estimator.export_savedmodel
    export_saved_model_fn(
        export_dir_base=export_dir_base,
        serving_input_receiver_fn=serving_input_fn)

  @tf_compat.skip_for_tf2
  def test_last_layer_fn(self):
    head = tf.contrib.estimator.regression_head(
        loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01)
    feature_columns = [tf.feature_column.numeric_column("input_1", shape=[2])]
    cand_pool = [
        tf.estimator.DNNEstimator(
            head=head,
            feature_columns=feature_columns,
            optimizer=optimizer,
            hidden_units=[3])
    ]
    input_features = {}
    features = {"input_1": [[1., 0.]]}
    labels = [[1.]]
    for key, feature in features.items():
      input_features[key] = tf.constant(feature, name=key)
    input_labels = tf.constant(labels, name="labels")

    class _FakeSummary(object):
      """A fake adanet.Summary."""

      def scalar(self, name, tensor, family=None):
        del name, tensor, family
        return "fake_scalar"

      @contextlib.contextmanager
      def current_scope(self):
        yield

    def _adanet_last_layer_fn(estimator_spec):
      del estimator_spec
      return input_labels

    # Call with custom last_layer_fn which simply returns the labels tensor.
    generator = _GeneratorFromCandidatePool(
        cand_pool, logits_fn=None, last_layer_fn=_adanet_last_layer_fn)
    candidates = generator.generate_candidates(
        previous_ensemble=None,
        iteration_number=None,
        previous_ensemble_reports=None,
        all_reports=None,
        config=tf.estimator.RunConfig())
    subnetwork = candidates[0].build_subnetwork(input_features,
                                                input_labels, None, False, 1,
                                                _FakeSummary(), None)

    self.assertEqual(input_labels, subnetwork.last_layer)

  @tf_compat.skip_for_tf2
  def test_extra_checkpoint_saver_hook(self):
    """Tests b/122795064."""

    features = {"input_1": [[1., 0.]]}
    labels = [[1.]]

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    head = tf.contrib.estimator.binary_classification_head(
        loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01)
    feature_columns = [tf.feature_column.numeric_column("input_1", shape=[2])]

    estimator = AutoEnsembleEstimator(
        head=head,
        candidate_pool=[
            tf.estimator.LinearClassifier(
                n_classes=2,
                feature_columns=feature_columns,
                optimizer=optimizer),
            tf.estimator.DNNClassifier(
                n_classes=2,
                feature_columns=feature_columns,
                optimizer=optimizer,
                hidden_units=[3]),
        ],
        max_iteration_steps=3,
        force_grow=True,
        model_dir=self.test_subdirectory,
        config=run_config)

    ckpt_dir = os.path.join(self.test_subdirectory)
    hooks = [tf.train.CheckpointSaverHook(ckpt_dir, save_steps=1)]

    def train_input_fn():
      input_features = {}
      for key, feature in features.items():
        input_features[key] = tf.constant(feature, name=key)
      input_labels = tf.constant(labels, name="labels")
      return input_features, input_labels

    estimator.train(input_fn=train_input_fn, max_steps=6, hooks=hooks)


if __name__ == "__main__":
  tf.test.main()
