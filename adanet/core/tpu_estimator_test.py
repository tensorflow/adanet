"""Tests AdaNet TPU estimator.

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

import itertools
import json
import os

from absl.testing import parameterized
from adanet import tf_compat
from adanet.core import testing_utils as tu
from adanet.core.tpu_estimator import TPUEstimator
from adanet.subnetwork import Builder
from adanet.subnetwork import Report
from adanet.subnetwork import SimpleGenerator
from adanet.subnetwork import Subnetwork
import numpy as np
import tensorflow as tf



class _DNNBuilder(Builder):
  """A simple DNN subnetwork builder."""

  def __init__(self,
               name,
               feature_columns=None,
               learning_rate=.01,
               layer_size=16,
               seed=13,
               use_tpu=False):
    self._name = name
    self._feature_columns = feature_columns
    self._learning_rate = learning_rate
    self._layer_size = layer_size
    self._seed = seed
    self._use_tpu = use_tpu

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
    with tf.variable_scope("dnn"):
      persisted_tensors = {}
      with tf.variable_scope("hidden_layer"):
        if self._feature_columns:
          input_layer = tf.feature_column.input_layer(
              features=features, feature_columns=self._feature_columns)
        else:
          input_layer = features["x"]
        w = tf.get_variable(
            shape=[input_layer.shape[1], self._layer_size],
            initializer=tf.glorot_uniform_initializer(seed=seed),
            name="weight")
        hidden_layer = tf.matmul(input_layer, w)

      if previous_ensemble:
        other_hidden_layer = previous_ensemble.weighted_subnetworks[
            -1].subnetwork.persisted_tensors["hidden_layer"]
        hidden_layer = tf.concat([hidden_layer, other_hidden_layer], axis=1)

      # Use a leaky-relu activation so that gradients can flow even when
      # outputs are negative. Leaky relu has a non-zero slope when x < 0.
      # Otherwise success at learning is completely dependent on random seed.
      hidden_layer = tf.nn.leaky_relu(hidden_layer, alpha=.2)
      persisted_tensors["hidden_layer"] = hidden_layer
      if training:
        # This change will only be in the next iteration if
        # `freeze_training_graph` is `True`.
        persisted_tensors["hidden_layer"] = 2 * hidden_layer

    with tf.variable_scope("logits"):
      logits = tf.layers.dense(
          hidden_layer,
          logits_dimension,
          kernel_initializer=tf.glorot_uniform_initializer(seed=seed))

    summary.scalar("scalar", 3)
    summary.image("image", tf.ones([1, 3, 3, 1]))
    with tf.variable_scope("nested"):
      summary.scalar("scalar", 5)

    return Subnetwork(
        last_layer=logits,
        logits=logits,
        complexity=3,
        persisted_tensors=persisted_tensors)

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=self._learning_rate)
    if self._use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    return optimizer.minimize(loss, var_list=var_list)

  def build_subnetwork_report(self):
    return Report(
        hparams={"layer_size": self._layer_size},
        attributes={"complexity": tf.constant(3, dtype=tf.int32)},
        metrics={
            "moo": (tf.constant(3,
                                dtype=tf.int32), tf.constant(3, dtype=tf.int32))
        })


class _NanLossBuilder(Builder):
  """A subnetwork builder always produces a NaN loss."""

  @property
  def name(self):
    return "nan"

  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    logits = tf_compat.v1.layers.dense(
        features["x"],
        logits_dimension,
        kernel_initializer=tf_compat.v1.glorot_uniform_initializer(
            seed=42)) * np.nan
    return Subnetwork(last_layer=logits, logits=logits, complexity=0)

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    return tf.no_op()


# TODO: merge this function with _check_eventfile_for_keyword and place
# in test_utils.
def _get_summary_value(keyword, dir_):
  """Returns the latest summary value for the given keyword in TF events."""

  tf.summary.FileWriterCache.clear()

  filenames = os.path.join(dir_, "events*")
  event_paths = tf.gfile.Glob(filenames)
  if not event_paths:
    raise ValueError("Path '{!r}' not found.".format(filenames))
  for last_event in tf.train.summary_iterator(event_paths[-1]):
    if last_event.summary is not None:
      for value in last_event.summary.value:
        if keyword == value.tag:
          if value.HasField("simple_value"):
            return value.simple_value
          if value.HasField("image"):
            return (value.image.height, value.image.width,
                    value.image.colorspace)
          if value.HasField("tensor"):
            return value.tensor.string_val

  raise ValueError("Keyword '{}' not found in path '{}'.".format(
      keyword, filenames))


class TPUEstimatorTest(tu.AdanetTestCase):

  def setUp(self):
    super(TPUEstimatorTest, self).setUp()

    if not tf_compat.version_greater_or_equal("1.14.0"):
      self.skipTest("TPUEmbedding not supported in version 1.13.0 and below.")

    # TPUConfig initializes model_dir from TF_CONFIG and checks that the user
    # provided model_dir matches the TF_CONFIG one.
    tf_config = {"model_dir": self.test_subdirectory}
    os.environ["TF_CONFIG"] = json.dumps(tf_config)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "not_use_tpu",
          "use_tpu":
              False,
          "subnetwork_generator":
              SimpleGenerator([_DNNBuilder("dnn", use_tpu=False)]),
          "want_loss":
              0.41315794,
      },
  )
  def test_tpu_estimator_simple_lifecycle(self, use_tpu, subnetwork_generator,
                                          want_loss):
    config = tf.contrib.tpu.RunConfig(master="", tf_random_seed=42)
    estimator = TPUEstimator(
        head=tu.head(),
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=10,
        model_dir=self.test_subdirectory,
        config=config,
        use_tpu=use_tpu,
        train_batch_size=64 if use_tpu else 0)
    max_steps = 30

    xor_features = [[1., 0.], [0., 0], [0., 1.], [1., 1.]]
    xor_labels = [[1.], [0.], [1.], [0.]]
    train_input_fn = tu.dummy_input_fn(xor_features, xor_labels)

    # Train.
    estimator.train(
        input_fn=train_input_fn, steps=None, max_steps=max_steps, hooks=None)

    # Evaluate.
    eval_results = estimator.evaluate(
        input_fn=train_input_fn, steps=1, hooks=None)

    # Predict.
    predictions = estimator.predict(
        input_fn=tu.dataset_input_fn(features=[0., 0.], labels=None))

    # Export SavedModel.
    def serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      return tf.estimator.export.ServingInputReceiver(
          features={"x": tf.constant([[0., 0.]], name="serving_x")},
          receiver_tensors=serialized_example)

    export_saved_model_fn = getattr(estimator, "export_saved_model", None)
    if not callable(export_saved_model_fn):
      export_saved_model_fn = estimator.export_savedmodel
    export_saved_model_fn(
        export_dir_base=estimator.model_dir,
        serving_input_receiver_fn=serving_input_fn)

    self.assertAlmostEqual(want_loss, eval_results["loss"], places=2)
    self.assertEqual(max_steps, eval_results["global_step"])
    self.assertEqual(2, eval_results["iteration"])
    for prediction in predictions:
      self.assertIsNotNone(prediction["predictions"])


  @parameterized.named_parameters(
      {
          "testcase_name": "not_use_tpu",
          "use_tpu": False,
          "want_loss": 0.55584925,
          "want_adanet_loss": .64416,
          "want_eval_summary_loss": 0.555849,
          "want_predictions": 0.46818,
      },
  )
  def test_tpu_estimator_summaries(self, use_tpu, want_loss, want_adanet_loss,
                                   want_eval_summary_loss, want_predictions):
    max_steps = 10
    config = tf.contrib.tpu.RunConfig(
        tf_random_seed=42,
        save_summary_steps=max_steps,
        log_step_count_steps=max_steps)
    assert config.log_step_count_steps

    def metric_fn(predictions):
      return {
          "predictions": tf_compat.v1.metrics.mean(predictions["predictions"])
      }

    estimator = TPUEstimator(
        head=tu.head(),
        subnetwork_generator=SimpleGenerator(
            [_DNNBuilder("dnn", use_tpu=use_tpu)]),
        max_iteration_steps=max_steps,
        model_dir=self.test_subdirectory,
        metric_fn=metric_fn,
        config=config,
        use_tpu=use_tpu,
        train_batch_size=64 if use_tpu else 0)
    xor_features = [[1., 0.], [0., 0], [0., 1.], [1., 1.]]
    xor_labels = [[1.], [0.], [1.], [0.]]
    train_input_fn = tu.dummy_input_fn(xor_features, xor_labels)

    estimator.train(input_fn=train_input_fn, max_steps=max_steps)
    eval_results = estimator.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAlmostEqual(want_loss, eval_results["loss"], places=2)
    self.assertEqual(max_steps, eval_results["global_step"])
    self.assertEqual(0, eval_results["iteration"])

    subnetwork_subdir = os.path.join(self.test_subdirectory,
                                     "subnetwork/t0_dnn")

    ensemble_subdir = os.path.join(
        self.test_subdirectory, "ensemble/t0_dnn_grow_complexity_regularized")

    # TODO: Why is the adanet_loss written to 'loss'?
    self.assertAlmostEqual(
        want_adanet_loss,
        _get_summary_value("loss", self.test_subdirectory),
        places=1)
    self.assertEqual(
        0.,
        _get_summary_value("iteration/adanet/iteration",
                           self.test_subdirectory))
    self.assertAlmostEqual(
        3., _get_summary_value("scalar", subnetwork_subdir), places=3)
    self.assertEqual((3, 3, 1),
                     _get_summary_value("image/image/0", subnetwork_subdir))
    self.assertAlmostEqual(
        5., _get_summary_value("nested/scalar", subnetwork_subdir), places=3)
    self.assertAlmostEqual(
        want_adanet_loss,
        _get_summary_value("adanet_loss/adanet/adanet_weighted_ensemble",
                           ensemble_subdir),
        places=1)
    self.assertAlmostEqual(
        0.,
        _get_summary_value(
            "complexity_regularization/adanet/adanet_weighted_ensemble",
            ensemble_subdir),
        places=1)
    self.assertAlmostEqual(
        1.,
        _get_summary_value(
            "mixture_weight_norms/adanet/"
            "adanet_weighted_ensemble/subnetwork_0", ensemble_subdir),
        places=1)

    # Eval metric summaries are always written out during eval.
    subnetwork_eval_subdir = os.path.join(subnetwork_subdir, "eval")
    self.assertAlmostEqual(
        want_eval_summary_loss,
        _get_summary_value("loss", subnetwork_eval_subdir),
        places=1)
    self.assertAlmostEqual(
        want_eval_summary_loss,
        _get_summary_value("average_loss", subnetwork_eval_subdir),
        places=1)
    self.assertAlmostEqual(
        want_predictions,
        _get_summary_value("predictions", subnetwork_eval_subdir),
        places=3)

    eval_subdir = os.path.join(self.test_subdirectory, "eval")
    ensemble_eval_subdir = os.path.join(ensemble_subdir, "eval")
    for subdir in [ensemble_eval_subdir, eval_subdir]:
      self.assertEqual([b"| dnn |"],
                       _get_summary_value("architecture/adanet/ensembles/0",
                                          subdir))
      if subdir == eval_subdir:
        self.assertAlmostEqual(
            want_loss, _get_summary_value("loss", subdir), places=1)
      self.assertAlmostEqual(
          want_eval_summary_loss,
          _get_summary_value("average_loss", subdir),
          places=1)


if __name__ == "__main__":
  tf.test.main()
