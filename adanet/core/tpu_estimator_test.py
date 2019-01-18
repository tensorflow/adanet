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

import os

from absl.testing import parameterized
from adanet.core import testing_utils as tu
from adanet.core.subnetwork import Builder
from adanet.core.subnetwork import Report
from adanet.core.subnetwork import SimpleGenerator
from adanet.core.subnetwork import Subnetwork
from adanet.core.tpu_estimator import TPUEstimator
from distutils.version import LooseVersion
import tensorflow as tf


class _DNNBuilder(Builder):
  """A simple DNN subnetwork builder."""

  def __init__(self,
               name,
               learning_rate=.001,
               mixture_weight_learning_rate=.001,
               layer_size=1,
               seed=13,
               use_tpu=False):
    self._name = name
    self._learning_rate = learning_rate
    self._mixture_weight_learning_rate = mixture_weight_learning_rate
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
        w = tf.get_variable(
            shape=[2, self._layer_size],
            initializer=tf.glorot_uniform_initializer(seed=seed),
            name="weight")
        hidden_layer = tf.matmul(features["x"], w)

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
    batch_size = features["x"].get_shape().as_list()[0]
    summary.image("image", tf.ones([batch_size, 3, 3, 1]))
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
    return optimizer.minimize(loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=self._mixture_weight_learning_rate)
    if self._use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    return optimizer.minimize(loss, var_list=var_list)

  def build_subnetwork_report(self):
    return Report(
        hparams={"layer_size": self._layer_size},
        attributes={"complexity": tf.constant(3, dtype=tf.int32)},
        metrics={
            "moo": (tf.constant(3, dtype=tf.int32),
                    tf.constant(3, dtype=tf.int32))
        })


def _check_eventfile_for_keyword(keyword, dir_):
  """Checks event files for the keyword."""

  tf.summary.FileWriterCache.clear()

  # Get last `Event` written.
  filenames = os.path.join(dir_, "events*")
  event_paths = tf.gfile.Glob(filenames)
  if not event_paths:
    raise ValueError("Path '{}' not found.".format(filenames))

  # There can be multiple events files for summaries.
  for event_path in event_paths:
    for last_event in tf.train.summary_iterator(event_path):
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
    if LooseVersion(tf.VERSION) < LooseVersion("1.11.0"):
      self.skipTest("TPUEstimatorSpec does not support `training_hooks`"
                    "TF v1.11.0.")

  @parameterized.named_parameters(
      {
          "testcase_name": "not_use_tpu",
          "use_tpu": False,
      },
  )
  def test_tpu_estimator_simple_lifecycle(self, use_tpu):
    config = tf.contrib.tpu.RunConfig(master="", tf_random_seed=42)
    estimator = TPUEstimator(
        head=tu.head(),
        subnetwork_generator=SimpleGenerator(
            [_DNNBuilder("dnn", use_tpu=use_tpu)]),
        max_iteration_steps=200,
        mixture_weight_initializer=tf.zeros_initializer(),
        use_bias=True,
        model_dir=self.test_subdirectory,
        config=config,
        use_tpu=use_tpu,
        train_batch_size=64 if use_tpu else 0)
    max_steps = 300

    xor_features = [[1., 0.], [0., 0], [0., 1.], [1., 1.]]
    xor_labels = [[1.], [0.], [1.], [0.]]
    train_input_fn = tu.dummy_input_fn(xor_features, xor_labels)

    # Train.
    estimator.train(
        input_fn=train_input_fn, steps=None, max_steps=max_steps, hooks=None)

    # Evaluate.
    eval_results = estimator.evaluate(
        input_fn=train_input_fn, steps=10, hooks=None)

    # Predict.
    # TODO: skip predictions on TF versions 1.11 and 1.12 since
    # some TPU hooks seem to be failing on predict.
    predictions = []
    tf_version = LooseVersion(tf.VERSION)
    if (tf_version != LooseVersion("1.11.0") and
        tf_version != LooseVersion("1.12.0")):
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

    self.assertAlmostEqual(0.32416, eval_results["loss"], places=3)
    self.assertEqual(max_steps, eval_results["global_step"])
    for prediction in predictions:
      self.assertIsNotNone(prediction["predictions"])

  @parameterized.named_parameters(
      {
          "testcase_name": "not_use_tpu",
          "use_tpu": False,
      },
  )
  def test_tpu_estimator_summaries(self, use_tpu):
    config = tf.contrib.tpu.RunConfig(
        tf_random_seed=42, save_summary_steps=2, log_step_count_steps=1)
    assert config.log_step_count_steps
    estimator = TPUEstimator(
        head=tu.head(),
        subnetwork_generator=SimpleGenerator(
            [_DNNBuilder("dnn", use_tpu=use_tpu)]),
        max_iteration_steps=200,
        model_dir=self.test_subdirectory,
        config=config,
        use_tpu=use_tpu,
        train_batch_size=64 if use_tpu else 0)
    xor_features = [[1., 0.], [0., 0], [0., 1.], [1., 1.]]
    xor_labels = [[1.], [0.], [1.], [0.]]
    train_input_fn = tu.dummy_input_fn(xor_features, xor_labels)

    estimator.train(input_fn=train_input_fn, max_steps=3)
    estimator.evaluate(input_fn=train_input_fn, steps=3)

    ensemble_loss = .5
    self.assertAlmostEqual(
        ensemble_loss,
        _check_eventfile_for_keyword("loss", self.test_subdirectory),
        places=1)
    self.assertIsNotNone(
        _check_eventfile_for_keyword("global_step/sec", self.test_subdirectory))
    eval_subdir = os.path.join(self.test_subdirectory, "eval")
    self.assertAlmostEqual(
        ensemble_loss,
        _check_eventfile_for_keyword("loss", eval_subdir),
        places=1)
    self.assertEqual(
        0.,
        _check_eventfile_for_keyword("iteration/adanet/iteration",
                                     self.test_subdirectory))

    candidate_subdir = os.path.join(self.test_subdirectory, "candidate/t0_dnn")
    self.assertAlmostEqual(
        3., _check_eventfile_for_keyword("scalar", candidate_subdir), places=3)
    self.assertEqual((3, 3, 1),
                     _check_eventfile_for_keyword("image/image/0",
                                                  candidate_subdir))
    self.assertAlmostEqual(
        5.,
        _check_eventfile_for_keyword("nested/scalar", candidate_subdir),
        places=1)
    self.assertAlmostEqual(
        ensemble_loss,
        _check_eventfile_for_keyword(
            "adanet_loss/adanet/adanet_weighted_ensemble", candidate_subdir),
        places=1)
    self.assertAlmostEqual(
        0.,
        _check_eventfile_for_keyword(
            "complexity_regularization/adanet/adanet_weighted_ensemble",
            candidate_subdir),
        places=1)
    self.assertAlmostEqual(
        1.,
        _check_eventfile_for_keyword(
            "mixture_weight_norms/adanet/"
            "adanet_weighted_ensemble/subnetwork_0", candidate_subdir),
        places=1)


if __name__ == "__main__":
  tf.test.main()
