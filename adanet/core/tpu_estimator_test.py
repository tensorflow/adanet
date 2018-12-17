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

import contextlib
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

from tensorflow.contrib.tpu.python.tpu import tpu_function


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


@contextlib.contextmanager
def fake_run_on_tpu():
  """Fakes TPU existence when running TPU tests on CPU/GPU."""

  original_number_of_shards_fn = tpu_function.TpuContext.number_of_shards
  tpu_function.TpuContext.number_of_shards = 1
  try:
    yield
  finally:
    tpu_function.TpuContext.number_of_shards = original_number_of_shards_fn


def _summaries_exist(dir_path):
  """Returns whether the given dir contains non-empty tf.Summaries."""

  tf.summary.FileWriterCache.clear()
  filenames = os.path.join(dir_path, "events*")
  event_paths = tf.gfile.Glob(filenames)
  if not event_paths:
    raise ValueError("Path {!r} not found.".format(filenames))

  for last_event in tf.train.summary_iterator(event_paths[-1]):
    summary = last_event.summary
    if summary and summary.value:
      return True
  return False


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
    tf.logging.info("%s", eval_results)

    # Predict.
    # TODO: skip predictions on TF versions 1.11 and 1.12 since
    # some TPU hooks seem to be failing on predict.
    predictions = []
    tf_verison = LooseVersion(tf.VERSION)
    if (tf_verison != LooseVersion("1.11") and
        tf_verison != LooseVersion("1.12")):
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

  def test_tpu_estimator_summaries(self):
    config = tf.contrib.tpu.RunConfig(tf_random_seed=42)
    estimator = TPUEstimator(
        head=tu.head(),
        subnetwork_generator=SimpleGenerator([_DNNBuilder("dnn")]),
        max_iteration_steps=200,
        model_dir=self.test_subdirectory,
        config=config,
        use_tpu=False)
    train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])

    with fake_run_on_tpu():
      estimator.train(input_fn=train_input_fn, max_steps=3)
    estimator.evaluate(input_fn=train_input_fn, steps=3)

    self.assertFalse(
        _summaries_exist(self.test_subdirectory + "/candidate/t0_dnn"))
    self.assertTrue(
        _summaries_exist(self.test_subdirectory + "/candidate/t0_dnn/eval"))


if __name__ == "__main__":
  tf.test.main()
