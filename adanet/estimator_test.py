"""Test AdaNet estimator single graph implementation.

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
from adanet.base_learner import BaseLearner
from adanet.base_learner import BaseLearnerBuilder
from adanet.base_learner import SimpleBaseLearnerBuilderGenerator
from adanet.base_learner_report import BaseLearnerReport
from adanet.base_learner_report import MaterializedBaseLearnerReport
from adanet.ensemble import MixtureWeightType
from adanet.estimator import Estimator
from adanet.evaluator import Evaluator
from adanet.report_materializer import ReportMaterializer
import adanet.testing_utils as tu
import tensorflow as tf

from tensorflow.python.estimator.export import export


class _DNNBaseLearnerBuilder(BaseLearnerBuilder):
  """A simple DNN base learner builder."""

  def __init__(self,
               name,
               mixture_weight_learning_rate=3.,
               return_penultimate_layer=True,
               layer_size=1,
               seed=13):
    self._name = name
    self._mixture_weight_learning_rate = mixture_weight_learning_rate
    self._return_penultimate_layer = return_penultimate_layer
    self._layer_size = layer_size
    self._seed = seed

  @property
  def name(self):
    return self._name

  def build_base_learner(self,
                         features,
                         logits_dimension,
                         training,
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
        other_hidden_layer = previous_ensemble.weighted_base_learners[
            -1].base_learner.persisted_tensors["hidden_layer"]
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

    last_layer = hidden_layer

    with tf.variable_scope("logits"):
      logits = tf.layers.dense(
          hidden_layer,
          logits_dimension,
          kernel_initializer=tf.glorot_uniform_initializer(seed=seed))

    with tf.name_scope(""):
      summary.scalar("scalar", 3)

    return BaseLearner(
        last_layer=last_layer if self._return_penultimate_layer else logits,
        logits=logits,
        complexity=3,
        persisted_tensors=persisted_tensors)

  def build_base_learner_train_op(self, loss, var_list, labels, summary):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=3.)
    return optimizer.minimize(loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     summary):
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=self._mixture_weight_learning_rate)
    return optimizer.minimize(loss, var_list=var_list)

  def build_base_learner_report(self):
    return BaseLearnerReport(
        hparams={"layer_size": self._layer_size},
        attributes={"complexity": tf.constant(3, dtype=tf.int32)},
        metrics={
            "moo": (tf.constant(3, dtype=tf.int32),
                    tf.constant(3, dtype=tf.int32))
        })


class _SimpleBaseLearnerBuilder(BaseLearnerBuilder):
  """A simple base learner builder that takes feature_columns."""

  def __init__(self, name, feature_columns, seed=42):
    self._name = name
    self._feature_columns = feature_columns
    self._seed = seed

  @property
  def name(self):
    return self._name

  def build_base_learner(self,
                         features,
                         logits_dimension,
                         training,
                         summary,
                         previous_ensemble=None):
    seed = self._seed
    if previous_ensemble:
      # Increment seed so different iterations don't learn the exact same thing.
      seed += 1

    with tf.variable_scope("simple"):
      input_layer = tf.feature_column.input_layer(
          features=features, feature_columns=self._feature_columns)
      last_layer = input_layer

    with tf.variable_scope("logits"):
      logits = tf.layers.dense(
          last_layer,
          logits_dimension,
          kernel_initializer=tf.glorot_uniform_initializer(seed=seed))

    return BaseLearner(
        last_layer=last_layer,
        logits=logits,
        complexity=1,
        persisted_tensors={},
    )

  def build_base_learner_train_op(self, loss, var_list, labels, summary):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=3.)
    return optimizer.minimize(loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     summary):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=3.)
    return optimizer.minimize(loss, var_list=var_list)


class _LinearBaseLearnerBuilder(BaseLearnerBuilder):
  """A simple linear base learner builder."""

  def __init__(self, name, mixture_weight_learning_rate=.1, seed=42):
    self._name = name
    self._mixture_weight_learning_rate = mixture_weight_learning_rate
    self._seed = seed

  @property
  def name(self):
    return self._name

  def build_base_learner(self,
                         features,
                         logits_dimension,
                         training,
                         summary,
                         previous_ensemble=None):

    logits = tf.layers.dense(
        features["x"],
        logits_dimension,
        kernel_initializer=tf.glorot_uniform_initializer(seed=self._seed))

    return BaseLearner(
        last_layer=features["x"],
        logits=logits,
        complexity=1,
        persisted_tensors={},
    )

  def build_base_learner_train_op(self, loss, var_list, labels, summary):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1)
    return optimizer.minimize(loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     summary):
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=self._mixture_weight_learning_rate)
    return optimizer.minimize(loss, var_list=var_list)


class _ModifierSessionRunHook(tf.train.SessionRunHook):
  """Modifies the graph by adding a variable."""

  def __init__(self):
    self._begun = False

  def begin(self):
    """Adds a variable to the graph.

    Raises:
      ValueError: If we've already begun a run.
    """

    if self._begun:
      raise ValueError("begin called twice without end.")
    self._begun = True
    _ = tf.get_variable(name="foo", initializer="foo")

  def end(self, session):
    """Adds a variable to the graph.

    Args:
      session: A `tf.Session` object that can be used to run ops.

    Raises:
      ValueError: If we've not begun a run.
    """

    _ = session
    if not self._begun:
      raise ValueError("end called without begin.")
    self._begun = False


def _head():
  return tf.contrib.estimator.binary_classification_head(
      loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)


class EstimatorTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    # Setup and cleanup test directory.
    self.test_subdirectory = os.path.join(tf.flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.mkdir(self.test_subdirectory)

  def tearDown(self):
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  @parameterized.named_parameters({
      "testcase_name":
          "one_step",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([_DNNBaseLearnerBuilder("dnn")]),
      "max_iteration_steps":
          1,
      "steps":
          1,
      "max_steps":
          None,
      "want_accuracy":
          .75,
      "want_loss":
          .69314,
  }, {
      "testcase_name":
          "single_builder_max_steps",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([_DNNBaseLearnerBuilder("dnn")]),
      "max_iteration_steps":
          200,
      "max_steps":
          300,
      "want_accuracy":
          1.,
      "want_loss":
          .00780,
  }, {
      "testcase_name":
          "single_builder_steps",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([_DNNBaseLearnerBuilder("dnn")]),
      "max_iteration_steps":
          200,
      "steps":
          300,
      "max_steps":
          None,
      "want_accuracy":
          1.,
      "want_loss":
          .00780,
  }, {
      "testcase_name":
          "single_builder_no_bias",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([_DNNBaseLearnerBuilder("dnn")]),
      "max_iteration_steps":
          200,
      "use_bias":
          False,
      "want_accuracy":
          1.,
      "want_loss":
          .34461,
  }, {
      "testcase_name":
          "single_builder_scalar_mixture_weight",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator(
              [_DNNBaseLearnerBuilder("dnn", return_penultimate_layer=False)]),
      "max_iteration_steps":
          200,
      "mixture_weight_type":
          MixtureWeightType.SCALAR,
      "want_accuracy":
          1.,
      "want_loss":
          3.1415e-6,
  }, {
      "testcase_name":
          "single_builder_vector_mixture_weight",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator(
              [_DNNBaseLearnerBuilder("dnn", return_penultimate_layer=False)]),
      "max_iteration_steps":
          200,
      "mixture_weight_type":
          MixtureWeightType.VECTOR,
      "want_accuracy":
          1.,
      "want_loss":
          3.1415e-6,
  }, {
      "testcase_name":
          "single_builder_replicate_ensemble_in_training",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([_DNNBaseLearnerBuilder("dnn")]),
      "replicate_ensemble_in_training":
          True,
      "max_iteration_steps":
          200,
      "max_steps":
          300,
      "want_accuracy":
          1.,
      "want_loss":
          .11910,
  }, {
      "testcase_name":
          "single_builder_with_hook",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([_DNNBaseLearnerBuilder("dnn")]),
      "max_iteration_steps":
          200,
      "hooks": [_ModifierSessionRunHook()],
      "want_accuracy":
          1.,
      "want_loss":
          .00780,
  }, {
      "testcase_name":
          "high_max_iteration_steps",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([_DNNBaseLearnerBuilder("dnn")]),
      "max_iteration_steps":
          500,
      "want_accuracy":
          .75,
      "want_loss":
          .59545,
  }, {
      "testcase_name":
          "two_builders",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([
              _DNNBaseLearnerBuilder("dnn"),
              _DNNBaseLearnerBuilder("dnn2", seed=99)
          ]),
      "max_iteration_steps":
          200,
      "want_accuracy":
          1.,
      "want_loss":
          .00780,
  }, {
      "testcase_name":
          "two_builders_different_layer_sizes",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([
              _DNNBaseLearnerBuilder("dnn"),
              _DNNBaseLearnerBuilder("dnn2", layer_size=3)
          ]),
      "max_iteration_steps":
          200,
      "want_accuracy":
          1.,
      "want_loss":
          .00176,
  }, {
      "testcase_name":
          "evaluator_good_input",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([
              _DNNBaseLearnerBuilder("dnn"),
              _DNNBaseLearnerBuilder("dnn2", layer_size=3)
          ]),
      "evaluator":
          Evaluator(input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=3),
      "max_iteration_steps":
          200,
      "want_accuracy":
          1.,
      "want_loss":
          .00176,
  }, {
      "testcase_name":
          "evaluator_bad_input",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([
              _DNNBaseLearnerBuilder("dnn"),
              _DNNBaseLearnerBuilder("dnn2", layer_size=3)
          ]),
      "evaluator":
          Evaluator(input_fn=tu.dummy_input_fn([[1., 1.]], [[1.]]), steps=3),
      "max_iteration_steps":
          200,
      "want_accuracy":
          1.,
      "want_loss":
          .00780,
  }, {
      "testcase_name":
          "report_materializer",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([
              _DNNBaseLearnerBuilder("dnn"),
              _DNNBaseLearnerBuilder("dnn2", layer_size=3)
          ]),
      "evaluator":
          Evaluator(input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=3),
      "report_materializer":
          ReportMaterializer(
              input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=3),
      "max_iteration_steps":
          200,
      "want_accuracy":
          1.,
      "want_loss":
          .00176,
  })
  def test_lifecycle(self,
                     base_learner_builder_generator,
                     want_accuracy,
                     want_loss,
                     max_iteration_steps,
                     mixture_weight_type=MixtureWeightType.MATRIX,
                     evaluator=None,
                     report_materializer=None,
                     use_bias=True,
                     replicate_ensemble_in_training=False,
                     hooks=None,
                     max_steps=300,
                     steps=None):
    """Train entire estimator lifecycle using XOR dataset."""

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    estimator = Estimator(
        head=_head(),
        base_learner_builder_generator=base_learner_builder_generator,
        max_iteration_steps=max_iteration_steps,
        mixture_weight_type=mixture_weight_type,
        mixture_weight_initializer=tf.zeros_initializer(),
        warm_start_mixture_weights=True,
        evaluator=evaluator,
        report_materializer=report_materializer,
        use_bias=use_bias,
        replicate_ensemble_in_training=replicate_ensemble_in_training,
        model_dir=self.test_subdirectory,
        config=run_config)

    xor_features = [[1., 0.], [0., 0], [0., 1.], [1., 1.]]
    xor_labels = [[1.], [0.], [1.], [0.]]
    train_input_fn = tu.dummy_input_fn(xor_features, xor_labels)

    # Train.
    estimator.train(
        input_fn=train_input_fn, steps=steps, max_steps=max_steps, hooks=hooks)

    # Evaluate.
    eval_results = estimator.evaluate(
        input_fn=train_input_fn, steps=10, hooks=hooks)
    self.assertAlmostEqual(want_accuracy, eval_results["accuracy"], places=3)
    self.assertAlmostEqual(want_loss, eval_results["loss"], places=5)
    self.assertEqual(max_steps or steps, eval_results["global_step"])

    # Predict.
    predictions = estimator.predict(
        input_fn=tu.dataset_input_fn(features=[0., 0.], labels=None))
    for prediction in predictions:
      self.assertIsNotNone(prediction["classes"])
      self.assertIsNotNone(prediction["probabilities"])

    # Export SavedModel.
    def serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      return tf.estimator.export.ServingInputReceiver(
          features={"x": tf.constant([[0., 0.]], name="serving_x")},
          receiver_tensors=serialized_example)

    estimator.export_savedmodel(
        export_dir_base=self.test_subdirectory,
        serving_input_receiver_fn=serving_input_fn)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "hash_bucket_with_one_hot",
          "feature_column": (tf.feature_column.indicator_column(
              categorical_column=(
                  tf.feature_column.categorical_column_with_hash_bucket(
                      key="human_names", hash_bucket_size=4, dtype=tf.string)))
                            ),
      }, {
          "testcase_name":
              "vocab_list_with_one_hot",
          "feature_column": (tf.feature_column.indicator_column(
              categorical_column=(
                  tf.feature_column.categorical_column_with_vocabulary_list(
                      key="human_names",
                      vocabulary_list=["alice", "bob"],
                      dtype=tf.string)))),
      }, {
          "testcase_name":
              "hash_bucket_with_embedding",
          "feature_column": (tf.feature_column.embedding_column(
              categorical_column=(
                  tf.feature_column.categorical_column_with_hash_bucket(
                      key="human_names", hash_bucket_size=4, dtype=tf.string)),
              dimension=2)),
      }, {
          "testcase_name":
              "vocab_list_with_embedding",
          "feature_column": (tf.feature_column.embedding_column(
              categorical_column=(
                  tf.feature_column.categorical_column_with_vocabulary_list(
                      key="human_names",
                      vocabulary_list=["alice", "bob"],
                      dtype=tf.string)),
              dimension=2)),
      })
  def test_categorical_columns(self, feature_column):
    estimator = Estimator(
        head=tf.contrib.estimator.binary_classification_head(),
        base_learner_builder_generator=SimpleBaseLearnerBuilderGenerator([
            _SimpleBaseLearnerBuilder(
                name="simple", feature_columns=[feature_column])
        ]),
        mixture_weight_initializer=tf.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=1,
        model_dir=self.test_subdirectory)

    def train_input_fn():
      input_features = {
          "human_names": tf.constant([["alice"], ["bob"]], name="human_names")
      }
      input_labels = tf.constant([[1.], [0.]], name="starts_with_a")
      return input_features, input_labels

    estimator.train(input_fn=train_input_fn, max_steps=3)

  @parameterized.named_parameters({
      "testcase_name": "no_base_learner_builder_generator",
      "base_learner_builder_generator": None,
      "max_iteration_steps": 100,
  }, {
      "testcase_name":
          "negative_max_iteration_steps",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([_DNNBaseLearnerBuilder("dnn")]),
      "max_iteration_steps":
          -1,
  }, {
      "testcase_name":
          "zero_max_iteration_steps",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([_DNNBaseLearnerBuilder("dnn")]),
      "max_iteration_steps":
          0,
  }, {
      "testcase_name":
          "steps_and_max_steps",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([_DNNBaseLearnerBuilder("dnn")]),
      "max_iteration_steps":
          1,
      "steps":
          1,
      "max_steps":
          1,
  }, {
      "testcase_name":
          "zero_steps",
      "base_learner_builder_generator":
          SimpleBaseLearnerBuilderGenerator([_DNNBaseLearnerBuilder("dnn")]),
      "max_iteration_steps":
          1,
      "steps":
          0,
      "max_steps":
          None,
  })
  def test_train_error(self,
                       base_learner_builder_generator,
                       max_iteration_steps,
                       steps=None,
                       max_steps=10):
    with self.assertRaises(ValueError):
      estimator = Estimator(
          head=_head(),
          base_learner_builder_generator=base_learner_builder_generator,
          mixture_weight_initializer=tf.zeros_initializer(),
          warm_start_mixture_weights=True,
          max_iteration_steps=max_iteration_steps,
          model_dir=self.test_subdirectory)
      train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])
      estimator.train(input_fn=train_input_fn, steps=steps, max_steps=max_steps)


class CheckpointTest(parameterized.TestCase, tf.test.TestCase):
  """Tests estimator checkpoints."""

  def setUp(self):
    # Setup and cleanup test directory.
    self.test_subdirectory = os.path.join(tf.flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.mkdir(self.test_subdirectory)

  def tearDown(self):
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  @parameterized.named_parameters({
      "testcase_name": "single_iteration",
      "max_iteration_steps": 3,
      "keep_checkpoint_max": 3,
      "want_num_checkpoints": 3,
  }, {
      "testcase_name": "single_iteration_keep_one",
      "max_iteration_steps": 3,
      "keep_checkpoint_max": 1,
      "want_num_checkpoints": 1,
  }, {
      "testcase_name": "three_iterations",
      "max_iteration_steps": 1,
      "keep_checkpoint_max": 3,
      "want_num_checkpoints": 3,
  }, {
      "testcase_name": "three_iterations_keep_one",
      "max_iteration_steps": 1,
      "keep_checkpoint_max": 1,
      "want_num_checkpoints": 1,
  })
  def test_checkpoints(self,
                       max_iteration_steps,
                       keep_checkpoint_max,
                       want_num_checkpoints,
                       max_steps=3):
    config = tf.estimator.RunConfig(
        save_checkpoints_steps=1,
        keep_checkpoint_max=keep_checkpoint_max,
    )
    base_learner_builder_generator = SimpleBaseLearnerBuilderGenerator(
        [_DNNBaseLearnerBuilder("dnn")])
    estimator = Estimator(
        head=_head(),
        base_learner_builder_generator=base_learner_builder_generator,
        mixture_weight_initializer=tf.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=max_iteration_steps,
        config=config,
        model_dir=self.test_subdirectory)
    train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])
    estimator.train(input_fn=train_input_fn, max_steps=max_steps)

    checkpoints = tf.gfile.Glob(os.path.join(self.test_subdirectory, "*.meta"))
    self.assertEqual(want_num_checkpoints, len(checkpoints))


def _check_eventfile_for_keyword(keyword, dir_):
  """Checks event files for the keyword."""

  tf.summary.FileWriterCache.clear()

  # Get last `Event` written.
  filenames = os.path.join(dir_, "events*")
  event_paths = tf.gfile.Glob(filenames)
  if not event_paths:
    raise ValueError("Path '{}' not found.".format(filenames))

  for last_event in tf.train.summary_iterator(event_paths[-1]):
    if last_event.summary is not None:
      for value in last_event.summary.value:
        if keyword == value.tag:
          return value.simple_value

  raise ValueError("Keyword '{}' not found in path '{}'.".format(
      keyword, filenames))


class _FakeMetric(object):
  """A fake metric."""

  def __init__(self, value, dtype):
    self._value = value
    self._dtype = dtype

  def to_metric(self):
    tensor = tf.convert_to_tensor(self._value, dtype=self._dtype)
    return (tensor, tensor)


class _EvalMetricsHead(object):
  """A fake head with the given evaluation metrics."""

  def __init__(self, fake_metrics):
    self._fake_metrics = fake_metrics

  @property
  def logits_dimension(self):
    return 1

  def create_estimator_spec(self,
                            features,
                            mode,
                            logits,
                            labels=None,
                            train_op_fn=None):
    del features  # Unused

    metric_ops = None
    if self._fake_metrics:
      metric_ops = {}
      for k, fake_metric in self._fake_metrics.items():
        metric_ops[k] = fake_metric.to_metric()
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=logits,
        loss=tf.reduce_mean(labels - logits),
        eval_metric_ops=metric_ops,
        train_op=train_op_fn(1))


class SummaryWriterTest(parameterized.TestCase, tf.test.TestCase):
  """Test that Tensorboard summaries get written correctly."""

  def setUp(self):
    # Setup and cleanup test directory.
    self.test_subdirectory = os.path.join(tf.flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.mkdir(self.test_subdirectory)

  def tearDown(self):
    shutil.rmtree(self.test_subdirectory, ignore_errors=False)

  def test_summaries(self):
    """Tests that summaries are written to candidate directory."""

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    base_learner_builder_generator = SimpleBaseLearnerBuilderGenerator(
        [_DNNBaseLearnerBuilder("dnn", mixture_weight_learning_rate=.01)])
    estimator = Estimator(
        head=_head(),
        base_learner_builder_generator=base_learner_builder_generator,
        mixture_weight_initializer=tf.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=10,
        config=run_config,
        model_dir=self.test_subdirectory)
    train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])
    estimator.train(input_fn=train_input_fn, max_steps=3)

    ensemble_loss = .693
    self.assertAlmostEqual(
        ensemble_loss,
        _check_eventfile_for_keyword("loss", self.test_subdirectory),
        places=3)
    self.assertEqual(
        0.,
        _check_eventfile_for_keyword("iteration/adanet/iteration",
                                     self.test_subdirectory))

    candidate_subdir = os.path.join(self.test_subdirectory, "candidate/dnn")
    self.assertAlmostEqual(
        3., _check_eventfile_for_keyword("scalar", candidate_subdir), places=3)
    self.assertAlmostEqual(
        .567,
        _check_eventfile_for_keyword(
            "adanet_loss_ema/adanet/adanet_weighted_ensemble",
            candidate_subdir),
        places=3)
    self.assertAlmostEqual(
        0.,
        _check_eventfile_for_keyword(
            "complexity_regularization/adanet/adanet_weighted_ensemble",
            candidate_subdir),
        places=3)
    self.assertAlmostEqual(
        0.,
        _check_eventfile_for_keyword(
            "mixture_weight_norms/adanet/"
            "adanet_weighted_ensemble/base_learner_0", candidate_subdir),
        places=3)

  @parameterized.named_parameters(
      {
          "testcase_name": "none_metrics",
          "head": _EvalMetricsHead(None),
          "want_summaries": [],
          "want_loss": .910,
      }, {
          "testcase_name": "empty_metrics",
          "head": _EvalMetricsHead({}),
          "want_summaries": [],
          "want_loss": .910,
      }, {
          "testcase_name": "evaluation_name",
          "head": _EvalMetricsHead({}),
          "evaluation_name": "continuous",
          "want_summaries": [],
          "want_loss": .910,
          "global_subdir": "eval_continuous",
          "candidate_subdir": "candidate/linear/eval_continuous",
      }, {
          "testcase_name":
              "regression_head",
          "head":
              tf.contrib.estimator.regression_head(
                  loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),
          "want_summaries": [
              "average_loss", "average_loss/adanet/adanet_weighted_ensemble"
          ],
          "want_loss":
              .691,
      }, {
          "testcase_name":
              "binary_classification_head",
          "head":
              tf.contrib.estimator.binary_classification_head(
                  loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE),
          "want_summaries": [
              "average_loss", "average_loss/adanet/adanet_weighted_ensemble"
          ],
          "want_loss":
              .671,
      }, {
          "testcase_name":
              "all_metrics",
          "head":
              _EvalMetricsHead({
                  "float32":
                      _FakeMetric(1., tf.float32),
                  "float64":
                      _FakeMetric(1., tf.float64),
                  "serialized_summary":
                      _FakeMetric(
                          tf.Summary(value=[
                              tf.Summary.Value(
                                  tag="summary_tag", simple_value=1.)
                          ]).SerializeToString(),
                          tf.string),
              }),
          "want_summaries": [
              "float32",
              "float64",
              "serialized_summary/0",
              "float32/adanet/adanet_weighted_ensemble",
              "float64/adanet/adanet_weighted_ensemble",
              "serialized_summary/adanet/adanet_weighted_ensemble/0",
          ],
          "want_loss":
              .910,
      })
  def test_eval_metrics(self,
                        head,
                        want_loss,
                        want_summaries,
                        evaluation_name=None,
                        global_subdir="eval",
                        candidate_subdir="candidate/linear/eval"):
    """Test that AdaNet evaluation metrics get persisted correctly."""

    seed = 42
    run_config = tf.estimator.RunConfig(tf_random_seed=seed)
    base_learner_builder_generator = SimpleBaseLearnerBuilderGenerator([
        _LinearBaseLearnerBuilder(
            "linear", mixture_weight_learning_rate=.01, seed=seed)
    ])
    estimator = Estimator(
        head=head,
        base_learner_builder_generator=base_learner_builder_generator,
        mixture_weight_initializer=tf.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=6,
        config=run_config,
        model_dir=self.test_subdirectory)
    train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])
    estimator.train(input_fn=train_input_fn, max_steps=3)

    eval_set_size = 5

    def eval_input_fn():
      """Generates a stream of random features."""
      feature_dataset = tf.data.Dataset.range(eval_set_size).map(
          lambda i: tf.stack([tf.to_float(i), tf.to_float(i)]))
      label_dataset = tf.data.Dataset.range(eval_set_size).map(
          lambda _: tf.constant(1.))
      dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))
      iterator = dataset.batch(1).make_one_shot_iterator()
      features, labels = iterator.get_next()
      return {"x": features}, labels

    metrics = estimator.evaluate(
        input_fn=eval_input_fn, steps=eval_set_size, name=evaluation_name)
    self.assertAlmostEqual(want_loss, metrics["loss"], places=3)

    global_subdir = os.path.join(self.test_subdirectory, global_subdir)
    candidate_subdir = os.path.join(self.test_subdirectory, candidate_subdir)
    self.assertAlmostEqual(metrics["loss"],
                           _check_eventfile_for_keyword("loss", global_subdir))
    self.assertAlmostEqual(
        metrics["loss"],
        _check_eventfile_for_keyword(
            "loss/adanet/adanet_weighted_ensemble",
            candidate_subdir,
        ),
        msg="Candidate loss and reported loss should be equal.")
    self.assertIsNotNone(
        _check_eventfile_for_keyword(
            "loss/adanet/uniform_average_ensemble",
            candidate_subdir,
        ))
    self.assertIsNotNone(
        _check_eventfile_for_keyword(
            "loss/adanet/base_learner",
            candidate_subdir,
        ))
    for metric in want_summaries:
      self.assertIsNotNone(
          _check_eventfile_for_keyword(metric, global_subdir),
          msg="{} should be under 'eval'.".format(metric))


class MembersOverrideTest(tf.test.TestCase):
  """Tests b/77494544 fix."""

  def test_assert_members_are_not_overridden(self):
    """Assert that AdaNet estimator does not break other estimators."""

    config = tf.estimator.RunConfig()
    base_learner_builder_generator = SimpleBaseLearnerBuilderGenerator(
        [_DNNBaseLearnerBuilder("dnn")])
    adanet = Estimator(
        head=_head(),
        base_learner_builder_generator=base_learner_builder_generator,
        mixture_weight_initializer=tf.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=10,
        config=config)
    self.assertIsNotNone(adanet)
    linear = tf.contrib.estimator.LinearEstimator(
        head=_head(), feature_columns=[tf.feature_column.numeric_column("x")])
    self.assertIsNotNone(linear)


def _dummy_feature_dict_input_fn(features, labels):
  """Returns an input_fn that returns feature and labels `Tensors`."""

  def _input_fn():
    input_features = {}
    for key, feature in features.items():
      input_features[key] = tf.constant(feature, name=key)
    input_labels = tf.constant(labels, name="labels")
    return input_features, input_labels

  return _input_fn


class DifferentFeaturesPerModeTest(parameterized.TestCase, tf.test.TestCase):
  """Tests b/109751254."""

  def setUp(self):
    # Setup and cleanup test directory.
    self.test_subdirectory = os.path.join(tf.flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  def tearDown(self):
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  @parameterized.named_parameters({
      "testcase_name": "extra_train_features",
      "train_features": {
          "x": [[1., 0.]],
          "extra": [[1., 0.]],
      },
      "eval_features": {
          "x": [[1., 0.]],
      },
      "predict_features": {
          "x": [[1., 0.]],
      },
  }, {
      "testcase_name": "extra_eval_features",
      "train_features": {
          "x": [[1., 0.]],
      },
      "eval_features": {
          "x": [[1., 0.]],
          "extra": [[1., 0.]],
      },
      "predict_features": {
          "x": [[1., 0.]],
      },
  }, {
      "testcase_name": "extra_predict_features",
      "train_features": {
          "x": [[1., 0.]],
      },
      "eval_features": {
          "x": [[1., 0.]],
      },
      "predict_features": {
          "x": [[1., 0.]],
          "extra": [[1., 0.]],
      },
  })
  def test_different_features_per_mode(self, train_features, eval_features,
                                       predict_features):
    """Tests tests different numbers of features per mode."""

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    base_learner_builder_generator = SimpleBaseLearnerBuilderGenerator(
        [_DNNBaseLearnerBuilder("dnn")])
    estimator = Estimator(
        head=_head(),
        base_learner_builder_generator=base_learner_builder_generator,
        mixture_weight_initializer=tf.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=1,
        model_dir=self.test_subdirectory,
        config=run_config)

    labels = [[1.]]
    train_input_fn = _dummy_feature_dict_input_fn(train_features, labels)

    # Train.
    estimator.train(input_fn=train_input_fn, max_steps=2)

    # Evaluate.
    eval_input_fn = _dummy_feature_dict_input_fn(eval_features, labels)
    estimator.evaluate(input_fn=eval_input_fn, steps=1)

    # Predict.
    predict_input_fn = _dummy_feature_dict_input_fn(predict_features, None)
    estimator.predict(input_fn=predict_input_fn)

    # Export SavedModel.
    def serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      features = {}
      for key, value in predict_features.items():
        features[key] = tf.constant(value)
      return tf.estimator.export.ServingInputReceiver(
          features=features, receiver_tensors=serialized_example)

    estimator.export_savedmodel(
        export_dir_base=self.test_subdirectory,
        serving_input_receiver_fn=serving_input_fn)


class ExportSavedModelForModeTest(parameterized.TestCase, tf.test.TestCase):
  """Tests b/110435640."""

  def setUp(self):
    # Setup and cleanup test directory.
    self.test_subdirectory = os.path.join(tf.flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  def tearDown(self):
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  def test_export_saved_model_for_mode(self):
    """Tests new SavedModel exporting functionality."""

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    base_learner_builder_generator = SimpleBaseLearnerBuilderGenerator(
        [_DNNBaseLearnerBuilder("dnn")])
    estimator = Estimator(
        head=_head(),
        base_learner_builder_generator=base_learner_builder_generator,
        mixture_weight_initializer=tf.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=1,
        model_dir=self.test_subdirectory,
        config=run_config)

    features = {"x": [[1., 0.]]}
    labels = [[1.]]
    train_input_fn = _dummy_feature_dict_input_fn(features, labels)

    # Train.
    estimator.train(input_fn=train_input_fn, max_steps=2)

    # Export SavedModel.
    def serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      for key, value in features.items():
        features[key] = tf.constant(value)
      return tf.estimator.export.ServingInputReceiver(
          features=features, receiver_tensors=serialized_example)

    tf.contrib.estimator.export_saved_model_for_mode(
        estimator,
        export_dir_base=self.test_subdirectory,
        input_receiver_fn=serving_input_fn,
        mode=tf.estimator.ModeKeys.PREDICT)


class ExportSavedModelForEvalTest(parameterized.TestCase, tf.test.TestCase):
  """Tests b/110991908."""

  def setUp(self):
    # Setup and cleanup test directory.
    self.test_subdirectory = os.path.join(tf.flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  def tearDown(self):
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  def test_export_saved_model_for_mode(self):
    """Tests new SavedModel exporting functionality."""

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    base_learner_builder_generator = SimpleBaseLearnerBuilderGenerator(
        [_DNNBaseLearnerBuilder("dnn")])
    estimator = Estimator(
        head=_head(),
        base_learner_builder_generator=base_learner_builder_generator,
        mixture_weight_initializer=tf.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=1,
        model_dir=self.test_subdirectory,
        config=run_config)

    features = {"x": [[1., 0.]]}
    labels = [[1.]]
    train_input_fn = _dummy_feature_dict_input_fn(features, labels)

    # Train.
    estimator.train(input_fn=train_input_fn, max_steps=2)

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

    subdir = tf.gfile.ListDirectory(export_dir_base)[0]

    with self.test_session() as sess:
      meta_graph_def = tf.saved_model.loader.load(
          sess, ["eval"], os.path.join(export_dir_base, subdir))
      signature_def = meta_graph_def.signature_def.get("eval")

      # Read zero metric.
      self.assertAlmostEqual(
          0.,
          sess.run(
              tf.saved_model.utils.get_tensor_from_tensor_info(
                  signature_def.outputs["metrics/average_loss/value"])),
          places=3)

      # Run metric update op.
      sess.run(
          tf.saved_model.utils.get_tensor_from_tensor_info(
              signature_def.outputs["metrics/average_loss/update_op"]))

      # Read metric again; it should no longer be zero.
      self.assertAlmostEqual(
          .201,
          sess.run(
              tf.saved_model.utils.get_tensor_from_tensor_info(
                  signature_def.outputs["metrics/average_loss/value"])),
          places=3)


class ReportGenerationTest(parameterized.TestCase, tf.test.TestCase):
  """Tests report generation."""

  def setUp(self):
    # Setup and cleanup test directory.
    self.test_subdirectory = os.path.join(tf.flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.mkdir(self.test_subdirectory)

  def tearDown(self):
    shutil.rmtree(self.test_subdirectory, ignore_errors=False)

  @parameterized.named_parameters(
      {
          "testcase_name":
              "one_iteration_one_base_learner",
          "base_learner_builder_generator":
              SimpleBaseLearnerBuilderGenerator(
                  [_DNNBaseLearnerBuilder("dnn", layer_size=1)]),
          "num_iterations":
              1,
          "want_materialized_iteration_reports": [[
              MaterializedBaseLearnerReport(
                  hparams={"layer_size": 1},
                  attributes={
                      "name": "dnn",
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=False,  # unused by test
              ),
          ]],
      },
      {
          "testcase_name":
              "three_iterations_one_base_learner",
          "base_learner_builder_generator":
              SimpleBaseLearnerBuilderGenerator(
                  [_DNNBaseLearnerBuilder("dnn", layer_size=1)]),
          "num_iterations":
              3,
          "want_materialized_iteration_reports": [
              [
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 1},
                      attributes={
                          "name": "dnn",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=False,  # usused by test
                  )
              ],
              [
                  MaterializedBaseLearnerReport(
                      hparams={},
                      attributes={
                          "name": "previous_ensemble",
                      },
                      metrics={},
                      included_in_final_ensemble=False,  # unused by test
                  ),
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 1},
                      attributes={
                          "name": "dnn",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=False,  # unused by test
                  ),
              ],
              [
                  MaterializedBaseLearnerReport(
                      hparams={},
                      attributes={
                          "name": "previous_ensemble",
                      },
                      metrics={},
                      included_in_final_ensemble=False,  # unused by test
                  ),
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 1},
                      attributes={
                          "name": "dnn",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=False,  # unused by test
                  ),
              ],
          ],
      },
      {
          "testcase_name":
              "one_iteration_three_base_learners",
          "base_learner_builder_generator":
              SimpleBaseLearnerBuilderGenerator([
                  _DNNBaseLearnerBuilder("dnn_1", layer_size=1),
                  _DNNBaseLearnerBuilder("dnn_2", layer_size=2),
                  _DNNBaseLearnerBuilder("dnn_3", layer_size=3)
              ]),
          "num_iterations":
              1,
          "want_materialized_iteration_reports": [[
              MaterializedBaseLearnerReport(
                  hparams={"layer_size": 1},
                  attributes={
                      "name": "dnn_1",
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
              ),
              MaterializedBaseLearnerReport(
                  hparams={"layer_size": 2},
                  attributes={
                      "name": "dnn_2",
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
              ),
              MaterializedBaseLearnerReport(
                  hparams={"layer_size": 3},
                  attributes={
                      "name": "dnn_3",
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
              ),
          ]],
      },
      {
          "testcase_name":
              "three_iterations_three_base_learners",
          "base_learner_builder_generator":
              SimpleBaseLearnerBuilderGenerator([
                  _DNNBaseLearnerBuilder("dnn_1", layer_size=1),
                  _DNNBaseLearnerBuilder("dnn_2", layer_size=2),
                  _DNNBaseLearnerBuilder("dnn_3", layer_size=3)
              ]),
          "num_iterations":
              3,
          "want_materialized_iteration_reports": [
              [
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 1},
                      attributes={
                          "name": "dnn_1",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                  ),
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 2},
                      attributes={
                          "name": "dnn_2",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                  ),
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 3},
                      attributes={
                          "name": "dnn_3",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                  ),
              ],
              [
                  MaterializedBaseLearnerReport(
                      hparams={},
                      attributes={
                          "name": "previous_ensemble",
                      },
                      metrics={},
                  ),
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 1},
                      attributes={
                          "name": "dnn_1",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                  ),
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 2},
                      attributes={
                          "name": "dnn_2",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                  ),
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 3},
                      attributes={
                          "name": "dnn_3",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                  ),
              ],
              [
                  MaterializedBaseLearnerReport(
                      hparams={},
                      attributes={
                          "name": "previous_ensemble",
                      },
                      metrics={},
                  ),
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 1},
                      attributes={
                          "name": "dnn_1",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                  ),
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 2},
                      attributes={
                          "name": "dnn_2",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                  ),
                  MaterializedBaseLearnerReport(
                      hparams={"layer_size": 3},
                      attributes={
                          "name": "dnn_3",
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                  ),
              ],
          ],
      })
  def testReportGeneration(self, base_learner_builder_generator, num_iterations,
                           want_materialized_iteration_reports):
    max_iteration_steps = 10
    max_steps = max_iteration_steps * num_iterations

    train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])
    estimator = Estimator(
        head=_head(),
        base_learner_builder_generator=base_learner_builder_generator,
        mixture_weight_initializer=tf.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=max_iteration_steps,
        report_materializer=ReportMaterializer(
            input_fn=train_input_fn, steps=10),
        model_dir=self.test_subdirectory)

    report_accessor = estimator._report_accessor

    estimator.train(input_fn=train_input_fn, max_steps=max_steps)

    materialized_iteration_reports = list(
        report_accessor.read_iteration_reports())
    self.assertEqual(num_iterations, len(materialized_iteration_reports))
    for i in range(num_iterations):
      want_materialized_base_learner_reports = (
          want_materialized_iteration_reports[i])
      materialized_base_learner_reports = materialized_iteration_reports[i]
      self.assertEqual(
          len(want_materialized_base_learner_reports),
          len(materialized_base_learner_reports))
      for (want_materialized_base_learner_report,
           materialized_base_learner_report) in zip(
               want_materialized_base_learner_reports,
               materialized_base_learner_reports):
        self.assertEqual(want_materialized_base_learner_report.hparams,
                         materialized_base_learner_report.hparams)
        for metric_key in want_materialized_base_learner_report.metrics:
          self.assertEqual(
              want_materialized_base_learner_report.metrics[metric_key],
              materialized_base_learner_report.metrics[metric_key])
        self.assertIn("adanet_loss", materialized_base_learner_report.metrics)

      # Compute argmin adanet loss.
      argmin_adanet_loss = 0
      smallest_known_adanet_loss = float("inf")
      for j, materialized_base_learner_report in enumerate(
          materialized_base_learner_reports):
        if (smallest_known_adanet_loss >
            materialized_base_learner_report.metrics["adanet_loss"]):
          smallest_known_adanet_loss = (
              materialized_base_learner_report.metrics["adanet_loss"])
          argmin_adanet_loss = j

      # Check that the base_learner with the lowest adanet loss is the one
      # that is included in the final ensemble.
      for j, materialized_base_learner_reports in enumerate(
          materialized_base_learner_reports):
        self.assertEqual(
            j == argmin_adanet_loss,
            materialized_base_learner_reports.included_in_final_ensemble)


if __name__ == "__main__":
  tf.test.main()
