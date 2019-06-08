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

import json
import os
import time

from absl import logging
from absl.testing import parameterized
from adanet import tf_compat
from adanet.core import testing_utils as tu
from adanet.core.estimator import Estimator
from adanet.core.evaluator import Evaluator
from adanet.core.report_materializer import ReportMaterializer
from adanet.ensemble import AllStrategy
from adanet.ensemble import GrowStrategy
from adanet.ensemble import MixtureWeightType
from adanet.ensemble import SoloStrategy
from adanet.subnetwork import Builder
from adanet.subnetwork import Generator
from adanet.subnetwork import MaterializedReport
from adanet.subnetwork import Report
from adanet.subnetwork import SimpleGenerator
from adanet.subnetwork import Subnetwork
from adanet.subnetwork import TrainOpSpec
import tensorflow as tf
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import multi_head as multi_head_lib
from tensorflow_estimator.python.estimator.head import regression_head

# Module path changed. Try importing from new and old location to maintain
# backwards compatibility.
# pylint: disable=g-import-not-at-top
try:
  from tensorflow_estimator.python.estimator.export import export
except ImportError:
  from tensorflow.python.estimator.export import export
# pylint: enable=g-import-not-at-top

logging.set_verbosity(logging.INFO)

XOR_FEATURES = [[1., 0.], [0., 0], [0., 1.], [1., 1.]]
XOR_LABELS = [[1.], [0.], [1.], [0.]]


class _DNNBuilder(Builder):
  """A simple DNN subnetwork builder."""

  def __init__(self,
               name,
               learning_rate=.001,
               mixture_weight_learning_rate=.001,
               return_penultimate_layer=True,
               layer_size=1,
               subnetwork_chief_hooks=None,
               subnetwork_hooks=None,
               mixture_weight_chief_hooks=None,
               mixture_weight_hooks=None,
               seed=13):
    self._name = name
    self._learning_rate = learning_rate
    self._mixture_weight_learning_rate = mixture_weight_learning_rate
    self._return_penultimate_layer = return_penultimate_layer
    self._layer_size = layer_size
    self._subnetwork_chief_hooks = subnetwork_chief_hooks
    self._subnetwork_hooks = subnetwork_hooks
    self._mixture_weight_chief_hooks = mixture_weight_chief_hooks
    self._mixture_weight_hooks = mixture_weight_hooks
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
    with tf_compat.v1.variable_scope("dnn"):
      persisted_tensors = {}
      with tf_compat.v1.variable_scope("hidden_layer"):
        w = tf_compat.v1.get_variable(
            shape=[2, self._layer_size],
            initializer=tf_compat.v1.glorot_uniform_initializer(seed=seed),
            name="weight")
        disjoint_op = tf.constant([1], name="disjoint_op")
        with tf_compat.v1.colocate_with(disjoint_op):  # tests b/118865235
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

    last_layer = hidden_layer

    with tf_compat.v1.variable_scope("logits"):
      logits = tf_compat.v1.layers.dense(
          hidden_layer,
          logits_dimension,
          kernel_initializer=tf_compat.v1.glorot_uniform_initializer(seed=seed))

    summary.scalar("scalar", 3)
    batch_size = features["x"].get_shape().as_list()[0]
    summary.image("image", tf.ones([batch_size, 3, 3, 1]))
    with tf_compat.v1.variable_scope("nested"):
      summary.scalar("scalar", 5)

    return Subnetwork(
        last_layer=last_layer if self._return_penultimate_layer else logits,
        logits=logits,
        complexity=3,
        persisted_tensors=persisted_tensors,
        shared=persisted_tensors)

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    optimizer = tf_compat.v1.train.GradientDescentOptimizer(
        learning_rate=self._learning_rate)
    train_op = optimizer.minimize(loss, var_list=var_list)
    if not self._subnetwork_hooks:
      return train_op
    return TrainOpSpec(train_op, self._subnetwork_chief_hooks,
                       self._subnetwork_hooks)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    optimizer = tf_compat.v1.train.GradientDescentOptimizer(
        learning_rate=self._mixture_weight_learning_rate)
    train_op = optimizer.minimize(loss, var_list=var_list)
    if not self._mixture_weight_hooks:
      return train_op
    return TrainOpSpec(train_op, self._mixture_weight_chief_hooks,
                       self._mixture_weight_hooks)

  def build_subnetwork_report(self):
    return Report(
        hparams={"layer_size": self._layer_size},
        attributes={"complexity": tf.constant(3, dtype=tf.int32)},
        metrics={
            "moo": (tf.constant(3,
                                dtype=tf.int32), tf.constant(3, dtype=tf.int32))
        })


class _SimpleBuilder(Builder):
  """A simple subnetwork builder that takes feature_columns."""

  def __init__(self, name, feature_columns, seed=42):
    self._name = name
    self._feature_columns = feature_columns
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
          features=features, feature_columns=self._feature_columns)
      last_layer = input_layer

    with tf_compat.v1.variable_scope("logits"):
      logits = tf_compat.v1.layers.dense(
          last_layer,
          logits_dimension,
          kernel_initializer=tf_compat.v1.glorot_uniform_initializer(seed=seed))

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

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    optimizer = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=.001)
    return optimizer.minimize(loss, var_list=var_list)


class _LinearBuilder(Builder):
  """A simple linear subnetwork builder."""

  def __init__(self, name, mixture_weight_learning_rate=.001, seed=42):
    self._name = name
    self._mixture_weight_learning_rate = mixture_weight_learning_rate
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

    logits = tf_compat.v1.layers.dense(
        features["x"],
        logits_dimension,
        kernel_initializer=tf_compat.v1.glorot_uniform_initializer(
            seed=self._seed))

    return Subnetwork(
        last_layer=features["x"],
        logits=logits,
        complexity=1,
        persisted_tensors={},
    )

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    optimizer = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=.001)
    return optimizer.minimize(loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    optimizer = tf_compat.v1.train.GradientDescentOptimizer(
        learning_rate=self._mixture_weight_learning_rate)
    return optimizer.minimize(loss, var_list=var_list)


class _FakeGenerator(Generator):
  """Generator that exposed generate_candidates' arguments."""

  def __init__(self, spy_fn, subnetwork_builders):
    """Checks the arguments passed to generate_candidates.

    Args:
      spy_fn: (iteration_number, previous_ensemble_reports, all_reports) -> ().
        Spies on the arguments passed to generate_candidates whenever it is
        called.
      subnetwork_builders: List of `Builder`s to return in every call to
        generate_candidates.
    """

    self._spy_fn = spy_fn
    self._subnetwork_builders = subnetwork_builders

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    """Spys on arguments passed in, then returns a fixed list of candidates."""

    del previous_ensemble  # unused
    self._spy_fn(iteration_number, previous_ensemble_reports, all_reports)
    return self._subnetwork_builders


class _WidthLimitingDNNBuilder(_DNNBuilder):
  """Limits the width of the previous_ensemble."""

  def __init__(self,
               name,
               learning_rate=.001,
               mixture_weight_learning_rate=.001,
               return_penultimate_layer=True,
               layer_size=1,
               width_limit=None,
               seed=13):
    if width_limit is not None and width_limit == 0:
      raise ValueError("width_limit must be at least 1 or None.")

    super(_WidthLimitingDNNBuilder,
          self).__init__(name, learning_rate, mixture_weight_learning_rate,
                         return_penultimate_layer, layer_size, seed)
    self._width_limit = width_limit

  def prune_previous_ensemble(self, previous_ensemble):
    indices = range(len(previous_ensemble.weighted_subnetworks))
    if self._width_limit is None:
      return indices
    if self._width_limit == 1:
      return []
    return indices[-self._width_limit + 1:]  # pylint: disable=invalid-unary-operand-type


class EstimatorTest(tu.AdanetTestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "one_step",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 1,
          "steps": 1,
          "max_steps": None,
          "want_loss": 0.49899703,
      },
      {
          "testcase_name": "none_max_iteration_steps",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": None,
          "steps": 300,
          "max_steps": None,
          "want_loss": 0.32487726,
      },
      {
          "testcase_name": "single_builder_max_steps",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 200,
          "max_steps": 300,
          "want_loss": 0.32420248,
      },
      {
          "testcase_name": "single_builder_steps",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 200,
          "steps": 300,
          "max_steps": None,
          "want_loss": 0.32420248,
      },
      {
          "testcase_name": "single_builder_no_bias",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 200,
          "use_bias": False,
          "want_loss": 0.496736,
      },
      {
          "testcase_name":
              "single_builder_subnetwork_hooks",
          "subnetwork_generator":
              SimpleGenerator([
                  _DNNBuilder(
                      "dnn",
                      subnetwork_chief_hooks=[
                          tu.ModifierSessionRunHook("chief_hook_var")
                      ],
                      subnetwork_hooks=[tu.ModifierSessionRunHook("hook_var")])
              ]),
          "max_iteration_steps":
              200,
          "use_bias":
              False,
          "want_loss":
              0.496736,
      },
      {
          "testcase_name":
              "single_builder_mixture_weight_hooks",
          "subnetwork_generator":
              SimpleGenerator([
                  _DNNBuilder(
                      "dnn",
                      mixture_weight_chief_hooks=[
                          tu.ModifierSessionRunHook("chief_hook_var")
                      ],
                      mixture_weight_hooks=[
                          tu.ModifierSessionRunHook("hook_var")
                      ])
              ]),
          "max_iteration_steps":
              200,
          "use_bias":
              False,
          "want_loss":
              0.496736,
      },
      {
          "testcase_name":
              "single_builder_scalar_mixture_weight",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn", return_penultimate_layer=False)]),
          "max_iteration_steps":
              200,
          "mixture_weight_type":
              MixtureWeightType.SCALAR,
          "want_loss":
              0.32317898,
      },
      {
          "testcase_name":
              "single_builder_vector_mixture_weight",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn", return_penultimate_layer=False)]),
          "max_iteration_steps":
              200,
          "mixture_weight_type":
              MixtureWeightType.VECTOR,
          "want_loss":
              0.32317898,
      },
      {
          "testcase_name": "single_builder_replicate_ensemble_in_training",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "replicate_ensemble_in_training": True,
          "max_iteration_steps": 200,
          "max_steps": 300,
          "want_loss": 0.32420215,
      },
      {
          "testcase_name": "single_builder_with_hook",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 200,
          "hooks": [tu.ModifierSessionRunHook()],
          "want_loss": 0.32420248,
      },
      {
          "testcase_name": "high_max_iteration_steps",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 500,
          "want_loss": 0.32487726,
      },
      {
          "testcase_name":
              "two_builders",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn"),
                   _DNNBuilder("dnn2", seed=99)]),
          "max_iteration_steps":
              200,
          "want_loss":
              0.27713922,
      },
      {
          "testcase_name":
              "two_builders_different_layer_sizes",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn"),
                   _DNNBuilder("dnn2", layer_size=3)]),
          "max_iteration_steps":
              200,
          "want_loss":
              0.29696745,
      },
      {
          "testcase_name":
              "two_builders_different_layer_sizes_three_iterations",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn"),
                   _DNNBuilder("dnn2", layer_size=3)]),
          "max_iteration_steps":
              100,
          "want_loss":
              0.26433355,
      },
      {
          "testcase_name":
              "width_limiting_builder_no_pruning",
          "subnetwork_generator":
              SimpleGenerator([_WidthLimitingDNNBuilder("no_pruning")]),
          "max_iteration_steps":
              75,
          "want_loss":
              0.32001898,
      },
      {
          "testcase_name":
              "width_limiting_builder_some_pruning",
          "subnetwork_generator":
              SimpleGenerator(
                  [_WidthLimitingDNNBuilder("some_pruning", width_limit=2)]),
          "max_iteration_steps":
              75,
          "want_loss":
              0.38592532,
      },
      {
          "testcase_name":
              "width_limiting_builder_prune_all",
          "subnetwork_generator":
              SimpleGenerator(
                  [_WidthLimitingDNNBuilder("prune_all", width_limit=1)]),
          "max_iteration_steps":
              75,
          "want_loss":
              0.43492866,
      },
      {
          "testcase_name":
              "width_limiting_builder_mixed",
          "subnetwork_generator":
              SimpleGenerator([
                  _WidthLimitingDNNBuilder("no_pruning"),
                  _WidthLimitingDNNBuilder("some_pruning", width_limit=2),
                  _WidthLimitingDNNBuilder("prune_all", width_limit=1)
              ]),
          "max_iteration_steps":
              75,
          "want_loss":
              0.32001898,
      },
      {
          "testcase_name":
              "evaluator_good_input",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn"),
                   _DNNBuilder("dnn2", layer_size=3)]),
          "evaluator":
              Evaluator(
                  input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=3),
          "max_iteration_steps":
              200,
          "want_loss":
              0.31241742,
      },
      {
          "testcase_name":
              "evaluator_bad_input",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn"),
                   _DNNBuilder("dnn2", layer_size=3)]),
          "evaluator":
              Evaluator(
                  input_fn=tu.dummy_input_fn([[1., 1.]], [[1.]]), steps=3),
          "max_iteration_steps":
              200,
          "want_loss":
              0.29696745,
      },
      {
          "testcase_name":
              "report_materializer",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn"),
                   _DNNBuilder("dnn2", layer_size=3)]),
          "report_materializer":
              ReportMaterializer(
                  input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=1),
          "max_iteration_steps":
              200,
          "want_loss":
              0.29696745,
      },
      {
          "testcase_name":
              "all_strategy",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn"),
                   _DNNBuilder("dnn2", layer_size=3)]),
          "ensemble_strategies": [AllStrategy()],
          "max_iteration_steps":
              200,
          "want_loss":
              0.29196805,
      },
      {
          "testcase_name":
              "solo_strategy",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn"),
                   _DNNBuilder("dnn2", layer_size=3)]),
          "ensemble_strategies": [SoloStrategy()],
          "max_iteration_steps":
              200,
          "want_loss":
              0.35249719,
      },
      {
          "testcase_name":
              "solo_strategy_three_iterations",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn"),
                   _DNNBuilder("dnn2", layer_size=3)]),
          "ensemble_strategies": [SoloStrategy()],
          "max_iteration_steps":
              100,
          "want_loss":
              0.36163166
      },
      {
          "testcase_name":
              "multi_ensemble_strategy",
          "subnetwork_generator":
              SimpleGenerator(
                  [_DNNBuilder("dnn"),
                   _DNNBuilder("dnn2", layer_size=3)]),
          "ensemble_strategies":
              [AllStrategy(), GrowStrategy(),
               SoloStrategy()],
          "max_iteration_steps":
              100,
          "want_loss":
              0.24838975,
      },
      {
          "testcase_name":
              "dataset_train_input_fn",
          "subnetwork_generator":
              SimpleGenerator([_DNNBuilder("dnn")]),
          # pylint: disable=g-long-lambda
          "train_input_fn":
              lambda: tf.data.Dataset.from_tensors(({
                  "x": XOR_FEATURES
              }, XOR_LABELS)).repeat(),
          # pylint: enable=g-long-lambda
          "max_iteration_steps":
              100,
          "want_loss":
              .32219219,
      })
  def test_lifecycle(self,
                     subnetwork_generator,
                     want_loss,
                     max_iteration_steps,
                     mixture_weight_type=MixtureWeightType.MATRIX,
                     evaluator=None,
                     use_bias=True,
                     replicate_ensemble_in_training=False,
                     hooks=None,
                     ensemble_strategies=None,
                     max_steps=300,
                     steps=None,
                     report_materializer=None,
                     train_input_fn=None):
    """Train entire estimator lifecycle using XOR dataset."""

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    estimator = Estimator(
        head=tu.head(),
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        mixture_weight_type=mixture_weight_type,
        mixture_weight_initializer=tf_compat.v1.zeros_initializer(),
        warm_start_mixture_weights=True,
        evaluator=evaluator,
        ensemble_strategies=ensemble_strategies,
        report_materializer=report_materializer,
        use_bias=use_bias,
        replicate_ensemble_in_training=replicate_ensemble_in_training,
        model_dir=self.test_subdirectory,
        config=run_config)

    if not train_input_fn:
      train_input_fn = tu.dummy_input_fn(XOR_FEATURES, XOR_LABELS)

    # Train.
    estimator.train(
        input_fn=train_input_fn, steps=steps, max_steps=max_steps, hooks=hooks)

    # Evaluate.
    eval_results = estimator.evaluate(
        input_fn=train_input_fn, steps=10, hooks=hooks)
    logging.info("%s", eval_results)
    self.assertAlmostEqual(want_loss, eval_results["loss"], places=3)
    self.assertEqual(max_steps or steps, eval_results["global_step"])

    # Predict.
    predictions = estimator.predict(
        input_fn=tu.dataset_input_fn(features=[0., 0.], labels=None))
    for prediction in predictions:
      self.assertIsNotNone(prediction["predictions"])

    # Export SavedModel.
    def serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf_compat.v1.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      return tf.estimator.export.ServingInputReceiver(
          features={"x": tf.constant([[0., 0.]], name="serving_x")},
          receiver_tensors=serialized_example)

    export_saved_model_fn = getattr(estimator, "export_saved_model", None)
    if not callable(export_saved_model_fn):
      export_saved_model_fn = estimator.export_savedmodel
    export_saved_model_fn(
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

    def train_input_fn():
      input_features = {
          "human_names": tf.constant([["alice"], ["bob"]], name="human_names")
      }
      input_labels = tf.constant([[1.], [0.]], name="starts_with_a")
      return input_features, input_labels

    report_materializer = ReportMaterializer(input_fn=train_input_fn, steps=1)
    estimator = Estimator(
        head=regression_head.RegressionHead(),
        subnetwork_generator=SimpleGenerator(
            [_SimpleBuilder(name="simple", feature_columns=[feature_column])]),
        report_materializer=report_materializer,
        mixture_weight_type=MixtureWeightType.MATRIX,
        mixture_weight_initializer=tf_compat.v1.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=1,
        use_bias=True,
        model_dir=self.test_subdirectory)

    estimator.train(input_fn=train_input_fn, max_steps=3)

  @parameterized.named_parameters(
      {
          "testcase_name": "no_subnetwork_generator",
          "subnetwork_generator": None,
          "max_iteration_steps": 100,
      }, {
          "testcase_name": "negative_max_iteration_steps",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": -1,
      }, {
          "testcase_name": "zero_max_iteration_steps",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 0,
      }, {
          "testcase_name": "steps_and_max_steps",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 1,
          "steps": 1,
          "max_steps": 1,
      }, {
          "testcase_name": "zero_steps",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 1,
          "steps": 0,
          "max_steps": None,
      })
  def test_train_error(self,
                       subnetwork_generator,
                       max_iteration_steps,
                       steps=None,
                       max_steps=10):
    report_materializer = ReportMaterializer(
        input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=1)
    with self.assertRaises(ValueError):
      estimator = Estimator(
          head=tu.head(),
          subnetwork_generator=subnetwork_generator,
          report_materializer=report_materializer,
          mixture_weight_type=MixtureWeightType.MATRIX,
          mixture_weight_initializer=tf_compat.v1.zeros_initializer(),
          warm_start_mixture_weights=True,
          max_iteration_steps=max_iteration_steps,
          use_bias=True,
          model_dir=self.test_subdirectory)
      train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])
      estimator.train(input_fn=train_input_fn, steps=steps, max_steps=max_steps)


class KerasCNNBuilder(Builder):
  """Builds a CNN subnetwork for AdaNet."""

  def __init__(self, learning_rate, seed=42):
    """Initializes a `SimpleCNNBuilder`.

    Args:
      learning_rate: The float learning rate to use.
      seed: The random seed.

    Returns:
      An instance of `SimpleCNNBuilder`.
    """
    self._learning_rate = learning_rate
    self._seed = seed

  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    """See `adanet.subnetwork.Builder`."""

    seed = self._seed
    if previous_ensemble:
      seed += len(previous_ensemble.weighted_subnetworks)
    images = list(features.values())[0]
    images = tf.reshape(images, [-1, 2, 2, 1])
    kernel_initializer = tf_compat.v1.keras.initializers.he_normal(seed=seed)
    x = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=1,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_initializer)(
            images)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        units=3, activation="relu", kernel_initializer=kernel_initializer)(
            x)
    logits = tf_compat.v1.layers.Dense(
        units=1, activation=None, kernel_initializer=kernel_initializer)(
            x)
    complexity = tf.constant(1)
    return Subnetwork(
        last_layer=x,
        logits=logits,
        complexity=complexity,
        persisted_tensors={})

  def build_subnetwork_train_op(self,
                                subnetwork,
                                loss,
                                var_list,
                                labels,
                                iteration_step,
                                summary,
                                previous_ensemble=None):
    optimizer = tf_compat.v1.train.GradientDescentOptimizer(self._learning_rate)
    return optimizer.minimize(loss=loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    return tf.no_op()

  @property
  def name(self):
    return "simple_cnn"


class EstimatorKerasLayersTest(tu.AdanetTestCase):

  def test_lifecycle(self):
    """Train entire estimator lifecycle using XOR dataset."""

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    estimator = Estimator(
        head=tu.head(),
        subnetwork_generator=SimpleGenerator(
            [KerasCNNBuilder(learning_rate=.001)]),
        max_iteration_steps=3,
        evaluator=Evaluator(
            input_fn=tu.dummy_input_fn([[1., 1., .1, .1]], [[0.]]), steps=3),
        model_dir=self.test_subdirectory,
        config=run_config)

    xor_features = [[1., 0., 1., 0.], [0., 0., 0., 0.], [0., 1., 0., 1.],
                    [1., 1., 1., 1.]]
    xor_labels = [[1.], [0.], [1.], [0.]]
    train_input_fn = tu.dummy_input_fn(xor_features, xor_labels)

    # Train.
    estimator.train(input_fn=train_input_fn, max_steps=9)

    # Evaluate.
    eval_results = estimator.evaluate(input_fn=train_input_fn, steps=3)
    logging.info("%s", eval_results)
    want_loss = 0.16915826
    if tf_compat.version_greater_or_equal("1.10.0"):
      # After TF v1.10.0 the loss computed from a neural network using Keras
      # layers changed, however it is not clear why.
      want_loss = 0.26195815
    self.assertAlmostEqual(want_loss, eval_results["loss"], places=3)

    # Predict.
    predictions = estimator.predict(
        input_fn=tu.dataset_input_fn(features=[0., 0., 0., 0.], labels=None))
    for prediction in predictions:
      self.assertIsNotNone(prediction["predictions"])

    # Export SavedModel.
    def serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf_compat.v1.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      return tf.estimator.export.ServingInputReceiver(
          features={"x": tf.constant([[0., 0., 0., 0.]], name="serving_x")},
          receiver_tensors=serialized_example)

    export_saved_model_fn = getattr(estimator, "export_saved_model", None)
    if not callable(export_saved_model_fn):
      export_saved_model_fn = estimator.export_savedmodel
    export_saved_model_fn(
        export_dir_base=self.test_subdirectory,
        serving_input_receiver_fn=serving_input_fn)


class MultiHeadBuilder(Builder):
  """Builds a subnetwork for AdaNet that uses dict labels."""

  def __init__(self, learning_rate=.001, split_logits=False, seed=42):
    """Initializes a `LabelsDictBuilder`.

    Args:
      learning_rate: The float learning rate to use.
      split_logits: Whether to return a dict of logits or a single concatenated
        logits `Tensor`.
      seed: The random seed.

    Returns:
      An instance of `MultiHeadBuilder`.
    """
    self._learning_rate = learning_rate
    self._split_logits = split_logits
    self._seed = seed

  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    """See `adanet.subnetwork.Builder`."""

    seed = self._seed
    if previous_ensemble:
      seed += len(previous_ensemble.weighted_subnetworks)
    kernel_initializer = tf_compat.v1.keras.initializers.he_normal(seed=seed)
    x = features["x"]
    logits = tf_compat.v1.layers.dense(
        x,
        units=logits_dimension,
        activation=None,
        kernel_initializer=kernel_initializer)
    if self._split_logits:
      # Return different logits, one for each head.
      logits1, logits2 = tf.split(logits, [1, 1], 1)
      logits = {
          "head1": logits1,
          "head2": logits2,
      }

    complexity = tf.constant(1)
    return Subnetwork(
        last_layer=logits,
        logits=logits,
        complexity=complexity,
        persisted_tensors={})

  def build_subnetwork_train_op(self,
                                subnetwork,
                                loss,
                                var_list,
                                labels,
                                iteration_step,
                                summary,
                                previous_ensemble=None):
    optimizer = tf_compat.v1.train.GradientDescentOptimizer(self._learning_rate)
    return optimizer.minimize(loss=loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    optimizer = tf_compat.v1.train.GradientDescentOptimizer(self._learning_rate)
    return optimizer.minimize(loss=loss, var_list=var_list)

  @property
  def name(self):
    return "multi_head"


def _tf_version_less_than_v1_10_0():
  # TODO: Figure out why this is needed for multi-head.
  return not tf_compat.version_greater_or_equal("1.10.0")


class EstimatorMultiHeadTest(tu.AdanetTestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "concatenated_logits",
          "builders": [MultiHeadBuilder()],
          "want_loss": 2.809 if _tf_version_less_than_v1_10_0() else 3.218,
      }, {
          "testcase_name": "split_logits",
          "builders": [MultiHeadBuilder(split_logits=True)],
          "want_loss": 2.814 if _tf_version_less_than_v1_10_0() else 3.224,
      })
  def test_lifecycle(self, builders, want_loss):
    """Train entire estimator lifecycle using XOR dataset."""

    run_config = tf.estimator.RunConfig(tf_random_seed=42)

    xor_features = [[1., 0., 1., 0.], [0., 0., 0., 0.], [0., 1., 0., 1.],
                    [1., 1., 1., 1.]]
    xor_labels = [[1.], [0.], [1.], [0.]]

    def train_input_fn():
      return {
          "x": tf.constant(xor_features)
      }, {
          "head1": tf.constant(xor_labels),
          "head2": tf.constant(xor_labels)
      }

    estimator = Estimator(
        head=multi_head_lib.MultiHead(heads=[
            regression_head.RegressionHead(
                name="head1", loss_reduction=tf_compat.SUM_OVER_BATCH_SIZE),
            regression_head.RegressionHead(
                name="head2", loss_reduction=tf_compat.SUM_OVER_BATCH_SIZE),
        ]),
        subnetwork_generator=SimpleGenerator(builders),
        max_iteration_steps=3,
        evaluator=Evaluator(input_fn=train_input_fn, steps=1),
        model_dir=self.test_subdirectory,
        config=run_config)

    # Train.
    estimator.train(input_fn=train_input_fn, max_steps=9)

    # Evaluate.
    eval_results = estimator.evaluate(input_fn=train_input_fn, steps=3)
    self.assertAlmostEqual(want_loss, eval_results["loss"], places=3)

    # Predict.
    predictions = estimator.predict(
        input_fn=tu.dataset_input_fn(features=[0., 0., 0., 0.], labels=None))
    for prediction in predictions:
      self.assertIsNotNone(prediction[("head1", "predictions")])
      self.assertIsNotNone(prediction[("head2", "predictions")])

    # Export SavedModel.
    def serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf_compat.v1.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      return tf.estimator.export.ServingInputReceiver(
          features={"x": tf.constant([[0., 0., 0., 0.]], name="serving_x")},
          receiver_tensors=serialized_example)

    export_saved_model_fn = getattr(estimator, "export_saved_model", None)
    if not callable(export_saved_model_fn):
      export_saved_model_fn = estimator.export_savedmodel
    export_saved_model_fn(
        export_dir_base=self.test_subdirectory,
        serving_input_receiver_fn=serving_input_fn)


class EstimatorCallingModelFnDirectlyTest(tu.AdanetTestCase):
  """Tests b/112108745. Warn users not to call model_fn directly."""

  def test_calling_model_fn_directly(self):
    subnetwork_generator = SimpleGenerator([_DNNBuilder("dnn")])
    report_materializer = ReportMaterializer(
        input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=1)
    estimator = Estimator(
        head=tu.head(),
        subnetwork_generator=subnetwork_generator,
        report_materializer=report_materializer,
        max_iteration_steps=3,
        use_bias=True,
        model_dir=self.test_subdirectory)
    model_fn = estimator.model_fn
    train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])
    tf_compat.v1.train.create_global_step()
    features, labels = train_input_fn()
    with self.assertRaises(UserWarning):
      model_fn(
          features=features,
          mode=tf.estimator.ModeKeys.TRAIN,
          labels=labels,
          config={})


class EstimatorCheckpointTest(tu.AdanetTestCase):
  """Tests estimator checkpoints."""

  @parameterized.named_parameters(
      {
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
    subnetwork_generator = SimpleGenerator([_DNNBuilder("dnn")])
    report_materializer = ReportMaterializer(
        input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=1)
    estimator = Estimator(
        head=tu.head(),
        subnetwork_generator=subnetwork_generator,
        report_materializer=report_materializer,
        mixture_weight_type=MixtureWeightType.MATRIX,
        mixture_weight_initializer=tf_compat.v1.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=max_iteration_steps,
        use_bias=True,
        config=config,
        model_dir=self.test_subdirectory)
    train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])
    estimator.train(input_fn=train_input_fn, max_steps=max_steps)

    checkpoints = tf.io.gfile.glob(
        os.path.join(self.test_subdirectory, "*.meta"))
    self.assertEqual(want_num_checkpoints, len(checkpoints))


def _check_eventfile_for_keyword(keyword, dir_):
  """Checks event files for the keyword."""

  tf_compat.v1.summary.FileWriterCache.clear()

  # Get last `Event` written.
  filenames = os.path.join(dir_, "events*")
  event_paths = tf.io.gfile.glob(filenames)
  if not event_paths:
    raise ValueError("Path '{}' not found.".format(filenames))

  for last_event in tf_compat.v1.train.summary_iterator(event_paths[-1]):
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


class _FakeMetric(object):
  """A fake metric."""

  def __init__(self, value, dtype):
    self._value = value
    self._dtype = dtype

  def to_metric(self):
    tensor = tf.convert_to_tensor(value=self._value, dtype=self._dtype)
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
        loss=tf.reduce_mean(input_tensor=labels - logits),
        eval_metric_ops=metric_ops,
        train_op=train_op_fn(1))


class EstimatorSummaryWriterTest(tu.AdanetTestCase):
  """Test that Tensorboard summaries get written correctly."""

  def test_summaries(self):
    """Tests that summaries are written to candidate directory."""

    run_config = tf.estimator.RunConfig(
        tf_random_seed=42, log_step_count_steps=2, save_summary_steps=2)
    subnetwork_generator = SimpleGenerator(
        [_DNNBuilder("dnn", mixture_weight_learning_rate=.001)])
    report_materializer = ReportMaterializer(
        input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=1)
    estimator = Estimator(
        head=tu.head(),
        subnetwork_generator=subnetwork_generator,
        report_materializer=report_materializer,
        mixture_weight_type=MixtureWeightType.MATRIX,
        mixture_weight_initializer=tf_compat.v1.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=10,
        use_bias=True,
        config=run_config,
        model_dir=self.test_subdirectory)
    train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])
    estimator.train(input_fn=train_input_fn, max_steps=3)

    ensemble_loss = 1.
    self.assertAlmostEqual(
        ensemble_loss,
        _check_eventfile_for_keyword("loss", self.test_subdirectory),
        places=3)
    self.assertIsNotNone(
        _check_eventfile_for_keyword("global_step/sec", self.test_subdirectory))
    self.assertEqual(
        0.,
        _check_eventfile_for_keyword("iteration/adanet/iteration",
                                     self.test_subdirectory))

    subnetwork_subdir = os.path.join(self.test_subdirectory,
                                     "subnetwork/t0_dnn")
    self.assertAlmostEqual(
        3., _check_eventfile_for_keyword("scalar", subnetwork_subdir), places=3)
    self.assertEqual((3, 3, 1),
                     _check_eventfile_for_keyword("image/image/0",
                                                  subnetwork_subdir))
    self.assertAlmostEqual(
        5.,
        _check_eventfile_for_keyword("nested/scalar", subnetwork_subdir),
        places=3)

    ensemble_subdir = os.path.join(
        self.test_subdirectory, "ensemble/t0_dnn_grow_complexity_regularized")
    self.assertAlmostEqual(
        ensemble_loss,
        _check_eventfile_for_keyword(
            "adanet_loss/adanet/adanet_weighted_ensemble", ensemble_subdir),
        places=3)
    self.assertAlmostEqual(
        0.,
        _check_eventfile_for_keyword(
            "complexity_regularization/adanet/adanet_weighted_ensemble",
            ensemble_subdir),
        places=3)
    self.assertAlmostEqual(
        0.,
        _check_eventfile_for_keyword(
            "mixture_weight_norms/adanet/"
            "adanet_weighted_ensemble/subnetwork_0", ensemble_subdir),
        places=3)

  @parameterized.named_parameters(
      {
          "testcase_name": "none_metrics",
          "head": _EvalMetricsHead(None),
          "want_summaries": [],
          "want_loss": -1.791,
      },
      {
          "testcase_name":
              "metrics_fn",
          "head":
              _EvalMetricsHead(None),
          # pylint: disable=g-long-lambda
          "metric_fn":
              lambda predictions: {
                  "avg": tf_compat.v1.metrics.mean(predictions)
              },
          # pylint: enable=g-long-lambda
          "want_summaries": ["avg"],
          "want_loss":
              -1.791,
      },
      {
          "testcase_name": "empty_metrics",
          "head": _EvalMetricsHead({}),
          "want_summaries": [],
          "want_loss": -1.791,
      },
      {
          "testcase_name":
              "evaluation_name",
          "head":
              _EvalMetricsHead({}),
          "evaluation_name":
              "continuous",
          "want_summaries": [],
          "want_loss":
              -1.791,
          "global_subdir":
              "eval_continuous",
          "subnetwork_subdir":
              "subnetwork/t0_dnn/eval_continuous",
          "ensemble_subdir":
              "ensemble/t0_dnn_grow_complexity_regularized/eval_continuous",
      },
      {
          "testcase_name":
              "regression_head",
          "head":
              regression_head.RegressionHead(
                  loss_reduction=tf_compat.SUM_OVER_BATCH_SIZE),
          "want_summaries": ["average_loss"],
          "want_loss":
              .256,
      },
      {
          "testcase_name":
              "binary_classification_head",
          "head":
              binary_class_head.BinaryClassHead(
                  loss_reduction=tf_compat.SUM_OVER_BATCH_SIZE),
          "learning_rate":
              .6,
          "want_summaries": ["average_loss", "accuracy", "recall"],
          "want_loss":
              0.122,
      },
      {
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
                          tf_compat.v1.Summary(value=[
                              tf_compat.v1.Summary.Value(
                                  tag="summary_tag", simple_value=1.)
                          ]).SerializeToString(), tf.string),
              }),
          "want_summaries": [
              "float32",
              "float64",
              "serialized_summary/0",
          ],
          "want_loss":
              -1.791,
      })
  def test_eval_metrics(
      self,
      head,
      want_loss,
      want_summaries,
      evaluation_name=None,
      metric_fn=None,
      learning_rate=.01,
      global_subdir="eval",
      subnetwork_subdir="subnetwork/t0_dnn/eval",
      ensemble_subdir="ensemble/t0_dnn_grow_complexity_regularized/eval"):
    """Test that AdaNet evaluation metrics get persisted correctly."""

    seed = 42
    run_config = tf.estimator.RunConfig(tf_random_seed=seed)
    subnetwork_generator = SimpleGenerator([
        _DNNBuilder(
            "dnn",
            learning_rate=learning_rate,
            mixture_weight_learning_rate=0.,
            layer_size=8,
            seed=seed)
    ])
    estimator = Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=100,
        metric_fn=metric_fn,
        config=run_config,
        model_dir=self.test_subdirectory)
    train_input_fn = tu.dummy_input_fn(XOR_FEATURES, XOR_LABELS)
    estimator.train(input_fn=train_input_fn, max_steps=100)

    metrics = estimator.evaluate(
        input_fn=train_input_fn, steps=1, name=evaluation_name)
    self.assertAlmostEqual(want_loss, metrics["loss"], places=3)

    global_subdir = os.path.join(self.test_subdirectory, global_subdir)
    subnetwork_subdir = os.path.join(self.test_subdirectory, subnetwork_subdir)
    ensemble_subdir = os.path.join(self.test_subdirectory, ensemble_subdir)
    self.assertAlmostEqual(
        want_loss,
        _check_eventfile_for_keyword("loss", subnetwork_subdir),
        places=3)
    for metric in want_summaries:
      self.assertIsNotNone(
          _check_eventfile_for_keyword(metric, subnetwork_subdir),
          msg="{} should be under 'eval'.".format(metric))
    for dir_ in [global_subdir, ensemble_subdir]:
      self.assertAlmostEqual(metrics["loss"],
                             _check_eventfile_for_keyword("loss", dir_))
      self.assertEqual([b"| dnn |"],
                       _check_eventfile_for_keyword(
                           "architecture/adanet/ensembles/0", dir_))
      for metric in want_summaries:
        self.assertTrue(
            _check_eventfile_for_keyword(metric, dir_) > 0.,
            msg="{} should be under 'eval'.".format(metric))


class EstimatorMembersOverrideTest(tu.AdanetTestCase):
  """Tests b/77494544 fix."""

  def test_assert_members_are_not_overridden(self):
    """Assert that AdaNet estimator does not break other estimators."""

    config = tf.estimator.RunConfig()
    subnetwork_generator = SimpleGenerator([_DNNBuilder("dnn")])
    report_materializer = ReportMaterializer(
        input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=1)
    adanet = Estimator(
        head=tu.head(),
        subnetwork_generator=subnetwork_generator,
        report_materializer=report_materializer,
        mixture_weight_type=MixtureWeightType.MATRIX,
        mixture_weight_initializer=tf_compat.v1.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=10,
        use_bias=True,
        config=config)
    self.assertIsNotNone(adanet)
    if hasattr(tf.estimator, "LinearEstimator"):
      estimator_fn = tf.estimator.LinearEstimator
    else:
      estimator_fn = tf.contrib.estimator.LinearEstimator
    linear = estimator_fn(
        head=tu.head(), feature_columns=[tf.feature_column.numeric_column("x")])
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


class EstimatorDifferentFeaturesPerModeTest(tu.AdanetTestCase):
  """Tests b/109751254."""

  @parameterized.named_parameters(
      {
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
    subnetwork_generator = SimpleGenerator([_DNNBuilder("dnn")])
    report_materializer = ReportMaterializer(
        input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=1)
    estimator = Estimator(
        head=tu.head(),
        subnetwork_generator=subnetwork_generator,
        report_materializer=report_materializer,
        mixture_weight_type=MixtureWeightType.MATRIX,
        mixture_weight_initializer=tf_compat.v1.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=1,
        use_bias=True,
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
      serialized_example = tf_compat.v1.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      features = {}
      for key, value in predict_features.items():
        features[key] = tf.constant(value)
      return tf.estimator.export.ServingInputReceiver(
          features=features, receiver_tensors=serialized_example)

    export_saved_model_fn = getattr(estimator, "export_saved_model", None)
    if not callable(export_saved_model_fn):
      export_saved_model_fn = estimator.export_savedmodel
    export_saved_model_fn(
        export_dir_base=self.test_subdirectory,
        serving_input_receiver_fn=serving_input_fn)


class EstimatorExportSavedModelForPredictTest(tu.AdanetTestCase):
  """Tests b/110435640."""

  def test_export_saved_model_for_predict(self):
    """Tests new SavedModel exporting functionality."""

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    subnetwork_generator = SimpleGenerator([_DNNBuilder("dnn")])
    report_materializer = ReportMaterializer(
        input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=1)
    estimator = Estimator(
        head=tu.head(),
        subnetwork_generator=subnetwork_generator,
        report_materializer=report_materializer,
        mixture_weight_type=MixtureWeightType.MATRIX,
        mixture_weight_initializer=tf_compat.v1.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=1,
        use_bias=True,
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
      serialized_example = tf_compat.v1.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      for key, value in features.items():
        features[key] = tf.constant(value)
      return tf.estimator.export.ServingInputReceiver(
          features=features, receiver_tensors=serialized_example)

    estimator.export_saved_model(
        export_dir_base=self.test_subdirectory,
        serving_input_receiver_fn=serving_input_fn,
        experimental_mode=tf.estimator.ModeKeys.PREDICT)


class EstimatorExportSavedModelForEvalTest(tu.AdanetTestCase):
  """Tests b/110991908."""

  def test_export_saved_model_for_eval(self):
    """Tests new SavedModel exporting functionality."""

    if _tf_version_less_than_v1_10_0():
      self.skipTest("export_saved_model_for_eval is not "
                    "supported before TF v1.10.0.")

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    subnetwork_generator = SimpleGenerator(
        [_DNNBuilder("dnn", layer_size=8, learning_rate=1.)])
    estimator = Estimator(
        head=binary_class_head.BinaryClassHead(),
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=100,
        model_dir=self.test_subdirectory,
        config=run_config)

    train_input_fn = tu.dummy_input_fn(XOR_FEATURES, XOR_LABELS)

    # Train.
    estimator.train(input_fn=train_input_fn, max_steps=300)

    metrics = estimator.evaluate(input_fn=train_input_fn, steps=1)
    self.assertAlmostEqual(.067, metrics["average_loss"], places=3)
    self.assertAlmostEqual(1., metrics["recall"], places=3)
    self.assertAlmostEqual(1., metrics["accuracy"], places=3)

    # Export SavedModel.
    def serving_input_fn():
      """Input fn for serving export, starting from serialized example."""
      serialized_example = tf_compat.v1.placeholder(
          dtype=tf.string, shape=(None), name="serialized_example")
      return export.SupervisedInputReceiver(
          features={"x": tf.constant(XOR_FEATURES)},
          labels=tf.constant(XOR_LABELS),
          receiver_tensors=serialized_example)

    export_dir_base = os.path.join(self.test_subdirectory, "export")
    try:
      estimator.export_saved_model(
          export_dir_base=export_dir_base,
          serving_input_receiver_fn=serving_input_fn,
          experimental_mode=tf.estimator.ModeKeys.EVAL)
    except AttributeError:
      pass

    try:
      tf.contrib.estimator.export_saved_model_for_mode(
          estimator,
          export_dir_base=export_dir_base,
          input_receiver_fn=serving_input_fn,
          mode=tf.estimator.ModeKeys.EVAL)
    except AttributeError:
      pass

    subdir = tf.io.gfile.listdir(export_dir_base)[0]

    with self.test_session() as sess:
      meta_graph_def = tf_compat.v1.saved_model.loader.load(
          sess, ["eval"], os.path.join(export_dir_base, subdir))
      signature_def = meta_graph_def.signature_def.get("eval")

      # Read zero metric.
      self.assertAlmostEqual(
          0.,
          sess.run(
              tf_compat.v1.saved_model.utils.get_tensor_from_tensor_info(
                  signature_def.outputs["metrics/average_loss/value"])),
          places=3)

      # Run metric update op.
      sess.run((tf_compat.v1.saved_model.utils.get_tensor_from_tensor_info(
          signature_def.outputs["metrics/average_loss/update_op"]),
                tf_compat.v1.saved_model.utils.get_tensor_from_tensor_info(
                    signature_def.outputs["metrics/accuracy/update_op"]),
                tf_compat.v1.saved_model.utils.get_tensor_from_tensor_info(
                    signature_def.outputs["metrics/recall/update_op"])))

      # Read metric again; it should no longer be zero.
      self.assertAlmostEqual(
          0.067,
          sess.run(
              tf_compat.v1.saved_model.utils.get_tensor_from_tensor_info(
                  signature_def.outputs["metrics/average_loss/value"])),
          places=3)
      self.assertAlmostEqual(
          1.,
          sess.run(
              tf_compat.v1.saved_model.utils.get_tensor_from_tensor_info(
                  signature_def.outputs["metrics/recall/value"])),
          places=3)

      self.assertAlmostEqual(
          1.,
          sess.run(
              tf_compat.v1.saved_model.utils.get_tensor_from_tensor_info(
                  signature_def.outputs["metrics/accuracy/value"])),
          places=3)


class EstimatorReportTest(tu.AdanetTestCase):
  """Tests report generation and usage."""

  def compare_report_lists(self, report_list1, report_list2):
    # Essentially assertEqual(report_list1, report_list2), but ignoring
    # the "metrics" attribute.

    def make_qualified_name(iteration_number, name):
      return "iteration_{}/{}".format(iteration_number, name)

    report_dict_1 = {
        make_qualified_name(report.iteration_number, report.name): report
        for report in report_list1
    }
    report_dict_2 = {
        make_qualified_name(report.iteration_number, report.name): report
        for report in report_list2
    }

    self.assertEqual(len(report_list1), len(report_list2))

    for qualified_name in report_dict_1.keys():
      report_1 = report_dict_1[qualified_name]
      report_2 = report_dict_2[qualified_name]
      self.assertEqual(
          report_1.hparams,
          report_2.hparams,
          msg="{} vs. {}".format(report_1, report_2))
      self.assertEqual(
          report_1.attributes,
          report_2.attributes,
          msg="{} vs. {}".format(report_1, report_2))
      self.assertEqual(
          report_1.included_in_final_ensemble,
          report_2.included_in_final_ensemble,
          msg="{} vs. {}".format(report_1, report_2))
      for metric_key, metric_value in report_1.metrics.items():
        self.assertEqual(
            metric_value,
            report_2.metrics[metric_key],
            msg="{} vs. {}".format(report_1, report_2))

  @parameterized.named_parameters(
      {
          "testcase_name": "one_iteration_one_subnetwork",
          "subnetwork_builders": [_DNNBuilder("dnn", layer_size=1),],
          "num_iterations": 1,
          "want_materialized_iteration_reports": [[
              MaterializedReport(
                  iteration_number=0,
                  name="dnn",
                  hparams={"layer_size": 1},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=True,
              ),
          ]],
          "want_previous_ensemble_reports": [],
          "want_all_reports": [],
      },
      {
          "testcase_name": "one_iteration_three_subnetworks",
          "subnetwork_builders": [
              # learning_rate is set to 0 for all but one Builder
              # to make sure that only one of them can learn.
              _DNNBuilder(
                  "dnn_1",
                  layer_size=1,
                  learning_rate=0.,
                  mixture_weight_learning_rate=0.),
              _DNNBuilder(
                  "dnn_2",
                  layer_size=2,
                  learning_rate=0.,
                  mixture_weight_learning_rate=0.),
              # fixing the match for dnn_3 to win.
              _DNNBuilder("dnn_3", layer_size=3),
          ],
          "num_iterations": 1,
          "want_materialized_iteration_reports": [[
              MaterializedReport(
                  iteration_number=0,
                  name="dnn_1",
                  hparams={"layer_size": 1},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=False,
              ),
              MaterializedReport(
                  iteration_number=0,
                  name="dnn_2",
                  hparams={"layer_size": 2},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=False,
              ),
              MaterializedReport(
                  iteration_number=0,
                  name="dnn_3",
                  hparams={"layer_size": 3},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=True,
              ),
          ]],
          "want_previous_ensemble_reports": [],
          "want_all_reports": [],
      },
      {
          "testcase_name":
              "three_iterations_one_subnetwork",
          "subnetwork_builders": [_DNNBuilder("dnn", layer_size=1),],
          "num_iterations":
              3,
          "want_materialized_iteration_reports": [
              [
                  MaterializedReport(
                      iteration_number=0,
                      name="dnn",
                      hparams={"layer_size": 1},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=True,
                  )
              ],
              [
                  MaterializedReport(
                      iteration_number=1,
                      name="previous_ensemble",
                      hparams={},
                      attributes={},
                      metrics={},
                      included_in_final_ensemble=False,
                  ),
                  MaterializedReport(
                      iteration_number=1,
                      name="dnn",
                      hparams={"layer_size": 1},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=True,
                  ),
              ],
              [
                  MaterializedReport(
                      iteration_number=2,
                      name="previous_ensemble",
                      hparams={},
                      attributes={},
                      metrics={},
                      included_in_final_ensemble=False,
                  ),
                  MaterializedReport(
                      iteration_number=2,
                      name="dnn",
                      hparams={"layer_size": 1},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=True,
                  ),
              ],
          ],
          "want_previous_ensemble_reports": [
              MaterializedReport(
                  iteration_number=0,
                  name="dnn",
                  hparams={"layer_size": 1},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=True,
              ),
              MaterializedReport(
                  iteration_number=1,
                  name="dnn",
                  hparams={"layer_size": 1},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=True,
              ),
          ],
          "want_all_reports": [
              MaterializedReport(
                  iteration_number=0,
                  name="dnn",
                  hparams={"layer_size": 1},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=True,
              ),
              MaterializedReport(
                  iteration_number=1,
                  name="previous_ensemble",
                  hparams={},
                  attributes={},
                  metrics={},
                  included_in_final_ensemble=False,
              ),
              MaterializedReport(
                  iteration_number=1,
                  name="dnn",
                  hparams={"layer_size": 1},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=True,
              ),
          ],
      },
      {
          "testcase_name":
              "three_iterations_three_subnetworks",
          "subnetwork_builders": [
              # learning_rate is set to 0 for all but one Builder
              # to make sure that only one of them can learn.
              _DNNBuilder(
                  "dnn_1",
                  layer_size=1,
                  learning_rate=0.,
                  mixture_weight_learning_rate=0.),
              _DNNBuilder(
                  "dnn_2",
                  layer_size=2,
                  learning_rate=0.,
                  mixture_weight_learning_rate=0.),
              # fixing the match for dnn_3 to win in every iteration.
              _DNNBuilder("dnn_3", layer_size=3),
          ],
          "num_iterations":
              3,
          "want_materialized_iteration_reports": [
              [
                  MaterializedReport(
                      iteration_number=0,
                      name="dnn_1",
                      hparams={"layer_size": 1},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=False,
                  ),
                  MaterializedReport(
                      iteration_number=0,
                      name="dnn_2",
                      hparams={"layer_size": 2},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=False,
                  ),
                  MaterializedReport(
                      iteration_number=0,
                      name="dnn_3",
                      hparams={"layer_size": 3},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=True,
                  ),
              ],
              [
                  MaterializedReport(
                      iteration_number=1,
                      name="previous_ensemble",
                      hparams={},
                      attributes={},
                      metrics={},
                      included_in_final_ensemble=False,
                  ),
                  MaterializedReport(
                      iteration_number=1,
                      name="dnn_1",
                      hparams={"layer_size": 1},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=False,
                  ),
                  MaterializedReport(
                      iteration_number=1,
                      name="dnn_2",
                      hparams={"layer_size": 2},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=False,
                  ),
                  MaterializedReport(
                      iteration_number=1,
                      name="dnn_3",
                      hparams={"layer_size": 3},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=True,
                  ),
              ],
              [
                  MaterializedReport(
                      iteration_number=2,
                      name="previous_ensemble",
                      hparams={},
                      attributes={},
                      metrics={},
                      included_in_final_ensemble=False,
                  ),
                  MaterializedReport(
                      iteration_number=2,
                      name="dnn_1",
                      hparams={"layer_size": 1},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=False,
                  ),
                  MaterializedReport(
                      iteration_number=2,
                      name="dnn_2",
                      hparams={"layer_size": 2},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=False,
                  ),
                  MaterializedReport(
                      iteration_number=2,
                      name="dnn_3",
                      hparams={"layer_size": 3},
                      attributes={
                          "complexity": 3,
                      },
                      metrics={
                          "moo": 3,
                      },
                      included_in_final_ensemble=True,
                  ),
              ],
          ],
          "want_previous_ensemble_reports": [
              MaterializedReport(
                  iteration_number=0,
                  name="dnn_3",
                  hparams={"layer_size": 3},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=True,
              ),
              MaterializedReport(
                  iteration_number=1,
                  name="dnn_3",
                  hparams={"layer_size": 3},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=True,
              ),
          ],
          "want_all_reports": [
              MaterializedReport(
                  iteration_number=0,
                  name="dnn_1",
                  hparams={"layer_size": 1},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=False,
              ),
              MaterializedReport(
                  iteration_number=0,
                  name="dnn_2",
                  hparams={"layer_size": 2},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=False,
              ),
              MaterializedReport(
                  iteration_number=0,
                  name="dnn_3",
                  hparams={"layer_size": 3},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=True,
              ),
              MaterializedReport(
                  iteration_number=1,
                  name="previous_ensemble",
                  hparams={},
                  attributes={},
                  metrics={},
                  included_in_final_ensemble=False,
              ),
              MaterializedReport(
                  iteration_number=1,
                  name="dnn_1",
                  hparams={"layer_size": 1},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=False,
              ),
              MaterializedReport(
                  iteration_number=1,
                  name="dnn_2",
                  hparams={"layer_size": 2},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=False,
              ),
              MaterializedReport(
                  iteration_number=1,
                  name="dnn_3",
                  hparams={"layer_size": 3},
                  attributes={
                      "complexity": 3,
                  },
                  metrics={
                      "moo": 3,
                  },
                  included_in_final_ensemble=True,
              ),
          ],
      },
  )
  def test_report_generation_and_usage(self, subnetwork_builders,
                                       num_iterations,
                                       want_materialized_iteration_reports,
                                       want_previous_ensemble_reports,
                                       want_all_reports):
    # Stores the iteration_number, previous_ensemble_reports and all_reports
    # arguments in the self._iteration_reports dictionary, overwriting what
    # was seen in previous iterations.
    spied_iteration_reports = {}

    def _spy_fn(iteration_number, previous_ensemble_reports, all_reports):
      spied_iteration_reports[iteration_number] = {
          "previous_ensemble_reports": previous_ensemble_reports,
          "all_reports": all_reports,
      }

    subnetwork_generator = _FakeGenerator(
        spy_fn=_spy_fn, subnetwork_builders=subnetwork_builders)

    max_iteration_steps = 5
    max_steps = max_iteration_steps * num_iterations + 1

    train_input_fn = tu.dummy_input_fn([[1., 0.]], [[1.]])
    estimator = Estimator(
        head=tu.head(),
        subnetwork_generator=subnetwork_generator,
        mixture_weight_type=MixtureWeightType.MATRIX,
        mixture_weight_initializer=tf_compat.v1.zeros_initializer(),
        warm_start_mixture_weights=True,
        max_iteration_steps=max_iteration_steps,
        use_bias=True,
        report_materializer=ReportMaterializer(
            input_fn=train_input_fn, steps=1),
        model_dir=self.test_subdirectory)

    report_accessor = estimator._report_accessor

    estimator.train(input_fn=train_input_fn, max_steps=max_steps)

    materialized_iteration_reports = list(
        report_accessor.read_iteration_reports())
    self.assertEqual(num_iterations, len(materialized_iteration_reports))
    for i in range(num_iterations):
      want_materialized_reports = (want_materialized_iteration_reports[i])
      materialized_reports = materialized_iteration_reports[i]
      self.compare_report_lists(want_materialized_reports, materialized_reports)

      # Compute argmin adanet loss.
      argmin_adanet_loss = 0
      smallest_known_adanet_loss = float("inf")
      for j, materialized_subnetwork_report in enumerate(materialized_reports):
        if (smallest_known_adanet_loss >
            materialized_subnetwork_report.metrics["adanet_loss"]):
          smallest_known_adanet_loss = (
              materialized_subnetwork_report.metrics["adanet_loss"])
          argmin_adanet_loss = j

      # Check that the subnetwork with the lowest adanet loss is the one
      # that is included in the final ensemble.
      for j, materialized_reports in enumerate(materialized_reports):
        self.assertEqual(j == argmin_adanet_loss,
                         materialized_reports.included_in_final_ensemble)

    # Check the arguments passed into the generate_candidates method of the
    # Generator.
    iteration_report = spied_iteration_reports[num_iterations - 1]
    self.compare_report_lists(want_previous_ensemble_reports,
                              iteration_report["previous_ensemble_reports"])
    self.compare_report_lists(want_all_reports, iteration_report["all_reports"])


class EstimatorForceGrowTest(tu.AdanetTestCase):
  """Tests the force_grow override.

  Uses linear subnetworks with the same seed. They will produce identical
  outputs, so unless the `force_grow` override is set, none of the new
  subnetworks will improve the AdaNet objective, and AdaNet will not add them to
  the ensemble.
  """

  @parameterized.named_parameters(
      {
          "testcase_name": "one_builder_no_force_grow",
          "builders":
              [_LinearBuilder("linear", mixture_weight_learning_rate=0.)],
          "force_grow": False,
          "want_subnetworks": 1,
      }, {
          "testcase_name": "one_builder",
          "builders":
              [_LinearBuilder("linear", mixture_weight_learning_rate=0.)],
          "force_grow": True,
          "want_subnetworks": 2,
      }, {
          "testcase_name": "two_builders",
          "builders": [
              _LinearBuilder("linear", mixture_weight_learning_rate=0.),
              _LinearBuilder("linear2", mixture_weight_learning_rate=0.)
          ],
          "force_grow": True,
          "want_subnetworks": 2,
      }, {
          "testcase_name":
              "two_builders_with_evaluator",
          "builders": [
              _LinearBuilder("linear", mixture_weight_learning_rate=0.),
              _LinearBuilder("linear2", mixture_weight_learning_rate=0.)
          ],
          "force_grow":
              True,
          "evaluator":
              Evaluator(
                  input_fn=tu.dummy_input_fn([[1., 1.]], [[0.]]), steps=1),
          "want_subnetworks":
              2,
      })
  def test_force_grow(self,
                      builders,
                      force_grow,
                      want_subnetworks,
                      evaluator=None):
    """Train entire estimator lifecycle using XOR dataset."""

    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    subnetwork_generator = SimpleGenerator(builders)
    estimator = Estimator(
        head=tu.head(),
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=1,
        evaluator=evaluator,
        force_grow=force_grow,
        model_dir=self.test_subdirectory,
        config=run_config)

    train_input_fn = tu.dummy_input_fn(XOR_FEATURES, XOR_LABELS)

    # Train for four iterations.
    estimator.train(input_fn=train_input_fn, max_steps=3)

    # Evaluate.
    eval_results = estimator.evaluate(input_fn=train_input_fn, steps=1)
    self.assertEqual(
        want_subnetworks,
        str(eval_results["architecture/adanet/ensembles"]).count(" linear "))


class EstimatorDebugTest(tu.AdanetTestCase):
  """Tests b/125483534. Detect NaNs in input_fns."""

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      {
          "testcase_name":
              "nan_features",
          "head":
              regression_head.RegressionHead(
                  name="y", loss_reduction=tf_compat.SUM_OVER_BATCH_SIZE),
          "input_fn":
              lambda: ({
                  "x": tf.math.log([[1., 0.]])
              }, tf.zeros([1, 1]))
      }, {
          "testcase_name":
              "nan_label",
          "head":
              regression_head.RegressionHead(
                  name="y", loss_reduction=tf_compat.SUM_OVER_BATCH_SIZE),
          "input_fn":
              lambda: ({
                  "x": tf.ones([1, 2])
              }, tf.math.log([[0.]]))
      }, {
          "testcase_name":
              "nan_labels_dict",
          "head":
              multi_head_lib.MultiHead(heads=[
                  regression_head.RegressionHead(
                      name="y", loss_reduction=tf_compat.SUM_OVER_BATCH_SIZE),
              ]),
          "input_fn":
              lambda: ({
                  "x": tf.ones([1, 2])
              }, {
                  "y": tf.math.log([[0.]])
              })
      })
  # pylint: enable=g-long-lambda
  def test_nans_from_input_fn(self, head, input_fn):
    subnetwork_generator = SimpleGenerator([_DNNBuilder("dnn")])
    estimator = Estimator(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=3,
        model_dir=self.test_subdirectory,
        debug=True)
    with self.assertRaises(tf.errors.InvalidArgumentError):
      estimator.train(input_fn=input_fn, max_steps=3)


class EstimatorEvaluateDuringTrainHookTest(tu.AdanetTestCase):
  """Tests b/129000842 with a hook that calls estimator.evaluate()."""

  def test_train(self):
    run_config = tf.estimator.RunConfig(tf_random_seed=42)
    subnetwork_generator = SimpleGenerator([_DNNBuilder("dnn")])
    estimator = Estimator(
        head=tu.head(),
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=1,
        model_dir=self.test_subdirectory,
        config=run_config)

    train_input_fn = tu.dummy_input_fn(XOR_FEATURES, XOR_LABELS)

    class EvalTrainHook(tf.estimator.SessionRunHook):

      def end(self, session):
        estimator.evaluate(input_fn=train_input_fn, steps=1)

    # This should not infinite loop.
    estimator.train(
        input_fn=train_input_fn, max_steps=3, hooks=[EvalTrainHook()])


class EstimatorTFLearnRunConfigTest(tu.AdanetTestCase):
  """Tests b/129483642 for tf.contrib.learn.RunConfig.

  Checks that TF_CONFIG is overwritten correctly when no cluster is specified
  in the RunConfig and the only task is of type chief.
  """

  def test_train(self):
    try:
      tf.contrib.learn.RunConfig(tf_random_seed=42)
    except AttributeError:
      self.skipTest("There is no tf.contrib in TF 2.0.")

    try:
      tf_config = {
          "task": {
              "type": "chief",
              "index": 0
          },
      }
      os.environ["TF_CONFIG"] = json.dumps(tf_config)
      run_config = tf.contrib.learn.RunConfig(tf_random_seed=42)
      run_config._is_chief = True  # pylint: disable=protected-access

      subnetwork_generator = SimpleGenerator([_DNNBuilder("dnn")])
      estimator = Estimator(
          head=tu.head(),
          subnetwork_generator=subnetwork_generator,
          max_iteration_steps=1,
          model_dir=self.test_subdirectory,
          config=run_config)
      train_input_fn = tu.dummy_input_fn(XOR_FEATURES, XOR_LABELS)

      # Will fail if TF_CONFIG is not overwritten correctly in
      # Estimator#prepare_next_iteration.
      estimator.train(input_fn=train_input_fn, max_steps=3)
    finally:
      # Revert TF_CONFIG environment variable in order to not break other tests.
      del os.environ["TF_CONFIG"]



if __name__ == "__main__":
  tf.test.main()
