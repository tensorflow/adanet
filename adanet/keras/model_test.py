"""An AdaNet Keras model implementation.

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

from absl.testing import parameterized
from adanet import tf_compat
from adanet.core import testing_utils as tu
from adanet.keras import model
from adanet.subnetwork import Builder
from adanet.subnetwork import SimpleGenerator
from adanet.subnetwork import Subnetwork
from adanet.subnetwork import TrainOpSpec
import tensorflow as tf
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

XOR_FEATURES = [[1., 0.], [0., 0.], [0., 1.], [1., 1.]]
XOR_LABELS = [[1.], [0.], [1.], [0.]]
XOR_CLASS_LABELS = [[1], [0], [1], [0]]


# TODO: Refactor this class into testing_utils.py to use within
#                   estimator_test.py and model_test.py.
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


class ModelTest(tu.AdanetTestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "one_step_binary_crossentropy_loss",
          "loss": "binary_crossentropy",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 1,
          "epochs": 1,
          "steps_per_epoch": 3,
          "want_loss": 0.7690,
      },
      {
          "testcase_name": "one_step_mse_loss",
          "loss": "mse",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 1,
          "epochs": 1,
          "steps_per_epoch": 3,
          "want_loss": 0.6354,
      },
      {
          "testcase_name": "one_step_sparse_categorical_crossentropy_loss",
          "loss": "sparse_categorical_crossentropy",
          "subnetwork_generator": SimpleGenerator([_DNNBuilder("dnn")]),
          "max_iteration_steps": 1,
          "epochs": 1,
          "steps_per_epoch": 3,
          "want_loss": 1.2521,
          "logits_dimension": 3,
          "dataset": lambda: tf.data.Dataset.from_tensors(({"x": XOR_FEATURES},  # pylint: disable=g-long-lambda
                                                           XOR_CLASS_LABELS))
      })
  @test_util.run_in_graph_and_eager_modes
  def test_lifecycle(self,
                     loss,
                     subnetwork_generator,
                     max_iteration_steps,
                     want_loss,
                     logits_dimension=1,
                     ensemblers=None,
                     ensemble_strategies=None,
                     evaluator=None,
                     adanet_loss_decay=0.9,
                     dataset=None,
                     epochs=None,
                     steps_per_epoch=None):

    keras_model = model.Model(
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        logits_dimension=logits_dimension,
        ensemblers=ensemblers,
        ensemble_strategies=ensemble_strategies,
        evaluator=evaluator,
        adanet_loss_decay=adanet_loss_decay,
        filepath=self.test_subdirectory)

    keras_model.compile(loss=loss)
    # Make sure we have access to metrics_names immediately after compilation.
    self.assertEqual(["loss"], keras_model.metrics_names)

    if dataset is None:
      dataset = lambda: tf.data.Dataset.from_tensors(  # pylint: disable=g-long-lambda
          ({"x": XOR_FEATURES}, XOR_LABELS)).repeat()

    keras_model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

    eval_results = keras_model.evaluate(dataset, steps=3)
    self.assertAlmostEqual(want_loss, eval_results["loss"], places=3)

    # TODO: Predict not currently working for BinaryClassHead and
    #                   MultiClassHead.
    if loss == "mse":
      prediction_data = lambda: tf.data.Dataset.from_tensors(({  # pylint: disable=g-long-lambda
          "x": XOR_FEATURES
      }))

      predictions = keras_model.predict(prediction_data)
      self.assertLen(predictions, 4)

  @test_util.run_in_graph_and_eager_modes
  def test_compile_exceptions(self):
    keras_model = model.Model(
        subnetwork_generator=SimpleGenerator([_DNNBuilder("dnn")]),
        max_iteration_steps=1)
    train_data = tf.data.Dataset.from_tensors(([[1., 1.]], [[1.]]))
    predict_data = tf.data.Dataset.from_tensors(([[1., 1.]]))

    with self.assertRaises(RuntimeError):
      keras_model.fit(train_data)

    with self.assertRaises(RuntimeError):
      keras_model.evaluate(train_data)

    with self.assertRaises(RuntimeError):
      keras_model.predict(predict_data)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
