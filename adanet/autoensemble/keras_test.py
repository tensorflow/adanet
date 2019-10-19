"""A Keras model that learns to ensemble.

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
from adanet.autoensemble.keras import AutoEnsemble
import tensorflow as tf
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator.head import regression_head


class KerasTest(parameterized.TestCase, tf.test.TestCase):

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      {
          "testcase_name": "dict_candidate_pool",
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
      })
  # pylint: enable=g-long-lambda

  @test_util.run_in_graph_and_eager_modes
  def test_auto_ensemble_lifecycle(self,
                                   candidate_pool):

    optimizer = lambda: tf.keras.optimizers.SGD(lr=.01)
    feature_columns = [tf.feature_column.numeric_column("x", shape=[2])]

    keras_model = AutoEnsemble(
        candidate_pool=candidate_pool(regression_head.RegressionHead(),
                                      feature_columns, optimizer),
        max_iteration_steps=10)
    keras_model.compile(loss="mse")
    self.assertEqual(["loss"], keras_model.metrics_names)

    train_data = lambda: tf.data.Dataset.from_tensors((  # pylint: disable=g-long-lambda
        {"x": [[1., 0.]]}, [[1.]])).repeat()
    keras_model.fit(train_data, epochs=1, steps_per_epoch=1)

    eval_results = keras_model.evaluate(train_data, steps=3)
    # TODO: Rewrite this test to be deterministic.
    self.assertIsNotNone(eval_results["loss"])

    predict_data = lambda: tf.data.Dataset.from_tensors(({"x": [[1., 0.]]}))
    predictions = keras_model.predict(predict_data)
    self.assertLen(predictions, 1)

if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
