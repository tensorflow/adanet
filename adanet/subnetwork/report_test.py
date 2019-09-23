"""Test AdaNet single graph subnetwork implementation.

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

from absl.testing import parameterized
from adanet.subnetwork.report import Report
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
# pylint: enable=g-direct-tensorflow-import


class ReportTest(parameterized.TestCase, tf.test.TestCase):

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      {
          "testcase_name": "empty",
          "hparams": {},
          "attributes": lambda: {},
          "metrics": lambda: {},
      }, {
          "testcase_name": "non_empty",
          "hparams": {
              "hoo": 1
          },
          "attributes": lambda: {
              "aoo": tf.constant(1)
          },
          "metrics": lambda: {
              "moo": (tf.constant(1), tf.constant(1))
          },
      }, {
          "testcase_name": "non_tensor_update_op",
          "hparams": {
              "hoo": 1
          },
          "attributes": lambda: {
              "aoo": tf.constant(1)
          },
          "metrics": lambda: {
              "moo": (tf.constant(1), tf.no_op())
          },
      })
  # pylint: enable=g-long-lambda
  @test_util.run_in_graph_and_eager_modes
  def test_new(self, hparams, attributes, metrics):
    with context.graph_mode():
      _ = tf.constant(0)  # Just to have a non-empty graph.
      report = Report(
          hparams=hparams, attributes=attributes(), metrics=metrics())
      self.assertEqual(hparams, report.hparams)
      self.assertEqual(
          self.evaluate(attributes()), self.evaluate(report.attributes))
      self.assertEqual(self.evaluate(metrics()), self.evaluate(report.metrics))

  @test_util.run_in_graph_and_eager_modes
  def test_drop_non_scalar_metric(self):
    """Tests b/118632346."""

    hparams = {"hoo": 1}
    attributes = {"aoo": tf.constant(1)}
    metrics = {
        "moo1": (tf.constant(1), tf.constant(1)),
        "moo2": (tf.constant([1, 1]), tf.constant([1, 1])),
    }
    want_metrics = metrics.copy()
    del want_metrics["moo2"]
    with self.test_session():
      report = Report(hparams=hparams, attributes=attributes, metrics=metrics)
      self.assertEqual(hparams, report.hparams)
      self.assertEqual(attributes, report.attributes)
      self.assertEqual(want_metrics, report.metrics)

  @parameterized.named_parameters(
      {
          "testcase_name": "tensor_hparams",
          "hparams": {
              "hoo": tf.constant(1)
          },
          "attributes": {},
          "metrics": {},
      }, {
          "testcase_name": "non_tensor_attributes",
          "hparams": {},
          "attributes": {
              "aoo": 1,
          },
          "metrics": {},
      }, {
          "testcase_name": "non_tuple_metrics",
          "hparams": {},
          "attributes": {},
          "metrics": {
              "moo": tf.constant(1)
          },
      }, {
          "testcase_name": "one_item_tuple_metrics",
          "hparams": {},
          "attributes": {},
          "metrics": {
              "moo": (tf.constant(1),)
          },
      })
  @test_util.run_in_graph_and_eager_modes
  def test_new_errors(self, hparams, attributes, metrics):
    with self.assertRaises(ValueError):
      Report(hparams=hparams, attributes=attributes, metrics=metrics)


if __name__ == "__main__":
  tf.test.main()
