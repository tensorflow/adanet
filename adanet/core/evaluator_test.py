"""Test AdaNet evaluator single graph implementation.

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
from adanet.core.evaluator import Evaluator
import adanet.core.testing_utils as tu
import numpy as np
import tensorflow as tf


def _fake_adanet_losses_0(input_fn):
  _, labels = input_fn()
  return [
      tf.reduce_sum(labels),
      tf.reduce_sum(labels * 2),
  ]


def _fake_adanet_losses_1(input_fn):
  _, labels = input_fn()
  return [
      tf.reduce_sum(labels * 2),
      tf.reduce_sum(labels),
  ]


class EvaluatorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "choose_index_0",
      "input_fn": tu.dummy_input_fn([[1., 2]], [[3.]]),
      "steps": 3,
      "adanet_losses": _fake_adanet_losses_0,
      "want_adanet_losses": [3, 6],
  }, {
      "testcase_name": "choose_index_1",
      "input_fn": tu.dummy_input_fn([[1., 2]], [[3.]]),
      "steps": 3,
      "adanet_losses": _fake_adanet_losses_1,
      "want_adanet_losses": [6, 3],
  }, {
      "testcase_name": "none_steps",
      "input_fn": tu.dataset_input_fn(),
      "steps": None,
      "adanet_losses": _fake_adanet_losses_1,
      "want_adanet_losses": [18, 9],
  }, {
      "testcase_name": "input_fn_out_of_range",
      "input_fn": tu.dataset_input_fn(),
      "steps": 3,
      "adanet_losses": _fake_adanet_losses_1,
      "want_adanet_losses": [18, 9],
  })
  def test_evaluate_no_metric_fn_falls_back_to_adanet_losses(
      self, input_fn, steps, adanet_losses, want_adanet_losses):
    adanet_losses = adanet_losses(input_fn)
    metrics = [{"adanet_loss": tf.metrics.mean(loss)} for loss in adanet_losses]
    with self.test_session() as sess:
      evaluator = Evaluator(input_fn=input_fn, steps=steps)
      adanet_losses = evaluator.evaluate(sess, ensemble_metrics=metrics)
      self.assertEqual(want_adanet_losses, adanet_losses)

  @parameterized.named_parameters(
      {
          "testcase_name": "minimize_returns_nanargmin",
          "objective": Evaluator.Objective.MAXIMIZE,
          "expected_objective_fn": np.nanargmax,
          "metric_fn": lambda x, y: None
      }, {
          "testcase_name": "maximize_returns_nanargmax",
          "objective": Evaluator.Objective.MINIMIZE,
          "expected_objective_fn": np.nanargmin,
          "metric_fn": lambda x, y: None
      })
  def test_objective(self, objective, expected_objective_fn, metric_fn=None):
    evaluator = Evaluator(input_fn=None, objective=objective)
    self.assertEqual(expected_objective_fn, evaluator.objective_fn)

  def test_objective_unsupported_objective(self):
    with self.assertRaises(ValueError):
      Evaluator(input_fn=None, objective="non_existent_objective")

  def test_evaluate(self):

    input_fn = tu.dummy_input_fn([[1., 2]], [[3.]])
    _, labels = input_fn()
    predictions = [labels * 2, labels * 3]
    metrics = []
    for preds in predictions:
      metrics.append({
          "mse": tf.metrics.mean_squared_error(labels, preds),
          "other_metric_1": (tf.constant(1), tf.constant(1)),
          "other_metric_2": (tf.constant(2), tf.constant(2))
      })

    with self.test_session() as sess:
      evaluator = Evaluator(input_fn=input_fn, metric_name="mse", steps=3)
      metrics = evaluator.evaluate(sess, ensemble_metrics=metrics)
      self.assertEqual([9, 36], metrics)

  def test_evaluate_invalid_metric(self):

    input_fn = tu.dummy_input_fn([[1., 2]], [[3.]])
    _, labels = input_fn()
    predictions = [labels * 2, labels * 3]
    metrics = []
    for preds in predictions:
      metrics.append({
          "mse": tf.metrics.mean_squared_error(labels, preds),
          "other_metric_1": (tf.constant(1), tf.constant(1)),
          "other_metric_2": (tf.constant(2), tf.constant(2))
      })

    with self.test_session() as sess:
      evaluator = Evaluator(input_fn=input_fn, metric_name="dne", steps=3)
      with self.assertRaises(KeyError):
        metrics = evaluator.evaluate(sess, ensemble_metrics=metrics)


if __name__ == "__main__":
  tf.test.main()
