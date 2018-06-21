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
from adanet.evaluator import Evaluator
import adanet.testing_utils as tu
import tensorflow as tf


def _fake_ensembles_0(input_fn):
  _, labels = input_fn()
  return [
      tu.dummy_ensemble("foo", adanet_loss=tf.reduce_sum(labels)),
      tu.dummy_ensemble("foo2", adanet_loss=tf.reduce_sum(labels * 2)),
  ]


def _fake_ensembles_1(input_fn):
  _, labels = input_fn()
  return [
      tu.dummy_ensemble("foo", adanet_loss=tf.reduce_sum(labels * 2)),
      tu.dummy_ensemble("foo2", adanet_loss=tf.reduce_sum(labels)),
  ]


def _fake_ensembles_2(input_fn):
  _, labels = input_fn()
  return [
      tu.dummy_ensemble(
          "foo",
          adanet_loss=tf.reduce_sum(labels * 2),
          eval_metric_ops={
              "a": (tf.constant(2.), tf.constant(2.)),
              "b": (tf.constant(3.), tf.constant(3.))
          }),
      tu.dummy_ensemble(
          "foo2",
          adanet_loss=tf.reduce_sum(labels),
          eval_metric_ops={
              "a": (tf.constant(4.), tf.constant(4.)),
              "c": (tf.constant(5.), tf.constant(5.))
          }),
  ]


class EvaluatorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "choose_index_0",
      "input_fn": tu.dummy_input_fn([[1., 2]], [[3.]]),
      "steps": 3,
      "ensembles": _fake_ensembles_0,
      "want_best_ensemble_index": 0,
  }, {
      "testcase_name": "choose_index_1",
      "input_fn": tu.dummy_input_fn([[1., 2]], [[3.]]),
      "steps": 3,
      "ensembles": _fake_ensembles_1,
      "want_best_ensemble_index": 1,
  }, {
      "testcase_name": "none_steps",
      "input_fn": tu.dataset_input_fn(),
      "steps": None,
      "ensembles": _fake_ensembles_1,
      "want_best_ensemble_index": 1,
  }, {
      "testcase_name": "input_fn_out_of_range",
      "input_fn": tu.dataset_input_fn(),
      "steps": 3,
      "ensembles": _fake_ensembles_1,
      "want_best_ensemble_index": 1,
  }, {
      "testcase_name": "minimize_metric",
      "input_fn": tu.dummy_input_fn([[1., 2]], [[3.]]),
      "steps": 3,
      "ensembles": _fake_ensembles_2,
      "want_best_ensemble_index": 1,
  })
  def test_best_ensemble_index(self, input_fn, steps, ensembles,
                               want_best_ensemble_index):
    with self.test_session() as sess:
      evaluator = Evaluator(input_fn=input_fn, steps=steps)
      best_ensemble_index = evaluator.best_ensemble_index(
          sess, ensembles(input_fn))
      self.assertEqual(want_best_ensemble_index, best_ensemble_index)


if __name__ == "__main__":
  tf.test.main()
