"""Test AdaNet single graph base learner implementation.

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
from adanet.core.base_learner import BaseLearner
import adanet.core.testing_utils as tu
import tensorflow as tf


class BaseLearnerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "empty_persisted_tensors",
      "last_layer": tu.dummy_tensor(),
      "logits": tu.dummy_tensor(),
      "complexity": tu.dummy_tensor(),
      "persisted_tensors": {},
  }, {
      "testcase_name": "persisted_tensors",
      "last_layer": tu.dummy_tensor(),
      "logits": tu.dummy_tensor(),
      "complexity": tu.dummy_tensor(),
      "persisted_tensors": {
          "hidden_layer": tu.dummy_tensor(),
      },
  }, {
      "testcase_name": "nested_persisted_tensors",
      "last_layer": tu.dummy_tensor(),
      "logits": tu.dummy_tensor(),
      "complexity": tu.dummy_tensor(),
      "persisted_tensors": {
          "hidden_layer": tu.dummy_tensor(),
          "nested": {
              "foo": tu.dummy_tensor(),
              "nested": {
                  "foo": tu.dummy_tensor(),
              },
          },
      },
  })
  def test_new(self, last_layer, logits, complexity, persisted_tensors):
    with self.test_session():
      got = BaseLearner(last_layer, logits, complexity, persisted_tensors)
      self.assertEqual(got.last_layer, last_layer)
      self.assertEqual(got.logits, logits)
      self.assertEqual(got.complexity, complexity)
      self.assertEqual(got.persisted_tensors, persisted_tensors)

  @parameterized.named_parameters({
      "testcase_name": "none_last_layer",
      "last_layer": None,
      "logits": tu.dummy_tensor(),
      "complexity": tu.dummy_tensor(),
      "persisted_tensors": {},
  }, {
      "testcase_name": "none_logits",
      "last_layer": tu.dummy_tensor(),
      "logits": None,
      "complexity": tu.dummy_tensor(),
      "persisted_tensors": {},
  }, {
      "testcase_name": "none_complexity",
      "last_layer": tu.dummy_tensor(),
      "logits": tu.dummy_tensor(),
      "complexity": None,
      "persisted_tensors": {},
  }, {
      "testcase_name": "none_persisted_tensors",
      "last_layer": tu.dummy_tensor(),
      "logits": tu.dummy_tensor(),
      "complexity": tu.dummy_tensor(),
      "persisted_tensors": None,
  }, {
      "testcase_name": "empty_list_persisted_tensors",
      "last_layer": tu.dummy_tensor(),
      "logits": tu.dummy_tensor(),
      "complexity": tu.dummy_tensor(),
      "persisted_tensors": [],
  }, {
      "testcase_name": "list_persisted_tensors",
      "last_layer": tu.dummy_tensor(),
      "logits": tu.dummy_tensor(),
      "complexity": tu.dummy_tensor(),
      "persisted_tensors": [1.],
  }, {
      "testcase_name": "empty_nested_persisted_tensors",
      "last_layer": tu.dummy_tensor(),
      "logits": tu.dummy_tensor(),
      "complexity": tu.dummy_tensor(),
      "persisted_tensors": {
          "value": tu.dummy_tensor(),
          "nested": {},
      },
  }, {
      "testcase_name": "empty_nested_persisted_tensors_recursive",
      "last_layer": tu.dummy_tensor(),
      "logits": tu.dummy_tensor(),
      "complexity": tu.dummy_tensor(),
      "persisted_tensors": {
          "value": tu.dummy_tensor(),
          "nested": {
              "value": tu.dummy_tensor(),
              "nested": {
                  "value": tu.dummy_tensor(),
                  "nested": {},
              },
          },
      },
  })
  def test_new_errors(self, last_layer, logits, complexity, persisted_tensors):
    with self.test_session():
      with self.assertRaises(ValueError):
        BaseLearner(last_layer, logits, complexity, persisted_tensors)


if __name__ == "__main__":
  tf.test.main()
