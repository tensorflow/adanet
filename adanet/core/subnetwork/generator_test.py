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

import collections

from absl.testing import parameterized
from adanet.core.subnetwork.generator import Builder
from adanet.core.subnetwork.generator import Subnetwork
import tensorflow as tf


def dummy_tensor(shape=(), random_seed=42):
  """Returns a randomly initialized tensor."""

  return tf.Variable(
      tf.random_normal(shape=shape, seed=random_seed),
      trainable=False).read_value()


class FakeSubnetwork(Builder):
  """Fake subnetwork builder."""

  @property
  def name(self):
    return "fake_subnetwork"

  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    return

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    return

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    return


class SubnetworkTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "no_persisted_tensors_nor_shared",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
  }, {
      "testcase_name": "empty_persisted_tensors",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "persisted_tensors": {},
  }, {
      "testcase_name": "dict_logits_and_last_layer",
      "last_layer": {
          "head1": dummy_tensor()
      },
      "logits": {
          "head1": dummy_tensor()
      },
      "complexity": dummy_tensor(),
      "persisted_tensors": {},
  }, {
      "testcase_name": "persisted_tensors",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "persisted_tensors": {
          "hidden_layer": dummy_tensor(),
      },
  }, {
      "testcase_name": "nested_persisted_tensors",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "persisted_tensors": {
          "hidden_layer": dummy_tensor(),
          "nested": {
              "foo": dummy_tensor(),
              "nested": {
                  "foo": dummy_tensor(),
              },
          },
      },
  }, {
      "testcase_name": "shared_primitive",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "shared": 1,
  }, {
      "testcase_name": "shared_dict",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "shared": {},
  }, {
      "testcase_name": "shared_lambda",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "shared": lambda x: x,
  }, {
      "testcase_name": "shared_object",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "shared": dummy_tensor(),
  })
  def test_new(self,
               last_layer,
               logits,
               complexity,
               persisted_tensors=None,
               shared=None):
    with self.test_session():
      got = Subnetwork(last_layer, logits, complexity, persisted_tensors,
                       shared)
      self.assertEqual(got.last_layer, last_layer)
      self.assertEqual(got.logits, logits)
      self.assertEqual(got.complexity, complexity)
      self.assertEqual(got.persisted_tensors, persisted_tensors)
      self.assertEqual(got.shared, shared)

  @parameterized.named_parameters({
      "testcase_name": "none_last_layer",
      "last_layer": None,
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "persisted_tensors": {},
  }, {
      "testcase_name": "none_logits",
      "last_layer": dummy_tensor(),
      "logits": None,
      "complexity": dummy_tensor(),
      "persisted_tensors": {},
  }, {
      "testcase_name": "none_complexity",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": None,
      "persisted_tensors": {},
  }, {
      "testcase_name": "empty_list_persisted_tensors",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "persisted_tensors": [],
  }, {
      "testcase_name": "list_persisted_tensors",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "persisted_tensors": [1.],
  }, {
      "testcase_name": "empty_nested_persisted_tensors",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "persisted_tensors": {
          "value": dummy_tensor(),
          "nested": {},
      },
  }, {
      "testcase_name": "empty_nested_persisted_tensors_recursive",
      "last_layer": dummy_tensor(),
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "persisted_tensors": {
          "value": dummy_tensor(),
          "nested": {
              "value": dummy_tensor(),
              "nested": {
                  "value": dummy_tensor(),
                  "nested": {},
              },
          },
      },
  }, {
      "testcase_name": "only_dict_logits",
      "last_layer": dummy_tensor(),
      "logits": {
          "head": dummy_tensor()
      },
      "complexity": dummy_tensor(),
      "persisted_tensors": {},
  }, {
      "testcase_name": "only_dict_last_layer",
      "last_layer": {
          "head": dummy_tensor()
      },
      "logits": dummy_tensor(),
      "complexity": dummy_tensor(),
      "persisted_tensors": {},
  })
  def test_new_errors(self, last_layer, logits, complexity, persisted_tensors):
    with self.test_session():
      with self.assertRaises(ValueError):
        Subnetwork(last_layer, logits, complexity, persisted_tensors)

  @parameterized.named_parameters({
      "testcase_name": "empty_previous",
      "previous_ensemble_size": 0,
      "expected_ensemble_size": 0,
      "expected_chosen_indices": []
  }, {
      "testcase_name": "one_subnetwork",
      "previous_ensemble_size": 1,
      "expected_ensemble_size": 1,
      "expected_chosen_indices": range(0, 1)
  }, {
      "testcase_name": "three_subnetwork",
      "previous_ensemble_size": 3,
      "expected_ensemble_size": 3,
      "expected_chosen_indices": range(0, 3)
  })
  def test_prune_previous_ensemble(self, previous_ensemble_size,
                                   expected_ensemble_size,
                                   expected_chosen_indices):
    builder = FakeSubnetwork()
    fake_ensemble = collections.namedtuple("ensemble", ["weighted_subnetworks"])
    previous_ensemble = fake_ensemble(
        weighted_subnetworks=[1] * previous_ensemble_size)
    self.assertEqual(expected_ensemble_size,
                     len(builder.prune_previous_ensemble(previous_ensemble)))
    if expected_chosen_indices:
      self.assertEqual(expected_chosen_indices,
                       builder.prune_previous_ensemble(previous_ensemble))


if __name__ == "__main__":
  tf.test.main()
