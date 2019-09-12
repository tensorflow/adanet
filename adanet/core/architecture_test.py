"""Test for the AdaNet architecture.

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
from adanet.core.architecture import _Architecture
import tensorflow as tf


class ArchitectureTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "empty",
      "subnetworks": [],
      "want": (),
  }, {
      "testcase_name": "single",
      "subnetworks": [(0, "linear")],
      "want": ((0, "linear"),),
  }, {
      "testcase_name": "different_iterations",
      "subnetworks": [(0, "linear"), (1, "dnn")],
      "want": ((0, "linear"), (1, "dnn")),
  }, {
      "testcase_name": "same_iterations",
      "subnetworks": [(0, "linear"), (0, "dnn"), (1, "dnn")],
      "want": ((0, "linear"), (0, "dnn"), (1, "dnn")),
  })
  def test_subnetworks(self, subnetworks, want):
    arch = _Architecture("foo", "dummy_ensembler_name")
    for subnetwork in subnetworks:
      arch.add_subnetwork(*subnetwork)
    self.assertEqual(want, arch.subnetworks)

  @parameterized.named_parameters({
      "testcase_name": "empty",
      "subnetworks": [],
      "want": (),
  }, {
      "testcase_name": "single",
      "subnetworks": [(0, "linear")],
      "want": ((0, ("linear",)),),
  }, {
      "testcase_name": "different_iterations",
      "subnetworks": [(0, "linear"), (1, "dnn")],
      "want": ((0, ("linear",)), (1, ("dnn",))),
  }, {
      "testcase_name": "same_iterations",
      "subnetworks": [(0, "linear"), (0, "dnn"), (1, "dnn")],
      "want": ((0, ("linear", "dnn")), (1, ("dnn",))),
  })
  def test_subnetworks_grouped_by_iteration(self, subnetworks, want):
    arch = _Architecture("foo", "dummy_ensembler_name")
    for subnetwork in subnetworks:
      arch.add_subnetwork(*subnetwork)
    self.assertEqual(want, arch.subnetworks_grouped_by_iteration)

  def test_set_and_add_replay_index(self):
    arch = _Architecture("foo", "dummy_ensembler_name")
    arch.set_replay_indices([1, 2, 3])
    self.assertAllEqual([1, 2, 3], arch.replay_indices)
    arch.add_replay_index(4)
    self.assertAllEqual([1, 2, 3, 4], arch.replay_indices)

  def test_serialization_lifecycle(self):
    arch = _Architecture("foo", "dummy_ensembler_name", replay_indices=[1, 2])
    arch.add_subnetwork(0, "linear")
    arch.add_subnetwork(0, "dnn")
    arch.add_subnetwork(1, "dnn")
    self.assertEqual("foo", arch.ensemble_candidate_name)
    self.assertEqual("dummy_ensembler_name", arch.ensembler_name)
    self.assertEqual(((0, ("linear", "dnn")), (1, ("dnn",))),
                     arch.subnetworks_grouped_by_iteration)
    global_step = 100
    serialized = arch.serialize(global_step)
    self.assertEqual(
        '{"ensemble_candidate_name": "foo", "ensembler_name": '
        '"dummy_ensembler_name", "global_step": 100, "replay_indices": [1, 2], '
        '"subnetworks": [{"builder_name": "linear", "iteration_number": 0}, '
        '{"builder_name": "dnn", "iteration_number": 0},'
        ' {"builder_name": "dnn", "iteration_number": 1}]}', serialized)
    deserialized_arch = _Architecture.deserialize(serialized)
    self.assertEqual(arch.ensemble_candidate_name,
                     deserialized_arch.ensemble_candidate_name)
    self.assertEqual(arch.ensembler_name,
                     deserialized_arch.ensembler_name)
    self.assertEqual(arch.subnetworks_grouped_by_iteration,
                     deserialized_arch.subnetworks_grouped_by_iteration)
    self.assertEqual(global_step, deserialized_arch.global_step)


if __name__ == "__main__":
  tf.test.main()
