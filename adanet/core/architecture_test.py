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
  def test_subnetworks(self, subnetworks, want):
    arch = _Architecture()
    for subnetwork in subnetworks:
      arch.add_subnetwork(*subnetwork)
    self.assertEqual(want, arch.subnetworks)

  def test_serialization_lifecycle(self):
    arch = _Architecture()
    arch.add_subnetwork(0, "linear")
    arch.add_subnetwork(0, "dnn")
    arch.add_subnetwork(1, "dnn")
    self.assertEqual(((0, ("linear", "dnn")), (1, ("dnn",))), arch.subnetworks)
    serialized = arch.serialize()
    self.assertEqual(
        b"\n\x08\x12\x06linear\n\x05\x12\x03dnn\n\x07\x08\x01\x12\x03dnn",
        serialized)
    deserialized_arch = _Architecture.deserialize(serialized)
    self.assertEqual(arch.subnetworks, deserialized_arch.subnetworks)


if __name__ == "__main__":
  tf.test.main()
