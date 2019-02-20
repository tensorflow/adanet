# Copyright 2019 The AdaNet Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Device placement function tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from adanet.core.devices import monkey_patch_default_variable_placement_strategy

import tensorflow as tf


class DevicesTest(parameterized.TestCase, tf.test.TestCase):

  def test_monkey_patch_default_variable_placement_strategy_no_ps(self):
    with monkey_patch_default_variable_placement_strategy():
      device_fn = tf.train.replica_device_setter(ps_tasks=0)
    self.assertIsNone(device_fn)

  @parameterized.named_parameters({
      "testcase_name": "one_ps",
      "num_tasks": 1,
      "op_names": ["foo", "bar", "baz"],
      "before_want_ps": ["/job:ps/task:0", "/job:ps/task:0", "/job:ps/task:0"],
      "after_want_ps": ["/job:ps/task:0", "/job:ps/task:0", "/job:ps/task:0"],
  }, {
      "testcase_name": "three_ps",
      "num_tasks": 3,
      "op_names": ["foo", "bar", "baz"],
      "before_want_ps": ["/job:ps/task:0", "/job:ps/task:1", "/job:ps/task:2"],
      "after_want_ps": ["/job:ps/task:2", "/job:ps/task:0", "/job:ps/task:1"],
  }, {
      "testcase_name": "reverse_three_ps",
      "num_tasks": 3,
      "op_names": ["baz", "bar", "foo"],
      "before_want_ps": ["/job:ps/task:0", "/job:ps/task:1", "/job:ps/task:2"],
      "after_want_ps": ["/job:ps/task:1", "/job:ps/task:0", "/job:ps/task:2"],
  }, {
      "testcase_name": "six_ps",
      "num_tasks": 6,
      "op_names": ["foo", "bar", "baz"],
      "before_want_ps": ["/job:ps/task:0", "/job:ps/task:1", "/job:ps/task:2"],
      "after_want_ps": ["/job:ps/task:2", "/job:ps/task:3", "/job:ps/task:4"],
  }, {
      "testcase_name": "reverse_six_ps",
      "num_tasks": 6,
      "op_names": ["baz", "bar", "foo"],
      "before_want_ps": ["/job:ps/task:0", "/job:ps/task:1", "/job:ps/task:2"],
      "after_want_ps": ["/job:ps/task:4", "/job:ps/task:3", "/job:ps/task:2"],
  })
  def test_monkey_patch_default_variable_placement_strategy(
      self, num_tasks, op_names, before_want_ps, after_want_ps):
    """Checks that ps placement is based on var name."""

    var_ops = [tf.Variable(0., name=op_name).op for op_name in op_names]
    before_device_fn = tf.train.replica_device_setter(ps_tasks=num_tasks)
    self.assertEqual(before_want_ps, [before_device_fn(op) for op in var_ops])

    with monkey_patch_default_variable_placement_strategy():
      after_device_fn = tf.train.replica_device_setter(ps_tasks=num_tasks)
    self.assertEqual(after_want_ps, [after_device_fn(op) for op in var_ops])

    # Check that monkey-patch is only for the context.
    before_device_fn = tf.train.replica_device_setter(ps_tasks=num_tasks)
    self.assertEqual(before_want_ps, [before_device_fn(op) for op in var_ops])


if __name__ == "__main__":
  tf.test.main()
