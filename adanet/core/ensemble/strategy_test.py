"""Test AdaNet single graph subnetwork implementation.

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

from adanet.core import ensemble
from adanet.core import subnetwork

import tensorflow as tf
mock = tf.test.mock


class StrategyTest(tf.test.TestCase):

  def setUp(self):
    self.fake_builder_1 = mock.create_autospec(spec=subnetwork.Builder)
    self.fake_builder_2 = mock.create_autospec(spec=subnetwork.Builder)

  def test_grow_strategy(self):
    self.assertEqual(
        ensemble.GrowStrategy().generate_ensemble_candidates(
            [self.fake_builder_1, self.fake_builder_2]), [
                ensemble.Candidate([self.fake_builder_1]),
                ensemble.Candidate([self.fake_builder_2])
            ])

  def test_all_strategy(self):
    self.assertEqual(
        ensemble.AllStrategy().generate_ensemble_candidates(
            [self.fake_builder_1, self.fake_builder_2]),
        [ensemble.Candidate([self.fake_builder_1, self.fake_builder_2])])


if __name__ == '__main__':
  tf.test.main()
