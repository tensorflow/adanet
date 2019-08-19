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

from adanet import ensemble
from adanet import subnetwork
import mock
import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import test_util
# pylint: enable=g-direct-tensorflow-import


class StrategyTest(tf.test.TestCase):

  def setUp(self):
    self.fake_builder_1 = mock.create_autospec(spec=subnetwork.Builder)
    self.fake_builder_2 = mock.create_autospec(spec=subnetwork.Builder)
    self.fake_builder_3 = mock.create_autospec(spec=subnetwork.Builder)
    self.fake_builder_4 = mock.create_autospec(spec=subnetwork.Builder)

  @test_util.run_in_graph_and_eager_modes
  def test_solo_strategy(self):
    want = [
        ensemble.Candidate("{}_solo".format(self.fake_builder_1.name),
                           [self.fake_builder_1], []),
        ensemble.Candidate("{}_solo".format(self.fake_builder_2.name),
                           [self.fake_builder_2], [])
    ]
    got = ensemble.SoloStrategy().generate_ensemble_candidates(
        [self.fake_builder_1, self.fake_builder_2], None)

    self.assertEqual(want, got)

  @test_util.run_in_graph_and_eager_modes
  def test_solo_strategy_with_previous_ensemble_subnetwork_builders(self):
    want = [
        ensemble.Candidate("{}_solo".format(self.fake_builder_1.name),
                           [self.fake_builder_1], []),
        ensemble.Candidate("{}_solo".format(self.fake_builder_2.name),
                           [self.fake_builder_2], [])
    ]
    got = ensemble.SoloStrategy().generate_ensemble_candidates(
        [self.fake_builder_1, self.fake_builder_2],
        [self.fake_builder_3, self.fake_builder_4])

    self.assertEqual(want, got)

  @test_util.run_in_graph_and_eager_modes
  def test_grow_strategy(self):
    want = [
        ensemble.Candidate("{}_grow".format(self.fake_builder_1.name),
                           [self.fake_builder_1], []),
        ensemble.Candidate("{}_grow".format(self.fake_builder_2.name),
                           [self.fake_builder_2], [])
    ]
    got = ensemble.GrowStrategy().generate_ensemble_candidates(
        [self.fake_builder_1, self.fake_builder_2], None)
    self.assertEqual(want, got)

  @test_util.run_in_graph_and_eager_modes
  def test_grow_strategy_with_previous_ensemble_subnetwork_builders(self):
    want = [
        ensemble.Candidate("{}_grow".format(self.fake_builder_1.name),
                           [self.fake_builder_1],
                           [self.fake_builder_3, self.fake_builder_4]),
        ensemble.Candidate("{}_grow".format(self.fake_builder_2.name),
                           [self.fake_builder_2],
                           [self.fake_builder_3, self.fake_builder_4])
    ]
    got = ensemble.GrowStrategy().generate_ensemble_candidates(
        [self.fake_builder_1, self.fake_builder_2],
        [self.fake_builder_3, self.fake_builder_4])
    self.assertEqual(want, got)

  @test_util.run_in_graph_and_eager_modes
  def test_all_strategy(self):
    want = [
        ensemble.Candidate("all", [self.fake_builder_1, self.fake_builder_2],
                           [])
    ]
    got = ensemble.AllStrategy().generate_ensemble_candidates(
        [self.fake_builder_1, self.fake_builder_2], None)
    self.assertEqual(want, got)

  @test_util.run_in_graph_and_eager_modes
  def test_all_strategy_with_previous_ensemble_subnetwork_builders(self):
    want = [
        ensemble.Candidate("all", [self.fake_builder_1, self.fake_builder_2],
                           [self.fake_builder_3, self.fake_builder_4])
    ]
    got = ensemble.AllStrategy().generate_ensemble_candidates(
        [self.fake_builder_1, self.fake_builder_2],
        [self.fake_builder_3, self.fake_builder_4])
    self.assertEqual(want, got)


if __name__ == "__main__":
  tf.test.main()
