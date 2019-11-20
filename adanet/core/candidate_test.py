"""Test AdaNet single graph candidate implementation.

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

import contextlib

from absl.testing import parameterized
from adanet import tf_compat
from adanet.core.candidate import _Candidate
from adanet.core.candidate import _CandidateBuilder
import adanet.core.testing_utils as tu
import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
# pylint: enable=g-direct-tensorflow-import


class CandidateTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "valid",
      "ensemble_spec": tu.dummy_ensemble_spec("foo"),
      "adanet_loss": [.1],
  })
  @test_util.run_in_graph_and_eager_modes
  def test_new(self, ensemble_spec, adanet_loss, variables=None):
    with self.test_session():
      got = _Candidate(ensemble_spec, adanet_loss, variables)
      self.assertEqual(got.ensemble_spec, ensemble_spec)
      self.assertEqual(got.adanet_loss, adanet_loss)

  @parameterized.named_parameters(
      {
          "testcase_name": "none_ensemble_spec",
          "ensemble_spec": None,
          "adanet_loss": [.1],
      }, {
          "testcase_name": "none_adanet_loss",
          "ensemble_spec": tu.dummy_ensemble_spec("foo"),
          "adanet_loss": None,
      })
  @test_util.run_in_graph_and_eager_modes
  def test_new_errors(self, ensemble_spec, adanet_loss, variables=None):
    with self.test_session():
      with self.assertRaises(ValueError):
        _Candidate(ensemble_spec, adanet_loss, variables)


class _FakeSummary(object):
  """A fake adanet.Summary."""

  def scalar(self, name, tensor, family=None):
    del name
    del tensor
    del family
    return "fake_scalar"

  @contextlib.contextmanager
  def current_scope(self):
    yield


class CandidateBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "evaluate",
          "training": False,
          "want_adanet_losses": [0., 0., 0.],
      }, {
          "testcase_name": "train_exactly_max_steps",
          "training": True,
          "want_adanet_losses": [1., .750, .583],
      }, {
          "testcase_name": "train_one_step_max_one_step",
          "training": True,
          "want_adanet_losses": [1.],
      }, {
          "testcase_name": "train_two_steps_max_two_steps",
          "training": True,
          "want_adanet_losses": [1., .750],
      }, {
          "testcase_name": "train_three_steps_max_four_steps",
          "training": True,
          "want_adanet_losses": [1., .750, .583],
      }, {
          "testcase_name": "eval_one_step",
          "training": False,
          "want_adanet_losses": [0.],
      })
  @test_util.run_in_graph_and_eager_modes
  def test_build_candidate(self, training, want_adanet_losses):
    # `Cadidate#build_candidate` will only ever be called in graph mode.
    with context.graph_mode():
      # A fake adanet_loss that halves at each train step: 1.0, 0.5, 0.25, ...
      fake_adanet_loss = tf.Variable(1.)
      fake_train_op = fake_adanet_loss.assign(fake_adanet_loss / 2)
      fake_ensemble_spec = tu.dummy_ensemble_spec(
          "new", adanet_loss=fake_adanet_loss, train_op=fake_train_op)

      builder = _CandidateBuilder()
      candidate = builder.build_candidate(
          ensemble_spec=fake_ensemble_spec,
          training=training,
          summary=_FakeSummary())
      self.evaluate(tf_compat.v1.global_variables_initializer())
      adanet_losses = []
      for _ in range(len(want_adanet_losses)):
        adanet_loss = self.evaluate(candidate.adanet_loss)
        adanet_losses.append(adanet_loss)
        self.evaluate(fake_train_op)

      # Verify that adanet_loss moving average works.
      self.assertAllClose(want_adanet_losses, adanet_losses, atol=1e-3)


if __name__ == "__main__":
  tf.test.main()
