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

from absl.testing import parameterized
from adanet.adanet.candidate import _Candidate
from adanet.adanet.candidate import _CandidateBuilder
import adanet.adanet.testing_utils as tu
import tensorflow as tf


class _FakeSummary(object):
  """A fake `Summary`."""

  def scalar(self, name, tensor):
    del name  # Unused
    del tensor  # Unused


class CandidateTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "valid",
      "ensemble": tu.dummy_ensemble("foo"),
      "adanet_loss": [.1],
      "is_training": True,
  })
  def test_new(self, ensemble, adanet_loss, is_training):
    with self.test_session():
      got = _Candidate(ensemble, adanet_loss, is_training)
      self.assertEqual(got.ensemble, ensemble)
      self.assertEqual(got.adanet_loss, adanet_loss)
      self.assertEqual(got.is_training, is_training)

  @parameterized.named_parameters({
      "testcase_name": "none_ensemble",
      "ensemble": None,
      "adanet_loss": [.1],
      "is_training": True,
  }, {
      "testcase_name": "none_adanet_loss",
      "ensemble": tu.dummy_ensemble("foo"),
      "adanet_loss": None,
      "is_training": True,
  }, {
      "testcase_name": "none_is_training",
      "ensemble": tu.dummy_ensemble("foo"),
      "adanet_loss": [.1],
      "is_training": None,
  })
  def test_new_errors(self, ensemble, adanet_loss, is_training):
    with self.test_session():
      with self.assertRaises(ValueError):
        _Candidate(ensemble, adanet_loss, is_training)


class CandidateBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "max_steps_eval",
      "ensemble": lambda: tu.dummy_ensemble("new"),
      "mode": tf.estimator.ModeKeys.EVAL,
      "max_steps": 2,
      "want_adanet_losses": [0., 0., 0.],
      "want_is_training": True,
  }, {
      "testcase_name": "max_steps_train",
      "ensemble": lambda: tu.dummy_ensemble("new"),
      "mode": tf.estimator.ModeKeys.TRAIN,
      "max_steps": 2,
      "want_adanet_losses": [-.188, -.219, -.226],
      "want_is_training": False,
  })
  def test_build_candidate(self,
                           ensemble,
                           mode,
                           want_adanet_losses,
                           want_is_training,
                           max_steps=None):
    with self.test_session() as sess:
      builder = _CandidateBuilder(max_steps=max_steps)
      summary = _FakeSummary()
      candidate = builder.build_candidate(
          ensemble=ensemble(), mode=mode, summary=summary)
      sess.run(tf.global_variables_initializer())
      adanet_losses = []
      is_training = True
      for _ in range(len(want_adanet_losses)):
        is_training, adanet_loss, _ = sess.run(
            (candidate.is_training, candidate.adanet_loss, candidate.update_op))
        adanet_losses.append(adanet_loss)
      self.assertAllClose(want_adanet_losses, adanet_losses, atol=1e-3)
      self.assertEqual(want_is_training, is_training)

  @parameterized.named_parameters({
      "testcase_name": "negative_max_steps",
      "max_steps": -1,
  }, {
      "testcase_name": "zero_max_steps",
      "max_steps": 0,
  })
  def test_init_errors(self, max_steps=None):
    with self.test_session():
      with self.assertRaises(ValueError):
        _CandidateBuilder(max_steps=max_steps)


if __name__ == "__main__":
  tf.test.main()
