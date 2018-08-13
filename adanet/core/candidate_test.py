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
from adanet.core.candidate import _Candidate
from adanet.core.candidate import _CandidateBuilder
import adanet.core.testing_utils as tu
import tensorflow as tf


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
      "testcase_name": "evaluate",
      "training": False,
      "max_steps": 3,
      "want_adanet_losses": [0., 0., 0.],
      "want_is_training": True,
  }, {
      "testcase_name": "train_exactly_max_steps",
      "training": True,
      "max_steps": 3,
      "want_adanet_losses": [1., .750, .583],
      "want_is_training": False,
  }, {
      "testcase_name": "train_extra_steps",
      "training": True,
      "max_steps": 2,
      "want_adanet_losses": [1., .750, .583],
      "want_is_training": False,
  }, {
      "testcase_name": "train_one_step_max_one_step",
      "training": True,
      "max_steps": 1,
      "want_adanet_losses": [1.],
      "want_is_training": False,
  }, {
      "testcase_name": "train_one_step_max_two_steps",
      "training": True,
      "max_steps": 2,
      "want_adanet_losses": [1.],
      "want_is_training": True,
  }, {
      "testcase_name": "train_two_steps_max_two_steps",
      "training": True,
      "max_steps": 2,
      "want_adanet_losses": [1., .750],
      "want_is_training": False,
  }, {
      "testcase_name": "train_three_steps_max_four_steps",
      "training": True,
      "max_steps": 4,
      "want_adanet_losses": [1., .750, .583],
      "want_is_training": True,
  }, {
      "testcase_name": "train_three_steps_max_five_steps",
      "training": True,
      "max_steps": 5,
      "want_adanet_losses": [1., .750, .583],
      "want_is_training": True,
  }, {
      "testcase_name": "eval_one_step",
      "training": False,
      "max_steps": 1,
      "want_adanet_losses": [0.],
      "want_is_training": True,
  }, {
      "testcase_name": "previous_best_training",
      "training": True,
      "is_previous_best": True,
      "max_steps": 4,
      "want_adanet_losses": [1., .750, .583],
      "want_is_training": False,
  })
  def test_build_candidate(self,
                           training,
                           max_steps,
                           want_adanet_losses,
                           want_is_training,
                           is_previous_best=False):
    # A fake adanet_loss that halves at each train step: 1.0, 0.5, 0.25, ...
    fake_adanet_loss = tf.Variable(1.)
    fake_train_op = tf.assign(fake_adanet_loss, fake_adanet_loss / 2)
    fake_ensemble = tu.dummy_ensemble(
        "new", adanet_loss=fake_adanet_loss, train_op=fake_train_op)

    iteration_step = tf.Variable(0)
    builder = _CandidateBuilder(max_steps=max_steps)
    candidate = builder.build_candidate(
        ensemble=fake_ensemble,
        training=training,
        iteration_step=iteration_step,
        summary=tf.summary,
        is_previous_best=is_previous_best)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      adanet_losses = []
      is_training = True
      for _ in range(len(want_adanet_losses)):
        is_training, adanet_loss = sess.run(
            (candidate.is_training, candidate.adanet_loss))
        adanet_losses.append(adanet_loss)
        sess.run((fake_train_op, tf.assign_add(iteration_step, 1)))

    # Verify that adanet_loss moving average works.
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
