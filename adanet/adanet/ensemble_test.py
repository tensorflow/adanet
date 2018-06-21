"""Test AdaNet ensemble single graph implementation.

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
from adanet.adanet.base_learner import BaseLearner
from adanet.adanet.base_learner import BaseLearnerBuilder
from adanet.adanet.ensemble import _EnsembleBuilder
import adanet.adanet.testing_utils as tu
import tensorflow as tf


class _BaseLearnerBuilder(BaseLearnerBuilder):

  def __init__(self,
               base_learner_train_op_fn,
               mixture_weights_train_op_fn,
               seed=42):
    self._base_learner_train_op_fn = base_learner_train_op_fn
    self._mixture_weights_train_op_fn = mixture_weights_train_op_fn
    self._seed = seed

  @property
  def name(self):
    return "test"

  def build_base_learner(self,
                         features,
                         logits_dimension,
                         training,
                         summary,
                         previous_ensemble=None):
    last_layer = tu.dummy_tensor(shape=(2, 3))
    logits = tf.layers.dense(
        last_layer,
        units=1,
        kernel_initializer=tf.glorot_uniform_initializer(seed=self._seed))
    return BaseLearner(
        last_layer=last_layer,
        logits=logits,
        complexity=2,
        persisted_tensors={})

  def build_base_learner_train_op(self, loss, var_list, labels, summary):
    return self._base_learner_train_op_fn(loss, var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     summary):
    return self._mixture_weights_train_op_fn(loss, var_list)


class _FakeSummary(object):
  """A fake `Summary`."""

  def scalar(self, name, tensor):
    del name  # Unused
    del tensor  # Unused

  def histogram(self, name, tensor):
    del name  # Unused
    del tensor  # Unused


class EnsembleBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "no_previous_ensemble",
      "want_logits": [[.016], [.117]],
      "want_loss": 1.338,
      "want_adanet_loss": 1.338,
      "want_complexity_regularization": 0.,
      "want_mixture_weight_vars": 1,
  }, {
      "testcase_name": "no_previous_ensemble_use_bias",
      "use_bias": True,
      "want_logits": [[0.013], [0.113]],
      "want_loss": 1.338,
      "want_adanet_loss": 1.338,
      "want_complexity_regularization": 0.,
      "want_mixture_weight_vars": 2,
  }, {
      "testcase_name": "no_previous_ensemble_predict_mode",
      "mode": tf.estimator.ModeKeys.PREDICT,
      "want_logits": [[0.], [0.]],
      "want_complexity_regularization": 0.,
  }, {
      "testcase_name": "no_previous_ensemble_lambda",
      "adanet_lambda": .01,
      "want_logits": [[.014], [.110]],
      "want_loss": 1.340,
      "want_adanet_loss": 1.343,
      "want_complexity_regularization": .003,
      "want_mixture_weight_vars": 1,
  }, {
      "testcase_name": "no_previous_ensemble_beta",
      "adanet_beta": .1,
      "want_logits": [[.006], [.082]],
      "want_loss": 1.349,
      "want_adanet_loss": 1.360,
      "want_complexity_regularization": .012,
      "want_mixture_weight_vars": 1,
  }, {
      "testcase_name": "no_previous_ensemble_lambda_and_beta",
      "adanet_lambda": .01,
      "adanet_beta": .1,
      "want_logits": [[.004], [.076]],
      "want_loss": 1.351,
      "want_adanet_loss": 1.364,
      "want_complexity_regularization": .013,
      "want_mixture_weight_vars": 1,
  }, {
      "testcase_name": "previous_ensemble",
      "ensemble_fn": lambda: tu.dummy_ensemble("test", random_seed=1),
      "adanet_lambda": .01,
      "adanet_beta": .1,
      "want_logits": [[.089], [.159]],
      "want_loss": 1.355,
      "want_adanet_loss": 1.398,
      "want_complexity_regularization": .043,
      "want_mixture_weight_vars": 2,
  }, {
      "testcase_name": "previous_ensemble_use_bias",
      "use_bias": True,
      "ensemble_fn": lambda: tu.dummy_ensemble("test", random_seed=1),
      "adanet_lambda": .01,
      "adanet_beta": .1,
      "want_logits": [[.075], [.146]],
      "want_loss": 1.354,
      "want_adanet_loss": 1.397,
      "want_complexity_regularization": .043,
      "want_mixture_weight_vars": 3,
  })
  def test_append_new_base_learner(self,
                                   want_logits,
                                   want_complexity_regularization,
                                   want_loss=None,
                                   want_adanet_loss=None,
                                   want_mixture_weight_vars=None,
                                   adanet_lambda=0.,
                                   adanet_beta=0.,
                                   ensemble_fn=lambda: None,
                                   use_bias=False,
                                   mode=tf.estimator.ModeKeys.TRAIN):
    seed = 64
    builder = _EnsembleBuilder(
        head=tf.contrib.estimator.binary_classification_head(
            loss_reduction=tf.losses.Reduction.SUM),
        adanet_lambda=adanet_lambda,
        adanet_beta=adanet_beta,
        use_bias=use_bias)

    features = {"x": tf.constant([[1.], [2.]])}
    labels = tf.constant([0, 1])

    def _base_learner_train_op_fn(loss, var_list):
      self.assertEqual(2, len(var_list))
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1)
      return optimizer.minimize(loss, var_list=var_list)

    def _mixture_weights_train_op_fn(loss, var_list):
      self.assertEqual(want_mixture_weight_vars, len(var_list))
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1)
      return optimizer.minimize(loss, var_list=var_list)

    ensemble = builder.append_new_base_learner(
        ensemble=ensemble_fn(),
        base_learner_builder=_BaseLearnerBuilder(
            _base_learner_train_op_fn, _mixture_weights_train_op_fn, seed),
        summary=_FakeSummary(),
        features=features,
        labels=labels,
        mode=mode)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      if mode == tf.estimator.ModeKeys.PREDICT:
        self.assertAllClose(want_logits, sess.run(ensemble.logits), atol=1e-3)
        self.assertIsNone(ensemble.loss)
        self.assertIsNone(ensemble.adanet_loss)
        self.assertIsNone(ensemble.complexity_regularized_loss)
        self.assertIsNone(ensemble.train_op)
        self.assertIsNotNone(ensemble.export_outputs)
        return

      # Verify that train_op works, previous loss should be greater than loss
      # after a train op.
      loss = sess.run(ensemble.loss)
      for _ in range(3):
        sess.run(ensemble.train_op)
      self.assertGreater(loss, sess.run(ensemble.loss))

      self.assertAllClose(want_logits, sess.run(ensemble.logits), atol=1e-3)

      # Bias should learn a non-zero value when used.
      if use_bias:
        self.assertNotAlmostEqual(0., sess.run(ensemble.bias))
      else:
        self.assertAlmostEqual(0., sess.run(ensemble.bias))

      self.assertAlmostEqual(
          want_complexity_regularization,
          sess.run(ensemble.complexity_regularization),
          places=3)
      self.assertAlmostEqual(want_loss, sess.run(ensemble.loss), places=3)
      self.assertAlmostEqual(
          want_adanet_loss, sess.run(ensemble.adanet_loss), places=3)
      self.assertAlmostEqual(
          want_adanet_loss,
          sess.run(ensemble.complexity_regularized_loss),
          places=3)


if __name__ == "__main__":
  tf.test.main()
