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
from adanet.core.base_learner import BaseLearner
from adanet.core.base_learner import BaseLearnerBuilder
from adanet.core.ensemble import _EnsembleBuilder
from adanet.core.ensemble import MixtureWeightType
import adanet.core.testing_utils as tu
import tensorflow as tf


class _BaseLearnerBuilder(BaseLearnerBuilder):

  def __init__(self,
               base_learner_train_op_fn,
               mixture_weights_train_op_fn,
               use_logits_last_layer,
               seed=42):
    self._base_learner_train_op_fn = base_learner_train_op_fn
    self._mixture_weights_train_op_fn = mixture_weights_train_op_fn
    self._use_logits_last_layer = use_logits_last_layer
    self._seed = seed

  @property
  def name(self):
    return "test"

  def build_base_learner(self,
                         features,
                         logits_dimension,
                         training,
                         iteration_step,
                         summary,
                         previous_ensemble=None):
    assert features is not None
    assert training is not None
    assert iteration_step is not None
    assert summary is not None
    last_layer = tu.dummy_tensor(shape=(2, 3))
    logits = tf.layers.dense(
        last_layer,
        units=logits_dimension,
        kernel_initializer=tf.glorot_uniform_initializer(seed=self._seed))
    return BaseLearner(
        last_layer=logits if self._use_logits_last_layer else last_layer,
        logits=logits,
        complexity=2,
        persisted_tensors={})

  def build_base_learner_train_op(self, loss, var_list, labels, iteration_step,
                                  summary):
    assert iteration_step is not None
    assert summary is not None
    return self._base_learner_train_op_fn(loss, var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    assert iteration_step is not None
    assert summary is not None
    return self._mixture_weights_train_op_fn(loss, var_list)


class EnsembleBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "no_previous_ensemble",
      "want_logits": [[.016], [.117]],
      "want_loss": 1.338,
      "want_adanet_loss": 1.338,
      "want_complexity_regularization": 0.,
      "want_mixture_weight_vars": 1,
  }, {
      "testcase_name": "default_mixture_weight_initializer_scalar",
      "mixture_weight_initializer": None,
      "mixture_weight_type": MixtureWeightType.SCALAR,
      "use_logits_last_layer": True,
      "want_logits": [[.580], [.914]],
      "want_loss": 1.362,
      "want_adanet_loss": 1.362,
      "want_complexity_regularization": 0.,
      "want_mixture_weight_vars": 1,
  }, {
      "testcase_name": "default_mixture_weight_initializer_vector",
      "mixture_weight_initializer": None,
      "mixture_weight_type": MixtureWeightType.VECTOR,
      "use_logits_last_layer": True,
      "want_logits": [[.580], [.914]],
      "want_loss": 1.362,
      "want_adanet_loss": 1.362,
      "want_complexity_regularization": 0.,
      "want_mixture_weight_vars": 1,
  }, {
      "testcase_name": "default_mixture_weight_initializer_matrix",
      "mixture_weight_initializer": None,
      "mixture_weight_type": MixtureWeightType.MATRIX,
      "want_logits": [[.016], [.117]],
      "want_loss": 1.338,
      "want_adanet_loss": 1.338,
      "want_complexity_regularization": 0.,
      "want_mixture_weight_vars": 1,
  }, {
      "testcase_name": "default_mixture_weight_initializer_matrix_on_logits",
      "mixture_weight_initializer": None,
      "mixture_weight_type": MixtureWeightType.MATRIX,
      "use_logits_last_layer": True,
      "want_logits": [[.030], [.047]],
      "want_loss": 1.378,
      "want_adanet_loss": 1.378,
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
  }, {
      "testcase_name": "previous_ensemble_no_warm_start",
      "ensemble_fn": lambda: tu.dummy_ensemble("test", random_seed=1),
      "warm_start_mixture_weights": False,
      "adanet_lambda": .01,
      "adanet_beta": .1,
      "want_logits": [[.007], [.079]],
      "want_loss": 1.351,
      "want_adanet_loss": 1.367,
      "want_complexity_regularization": .016,
      "want_mixture_weight_vars": 2,
  })
  def test_append_new_base_learner(
      self,
      want_logits,
      want_complexity_regularization,
      want_loss=None,
      want_adanet_loss=None,
      want_mixture_weight_vars=None,
      adanet_lambda=0.,
      adanet_beta=0.,
      ensemble_fn=lambda: None,
      use_bias=False,
      use_logits_last_layer=False,
      mixture_weight_type=MixtureWeightType.MATRIX,
      mixture_weight_initializer=tf.zeros_initializer(),
      warm_start_mixture_weights=True,
      mode=tf.estimator.ModeKeys.TRAIN):
    seed = 64
    builder = _EnsembleBuilder(
        head=tf.contrib.estimator.binary_classification_head(
            loss_reduction=tf.losses.Reduction.SUM),
        mixture_weight_type=mixture_weight_type,
        mixture_weight_initializer=mixture_weight_initializer,
        warm_start_mixture_weights=warm_start_mixture_weights,
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
        base_learner_builder=_BaseLearnerBuilder(_base_learner_train_op_fn,
                                                 _mixture_weights_train_op_fn,
                                                 use_logits_last_layer, seed),
        summary=tf.summary,
        features=features,
        iteration_step=tf.train.get_or_create_global_step(),
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
