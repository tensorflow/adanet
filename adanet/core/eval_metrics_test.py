"""Tests for AdaNet eval metrics.

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
from adanet import tf_compat
from adanet.core.architecture import _Architecture
from adanet.core.candidate import _Candidate
from adanet.core.ensemble_builder import _EnsembleSpec
from adanet.core.eval_metrics import _IterationMetrics
from adanet.core.eval_metrics import call_eval_metrics
import adanet.core.testing_utils as tu
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
# pylint: enable=g-direct-tensorflow-import


class MetricsTest(tu.AdanetTestCase):

  def setup_graph(self):
    # We only test the multi head since this is the general case.
    self._features = {"x": tf.constant([[1.], [2.]])}
    heads = ("head_1", "head_2")
    labels = tf.constant([0, 1])
    self._labels = {head: labels for head in heads}
    predictions = {(head, "predictions"): labels for head in heads}
    loss = tf.constant(2.)
    self._estimator_spec = tf_compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        predictions=predictions,
        eval_metrics=(self._spec_metric_fn, {
            "features": self._features,
            "labels": self._labels,
            "predictions": predictions,
            "loss": loss
        }))

  def _run_metrics(self, metrics):
    metric_ops = metrics
    if isinstance(metric_ops, tuple):
      metric_ops = call_eval_metrics(metric_ops)
    self.evaluate((tf_compat.v1.global_variables_initializer(),
                   tf_compat.v1.local_variables_initializer()))
    self.evaluate(metric_ops)
    return {k: self.evaluate(metric_ops[k][0]) for k in metric_ops}

  def _assert_tensors_equal(self, actual, expected):
    actual, expected = self.evaluate((actual, expected))
    self.assertEqual(actual, expected)

  def _spec_metric_fn(self, features, labels, predictions, loss):
    actual = [features, labels, predictions, loss]
    expected = [
        self._features, self._labels, self._estimator_spec.predictions,
        self._estimator_spec.loss
    ]
    self._assert_tensors_equal(actual, expected)
    return {"metric_1": tf_compat.v1.metrics.mean(tf.constant(1.))}

  def _metric_fn(self, features, predictions):
    actual = [features, predictions]
    expected = [self._features, self._estimator_spec.predictions]
    self._assert_tensors_equal(actual, expected)
    return {"metric_2": tf_compat.v1.metrics.mean(tf.constant(2.))}

  @parameterized.named_parameters(
      {
          "testcase_name": "use_tpu",
          "use_tpu": True,
      },
      {
          # TODO: Figure out why this gives error in TF 2.0:
          # ValueError: Please call update_state(...) on the "mean_1" metric.
          "testcase_name": "not_use_tpu",
          "use_tpu": False,
      })
  @test_util.run_in_graph_and_eager_modes
  def test_subnetwork_metrics(self, use_tpu):
    with context.graph_mode():
      self.setup_graph()
      spec = self._estimator_spec
      if not use_tpu:
        spec = spec.as_estimator_spec()
      metrics = tu.create_subnetwork_metrics_for_testing(
          self._metric_fn,
          use_tpu=use_tpu,
          features=self._features,
          labels=self._labels,
          estimator_spec=spec)
      actual = self._run_metrics(metrics.eval_metrics_tuple())

      expected = {"loss": 2., "metric_1": 1., "metric_2": 2.}
      self.assertEqual(actual, expected)

  @test_util.run_in_graph_and_eager_modes
  def test_subnetwork_metrics_user_metric_fn_overrides_metrics(self):
    with context.graph_mode():
      self.setup_graph()
      overridden_value = 100.

      def _overriding_metric_fn():
        value = tf.constant(overridden_value)
        return {"metric_1": tf_compat.v1.metrics.mean(value)}

      metrics = tu.create_subnetwork_metrics_for_testing(
          _overriding_metric_fn,
          features=self._features,
          labels=self._labels,
          estimator_spec=self._estimator_spec)

      actual = self._run_metrics(metrics.eval_metrics_tuple())

      expected = {"loss": 2., "metric_1": overridden_value}
      self.assertEqual(actual, expected)

  @test_util.run_in_graph_and_eager_modes
  def test_ensemble_metrics(self):
    with context.graph_mode():
      self.setup_graph()
      architecture = _Architecture("test_ensemble_candidate", "test_ensembler")
      architecture.add_subnetwork(iteration_number=0, builder_name="b_0_0")
      architecture.add_subnetwork(iteration_number=0, builder_name="b_0_1")
      architecture.add_subnetwork(iteration_number=1, builder_name="b_1_0")
      architecture.add_subnetwork(iteration_number=2, builder_name="b_2_0")

      metrics = tu.create_ensemble_metrics_for_testing(
          self._metric_fn,
          features=self._features,
          labels=self._labels,
          estimator_spec=self._estimator_spec,
          architecture=architecture)

      actual = self._run_metrics(metrics.eval_metrics_tuple())

      serialized_arch_proto = actual["architecture/adanet/ensembles"]
      expected_arch_string = b"| b_0_0 | b_0_1 | b_1_0 | b_2_0 |"
      self.assertIn(expected_arch_string, serialized_arch_proto)

  @parameterized.named_parameters(
      {
          "testcase_name": "use_tpu_evaluating",
          "use_tpu": True,
          "mode": tf.estimator.ModeKeys.EVAL,
      }, {
          "testcase_name": "use_tpu_not_evaluating",
          "use_tpu": True,
          "mode": tf.estimator.ModeKeys.TRAIN,
      }, {
          "testcase_name": "not_use_tpu_evaluating",
          "use_tpu": False,
          "mode": tf.estimator.ModeKeys.EVAL,
      }, {
          "testcase_name": "not_use_tpu_not_evaluating",
          "use_tpu": False,
          "mode": tf.estimator.ModeKeys.TRAIN,
      })
  @test_util.run_in_graph_and_eager_modes
  def test_iteration_metrics(self, use_tpu, mode):
    with context.graph_mode():
      self.setup_graph()
      best_candidate_index = 3
      candidates = []
      for i in range(10):

        def metric_fn(val=i):
          metric = tf.keras.metrics.Mean()
          metric.update_state(tf.constant(val))
          return {
              "ensemble_v1_metric": tf_compat.v1.metrics.mean(tf.constant(val)),
              "ensemble_keras_metric": metric
          }

        spec = _EnsembleSpec(
            name="ensemble_{}".format(i),
            ensemble=None,
            architecture=None,
            subnetwork_builders=None,
            predictions=None,
            step=None,
            eval_metrics=(metric_fn, {}))
        candidate = _Candidate(ensemble_spec=spec, adanet_loss=tf.constant(i))
        candidates.append(candidate)
      metrics = _IterationMetrics(1, candidates, subnetwork_specs=[])

      metrics_fn = (
          metrics.best_eval_metrics_tuple
          if use_tpu else metrics.best_eval_metric_ops)
      actual = self._run_metrics(
          metrics_fn(tf.constant(best_candidate_index), mode) or {})

      if mode == tf.estimator.ModeKeys.EVAL:
        expected = {
            "ensemble_v1_metric": best_candidate_index,
            "ensemble_keras_metric": best_candidate_index,
            "iteration": 1
        }
      else:
        expected = {}
      self.assertEqual(actual, expected)

  @test_util.run_in_graph_and_eager_modes
  def test_metric_ops_not_duplicated_on_cpu(self):

    with context.graph_mode():
      self.setup_graph()
      metric_fn = lambda: {"metric": tf.constant(5)}

      ensemble_metrics = tu.create_ensemble_metrics_for_testing(metric_fn)
      ensemble_ops1 = call_eval_metrics(ensemble_metrics.eval_metrics_tuple())
      ensemble_ops2 = call_eval_metrics(ensemble_metrics.eval_metrics_tuple())
      self.assertEqual(ensemble_ops1, ensemble_ops2)

      subnetwork_metrics = tu.create_subnetwork_metrics_for_testing(
          metric_fn)
      subnetwork_ops1 = call_eval_metrics(
          subnetwork_metrics.eval_metrics_tuple())
      subnetwork_ops2 = call_eval_metrics(
          subnetwork_metrics.eval_metrics_tuple())
      self.assertEqual(subnetwork_ops1, subnetwork_ops2)


if __name__ == "__main__":
  tf.test.main()
