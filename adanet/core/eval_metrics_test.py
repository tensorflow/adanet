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
from adanet.core.eval_metrics import _EnsembleMetrics
from adanet.core.eval_metrics import _IterationMetrics
from adanet.core.eval_metrics import _SubnetworkMetrics
from adanet.core.eval_metrics import call_eval_metrics
import adanet.core.testing_utils as tu
import tensorflow as tf


def _run_metrics(sess, metrics):
  metric_ops = metrics
  if isinstance(metric_ops, tuple):
    metric_ops = call_eval_metrics(metric_ops)
  if tf.executing_eagerly():
    results = {}
    for k in metric_ops:
      metric = metric_ops[k]
      if isinstance(metric, tf.keras.metrics.Metric):
        results[k] = metric_ops[k].result().numpy()
      else:
        results[k] = metric_ops[k][0]
    return results
  sess.run((tf_compat.v1.global_variables_initializer(),
            tf_compat.v1.local_variables_initializer()))
  sess.run(metric_ops)
  return {k: sess.run(metric_ops[k][0]) for k in metric_ops}


class MetricsTest(tu.AdanetTestCase):

  def setUp(self):
    super(MetricsTest, self).setUp()

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

  def _assert_tensors_equal(self, actual, expected):
    if not tf.executing_eagerly():
      with self.test_session() as sess:
        actual, expected = sess.run((actual, expected))
    self.assertEqual(actual, expected)

  def _spec_metric_fn(self, features, labels, predictions, loss):
    actual = [features, labels, predictions, loss]
    expected = [
        self._features, self._labels, self._estimator_spec.predictions,
        self._estimator_spec.loss
    ]
    self._assert_tensors_equal(actual, expected)
    if tf.executing_eagerly():
      metric = tf.metrics.Mean("mean_1")
      metric(tf.constant(1.))
      return {"metric_1": metric}
    return {"metric_1": tf_compat.v1.metrics.mean(tf.constant(1.))}

  def _metric_fn(self, features, predictions):
    actual = [features, predictions]
    expected = [self._features, self._estimator_spec.predictions]
    self._assert_tensors_equal(actual, expected)
    if tf.executing_eagerly():
      metric = tf.metrics.Mean("mean_2")
      metric(tf.constant(2.))
      return {"metric_2": metric}
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
  def test_subnetwork_metrics(self, use_tpu):
    spec = self._estimator_spec
    if not use_tpu:
      spec = spec.as_estimator_spec()
    metrics = _SubnetworkMetrics()
    metrics.create_eval_metrics(self._features, self._labels, spec,
                                self._metric_fn)

    with self.test_session() as sess:
      actual = _run_metrics(sess, metrics.eval_metrics_tuple())

    expected = {"loss": 2., "metric_1": 1., "metric_2": 2.}
    self.assertEqual(actual, expected)

  def test_subnetwork_metrics_user_metric_fn_overrides_metrics(self):

    overridden_value = 100.

    def _overriding_metric_fn():
      value = tf.constant(overridden_value)
      if tf.executing_eagerly():
        metric = tf.metrics.Mean()
        metric.update_state(value)
        return {"metric_1": metric}
      return {"metric_1": tf_compat.v1.metrics.mean(value)}

    metrics = _SubnetworkMetrics()
    metrics.create_eval_metrics(self._features, self._labels,
                                self._estimator_spec, _overriding_metric_fn)

    with self.test_session() as sess:
      actual = _run_metrics(sess, metrics.eval_metrics_tuple())

    expected = {"loss": 2., "metric_1": overridden_value}
    self.assertEqual(actual, expected)

  def test_ensemble_metrics(self):
    architecture = _Architecture("test_ensemble_candidate")
    architecture.add_subnetwork(iteration_number=0, builder_name="b_0_0")
    architecture.add_subnetwork(iteration_number=0, builder_name="b_0_1")
    architecture.add_subnetwork(iteration_number=1, builder_name="b_1_0")
    architecture.add_subnetwork(iteration_number=2, builder_name="b_2_0")

    metrics = _EnsembleMetrics()
    metrics.create_eval_metrics(self._features, self._labels,
                                self._estimator_spec, self._metric_fn,
                                architecture)

    with self.test_session() as sess:
      actual = _run_metrics(sess, metrics.eval_metrics_tuple())

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
  def test_iteration_metrics(self, use_tpu, mode):
    best_candidate_index = 3
    candidates = []
    for i in range(10):

      def metric_fn(val=i):
        if tf.executing_eagerly():
          metric = tf.metrics.Mean()
          metric(tf.constant(val))
          return {"ensemble_metric": metric}
        return {"ensemble_metric": tf_compat.v1.metrics.mean(tf.constant(val))}

      spec = _EnsembleSpec(
          name="ensemble_{}".format(i),
          ensemble=None,
          architecture=None,
          subnetwork_builders=None,
          predictions=None,
          eval_metrics=(metric_fn, {}))
      candidate = _Candidate(
          ensemble_spec=spec,
          adanet_loss=tf.constant(i),
          is_training=tf.constant(False))
      candidates.append(candidate)
    metrics = _IterationMetrics(candidates, subnetwork_specs=[])

    with self.test_session() as sess:
      metrics_fn = (
          metrics.best_eval_metrics_tuple
          if use_tpu else metrics.best_eval_metric_ops)
      actual = _run_metrics(
          sess,
          metrics_fn(tf.constant(best_candidate_index), mode) or {})

    if mode == tf.estimator.ModeKeys.EVAL:
      if tf.executing_eagerly():
        metric = tf.metrics.Mean()
        metric(best_candidate_index)
        return {"ensemble_metric": metric}
      expected = {"ensemble_metric": best_candidate_index}
    else:
      expected = {}
    self.assertEqual(actual, expected)


if __name__ == "__main__":
  tf.test.main()
