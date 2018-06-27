"""Test AdaNet materializer single graph implementation.

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
from adanet.base_learner_report import BaseLearnerReport
from adanet.base_learner_report import MaterializedBaseLearnerReport
from adanet.report_materializer import ReportMaterializer
import adanet.testing_utils as tu
import tensorflow as tf


class ReportMaterializerTest(parameterized.TestCase, tf.test.TestCase):

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters({
      "testcase_name": "one_empty_base_learner",
      "input_fn": tu.dummy_input_fn([[1., 2]], [[3.]]),
      "steps": 3,
      "base_learner_reports_fn": lambda features, labels: [
          BaseLearnerReport(hparams={}, attributes={}, metrics={}),
      ],
      "included_base_learner_indices": [0],
      "want_base_learner_reports_materialized": [
          MaterializedBaseLearnerReport(
              hparams={},
              attributes={},
              metrics={},
              included_in_final_ensemble=True,
          ),
      ],
  }, {
      "testcase_name": "one_base_learner",
      "input_fn": tu.dummy_input_fn([[1., 2]], [[3.]]),
      "steps": 3,
      "base_learner_reports_fn": lambda features, labels: [
          BaseLearnerReport(
              hparams={
                  "learning_rate": 1.e-5,
                  "optimizer": "sgd",
                  "num_layers": 0,
                  "use_side_inputs": True,
              },
              attributes={
                  "weight_norms": tf.constant(3.14),
                  "foo": tf.constant("bar"),
                  "parameters": tf.constant(7777),
                  "boo": tf.constant(True),
              },
              metrics={},
          ),
      ],
      "included_base_learner_indices": [0],
      "want_base_learner_reports_materialized": [
          MaterializedBaseLearnerReport(
              hparams={
                  "learning_rate": 1.e-5,
                  "optimizer": "sgd",
                  "num_layers": 0,
                  "use_side_inputs": True,
              },
              attributes={
                  "weight_norms": 3.14,
                  "foo": "bar",
                  "parameters": 7777,
                  "boo": True,
              },
              metrics={},
              included_in_final_ensemble=True,
          ),
      ],
  }, {
      "testcase_name": "two_base_learners",
      "input_fn": tu.dummy_input_fn([[1., 2]], [[3.]]),
      "steps": 3,
      "base_learner_reports_fn": lambda features, labels: [
          BaseLearnerReport(
              hparams={
                  "learning_rate": 1.e-5,
                  "optimizer": "sgd",
                  "num_layers": 0,
                  "use_side_inputs": True,
              },
              attributes={
                  "weight_norms": tf.constant(3.14),
                  "foo": tf.constant("bar"),
                  "parameters": tf.constant(7777),
                  "boo": tf.constant(True),
              },
              metrics={},
          ),
          BaseLearnerReport(
              hparams={
                  "learning_rate": 1.e-6,
                  "optimizer": "sgd",
                  "num_layers": 1,
                  "use_side_inputs": True,
              },
              attributes={
                  "weight_norms": tf.constant(3.1445),
                  "foo": tf.constant("baz"),
                  "parameters": tf.constant(7788),
                  "boo": tf.constant(True),
              },
              metrics={},
          ),
      ],
      "included_base_learner_indices": [1],
      "want_base_learner_reports_materialized": [
          MaterializedBaseLearnerReport(
              hparams={
                  "learning_rate": 1.e-5,
                  "optimizer": "sgd",
                  "num_layers": 0,
                  "use_side_inputs": True,
              },
              attributes={
                  "weight_norms": 3.14,
                  "foo": "bar",
                  "parameters": 7777,
                  "boo": True,
              },
              metrics={},
              included_in_final_ensemble=False,
          ),
          MaterializedBaseLearnerReport(
              hparams={
                  "learning_rate": 1.e-6,
                  "optimizer": "sgd",
                  "num_layers": 1,
                  "use_side_inputs": True,
              },
              attributes={
                  "weight_norms": 3.1445,
                  "foo": "baz",
                  "parameters": 7788,
                  "boo": True,
              },
              metrics={},
              included_in_final_ensemble=True,
          ),
      ],
  }, {
      "testcase_name": "two_base_learners_zero_included",
      "input_fn": tu.dummy_input_fn([[1., 2]], [[3.]]),
      "steps": 3,
      "base_learner_reports_fn": lambda features, labels: [
          BaseLearnerReport(
              hparams={},
              attributes={},
              metrics={},
          ),
          BaseLearnerReport(
              hparams={},
              attributes={},
              metrics={},
          ),
      ],
      "included_base_learner_indices": [],
      "want_base_learner_reports_materialized": [
          MaterializedBaseLearnerReport(
              hparams={},
              attributes={},
              metrics={},
              included_in_final_ensemble=False,
          ),
          MaterializedBaseLearnerReport(
              hparams={},
              attributes={},
              metrics={},
              included_in_final_ensemble=False,
          ),
      ],
  }, {
      "testcase_name": "two_base_learners_both_included",
      "input_fn": tu.dummy_input_fn([[1., 2]], [[3.]]),
      "steps": 3,
      "base_learner_reports_fn": lambda features, labels: [
          BaseLearnerReport(
              hparams={},
              attributes={},
              metrics={},
          ),
          BaseLearnerReport(
              hparams={},
              attributes={},
              metrics={},
          ),
      ],
      "included_base_learner_indices": [0, 1],
      "want_base_learner_reports_materialized": [
          MaterializedBaseLearnerReport(
              hparams={},
              attributes={},
              metrics={},
              included_in_final_ensemble=True,
          ),
          MaterializedBaseLearnerReport(
              hparams={},
              attributes={},
              metrics={},
              included_in_final_ensemble=True,
          ),
      ],
  }, {
      "testcase_name": "materialize_metrics",
      "input_fn": tu.dummy_input_fn([[1., 1.], [1., 1.], [1., 1.]],
                                    [[1.], [2.], [3.]]),
      "steps": 3,
      "base_learner_reports_fn": lambda features, labels: [
          BaseLearnerReport(
              hparams={},
              attributes={},
              metrics={"moo": tf.metrics.mean(labels)},
          ),
      ],
      "included_base_learner_indices": [0],
      "want_base_learner_reports_materialized": [
          MaterializedBaseLearnerReport(
              hparams={},
              attributes={},
              metrics={"moo": 2.},
              included_in_final_ensemble=True,
          ),
      ],
  })
  def test_materialize_base_learner_reports(
      self, input_fn, steps, base_learner_reports_fn,
      included_base_learner_indices, want_base_learner_reports_materialized):
    tf.constant(0.)  # dummy op so that the session graph is never empty.
    features, labels = input_fn()
    base_learner_reports = base_learner_reports_fn(features, labels)
    with self.test_session() as sess:
      sess.run(tf.initializers.local_variables())
      report_materializer = ReportMaterializer(input_fn=input_fn, steps=steps)
      base_learner_reports_materialized = (
          report_materializer.materialize_base_learner_reports(
              sess, base_learner_reports, included_base_learner_indices))
      self.assertEqual(
          len(want_base_learner_reports_materialized),
          len(base_learner_reports_materialized))
      for (want_base_learner_report_materialized,
           base_learner_report_materialized) in zip(
               want_base_learner_reports_materialized,
               base_learner_reports_materialized):
        self.assertEqual(
            set(want_base_learner_report_materialized.hparams.keys()),
            set(base_learner_report_materialized.hparams.keys()))
        for hparam_key, want_hparam in (
            want_base_learner_report_materialized.hparams.items()):
          if isinstance(want_hparam, float):
            self.assertAllClose(
                want_hparam,
                base_learner_report_materialized.hparams[hparam_key])
          else:
            self.assertEqual(
                want_hparam,
                base_learner_report_materialized.hparams[hparam_key])

        self.assertSetEqual(
            set(want_base_learner_report_materialized.attributes.keys()),
            set(base_learner_report_materialized.attributes.keys()))
        for attribute_key, want_attribute in (
            want_base_learner_report_materialized.attributes.items()):
          if isinstance(want_attribute, float):
            self.assertAllClose(
                want_attribute,
                base_learner_report_materialized.attributes[attribute_key])
          else:
            self.assertEqual(
                want_attribute,
                base_learner_report_materialized.attributes[attribute_key])

        self.assertSetEqual(
            set(want_base_learner_report_materialized.metrics.keys()),
            set(base_learner_report_materialized.metrics.keys()))
        for metric_key, want_metric in (
            want_base_learner_report_materialized.metrics.items()):
          if isinstance(want_metric, float):
            self.assertAllClose(
                want_metric,
                base_learner_report_materialized.metrics[metric_key])
          else:
            self.assertEqual(
                want_metric,
                base_learner_report_materialized.metrics[metric_key])


if __name__ == "__main__":
  tf.test.main()
