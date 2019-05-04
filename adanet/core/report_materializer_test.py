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
from adanet import subnetwork
from adanet import tf_compat
from adanet.core.report_materializer import ReportMaterializer
import adanet.core.testing_utils as tu
import tensorflow as tf


def decode(param):
  """Decodes the given param when it is bytes."""

  if isinstance(param, (float, int)):
    return param
  return param.decode("utf-8")


class ReportMaterializerTest(parameterized.TestCase, tf.test.TestCase):

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      {
          "testcase_name":
              "one_empty_subnetwork",
          "input_fn":
              tu.dummy_input_fn([[1., 2]], [[3.]]),
          "subnetwork_reports_fn":
              lambda features, labels: {
                  "foo":
                      subnetwork.Report(hparams={}, attributes={}, metrics={}),
              },
          "steps":
              3,
          "included_subnetwork_names": ["foo"],
          "want_materialized_reports": [
              subnetwork.MaterializedReport(
                  iteration_number=0,
                  name="foo",
                  hparams={},
                  attributes={},
                  metrics={},
                  included_in_final_ensemble=True,
              ),
          ],
      }, {
          "testcase_name":
              "one_subnetwork",
          "input_fn":
              tu.dummy_input_fn([[1., 2]], [[3.]]),
          "subnetwork_reports_fn":
              lambda features, labels: {
                  "foo":
                      subnetwork.Report(
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
              },
          "steps":
              3,
          "included_subnetwork_names": ["foo"],
          "want_materialized_reports": [
              subnetwork.MaterializedReport(
                  iteration_number=0,
                  name="foo",
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
          "testcase_name":
              "one_subnetwork_iteration_2",
          "input_fn":
              tu.dummy_input_fn([[1., 2]], [[3.]]),
          "subnetwork_reports_fn":
              lambda features, labels: {
                  "foo":
                      subnetwork.Report(
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
              },
          "steps":
              3,
          "iteration_number":
              2,
          "included_subnetwork_names": ["foo"],
          "want_materialized_reports": [
              subnetwork.MaterializedReport(
                  iteration_number=2,
                  name="foo",
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
          "testcase_name":
              "two_subnetworks",
          "input_fn":
              tu.dummy_input_fn([[1., 2]], [[3.]]),
          "subnetwork_reports_fn":
              lambda features, labels: {
                  "foo1":
                      subnetwork.Report(
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
                  "foo2":
                      subnetwork.Report(
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
              },
          "steps":
              3,
          "included_subnetwork_names": ["foo2"],
          "want_materialized_reports": [
              subnetwork.MaterializedReport(
                  iteration_number=0,
                  name="foo1",
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
              subnetwork.MaterializedReport(
                  iteration_number=0,
                  name="foo2",
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
          "testcase_name":
              "two_subnetworks_zero_included",
          "input_fn":
              tu.dummy_input_fn([[1., 2]], [[3.]]),
          "subnetwork_reports_fn":
              lambda features, labels: {
                  "foo1":
                      subnetwork.Report(
                          hparams={},
                          attributes={},
                          metrics={},
                      ),
                  "foo2":
                      subnetwork.Report(
                          hparams={},
                          attributes={},
                          metrics={},
                      ),
              },
          "steps":
              3,
          "included_subnetwork_names": [],
          "want_materialized_reports": [
              subnetwork.MaterializedReport(
                  iteration_number=0,
                  name="foo1",
                  hparams={},
                  attributes={},
                  metrics={},
                  included_in_final_ensemble=False,
              ),
              subnetwork.MaterializedReport(
                  iteration_number=0,
                  name="foo2",
                  hparams={},
                  attributes={},
                  metrics={},
                  included_in_final_ensemble=False,
              ),
          ],
      }, {
          "testcase_name":
              "two_subnetworks_both_included",
          "input_fn":
              tu.dummy_input_fn([[1., 2]], [[3.]]),
          "subnetwork_reports_fn":
              lambda features, labels: {
                  "foo1":
                      subnetwork.Report(
                          hparams={},
                          attributes={},
                          metrics={},
                      ),
                  "foo2":
                      subnetwork.Report(
                          hparams={},
                          attributes={},
                          metrics={},
                      ),
              },
          "steps":
              3,
          "included_subnetwork_names": ["foo1", "foo2"],
          "want_materialized_reports": [
              subnetwork.MaterializedReport(
                  iteration_number=0,
                  name="foo1",
                  hparams={},
                  attributes={},
                  metrics={},
                  included_in_final_ensemble=True,
              ),
              subnetwork.MaterializedReport(
                  iteration_number=0,
                  name="foo2",
                  hparams={},
                  attributes={},
                  metrics={},
                  included_in_final_ensemble=True,
              ),
          ],
      }, {
          "testcase_name":
              "materialize_metrics",
          "input_fn":
              tu.dummy_input_fn([[1., 1.], [1., 1.], [1., 1.]],
                                [[1.], [2.], [3.]]),
          "subnetwork_reports_fn":
              lambda features, labels: {
                  "foo":
                      subnetwork.Report(
                          hparams={},
                          attributes={},
                          metrics={"moo": tf_compat.v1.metrics.mean(labels)},
                      ),
              },
          "steps":
              3,
          "included_subnetwork_names": ["foo"],
          "want_materialized_reports": [
              subnetwork.MaterializedReport(
                  iteration_number=0,
                  name="foo",
                  hparams={},
                  attributes={},
                  metrics={"moo": 2.},
                  included_in_final_ensemble=True,
              ),
          ],
      }, {
          "testcase_name":
              "materialize_metrics_none_steps",
          "input_fn":
              tu.dataset_input_fn([[1., 1.], [1., 1.], [1., 1.]],
                                  [[1.], [2.], [3.]]),
          "subnetwork_reports_fn":
              lambda features, labels: {
                  "foo":
                      subnetwork.Report(
                          hparams={},
                          attributes={},
                          metrics={"moo": tf_compat.v1.metrics.mean(labels)},
                      ),
              },
          "steps":
              None,
          "included_subnetwork_names": ["foo"],
          "want_materialized_reports": [
              subnetwork.MaterializedReport(
                  iteration_number=0,
                  name="foo",
                  hparams={},
                  attributes={},
                  metrics={"moo": 2.},
                  included_in_final_ensemble=True,
              ),
          ],
      }, {
          "testcase_name":
              "materialize_metrics_non_tensor_op",
          "input_fn":
              tu.dummy_input_fn([[1., 2]], [[3.]]),
          "subnetwork_reports_fn":
              lambda features, labels: {
                  "foo":
                      subnetwork.Report(
                          hparams={},
                          attributes={},
                          metrics={"moo": (tf.constant(42), tf.no_op())},
                      ),
              },
          "steps":
              3,
          "included_subnetwork_names": ["foo"],
          "want_materialized_reports": [
              subnetwork.MaterializedReport(
                  iteration_number=0,
                  name="foo",
                  hparams={},
                  attributes={},
                  metrics={"moo": 42},
                  included_in_final_ensemble=True,
              ),
          ],
      })
  def test_materialize_subnetwork_reports(self,
                                          input_fn,
                                          subnetwork_reports_fn,
                                          steps,
                                          iteration_number=0,
                                          included_subnetwork_names=None,
                                          want_materialized_reports=None):
    tf.constant(0.)  # dummy op so that the session graph is never empty.
    features, labels = input_fn()
    subnetwork_reports = subnetwork_reports_fn(features, labels)
    with self.test_session() as sess:
      sess.run(tf_compat.v1.initializers.local_variables())
      report_materializer = ReportMaterializer(input_fn=input_fn, steps=steps)
      materialized_reports = (
          report_materializer.materialize_subnetwork_reports(
              sess, iteration_number, subnetwork_reports,
              included_subnetwork_names))
      self.assertEqual(
          len(want_materialized_reports), len(materialized_reports))
      materialized_reports_dict = {
          blrm.name: blrm for blrm in materialized_reports
      }
      for want_materialized_report in want_materialized_reports:
        materialized_report = (
            materialized_reports_dict[want_materialized_report.name])
        self.assertEqual(iteration_number, materialized_report.iteration_number)
        self.assertEqual(
            set(want_materialized_report.hparams.keys()),
            set(materialized_report.hparams.keys()))
        for hparam_key, want_hparam in (
            want_materialized_report.hparams.items()):
          if isinstance(want_hparam, float):
            self.assertAllClose(want_hparam,
                                materialized_report.hparams[hparam_key])
          else:
            self.assertEqual(want_hparam,
                             materialized_report.hparams[hparam_key])

        self.assertSetEqual(
            set(want_materialized_report.attributes.keys()),
            set(materialized_report.attributes.keys()))
        for attribute_key, want_attribute in (
            want_materialized_report.attributes.items()):
          if isinstance(want_attribute, float):
            self.assertAllClose(
                want_attribute,
                decode(materialized_report.attributes[attribute_key]))
          else:
            self.assertEqual(
                want_attribute,
                decode(materialized_report.attributes[attribute_key]))

        self.assertSetEqual(
            set(want_materialized_report.metrics.keys()),
            set(materialized_report.metrics.keys()))
        for metric_key, want_metric in (
            want_materialized_report.metrics.items()):
          if isinstance(want_metric, float):
            self.assertAllClose(want_metric,
                                decode(materialized_report.metrics[metric_key]))
          else:
            self.assertEqual(want_metric,
                             decode(materialized_report.metrics[metric_key]))


if __name__ == "__main__":
  tf.test.main()
