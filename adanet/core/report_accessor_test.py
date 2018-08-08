"""Tests for run_report_accessor.py.

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

from adanet.core.base_learner_report import MaterializedBaseLearnerReport
from adanet.core.report_accessor import _ReportAccessor
import tensorflow as tf


class ReportAccessorTest(tf.test.TestCase):

  def test_read_from_empty_file(self):
    report_accessor = _ReportAccessor(self.get_temp_dir())
    self.assertEqual([], list(report_accessor.read_iteration_reports()))

  def test_add_to_empty_file(self):
    report_accessor = _ReportAccessor(self.get_temp_dir())
    materialized_base_learner_reports = [
        MaterializedBaseLearnerReport(
            iteration_number=0,
            name="foo",
            hparams={
                "p1": 1,
                "p2": "hoo",
                "p3": True,
            },
            attributes={
                "a1": 1,
                "a2": "aoo",
                "a3": True,
            },
            metrics={
                "m1": 1,
                "m2": "moo",
                "m3": True,
            },
            included_in_final_ensemble=True,
        ),
    ]

    report_accessor.write_iteration_report(
        iteration_number=0,
        materialized_base_learner_reports=materialized_base_learner_reports,
    )
    actual_iteration_reports = list(report_accessor.read_iteration_reports())

    self.assertEqual(1, len(actual_iteration_reports))
    self.assertEqual(materialized_base_learner_reports,
                     actual_iteration_reports[0])

  def test_add_to_existing_file(self):
    materialized_base_learner_reports = [
        [
            MaterializedBaseLearnerReport(
                iteration_number=0,
                name="foo1",
                hparams={
                    "p1": 11,
                    "p2": "hoo",
                    "p3": True,
                },
                attributes={
                    "a1": 11,
                    "a2": "aoo",
                    "a3": True,
                },
                metrics={
                    "m1": 11,
                    "m2": "moo",
                    "m3": True,
                },
                included_in_final_ensemble=False,
            ),
            MaterializedBaseLearnerReport(
                iteration_number=0,
                name="foo2",
                hparams={
                    "p1": 12,
                    "p2": "hoo",
                    "p3": True,
                },
                attributes={
                    "a1": 12,
                    "a2": "aoo",
                    "a3": True,
                },
                metrics={
                    "m1": 12,
                    "m2": "moo",
                    "m3": True,
                },
                included_in_final_ensemble=True,
            ),
        ],
        [
            MaterializedBaseLearnerReport(
                iteration_number=1,
                name="foo1",
                hparams={
                    "p1": 21,
                    "p2": "hoo",
                    "p3": True,
                },
                attributes={
                    "a1": 21,
                    "a2": "aoo",
                    "a3": True,
                },
                metrics={
                    "m1": 21,
                    "m2": "moo",
                    "m3": True,
                },
                included_in_final_ensemble=True,
            ),
            MaterializedBaseLearnerReport(
                iteration_number=1,
                name="foo2",
                hparams={
                    "p1": 22,
                    "p2": "hoo",
                    "p3": True,
                },
                attributes={
                    "a1": 22,
                    "a2": "aoo",
                    "a3": True,
                },
                metrics={
                    "m1": 22,
                    "m2": "moo",
                    "m3": True,
                },
                included_in_final_ensemble=False,
            ),
        ],
        [
            MaterializedBaseLearnerReport(
                iteration_number=2,
                name="foo1",
                hparams={
                    "p1": 31,
                    "p2": "hoo",
                    "p3": True,
                },
                attributes={
                    "a1": 31,
                    "a2": "aoo",
                    "a3": True,
                },
                metrics={
                    "m1": 31,
                    "m2": "moo",
                    "m3": True,
                },
                included_in_final_ensemble=False,
            ),
            MaterializedBaseLearnerReport(
                iteration_number=2,
                name="foo2",
                hparams={
                    "p1": 32,
                    "p2": "hoo",
                    "p3": True,
                },
                attributes={
                    "a1": 32,
                    "a2": "aoo",
                    "a3": True,
                },
                metrics={
                    "m1": 32,
                    "m2": "moo",
                    "m3": True,
                },
                included_in_final_ensemble=True,
            ),
        ],
    ]

    report_accessor = _ReportAccessor(self.get_temp_dir())

    report_accessor.write_iteration_report(
        0, materialized_base_learner_reports[0])
    report_accessor.write_iteration_report(
        1, materialized_base_learner_reports[1])
    report_accessor.write_iteration_report(
        2, materialized_base_learner_reports[2])
    actual_reports = report_accessor.read_iteration_reports()

    self.assertEqual(materialized_base_learner_reports, actual_reports)


if __name__ == "__main__":
  tf.test.main()
