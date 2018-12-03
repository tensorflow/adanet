"""Materializes the subnetwork.Reports.

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

import math

from adanet.core import subnetwork
import tensorflow as tf


class ReportMaterializer(object):
  """Materializes reports.

  Specifically it materializes a subnetwork's :class:`adanet.subnetwork.Report`
  instances into :class:`adanet.subnetwork.MaterializedReport` instances.

  Requires an input function `input_fn` that returns a tuple of:

  * features: Dictionary of string feature name to `Tensor`.
  * labels: `Tensor` of labels.

  Args:
    input_fn: The input function.
    steps: Number of steps for which to materialize the ensembles. If an
      `OutOfRangeError` occurs, materialization stops. If set to None, will
      iterate the dataset until all inputs are exhausted.

  Returns:
    A `ReportMaterializer` instance.
  """

  def __init__(self, input_fn, steps=None):
    self._input_fn = input_fn
    self._steps = steps
    super(ReportMaterializer, self).__init__()

  @property
  def input_fn(self):
    """Returns the input_fn that materialize_subnetwork_reports would run on.

    Even though this property appears to be unused, it would be used to build
    the AdaNet model graph inside AdaNet estimator.train(). After the graph is
    built, the queue_runners are started and the initializers are run,
    AdaNet estimator.train() passes its tf.Session as an argument to
    materialize_subnetwork_reports(), thus indirectly making input_fn
    available to materialize_subnetwork_reports.
    """
    return self._input_fn

  @property
  def steps(self):
    """Return the number of steps."""
    return self._steps

  def materialize_subnetwork_reports(self, sess, iteration_number,
                                     subnetwork_reports,
                                     included_subnetwork_names):
    """Materializes the Tensor objects in subnetwork_reports using sess.

    This converts the Tensors in subnetwork_reports to ndarrays, logs the
    progress, converts the ndarrays to python primitives, then packages them
    into `adanet.subnetwork.MaterializedReports`.

    Args:
      sess: `Session` instance with most recent variable values loaded.
      iteration_number: Integer iteration number.
      subnetwork_reports: Dict mapping string names to `subnetwork.Report`
        objects to be materialized.
      included_subnetwork_names: List of string names of the
        `subnetwork.Report`s that are included in the final ensemble.

    Returns:
      List of `adanet.subnetwork.MaterializedReport` objects.
    """

    # A metric is a tuple where the first element is a Tensor and
    # the second element is an update op. We collate the update ops here.
    metric_update_ops = []
    for subnetwork_report in subnetwork_reports.values():
      for metric_tuple in subnetwork_report.metrics.values():
        metric_update_ops.append(metric_tuple[1])

    # Extract the Tensors to be materialized.
    tensors_to_materialize = {}
    for name, subnetwork_report in subnetwork_reports.items():
      metrics = {
          metric_key: metric_tuple[0]
          for metric_key, metric_tuple in subnetwork_report.metrics.items()
      }
      tensors_to_materialize[name] = {
          "attributes": subnetwork_report.attributes,
          "metrics": metrics
      }

    if self.steps is None:
      logging_frequency = 1000
    elif self.steps < 10:
      logging_frequency = 1
    else:
      logging_frequency = math.floor(self.steps / 10.)

    steps_completed = 0
    while True:
      if self.steps is not None and steps_completed == self.steps:
        break
      try:
        steps_completed += 1
        if (steps_completed % logging_frequency == 0 or
            self.steps == steps_completed):
          tf.logging.info("Report materialization [%d/%s]", steps_completed,
                          self.steps or "??")

        sess.run(metric_update_ops)
      except tf.errors.OutOfRangeError:
        tf.logging.info(
            "Encountered end of input during report materialization")
        break

    materialized_tensors_dict = sess.run(tensors_to_materialize)
    tf.logging.info("Materialized subnetwork_reports.")

    # Convert scalar ndarrays into python primitives, then place them into
    # subnetwork.MaterializedReports.
    materialized_reports = []
    for name, materialized_tensors in materialized_tensors_dict.items():
      attributes = {
          key: value.item() if hasattr(value, "item") else value
          for key, value in materialized_tensors["attributes"].items()
      }
      metrics = {
          key: value.item() if hasattr(value, "item") else value
          for key, value in materialized_tensors["metrics"].items()
      }
      materialized_reports.append(
          subnetwork.MaterializedReport(
              iteration_number=iteration_number,
              name=name,
              hparams=subnetwork_reports[name].hparams,
              attributes=attributes,
              metrics=metrics,
              included_in_final_ensemble=(name in included_subnetwork_names)))
    return materialized_reports
