"""Materializes the BaseLearnerReports.

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

from adanet.core.base_learner_report import MaterializedBaseLearnerReport
import tensorflow as tf


class ReportMaterializer(object):
  """Materializes BaseLearnerReport."""

  def __init__(self, input_fn, steps):
    """Initializes an `ReportMaterializer` instance.

    Args:
      input_fn: Input function returning a tuple of:
        features - Dictionary of string feature name to `Tensor`.
        labels - `Tensor` of labels.
      steps: Number of steps for which to materialize the ensembles. If an
        `OutOfRangeError` occurs, materialization stops. If set to None, will
        iterate the dataset until all inputs are exhausted.

    Returns:
      A `ReportMaterializer` instance.
    """

    self._input_fn = input_fn
    self._steps = steps
    super(ReportMaterializer, self).__init__()

  @property
  def input_fn(self):
    """Returns the input_fn that materialize_base_learner_reports would run on.

    Even though this property appears to be unused, it would be used to build
    the AdaNet model graph inside AdaNet estimator.train(). After the graph is
    built, the queue_runners are started and the initializers are run,
    AdaNet estimator.train() passes its tf.Session as an argument to
    materialize_base_learner_reports(), thus indirectly making input_fn
    available to materialize_base_learner_reports.
    """
    return self._input_fn

  @property
  def steps(self):
    """Return the number of steps."""
    return self._steps

  def materialize_base_learner_reports(self, sess, iteration_number,
                                       base_learner_reports,
                                       included_base_learner_names):
    """Materializes the Tensor objects in base_learner_reports using sess.

    This converts the Tensors in base_learner_reports to ndarrays, logs the
    progress, converts the ndarrays to python primitives, then packages them
    into `MaterializedBaseLearnerReports`.

    Args:
      sess: `Session` instance with most recent variable values loaded.
      iteration_number: Integer iteration number.
      base_learner_reports: Dict mapping string names to `BaseLearnerReport`
        objects to be materialized.
      included_base_learner_names: List of string names of the
        `BaseLearnerReport`s that are included in the final ensemble.

    Returns:
      List of `MaterializedBaseLearnerReport` objects.
    """

    # Extract the Tensors to be materialized.
    tensors_to_materialize = {
        name: {
            "attributes": base_learner_report.attributes,
            "metrics": {
                # A metric is really a tuple where the first element is a Tensor
                # and the second element is an update op that evaluates to a
                # Tensor.
                # Using the second element ensures that the Tensor is up to date
                # and allows us to avoid reading and updating the metric Tensor
                # in a non-deterministic order.
                metric_key: metric_tuple[1]
                for metric_key, metric_tuple
                in base_learner_report.metrics.items()
            },
        }
        for name, base_learner_report in base_learner_reports.items()
    }

    steps_completed = 0
    materialized_tensors_dict = {}
    while True:
      if self.steps is not None and steps_completed == self.steps:
        break
      try:
        materialized_tensors_dict = sess.run(tensors_to_materialize)

        steps_completed += 1
        log_frequency = (1 if (self.steps is None or self.steps < 10) else
                         math.floor(self.steps / 10.))
        if self.steps is None:
          tf.logging.info("Report materialization [%d]", steps_completed)
        elif (steps_completed % log_frequency == 0 or
              self.steps == steps_completed):
          tf.logging.info("Report materialization [%d/%d]", steps_completed,
                          self.steps)
      except tf.errors.OutOfRangeError:
        tf.logging.info(
            "Encountered end of input during report materialization")
        break

    tf.logging.info("Materialized base_learner_reports.")

    # Convert scalar ndarrays into python primitives, then place them into
    # MaterializedBaseLearnerReports.
    return [
        MaterializedBaseLearnerReport(
            iteration_number=iteration_number,
            name=name,
            hparams=base_learner_reports[name].hparams,
            attributes={
                key: value.item() if hasattr(value, "item") else value
                for key, value in materialized_tensors["attributes"].items()
            },
            metrics={
                key: value.item() if hasattr(value, "item") else value
                for key, value in materialized_tensors["metrics"].items()
            },
            included_in_final_ensemble=(name in included_base_learner_names),
        )
        for name, materialized_tensors in materialized_tensors_dict.items()
    ]
