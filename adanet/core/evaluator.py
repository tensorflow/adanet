"""An AdaNet evaluator implementation in Tensorflow using a single graph.

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

from absl import logging
from adanet import tf_compat
import numpy as np
import tensorflow.compat.v2 as tf


# TODO: Remove uses of Evaluator once AdaNet Ranker is implemented.
class Evaluator(object):
  """Evaluates candidate ensemble performance."""

  class Objective(object):
    """The Evaluator objective for the metric being optimized.

    Two objectives are currently supported:
      - MINIMIZE: Lower is better for the metric being optimized.
      - MAXIMIZE: Higher is better for the metric being optimized.
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

  def __init__(self,
               input_fn,
               metric_name="adanet_loss",
               objective=Objective.MINIMIZE,
               steps=None):
    """Initializes a new Evaluator instance.

    Args:
      input_fn: Input function returning a tuple of: features - Dictionary of
        string feature name to `Tensor`. labels - `Tensor` of labels.
      metric_name: The name of the evaluation metrics to use when choosing the
        best ensemble. Must refer to a valid evaluation metric.
      objective: Either `Objective.MINIMIZE` or `Objective.MAXIMIZE`.
      steps: Number of steps for which to evaluate the ensembles. If an
        `OutOfRangeError` occurs, evaluation stops. If set to None, will iterate
        the dataset until all inputs are exhausted.

    Returns:
      An :class:`adanet.Evaluator` instance.
    """
    self._input_fn = input_fn
    self._steps = steps
    self._metric_name = metric_name
    self._objective = objective
    if objective == self.Objective.MINIMIZE:
      self._objective_fn = np.nanargmin
    elif objective == self.Objective.MAXIMIZE:
      self._objective_fn = np.nanargmax
    else:
      raise ValueError(
          "Evaluator objective must be one of MINIMIZE or MAXIMIZE.")

  @property
  def input_fn(self):
    """Return the input_fn."""
    return self._input_fn

  @property
  def steps(self):
    """Return the number of evaluation steps."""
    return self._steps

  @property
  def metric_name(self):
    """Returns the name of the metric being optimized."""
    return self._metric_name

  @property
  def objective_fn(self):
    """Returns a fn which selects the best metric based on the objective."""
    return self._objective_fn

  def evaluate(self, sess, ensemble_metrics):
    """Evaluates the given AdaNet objectives on the data from `input_fn`.

    The candidates are fed the same batches of features and labels as
    provided by `input_fn`, and their losses are computed and summed over
    `steps` batches.

    Args:
      sess: `Session` instance with most recent variable values loaded.
      ensemble_metrics: A list dictionaries of `tf.metrics` for each candidate
        ensemble.

    Returns:
      List of evaluated metrics.
    """

    evals_completed = 0
    if self.steps is None:
      logging_frequency = 1000
    elif self.steps < 10:
      logging_frequency = 1
    else:
      logging_frequency = math.floor(self.steps / 10.)

    objective_metrics = [em[self._metric_name] for em in ensemble_metrics]

    sess.run(tf_compat.v1.local_variables_initializer())
    while True:
      if self.steps is not None and evals_completed == self.steps:
        break
      try:
        evals_completed += 1
        if (evals_completed % logging_frequency == 0 or
            self.steps == evals_completed):
          logging.info("Ensemble evaluation [%d/%s]", evals_completed,
                       self.steps or "??")
        sess.run(objective_metrics)
      except tf.errors.OutOfRangeError:
        logging.info("Encountered end of input after %d evaluations",
                     evals_completed)
        break

    # Evaluating the first element is idempotent for metric tuples.
    return sess.run([metric[0] for metric in objective_metrics])
