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

import numpy as np
import tensorflow as tf


class Evaluator(object):
  """An Evaluator selects best trained ensembles."""

  def __init__(self, input_fn, steps):
    """Initializes an `Evaluator` instance.

    Args:
      input_fn: Input function returning a tuple of:
        features - Dictionary of string feature name to `Tensor`.
        labels - `Tensor` of labels.
      steps: Number of steps for which to evaluate the ensembles. If an
        `OutOfRangeError` occurs, evaluation stops. If set to None, will
        iterate the dataset until all inputs are exhausted.

    Returns:
      An `Evaluator` instance.
    """

    self._input_fn = input_fn
    self._steps = steps
    super(Evaluator, self).__init__()

  @property
  def input_fn(self):
    """Return the input_fn."""
    return self._input_fn

  @property
  def steps(self):
    """Return the number of evaluation steps."""
    return self._steps

  # TODO: Rename to "best_candidate_index" and take a
  # list of "adanet_loss" tensors instead of a list of ensembles as an
  # argument
  def best_ensemble_index(self, sess, ensembles):
    """Returns the index of the ensemble with the lowest AdaNet loss.

    The ensembles are each fed the same batches of features and labels as
    provided by `input_fn`, and their losses are computed and summed over
    `steps` batches.

    Args:
      sess: `Session` instance with most recent variable values loaded.
      ensembles: List of trained `Ensemble` objects to compare.

    Returns:
      Index of the candidate with the lowest AdaNet loss.
    """

    total = None
    evals_completed = 0
    while True:
      if self.steps is not None and evals_completed == self.steps:
        break
      try:
        losses = sess.run([ensemble.adanet_loss for ensemble in ensembles])
        if total is not None:
          total += np.array(losses)
        else:
          total = np.array(losses)

        evals_completed += 1
        log_frequency = (1 if (self.steps is None or self.steps < 10) else
                         math.floor(self.steps / 10.))
        if self.steps is None:
          tf.logging.info("Ensemble evaluation [%d]", evals_completed)
        elif (evals_completed % log_frequency == 0 or
              self.steps == evals_completed):
          tf.logging.info("Ensemble evaluation [%d/%d]", evals_completed,
                          self.steps)
      except tf.errors.OutOfRangeError:
        tf.logging.info("Encountered end of input during ensemble evaluation")
        break

    assert len(total) == len(ensembles)

    values = []
    for i in range(len(ensembles)):
      metric_name = "adanet_loss"
      values.append("{}/{} = {:.6f}".format(metric_name, ensembles[i].name,
                                            total[i]))
    tf.logging.info("Computed ensemble metrics: %s", ", ".join(values))
    return np.argmin(total)
