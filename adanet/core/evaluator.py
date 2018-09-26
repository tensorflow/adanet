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

from six.moves import range
import tensorflow as tf


class Evaluator(object):
  """An Evaluator selects best trained ensembles."""

  def __init__(self, input_fn, steps=None):
    """Initializes an `Evaluator` instance.

    Args:
      input_fn: Input function returning a tuple of: features - Dictionary of
        string feature name to `Tensor`. labels - `Tensor` of labels.
      steps: Number of steps for which to evaluate the ensembles. If an
        `OutOfRangeError` occurs, evaluation stops. If set to None, will iterate
        the dataset until all inputs are exhausted.

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

    evals_completed = 0
    if self.steps is None:
      logging_frequency = 1000
    elif self.steps < 10:
      logging_frequency = 1
    else:
      logging_frequency = math.floor(self.steps / 10.)

    adanet_losses = [
        tf.metrics.mean(ensemble.adanet_loss) for ensemble in ensembles
    ]
    sess.run(tf.local_variables_initializer())
    while True:
      if self.steps is not None and evals_completed == self.steps:
        break
      try:
        evals_completed += 1
        if (evals_completed % logging_frequency == 0 or
            self.steps == evals_completed):
          tf.logging.info("Ensemble evaluation [%d/%s]", evals_completed,
                          self.steps or "??")
        sess.run(adanet_losses)
      except tf.errors.OutOfRangeError:
        tf.logging.info("Encountered end of input after %d evaluations",
                        evals_completed)
        break

    # Losses are metric op tuples. Evaluating the first element is idempotent.
    adanet_losses = [loss[0] for loss in adanet_losses]
    evaluated_adanet_losses, best_ensemble_index = sess.run(
        (adanet_losses, tf.argmin(adanet_losses)))
    values = []
    for i in range(len(ensembles)):
      metric_name = "adanet_loss"
      values.append("{}/{} = {:.6f}".format(metric_name, ensembles[i].name,
                                            evaluated_adanet_losses[i]))
    tf.logging.info("Computed ensemble metrics: %s", ", ".join(values))
    return best_ensemble_index
