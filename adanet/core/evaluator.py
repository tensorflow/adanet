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

import tensorflow as tf


class Evaluator(object):
  """Evaluates candidate ensemble performance.

  Args:
    input_fn: Input function returning a tuple of: features - Dictionary of
      string feature name to `Tensor`. labels - `Tensor` of labels.
    steps: Number of steps for which to evaluate the ensembles. If an
      `OutOfRangeError` occurs, evaluation stops. If set to None, will iterate
      the dataset until all inputs are exhausted.

  Returns:
    An :class:`adanet.Evaluator` instance.
  """

  def __init__(self, input_fn, steps=None):
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

  def evaluate_adanet_losses(self, sess, adanet_losses):
    """Evaluates the given AdaNet objectives on the data from `input_fn`.

    The candidates are fed the same batches of features and labels as
    provided by `input_fn`, and their losses are computed and summed over
    `steps` batches.

    Args:
      sess: `Session` instance with most recent variable values loaded.
      adanet_losses: List of AdaNet loss `Tensors`.

    Returns:
      List of evaluated AdaNet losses.
    """

    evals_completed = 0
    if self.steps is None:
      logging_frequency = 1000
    elif self.steps < 10:
      logging_frequency = 1
    else:
      logging_frequency = math.floor(self.steps / 10.)

    adanet_losses = [
        tf.metrics.mean(adanet_loss) for adanet_loss in adanet_losses
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
    return sess.run(adanet_losses)
