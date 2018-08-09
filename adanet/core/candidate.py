"""The AdaNet candidate implementation in Tensorflow using a single graph.

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

import collections

import tensorflow as tf

from tensorflow.python.training import moving_averages


class _Candidate(
    collections.namedtuple("_Candidate", [
        "ensemble", "adanet_loss", "is_training", "update_op",
        "is_previous_best"
    ])):
  """An AdaNet candidate.

  A `_Candidate` tracks the progress of a candidate base learner's training
  within an ensemble, as well as their AdaNet loss over time.
  """

  def __new__(cls,
              ensemble,
              adanet_loss,
              is_training,
              update_op=None,
              is_previous_best=False):
    """Creates a validated `_Candidate` instance.

    Args:
      ensemble: The `Ensemble` instance to track.
      adanet_loss: float `Tensor` representing the ensemble's AdaNet loss
        on the training set as defined in Equation (4) of the paper.
      is_training: bool `Tensor` indicating if training is ongoing.
      update_op: Optional op for updating the candidate's variables.
      is_previous_best: bool identifying whether this ensemble came from the
        previous iteration.

    Returns:
      A validated `_Candidate` object.

    Raises:
      ValueError: If validation fails.
    """

    if ensemble is None:
      raise ValueError("ensemble is required")
    if adanet_loss is None:
      raise ValueError("adanet_loss is required")
    if is_training is None:
      raise ValueError("is_training is required")
    if update_op is None:
      update_op = tf.no_op()
    return super(_Candidate, cls).__new__(
        cls,
        ensemble=ensemble,
        adanet_loss=adanet_loss,
        is_training=is_training,
        update_op=update_op,
        is_previous_best=is_previous_best)


class _CandidateBuilder(object):
  """Builds AdaNet candidates."""

  def __init__(self, max_steps, adanet_loss_decay=.999):
    """Creates a `_CandidateBuilder` instance.

    Args:
      max_steps: Total number steps to train this candidates.
      adanet_loss_decay: Float. The adanet loss is tracked as an exponential
        moving average, so this is the decay rate to use.

    Returns:
      A `_CandidateBuilder` object.

    Raises:
      ValueError: If `max_steps` is <= 0.
    """

    if max_steps <= 0:
      raise ValueError("max_steps must be > 0.")

    self._max_steps = max_steps
    self._adanet_loss_decay = adanet_loss_decay
    super(_CandidateBuilder, self).__init__()

  def build_candidate(self, ensemble, training, summary,
                      is_previous_best=False):
    """Builds and returns an AdaNet candidate.

    When creating a candidate from a ensemble, it creates a train loss
    to track its train loss over time. From this information, it compares its
    performance against the previous best candidate, and determines whether
    it should keep training.

    Args:
      ensemble: `Ensemble` instance to track.
      training: A python boolean indicating whether the graph is in training
        mode or prediction mode.
      summary: A `Summary` for recording summaries for TensorBoard.
      is_previous_best: Bool identifying whether this ensemble came from a
        previous iteration. If `True`, `is_training` will be `False` since its
        weights are frozen.

    Returns:
      A _Candidate instance.
    """

    candidate_scope = "candidate_{}".format(ensemble.name)

    with tf.variable_scope(candidate_scope):
      adanet_loss = tf.get_variable(
          "adanet_loss", initializer=0., trainable=False)

      # Counter variable to track the number of steps.
      step_counter = tf.get_variable(
          "step_counter", initializer=1, trainable=False)

      # Update train loss during training so that it's available in other modes.
      update_op = tf.assign_add(step_counter, 1) if training else tf.no_op()

      if is_previous_best:
        # This candidate is frozen, so it is already done training.
        is_training = tf.constant(False, name="is_training")
      else:
        # Train this candidate for `max_steps` steps.
        with tf.control_dependencies([update_op]):
          is_training = tf.less(
              step_counter.read_value(), self._max_steps, name="is_training")

      if training:
        with tf.control_dependencies([ensemble.adanet_loss]):
          update_adanet_loss_op = moving_averages.assign_moving_average(
              adanet_loss, ensemble.adanet_loss, decay=self._adanet_loss_decay)
        with tf.control_dependencies([update_adanet_loss_op]):
          adanet_loss = adanet_loss.read_value()

      with tf.name_scope(""):
        summary.scalar(
            "complexity_regularization/adanet/adanet_weighted_ensemble",
            ensemble.complexity_regularization)
        summary.scalar("adanet_loss/adanet/adanet_weighted_ensemble",
                       adanet_loss)
      return _Candidate(
          ensemble=ensemble,
          adanet_loss=adanet_loss,
          is_training=is_training,
          update_op=update_op,
          is_previous_best=is_previous_best)
