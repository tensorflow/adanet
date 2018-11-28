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
    collections.namedtuple(
        "_Candidate",
        ["ensemble_spec", "adanet_loss", "is_training", "is_previous_best"])):
  """An AdaNet candidate.

  A `_Candidate` tracks the progress of a candidate subnetwork's training
  within an ensemble, as well as their AdaNet loss over time.
  """

  def __new__(cls,
              ensemble_spec,
              adanet_loss,
              is_training,
              is_previous_best=False):
    """Creates a validated `_Candidate` instance.

    Args:
      ensemble_spec: The `_EnsembleSpec` instance to track.
      adanet_loss: float `Tensor` representing the ensemble's AdaNet loss on the
        training set as defined in Equation (4) of the paper.
      is_training: bool `Tensor` indicating if training is ongoing.
      is_previous_best: bool identifying whether this ensemble came from the
        previous iteration.

    Returns:
      A validated `_Candidate` object.

    Raises:
      ValueError: If validation fails.
    """

    if ensemble_spec is None:
      raise ValueError("ensemble_spec is required")
    if adanet_loss is None:
      raise ValueError("adanet_loss is required")
    if is_training is None:
      raise ValueError("is_training is required")
    return super(_Candidate, cls).__new__(
        cls,
        ensemble_spec=ensemble_spec,
        adanet_loss=adanet_loss,
        is_training=is_training,
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

  def build_candidate(self,
                      ensemble_spec,
                      training,
                      iteration_step,
                      summary,
                      is_previous_best=False):
    """Builds and returns an AdaNet candidate.

    Args:
      ensemble_spec: `_EnsembleSpec` instance to track.
      training: A python boolean indicating whether the graph is in training
        mode or prediction mode.
      iteration_step: Integer `Tensor` representing the step since the beginning
        of the current iteration, as opposed to the global step.
      summary: A `Summary` for recording summaries for TensorBoard.
      is_previous_best: Bool identifying whether this ensemble came from a
        previous iteration. If `True`, `is_training` will be `False` since its
        weights are frozen.

    Returns:
      A _Candidate instance.
    """

    candidate_scope = "candidate_{}".format(ensemble_spec.name)

    with tf.variable_scope(candidate_scope):
      adanet_loss = tf.get_variable(
          "adanet_loss", initializer=0., trainable=False)

      if is_previous_best:
        # This candidate is frozen, so it is already done training.
        is_training = tf.constant(False, name="is_training")
      else:
        # Train this candidate for `max_steps` steps.
        # NOTE: During training, the iteration step gets incremented at the very
        # end of the computation graph, so we need to account for that here.
        is_training = tf.less(
            iteration_step + 1 if training else 0,
            self._max_steps,
            name="is_training")

      if training:
        update_adanet_loss_op = moving_averages.assign_moving_average(
            adanet_loss,
            ensemble_spec.adanet_loss,
            decay=self._adanet_loss_decay)
        with tf.control_dependencies([update_adanet_loss_op]):
          adanet_loss = adanet_loss.read_value()

      with summary.current_scope():
        summary.scalar("adanet_loss/adanet/adanet_weighted_ensemble",
                       adanet_loss)
      return _Candidate(
          ensemble_spec=ensemble_spec,
          adanet_loss=adanet_loss,
          is_training=is_training,
          is_previous_best=is_previous_best)
