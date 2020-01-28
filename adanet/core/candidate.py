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

from adanet import tf_compat
import tensorflow as tf


class _Candidate(
    collections.namedtuple("_Candidate",
                           ["ensemble_spec", "adanet_loss", "variables"])):
  """An AdaNet candidate.

  A `_Candidate` tracks the progress of a candidate subnetwork's training
  within an ensemble, as well as their AdaNet loss over time.
  """

  def __new__(cls, ensemble_spec, adanet_loss, variables):
    """Creates a validated `_Candidate` instance.

    Args:
      ensemble_spec: The `_EnsembleSpec` instance to track.
      adanet_loss: float `Tensor` representing the ensemble's AdaNet loss on the
        training set as defined in Equation (4) of the paper.
      variables: List of `tf.Variable` instances associated with the ensemble.

    Returns:
      A validated `_Candidate` object.

    Raises:
      ValueError: If validation fails.
    """

    if ensemble_spec is None:
      raise ValueError("ensemble_spec is required")
    if adanet_loss is None:
      raise ValueError("adanet_loss is required")
    return super(_Candidate, cls).__new__(
        cls,
        ensemble_spec=ensemble_spec,
        adanet_loss=adanet_loss,
        variables=variables)


class _CandidateBuilder(object):
  """Builds AdaNet candidates."""

  def __init__(self, adanet_loss_decay=.999):
    """Creates a `_CandidateBuilder` instance.

    Args:
      adanet_loss_decay: Float. The adanet loss is tracked as an exponential
        moving average, so this is the decay rate to use.

    Returns:
      A `_CandidateBuilder` object.
    """

    self._adanet_loss_decay = adanet_loss_decay
    super(_CandidateBuilder, self).__init__()

  def build_candidate(self,
                      ensemble_spec,
                      training,
                      summary,
                      rebuilding=False,
                      track_moving_average=True):
    """Builds and returns an AdaNet candidate.

    Args:
      ensemble_spec: `_EnsembleSpec` instance to track.
      training: A python boolean indicating whether the graph is in training
        mode or prediction mode.
      summary: A `Summary` for recording summaries for TensorBoard.
      rebuilding: Boolean whether the iteration is being rebuilt only to restore
        the previous best subnetworks and ensembles.
      track_moving_average: Bool whether to track the moving average of the
        ensemble's adanet loss.

    Returns:
      A _Candidate instance.
    """

    from tensorflow.python.training import moving_averages  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

    candidate_scope = "candidate_{}".format(ensemble_spec.name)

    with tf_compat.v1.variable_scope(candidate_scope):
      adanet_loss = ensemble_spec.adanet_loss
      variables = []
      if track_moving_average:
        loss_to_track = ensemble_spec.adanet_loss
        if loss_to_track is None:
          # Dummy loss so that we always create moving averages variables.
          # If we pass a None loss assign_moving_average raises an exception.
          loss_to_track = tf.constant(99999., name="dummy_adanet_loss")

        adanet_loss_var = tf_compat.v1.get_variable(
            "adanet_loss", initializer=0., trainable=False)
        update_adanet_loss_op = moving_averages.assign_moving_average(
            adanet_loss_var, loss_to_track, decay=self._adanet_loss_decay)
        # Get the two moving average variables created by assign_moving_average.
        # Since these two variables are the most recently created ones, we can
        # slice them both from the end of the global variables collection.
        variables = [adanet_loss_var] + tf_compat.v1.global_variables()[-2:]
        if training and not rebuilding:
          with tf.control_dependencies([update_adanet_loss_op]):
            adanet_loss = adanet_loss_var.read_value()
        else:
          adanet_loss = adanet_loss_var.read_value()

      with summary.current_scope():
        summary.scalar("adanet_loss/adanet/adanet_weighted_ensemble",
                       adanet_loss)

      return _Candidate(
          ensemble_spec=ensemble_spec,
          adanet_loss=adanet_loss,
          variables=variables)
