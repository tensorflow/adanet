# Copyright 2019 The AdaNet Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adanet implementation for an ensembler for the mean of subnetwork logits."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from adanet.ensemble.ensembler import Ensemble
from adanet.ensemble.ensembler import Ensembler
import tensorflow as tf


class MeanEnsemble(
    collections.namedtuple('MeanEnsemble',
                           ['logits', 'subnetworks', 'predictions']),
    Ensemble):
  r"""Mean ensemble.

  Attributes:
    logits: Logits :class:`tf.Tensor` or dict of string to logits
      :class:`tf.Tensor` (for multi-head).
    subnetworks: List of :class:`adanet.subnetwork.Subnetwork` instances that
      form this ensemble.
    predictions: Optional dict mapping prediction keys to Tensors. MeanEnsembler
      can export mean_last_layer if the subnetworks have the last_layer of the
      same dimension.
  """
  # Key in predictions and export_outputs for mean of last_layer.
  MEAN_LAST_LAYER = 'mean_last_layer'

  def __new__(cls,
              logits,
              subnetworks=None,
              predictions=None):
    return super(MeanEnsemble, cls).__new__(
        cls,
        logits=logits,
        subnetworks=list(subnetworks or []),
        predictions=predictions)


class MeanEnsembler(Ensembler):
  # pyformat: disable
  r"""Ensembler that takes the mean of logits returned by its subnetworks.

  Attributes:
    name: Optional name for the ensembler. Defaults to 'complexity_regularized'.
    add_mean_last_layer_predictions: Set to True to add mean of last_layer in
      subnetworks in estimator's predictions and export outputs.
  """
  # pyformat: enable

  def __init__(self,
               name=None, add_mean_last_layer_predictions=False):
    self._name = name
    self._add_mean_last_layer_predictions = add_mean_last_layer_predictions

  @property
  def name(self):
    if self._name:
      return self._name
    return 'mean'

  def _assert_last_layer_compatible_shapes(self, tensors):
    if not tensors:
      return True
    first_shape = tensors[0].shape
    for tensor in tensors:
      try:
        first_shape.assert_is_compatible_with(tensor.shape)
      except ValueError:
        raise ValueError(
            'Shape of `last_layer` tensors must be same if setting '
            '`add_mean_last_layer_predictions` to True. Found %s vs %s.'
            % (first_shape, tensor.shape))
    return True

  def build_ensemble(self, subnetworks, previous_ensemble_subnetworks, features,
                     labels, logits_dimension, training, iteration_step,
                     summary, previous_ensemble):
    del features, labels, logits_dimension, training, iteration_step  # unused
    del previous_ensemble_subnetworks  # unused

    if isinstance(subnetworks[0].logits, dict):
      mean_logits = {
          key: tf.math.reduce_mean(
              tf.stack([s.logits[key] for s in subnetworks]), axis=0)
          for key in subnetworks[0].logits
      }
    else:
      mean_logits = tf.math.reduce_mean(
          tf.stack([s.logits for s in subnetworks]), axis=0)

    mean_last_layer = None
    if self._add_mean_last_layer_predictions:
      mean_last_layer = {}
      if isinstance(subnetworks[0].last_layer, dict):
        for key in subnetworks[0].logits:
          last_layers = [s.last_layer[key] for s in subnetworks]
          self._assert_last_layer_compatible_shapes(last_layers)
          mean_last_layer['{}_{}'.format(MeanEnsemble.MEAN_LAST_LAYER,
                                         key)] = tf.math.reduce_mean(
                                             tf.stack(last_layers), axis=0)
      else:
        last_layers = [subnetwork.last_layer for subnetwork in subnetworks]
        self._assert_last_layer_compatible_shapes(last_layers)
        mean_last_layer = {
            MeanEnsemble.MEAN_LAST_LAYER:
                tf.math.reduce_mean(tf.stack(last_layers), axis=0)
        }

    return MeanEnsemble(
        subnetworks=subnetworks,
        logits=mean_logits,
        predictions=mean_last_layer)

  def build_train_op(self, ensemble, loss, var_list, labels, iteration_step,
                     summary, previous_ensemble):
    del ensemble, loss, var_list, labels, iteration_step, summary  # unused
    del previous_ensemble  # unused
    return tf.no_op()
