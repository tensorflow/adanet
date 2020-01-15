# Lint as: python3
# Copyright 2019 The AdaNet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An AdaNet ensemble implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from typing import Sequence

import tensorflow as tf


class EnsembleModel(tf.keras.Model):
  """An ensemble of Keras models."""

  def __init__(self, submodels: Sequence[tf.keras.Model],
               freeze_submodels: bool = True):
    """Initializes an EnsembleModel.

    Args:
      submodels: A list of `tf.keras.Model` that compose the ensemble.
      freeze_submodels: Whether to freeze the weights of submodels.
    """

    super().__init__()
    if freeze_submodels:
      for submodel in submodels:
        for layer in submodel.layers:
          layer.trainable = False
    self._submodels = submodels

  @property
  def submodels(self) -> Sequence[tf.keras.Model]:
    return self._submodels

  def call(self, inputs):
    raise NotImplementedError


class MeanEnsemble(EnsembleModel):
  """An ensemble that averages submodel outputs."""

  def call(self, inputs):
    if len(self._submodels) == 1:
      return self._submodels[0](inputs)

    submodel_outputs = []
    for submodel in self._submodels:
      submodel_outputs.append(submodel(inputs))
    return tf.keras.layers.average(submodel_outputs)


class WeightedEnsemble(EnsembleModel):
  """An ensemble that linearly combines submodel outputs."""

  # TODO: Extract output shapes from submodels instead of passing in
  # as argument.
  def __init__(self, submodels: Sequence[tf.keras.Model], output_units: int):
    """Initializes a WeightedEnsemble.

    Args:
        submodels: A list of `adanet.keras.SubModel` that compose the ensemble.
        output_units: The output size of the last layer of each submodel.
    """

    super().__init__(submodels)
    self.dense = tf.keras.layers.Dense(units=output_units)

  def call(self, inputs):
    submodel_outputs = []
    for submodel in self.submodels:
      submodel_outputs.append(submodel(inputs))
    return self.dense(tf.stack(submodel_outputs))
