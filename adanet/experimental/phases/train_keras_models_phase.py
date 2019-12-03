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
"""A phase in the AdaNet workflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from adanet.experimental.phases.model_phase import ModelPhase
from adanet.experimental.work_units.keras_trainer import KerasTrainer
from adanet.experimental.work_units.work_unit import WorkUnit
import tensorflow as tf
from typing import Iterator, Sequence


class TrainKerasModelsPhase(ModelPhase):
  """Trains Keras models."""

  def __init__(self, models: Sequence[tf.keras.Model],
               dataset: tf.data.Dataset):
    # TODO: Consume arbitary fit inputs.
    # Dataset should be wrapped inside a work unit.
    # For instance when you create KerasTrainer work unit the dataset is
    # encapsulated inside that work unit.
    # What if you want to run on different (parts of the) datasets
    # what if a work units consumes numpy arrays?
    self._models = models
    self._dataset = dataset

  def work_units(self) -> Iterator[WorkUnit]:
    for model in self._models:
      yield KerasTrainer(model, self._dataset, self.storage)

  def get_best_models(self, num_models) -> Sequence[tf.keras.Model]:
    return self.storage.get_best_models(num_models)
