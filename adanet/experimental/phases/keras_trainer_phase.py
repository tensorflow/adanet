# Lint as: python3
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

from typing import Callable, Iterable, Iterator, Union
from adanet.experimental.phases.phase import DatasetProvider
from adanet.experimental.phases.phase import ModelProvider
from adanet.experimental.storages.in_memory_storage import InMemoryStorage
from adanet.experimental.storages.storage import Storage
from adanet.experimental.work_units.keras_trainer_work_unit import KerasTrainerWorkUnit
from adanet.experimental.work_units.work_unit import WorkUnit
import tensorflow as tf


class KerasTrainerPhase(DatasetProvider, ModelProvider):
  """Trains Keras models."""

  def __init__(self,
               models: Union[Iterable[tf.keras.Model],
                             Callable[[], Iterable[tf.keras.Model]]],
               storage: Storage = InMemoryStorage()):
    # TODO: Consume arbitary fit inputs.
    # Dataset should be wrapped inside a work unit.
    # For instance when you create KerasTrainer work unit the dataset is
    # encapsulated inside that work unit.
    # What if you want to run on different (parts of the) datasets
    # what if a work units consumes numpy arrays?
    super().__init__(storage)
    self._models = models

  def work_units(self, previous_phase: DatasetProvider) -> Iterator[WorkUnit]:
    self._train_dataset = previous_phase.get_train_dataset()
    self._eval_dataset = previous_phase.get_eval_dataset()
    models = self._models
    if callable(models):
      models = models()
    for model in models:
      yield KerasTrainerWorkUnit(model, self._train_dataset, self._eval_dataset,
                                 self._storage)

  def get_models(self) -> Iterable[tf.keras.Model]:
    return self._storage.get_models()

  def get_best_models(self, num_models) -> Iterable[tf.keras.Model]:
    return self._storage.get_best_models(num_models)

  def get_train_dataset(self) -> tf.data.Dataset:
    return self._train_dataset

  def get_eval_dataset(self) -> tf.data.Dataset:
    return self._eval_dataset
