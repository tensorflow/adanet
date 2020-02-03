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
"""A phase in the AdaNet workflow."""

import abc

from typing import Iterable, Iterator, Optional
from adanet.experimental.storages.in_memory_storage import InMemoryStorage
from adanet.experimental.storages.storage import Storage
from adanet.experimental.work_units.work_unit import WorkUnit
import tensorflow.compat.v2 as tf


class Phase(abc.ABC):
  """A stage in a linear workflow."""

  def __init__(self, storage: Storage = InMemoryStorage()):
    self._storage = storage

  # TODO: Find a better way to ensure work_units only gets called
  # once per phase.
  @abc.abstractmethod
  def work_units(self, previous_phase: Optional['Phase']) -> Iterator[WorkUnit]:
    pass


class DatasetProvider(Phase, abc.ABC):
  """An interface for a phase that produces datasets."""

  def __init__(self, storage: Storage = InMemoryStorage()):
    """Initializes a Phase.

    Args:
      storage: A `Storage` instance.
    """

    super().__init__(storage)
    self._train_dataset = None
    self._eval_dataset = None

  @abc.abstractmethod
  def get_train_dataset(self) -> tf.data.Dataset:
    """Returns the dataset for train data."""
    pass

  @abc.abstractmethod
  def get_eval_dataset(self) -> tf.data.Dataset:
    """Returns the dataset for eval data."""
    pass


class ModelProvider(Phase, abc.ABC):
  """An interface for a phase that produces models."""

  @abc.abstractmethod
  def get_models(self) -> Iterable[tf.keras.Model]:
    """Returns the models produced by this phase."""
    pass

  @abc.abstractmethod
  def get_best_models(self, num_models: int = 1) -> Iterable[tf.keras.Model]:
    """Returns the `k` best models produced by this phase."""
    pass

