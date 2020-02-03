# Lint as: python3
# Copyright 2020 The AdaNet Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A phase that repeats its inner phases."""

from typing import Callable, Iterable, Iterator, List
from adanet.experimental.phases.phase import DatasetProvider
from adanet.experimental.phases.phase import ModelProvider
from adanet.experimental.phases.phase import Phase
from adanet.experimental.work_units.work_unit import WorkUnit
import tensorflow.compat.v2 as tf


class RepeatPhase(DatasetProvider, ModelProvider):
  """A phase that repeats its inner phases."""

  def __init__(self,
               phase_factory: List[Callable[..., Phase]],
               repetitions: int):
    self._phase_factory = phase_factory
    self._repetitions = repetitions
    self._final_phase = None
    """Initializes a RepeatPhase.

    Args:
      phase_factory: A list of callables that return `Phase` instances.
      repetitions: Number of times to repeat the phases in the phase factory.
    """

  def work_units(self, previous_phase: DatasetProvider) -> Iterator[WorkUnit]:
    for _ in range(self._repetitions):
      # Each repetition, the "first" previous phase is the one preceeding the
      # repeat phase itself.
      prev_phase = previous_phase
      for phase in self._phase_factory:
        phase = phase()
        for work_unit in phase.work_units(prev_phase):
          yield work_unit
        prev_phase = phase
    self._final_phase = prev_phase

  def get_train_dataset(self) -> tf.data.Dataset:
    if not isinstance(self._final_phase, DatasetProvider):
      raise NotImplementedError(
          'The last phase in repetition does not provide datasets.')
    return self._final_phase.get_train_dataset()

  def get_eval_dataset(self) -> tf.data.Dataset:
    if not isinstance(self._final_phase, DatasetProvider):
      raise NotImplementedError(
          'The last phase in repetition does not provide datasets.')
    return self._final_phase.get_eval_dataset()

  def get_models(self) -> Iterable[tf.keras.Model]:
    if not isinstance(self._final_phase, ModelProvider):
      raise NotImplementedError(
          'The last phase in repetition does not provide models.')
    return self._final_phase.get_models()

  def get_best_models(self, num_models=1) -> Iterable[tf.keras.Model]:
    if not isinstance(self._final_phase, ModelProvider):
      raise NotImplementedError(
          'The last phase in repetition does not provide models.')
    return self._final_phase.get_best_models(num_models)
