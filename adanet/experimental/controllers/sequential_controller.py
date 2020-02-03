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
"""A manual controller for model search."""

from typing import Iterator, Sequence
from adanet.experimental.controllers.controller import Controller
from adanet.experimental.phases.phase import ModelProvider
from adanet.experimental.phases.phase import Phase
from adanet.experimental.work_units.work_unit import WorkUnit
import tensorflow.compat.v2 as tf


class SequentialController(Controller):
  """A controller where the user specifies the sequences of phase to execute."""

  # TODO: Add checks to make sure phases are valid.
  def __init__(self, phases: Sequence[Phase]):
    """Initializes a SequentialController.

    Args:
      phases: A list of `Phase` instances.
    """

    self._phases = phases

  def work_units(self) -> Iterator[WorkUnit]:
    previous_phase = None
    for phase in self._phases:
      for work_unit in phase.work_units(previous_phase):
        yield work_unit
      previous_phase = phase

  def get_best_models(self, num_models: int) -> Sequence[tf.keras.Model]:
    final_phase = self._phases[-1]
    if isinstance(final_phase, ModelProvider):
      return self._phases[-1].get_best_models(num_models)
    raise RuntimeError('Final phase does not provide models.')
