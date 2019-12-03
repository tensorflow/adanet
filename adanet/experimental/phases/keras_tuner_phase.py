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
from adanet.experimental.work_units.keras_tuner_work_unit import KerasTunerWorkUnit
from adanet.experimental.work_units.work_unit import WorkUnit
from kerastuner.engine.tuner import Tuner
import tensorflow as tf
from typing import Iterator, Sequence


class KerasTunerPhase(ModelPhase):
  """Tunes Keras Model hyperparameters using the Keras Tuner."""

  def __init__(self, tuner: Tuner, *search_args, **search_kwargs):
    self._tuner = tuner
    self._search_args = search_args
    self._search_kwargs = search_kwargs

  def work_units(self) -> Iterator[WorkUnit]:
    yield KerasTunerWorkUnit(self._tuner, *self._search_args,
                             **self._search_kwargs)

  def get_best_models(self, num_models) -> Sequence[tf.keras.Model]:
    return self._tuner.get_best_models(num_models=num_models)
