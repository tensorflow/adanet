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

import sys

from typing import Callable, Iterable, Iterator, Union
from adanet.experimental.phases.phase import DatasetProvider
from adanet.experimental.phases.phase import ModelProvider
from adanet.experimental.work_units.keras_tuner_work_unit import KerasTunerWorkUnit
from adanet.experimental.work_units.work_unit import WorkUnit
from kerastuner.engine.tuner import Tuner
import tensorflow.compat.v2 as tf


class KerasTunerPhase(DatasetProvider, ModelProvider):
  """Tunes Keras Model hyperparameters using the Keras Tuner."""

  def __init__(self, tuner: Union[Callable[..., Tuner], Tuner], *search_args,
               **search_kwargs):
    """Initializes a KerasTunerPhase.

    Args:
      tuner: A `kerastuner.tuners.tuner.Tuner` instance or a callable that
        returns a `kerastuner.tuners.tuner.Tuner` instance.
      *search_args: Arguments to pass to the tuner search method.
      **search_kwargs: Keyword arguments to pass to the tuner search method.
    """

    if callable(tuner):
      self._tuner = tuner()
    else:
      self._tuner = tuner
    self._search_args = search_args
    self._search_kwargs = search_kwargs

  def work_units(self, previous_phase: DatasetProvider) -> Iterator[WorkUnit]:
    self._train_dataset = previous_phase.get_train_dataset()
    self._eval_dataset = previous_phase.get_eval_dataset()
    yield KerasTunerWorkUnit(
        self._tuner,
        x=self._train_dataset,
        validation_data=self._eval_dataset,
        *self._search_args,
        **self._search_kwargs)

  # TODO: Find a better way to get all models than to pass in a
  # large number.
  def get_models(self) -> Iterable[tf.keras.Model]:
    return self._tuner.get_best_models(num_models=sys.maxsize)

  def get_best_models(self, num_models) -> Iterable[tf.keras.Model]:
    return self._tuner.get_best_models(num_models=num_models)

  def get_train_dataset(self) -> tf.data.Dataset:
    return self._train_dataset

  def get_eval_dataset(self) -> tf.data.Dataset:
    return self._eval_dataset
