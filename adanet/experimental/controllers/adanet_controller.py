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
"""An AdaNet controller for model search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from typing import Callable, Iterator, List, Sequence, Union
from adanet.experimental.controllers.controller import Controller
from adanet.experimental.keras.ensemble_model import MeanEnsemble
from adanet.experimental.phases.phase import Phase
from adanet.experimental.storages.in_memory_storage import InMemoryStorage
from adanet.experimental.storages.storage import Storage
from adanet.experimental.work_units.keras_trainer import KerasTrainer
from adanet.experimental.work_units.work_unit import WorkUnit

import tensorflow as tf


class AdaNetCandidatePhase(Phase):
  """Generates and trains neural networks with various layer depths."""

  def __init__(self, dataset: tf.data.Dataset,
               candidates_per_iteration: int,
               optimizer: Union[str, tf.keras.optimizers.Optimizer],
               loss: Union[str, tf.keras.losses.Loss],
               output_units: int,
               metrics: List[Union[str, tf.keras.metrics.Metric]] = None,
               units_per_layer: int = 128,
               layer_activation: Union[str, Callable[..., tf.Tensor]] = 'relu',
               output_activation: Union[str, Callable[...,
                                                      tf.Tensor]] = 'linear'):
    self._dataset = dataset
    self._candidates_per_iteration = candidates_per_iteration
    self._optimizer = optimizer
    self._loss = loss
    self._metrics = metrics
    self._units_per_layer = units_per_layer
    self._output_units = output_units
    self._layer_activation = layer_activation
    self._output_activation = output_activation
    self._candidate_storage = None

  # TODO: Add warning about build not being called.
  def build(self, candidate_storage: Storage):
    self._candidate_storage = candidate_storage

  def work_units(self) -> Iterator[WorkUnit]:
    for network in self._generate_networks():
      yield KerasTrainer(network, self._dataset, self._candidate_storage)

  def _generate_networks(self) -> Iterator[tf.keras.Model]:
    best_candidate = self._candidate_storage.get_best_models(num_models=1)
    if not best_candidate:
      num_layers = 0
    else:
      num_layers = len(best_candidate[0].layers)
    for i in range(self._candidates_per_iteration):
      model = tf.keras.Sequential()
      for _ in range(num_layers+i):
        model.add(tf.keras.layers.Dense(units=self._units_per_layer,
                                        activation=self._layer_activation))
      model.add(tf.keras.layers.Dense(units=self._output_units,
                                      activation=self._output_activation))
      model.compile(optimizer=self._optimizer,
                    loss=self._loss,
                    metrics=self._metrics)
      yield model


# TODO: Make this a more general phase.
class AdaNetEnsemblePhase(Phase):
  """Ensembles submodels."""

  def __init__(self, dataset: tf.data.Dataset,
               candidates_per_iteration: int,
               optimizer: Union[str, tf.keras.optimizers.Optimizer],
               loss: Union[str, tf.keras.losses.Loss],
               metrics: List[Union[str, tf.keras.metrics.Metric]] = None):
    self._dataset = dataset
    self._candidates_per_iteration = candidates_per_iteration
    self._optimizer = optimizer
    self._loss = loss
    self._metrics = metrics
    self._candidate_storage = None
    self._ensemble_storage = None

  def build(self, candidate_storage: Storage, ensemble_storage: Storage):
    self._candidate_storage = candidate_storage
    self._ensemble_storage = ensemble_storage

  @property
  def ensemble_storage(self):
    return self._ensemble_storage

  # TODO: Revisit how newest candidates are obtained within this
  # phase.
  def work_units(self) -> Iterator[WorkUnit]:
    best_candidates = self._candidate_storage.get_newest_models(
        num_models=self._candidates_per_iteration)
    best_ensemble = self._ensemble_storage.get_best_models(num_models=1)
    for candidate in best_candidates:
      if not best_ensemble:
        ensemble = MeanEnsemble([candidate])
      else:
        ensemble = MeanEnsemble(best_ensemble[0].submodels + [candidate])
      ensemble.compile(optimizer=self._optimizer,
                       loss=self._loss,
                       metrics=self._metrics)

      yield KerasTrainer(ensemble, self._dataset, self._ensemble_storage)


class AdaNetController(Controller):
  """A controller that trains candidate networks and ensembles them."""

  def __init__(self,
               candidate_phase: AdaNetCandidatePhase,
               ensemble_phase: AdaNetEnsemblePhase,
               iterations: int,
               candidate_storage: Storage = InMemoryStorage(),
               ensemble_storage: Storage = InMemoryStorage()):
    candidate_phase.build(candidate_storage)
    ensemble_phase.build(candidate_storage, ensemble_storage)
    self._candidate_phase = candidate_phase
    self._ensemble_phase = ensemble_phase
    self._iterations = iterations

  def work_units(self) -> Iterator[WorkUnit]:
    for _ in range(self._iterations):
      for work_unit in itertools.chain(self._candidate_phase.work_units(),
                                       self._ensemble_phase.work_units()):
        yield work_unit

  def get_best_models(self, num_models) -> Sequence[tf.keras.Model]:
    return self._ensemble_phase.ensemble_storage.get_best_models(num_models)
