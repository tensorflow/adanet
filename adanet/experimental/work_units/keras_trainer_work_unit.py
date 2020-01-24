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
"""A work unit for training, evaluating, and saving a Keras model."""

from adanet.experimental.storages.storage import ModelContainer
from adanet.experimental.storages.storage import Storage
from adanet.experimental.work_units import work_unit
import tensorflow as tf


class KerasTrainerWorkUnit(work_unit.WorkUnit):
  """Trains, evaluates, and saves a Keras model."""

  def __init__(self, model: tf.keras.Model,
               train_dataset: tf.data.Dataset,
               eval_dataset: tf.data.Dataset,
               storage: Storage):
    self._model = model
    self._train_dataset = train_dataset
    self._eval_dataset = eval_dataset
    self._storage = storage

  def execute(self):
    self._model.fit(self._train_dataset)
    results = self._model.evaluate(self._eval_dataset)
    # If the model was compiled with metrics, the results is a list of loss +
    # metric values. If the model was compiled without metrics, it is a loss
    # scalar.
    if not isinstance(results, list):
      results = [results]
    self._storage.save_model(ModelContainer(results[0], self._model, results))
