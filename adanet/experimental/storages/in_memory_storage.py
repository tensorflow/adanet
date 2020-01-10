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
"""A storage for persisting results and managing stage."""

import heapq

from typing import List
from adanet.experimental.storages.storage import ModelContainer
from adanet.experimental.storages.storage import Storage
import tensorflow as tf


class InMemoryStorage(Storage):
  """In memory storage for testing-only.

  Uses a priority queue under the hood to sort the models according to their
  score.

  Currently the only supported score is 'loss'.
  """

  def __init__(self):
    self._model_containers = []

  def save_model(self, model_container: ModelContainer):
    # We use a counter since heappush will compare on the second item in the
    # tuple in the case of a tie in the first item comparison. This is for the
    # off chance that two models have the same loss.
    heapq.heappush(self._model_containers, model_container)

  def get_models(self) -> List[tf.keras.Model]:
    return [c.model for c in self._model_containers]

  def get_best_models(self, num_models: int = 1) -> List[tf.keras.Model]:
    return [c.model
            for c in heapq.nsmallest(num_models, self._model_containers)]

  def get_model_metrics(self) -> List[List[float]]:
    return [c.metrics for c in self._model_containers]
