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
"""A storage for persisting results and managing stage."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import heapq

from adanet.experimental.storages.storage import Storage
import tensorflow as tf
from typing import Sequence


class InMemoryStorage(Storage):
  """In memory storage for testing-only.

  Uses a priority queue under the hood to sort the models according to their
  score.

  Currently the only supported score is 'loss'.
  """

  def __init__(self):
    self._id = 0
    self._models = []

  def save_model(self, model: tf.keras.Model, score: float) -> int:
    model_id = self._id
    heapq.heappush(self._models, (score, model_id, model))
    self._id += 1
    return model_id

  def load_model(self, model_id: int) -> tf.keras.Model:
    for _, id_, model in self._models:
      if id_ == model_id:
        return model
    raise ValueError("Model with id '{}' not found".format(model_id))

  def get_best_models(self, num_models) -> Sequence[tf.keras.Model]:
    return [m for _, _, m in heapq.nsmallest(num_models, self._models)]

  def get_newest_models(self, num_models) -> Sequence[tf.keras.Model]:
    return [
        m for _, m_id, m in self._models
        if m_id in [self._id - i for i in range(num_models)]
    ]
