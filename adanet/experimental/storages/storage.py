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

import abc

from typing import Iterable, List
import tensorflow.compat.v2 as tf


class ModelContainer:
  """A container for a model and its metadata."""

  def __init__(self, score: float, model: tf.keras.Model, metrics: List[float]):
    self.score = score
    self.model = model
    self.metrics = metrics

  def __eq__(self, other: 'ModelContainer'):
    return self.score == other.score

  def __lt__(self, other: 'ModelContainer'):
    return self.score < other.score


class Storage(abc.ABC):
  """A storage for persisting results and managing state."""

  @abc.abstractmethod
  def save_model(self, model_container: ModelContainer):
    """Stores a model and its metadata."""
    # TODO: How do we enforce that save_model is called only once per
    # model?
    pass

  @abc.abstractmethod
  def get_models(self) -> Iterable[tf.keras.Model]:
    """Returns all stored models."""
    pass

  @abc.abstractmethod
  def get_best_models(self, num_models: int = 1) -> Iterable[tf.keras.Model]:
    """Returns the top `num_models` stored models in descending order."""
    pass

  @abc.abstractmethod
  def get_model_metrics(self) -> Iterable[Iterable[float]]:
    """Returns the metrics for all stored models."""
    pass
