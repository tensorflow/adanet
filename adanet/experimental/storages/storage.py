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

import abc

import tensorflow as tf
from typing import Sequence


class Storage(abc.ABC):
  """A storage for persisting results and managing state."""

  @abc.abstractmethod
  def save_model(self, model: tf.keras.Model, score: float) -> int:
    # TODO: How do we enforce that save_model is called only once per
    # model?
    pass

  @abc.abstractmethod
  def load_model(self, model_id: int) -> tf.keras.Model:
    pass

  @abc.abstractmethod
  def get_best_models(self, num_models) -> Sequence[tf.keras.Model]:
    # TODO: Rethink get_best_model API since it's defined in Storage,
    # Phases, and Controllers.
    pass
