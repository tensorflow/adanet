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
"""A work unit for training, evaluating, and saving a Keras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from adanet.experimental.work_units import work_unit
import tensorflow as tf
from typing import Text


class KerasTrainer(work_unit.WorkUnit):
  """Trains, evaluates, and saves a Keras Model."""

  def __init__(self, model: tf.keras.Model, dataset: tf.data.Dataset,
               save_path: Text):
    # TODO: Extend with args passed from top level to fit below.
    self._model = model
    self._dataset = dataset
    self._save_path = save_path

  def execute(self):
    self._model.fit(self._dataset)
    self._model.evaluate(self._dataset)
    self._model.save(self._save_path)
