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
"""An AdaNet interface for model search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelSearch():
  """A Keras-like interface for performing a model search."""

  # TODO: Add default controller and scheduler.
  def __init__(self, controller, scheduler):  # pylint: disable=unused-argument
    self.ensemble = None

  def compile(self, optimizer, loss, metrics):
    pass

  def fit(self, dataset):
    pass

  # TODO: Since self.ensemble is a `tf.keras.Model`, may be able to
  # fully emulate evaluate, predict and save by simply passing in the arguments
  # to the underlying ensemble.
  def evaluate(self, dataset):
    if not self.ensemble:
      raise RuntimeError("Need to fit an ensemble model before evaluation.")
    self.ensemble.evaluate(dataset)

  def predict(self, dataset):
    if not self.ensemble:
      raise RuntimeError("Need to fit an ensemble model before prediction.")
    self.ensemble.predict(dataset)

  def save(self, filepath):
    if not self.ensemble:
      raise RuntimeError("Need to fit an ensemble model before saving.")
    self.ensemble.save(filepath)
