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

from adanet.experimental.controllers.controller import Controller
from adanet.experimental.schedulers.in_process import InProcess
from adanet.experimental.schedulers.scheduler import Scheduler
import tensorflow as tf


class ModelSearch(object):
  """A Keras-like interface for performing a model search."""

  def __init__(self, controller: Controller,
               scheduler: Scheduler = InProcess()):
    self._controller = controller
    self._scheduler = scheduler
    self._best_model = None  # type: tf.keras.Model

  def compile(self, optimizer, loss, metrics):
    pass

  def fit(self):
    self._scheduler.schedule(self._controller.work_units())

  # TODO: Since self._best_model is a `tf.keras.Model`, may be able
  # to fully emulate evaluate, predict and save by simply passing in the
  # arguments to the underlying ensemble.
  def evaluate(self, dataset: tf.data.Dataset):
    if not self._best_model:
      raise RuntimeError("Need to fit an ensemble model before evaluation.")
    self._best_model.evaluate(dataset)

  def predict(self, dataset: tf.data.Dataset):
    if not self._best_model:
      raise RuntimeError("Need to fit an ensemble model before prediction.")
    self._best_model.predict(dataset)

  def save(self, filepath):
    # TODO: Note that we should save more than just the best
    # model, so that a given model search run can be restored.
    if not self._best_model:
      raise RuntimeError("Need to fit an ensemble model before saving.")
    self._best_model.save(filepath)
