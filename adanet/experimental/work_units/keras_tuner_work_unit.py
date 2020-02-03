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

import os
import time

from adanet.experimental.work_units import work_unit
from kerastuner.engine.tuner import Tuner
import tensorflow.compat.v2 as tf


class KerasTunerWorkUnit(work_unit.WorkUnit):
  """Trains, evaluates and saves a tuned Keras model."""

  def __init__(self, tuner: Tuner, *search_args, **search_kwargs):
    self._tuner = tuner
    self._search_args = search_args
    self._search_kwargs = search_kwargs

  # TODO: Allow better customization of TensorBoard log_dir.
  def execute(self):
    log_dir = os.path.join('/tmp', str(int(time.time())))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 update_freq='batch')
    # We don't need to eval and store, because the Tuner does it for us.
    self._tuner.search(callbacks=[tensorboard], *self._search_args,
                       **self._search_kwargs)
