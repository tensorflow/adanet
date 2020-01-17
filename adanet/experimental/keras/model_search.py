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
from __future__ import google_type_annotations
from __future__ import print_function

from typing import Sequence

from adanet.experimental.controllers.controller import Controller
from adanet.experimental.schedulers.in_process_scheduler import InProcessScheduler
from adanet.experimental.schedulers.scheduler import Scheduler
import tensorflow as tf


class ModelSearch(object):
  """An AutoML pipeline manager."""

  def __init__(self,
               controller: Controller,
               scheduler: Scheduler = InProcessScheduler()):
    """Initializes a ModelSearch.

    Args:
      controller: A `Controller` instance.
      scheduler: A `Scheduler` instance.
    """

    self._controller = controller
    self._scheduler = scheduler

  def run(self):
    """Executes the training workflow to generate models."""
    self._scheduler.schedule(self._controller.work_units())

  def get_best_models(self, num_models) -> Sequence[tf.keras.Model]:
    """Returns the top models from the run."""
    return self._controller.get_best_models(num_models)
