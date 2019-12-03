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
from kerastuner.engine.tuner import Tuner


class KerasTunerWorkUnit(work_unit.WorkUnit):

  def __init__(self, tuner: Tuner, *search_args, **search_kwargs):
    self._tuner = tuner
    self._search_args = search_args
    self._search_kwargs = search_kwargs

  def execute(self):
    # We don't need to eval and store, because the Tuner does it for us.
    self._tuner.search(*self._search_args, **self._search_kwargs)
