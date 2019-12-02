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
"""A manual controller for model search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from adanet.experimental.controllers.controller import Controller
from adanet.experimental.phases.phase import Phase
from adanet.experimental.work_units.work_unit import WorkUnit
from typing import Iterator, Sequence


class StaticController(Controller):

  def __init__(self, phases: Sequence[Phase]):
    self._phases = phases

  def work_units(self) -> Iterator[WorkUnit]:
    for phase in self._phases:
      for work_unit in phase.work_units():
        yield work_unit
