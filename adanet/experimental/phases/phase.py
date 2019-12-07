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
"""A phase in the AdaNet workflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import abc

from adanet.experimental.storages.storage import Storage
from adanet.experimental.work_units.work_unit import WorkUnit
from typing import Iterator, Optional


class Phase(abc.ABC):
  """A stage in a linear workflow.

  A phase is only complete once all its work units complete, as a barrier.
  """

  # TODO: Remove this build function.
  def build(self, storage: Storage, previous: 'Phase' = None):
    self._storage = storage
    self._previous = previous

  @property
  def storage(self) -> Storage:
    return self._storage

  @property
  def previous(self) -> Optional['Phase']:
    return self._previous

  @abc.abstractmethod
  def work_units(self) -> Iterator[WorkUnit]:
    pass
