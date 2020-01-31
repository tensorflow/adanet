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
"""A scheduler for managing AdaNet phases."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from adanet.experimental.work_units.work_unit import WorkUnit
from typing import Iterator


class Scheduler(abc.ABC):
  """Abstract interface for a scheduler to be used in ModelFlow pipelines."""

  @abc.abstractmethod
  def schedule(self, work_units: Iterator[WorkUnit]):
    """Schedules and executes work units.

    Args:
      work_units: An iterator that yields `WorkUnit` instances.
    """
    pass
