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
"""An in process scheduler for managing AdaNet phases."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from adanet.experimental.schedulers import scheduler
from adanet.experimental.work_units.work_unit import WorkUnit
from typing import Iterator


class InProcessScheduler(scheduler.Scheduler):
  """A scheduler that executes in a single process."""

  def schedule(self, work_units: Iterator[WorkUnit]):
    """Schedules and execute work units in a single process.

    Args:
      work_units: An iterator that yields `WorkUnit` instances.
    """

    for work_unit in work_units:
      work_unit.execute()
