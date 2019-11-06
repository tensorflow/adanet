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
"""Device placement functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import hashlib


class _OpNameHashStrategy(object):
  """Returns the ps task index for placement using a hash of the op name."""

  def __init__(self, num_tasks):
    """Create a new `_OpNameHashStrategy`.

    Args:
      num_tasks: Number of ps tasks to cycle among.
    """

    self._num_tasks = num_tasks

  def __call__(self, op):
    """Choose a ps task index for the given `Operation`.

    Hashes the op name and assigns it to a ps task modulo the number of tasks.
    This ensures that variables with the same name are always placed on the same
    parameter server.

    Args:
      op: An `Operation` to be placed on ps.

    Returns:
      The ps task index to use for the `Operation`.
    """

    hashed = int(hashlib.sha256(op.name.encode("utf-8")).hexdigest(), 16)
    return hashed % self._num_tasks


@contextlib.contextmanager
def monkey_patch_default_variable_placement_strategy():
  """Monkey patches the default variable placement strategy.

  This strategy is used by tf.train.replica_device_setter. The new strategy
  allows workers to having different graphs from the chief.

  Yields:
    A context with the monkey-patched default variable placement strategy.
  """

  # Import here to avoid strict BUILD deps check.
  from tensorflow.python.training import device_setter  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  old_round_robin_strategy = device_setter._RoundRobinStrategy  # pylint: disable=protected-access
  setattr(device_setter, "_RoundRobinStrategy", _OpNameHashStrategy)
  try:
    yield
  finally:
    setattr(device_setter, "_RoundRobinStrategy", old_round_robin_strategy)
