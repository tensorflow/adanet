"""A simple timer implementation.

Copyright 2018 The AdaNet Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


class _CountDownTimer(object):
  """A simple count down timer implementation."""

  def __init__(self, duration_secs):
    """Initializes a `_CountDownTimer`.

    Args:
      duration_secs: Float seconds for countdown.

    Returns:
      A `_CountDownTimer` instance.
    """

    self._start_time_secs = time.time()
    self._duration_secs = duration_secs

  def secs_remaining(self):
    """Returns the remaining countdown seconds."""

    diff = self._duration_secs - (time.time() - self._start_time_secs)
    return max(0., diff)
