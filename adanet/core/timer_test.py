"""Tests for timer.

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

from adanet.core.timer import _CountDownTimer
import tensorflow as tf


class CountDownTimerTest(tf.test.TestCase):

  def test_secs_remaining_long(self):
    timer = _CountDownTimer(60)
    time.sleep(.1)
    secs_remaining = timer.secs_remaining()
    self.assertLess(0., secs_remaining)
    self.assertGreater(60., secs_remaining)

  def test_secs_remaining_short(self):
    timer = _CountDownTimer(.001)
    time.sleep(.1)
    secs_remaining = timer.secs_remaining()
    self.assertEqual(0., secs_remaining)

  def test_secs_remaining_zero(self):
    timer = _CountDownTimer(0.)
    time.sleep(.01)
    secs_remaining = timer.secs_remaining()
    self.assertEqual(0., secs_remaining)


if __name__ == "__main__":
  tf.test.main()
