"""Tests for AdaNet dictionary utilities.

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

from adanet.core import dict_utils
import tensorflow as tf


class DictUtilsTest(tf.test.TestCase):

  def test_flatten_dict(self):
    to_flatten = {
        "hello": {
            "world": 1,
            "sailor": 2,
        },
        "ada": {
            "net": 3,
            "boost": 4,
        },
        "nodict": 5,
    }

    actual = dict_utils.flatten_dict(to_flatten, delimiter="-")

    expected = {
        "hello-world": 1,
        "hello-sailor": 2,
        "ada-net": 3,
        "ada-boost": 4,
        "nodict": 5,
    }

    self.assertDictEqual(actual, expected)

  def test_unflatten_dict(self):
    flat_dict = {
        "hello-world": 1,
        "hello-sailor": 2,
        "ada-net": 3,
        "ada-boost": 4,
        "nodict": 5,
    }

    actual_wrong_delimiter = dict_utils.unflatten_dict(
        flat_dict, prefixes=["hello", "ada"], delimiter="/")
    actual_unflattened = dict_utils.unflatten_dict(
        flat_dict, prefixes=["ada", "unk"], delimiter="-")

    expected = {
        "hello-world": 1,
        "hello-sailor": 2,
        "ada": {
            "net": 3,
            "boost": 4,
        },
        "nodict": 5,
    }

    self.assertDictEqual(actual_wrong_delimiter, flat_dict)
    self.assertDictEqual(actual_unflattened, expected)


if __name__ == "__main__":
  tf.test.main()
