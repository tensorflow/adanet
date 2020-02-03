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
"""Utilities for unit-testing AdaNet Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Tuple

import numpy as np
import tensorflow.compat.v2 as tf


# TODO: Add ability to choose the problem type: regression,
# classification, multi-class etc.
def get_holdout_data(
    train_samples: int,
    test_samples: int,
    input_shape: Tuple[int],
    num_classes: int,
    random_seed: Optional[int] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Generates training and test data.

  Args:
    train_samples: Number of training samples to generate.
    test_samples: Number of training samples to generate.
    input_shape: Shape of the inputs.
    num_classes: Number of classes for the data and targets.
    random_seed: A random seed for numpy to use.

  Returns:
    A tuple of `tf.data.Datasets`.
  """
  if random_seed:
    np.random.seed(random_seed)

  num_sample = train_samples + test_samples
  templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
  y = np.random.randint(0, num_classes, size=(num_sample,))
  x = np.zeros((num_sample,) + input_shape, dtype=np.float32)
  for i in range(num_sample):
    x[i] = templates[y[i]] + np.random.normal(loc=0, scale=1., size=input_shape)

  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x[:train_samples], y[:train_samples]))
  test_dataset = tf.data.Dataset.from_tensor_slices(
      (x[train_samples:], y[train_samples:]))
  return train_dataset, test_dataset
