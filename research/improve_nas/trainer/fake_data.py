"""Fake dataset for testing and debugging.

Copyright 2019 The AdaNet Authors. All Rights Reserved.

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

import numpy as np
import tensorflow as tf


class FakeImageProvider(object):
  """A fake image data provider."""

  def __init__(self,
               num_examples=3,
               num_classes=3,
               image_dim=8,
               channels=1,
               seed=42):
    self._num_examples = num_examples
    self._num_classes = num_classes
    self._seed = seed
    self._channels = channels
    self._image_dim = image_dim

  def get_head(self, name=None):
    return tf.contrib.estimator.multi_class_head(
        self._num_classes, name=name, loss_reduction=tf.losses.Reduction.SUM)

  def _shape(self):
    return [self._image_dim, self._image_dim, self._channels]

  def get_input_fn(self,
                   partition,
                   mode,
                   batch_size):
    """See `data.Provider` get_input_fn."""

    del partition
    def input_fn(params=None):
      """Input_fn to return."""

      del params  # Unused.

      np.random.seed(self._seed)
      if mode == tf.estimator.ModeKeys.EVAL:
        np.random.seed(self._seed + 1)

      images = tf.to_float(
          tf.convert_to_tensor(
              np.random.rand(self._num_examples, *self._shape())))
      labels = tf.convert_to_tensor(
          np.random.randint(0, high=2, size=(self._num_examples, 1)))
      dataset = tf.data.Dataset.from_tensor_slices(({"x": images}, labels))
      if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat()
      dataset = dataset.batch(batch_size)
      iterator = dataset.make_one_shot_iterator()
      return iterator.get_next()

    return input_fn

  def get_feature_columns(self):
    feature_columns = [
        tf.feature_column.numeric_column(key="x", shape=self._shape()),
    ]
    return feature_columns
