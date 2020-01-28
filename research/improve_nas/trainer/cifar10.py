# Lint as: python3
"""CIFAR-10 data and convenience functions.

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

import functools
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import cifar10

# pylint: disable=g-import-not-at-top
try:
  from adanet.research.improve_nas.trainer import image_processing
except ImportError as e:
  from trainer import image_processing
# pylint: enable=g-import-not-at-top

FEATURES = 'x'
PreprocessingType = image_processing.PreprocessingType


class Provider(object):
  """A CIFAR-10 data provider."""

  def __init__(self,
               params_string='',
               seed=None):
    """Returns a CIFAR-10 `Provider`."""
    # For testing
    self._seed = seed
    default_params = tf.contrib.training.HParams(
        cutout=True, augmentation=PreprocessingType.BASIC)
    self._params = default_params.parse(params_string)

  def _preprocess_data(self, image, label, training, preprocess):
    """Apply Inception data augmentation and preprocessing."""

    # Unpack `Element` tuple.
    # image, label = element

    if preprocess:
      image_height, image_width = self._shape()[:2]
      if self._params.augmentation == PreprocessingType.BASIC:
        image = image_processing.resize_and_normalize(image, image_height,
                                                      image_width)
        if training:
          image = image_processing.basic_augmentation(image, image_height,
                                                      image_width, self._seed)
      else:
        raise ValueError('Unsupported data augmentation type: `%s`' %
                         self._params.augmentation)

      if training and self._params.cutout:
        # According to https://arxiv.org/abs/1708.04552, cutting out 16x16
        # works best.
        image = image_processing.cutout(image, pad_size=8, seed=self._seed)

    # Set shapes so that they are defined.
    image.set_shape(self._shape())
    if label is not None:
      label.set_shape([1])
    return {FEATURES: image}, label

  def _cifar10_dataset(self, partition):
    """Returns a partition of the CIFAR-10 `Dataset`."""
    cifar10_data = None
    try:
      cifar10_data = cifar10.load_data()
      tf.logging.info('Loaded cifar10.')
    except:  # pylint: disable=bare-except
      tf.logging.info(
          'Can not load cifar10 from internet. Creating dummy data for '
          'testing.')
      data = np.zeros((3, 32, 32, 3))
      labels = np.array([[5], [3], [9]])
      data[:, 0, 0] = [148, 141, 174]
      data[:, -1, 0, 0] = 128
      cifar10_data = ((data, labels), (data, labels))
    (x_train, y_train), (x_test, y_test) = cifar10_data
    x = None
    y = None
    if partition == 'train':
      x, y = x_train, y_train
    else:
      x, y = x_test, y_test

    dataset = tf.data.Dataset.from_tensor_slices((x, y.astype(np.int32)))
    return dataset.cache()

  def _shape(self):
    """Returns a 3-dimensional list with the shape of the image."""
    return [32, 32, 3]

  def get_input_fn(self,
                   partition,
                   mode,
                   batch_size,
                   preprocess=True,
                   use_tpu=False):
    """See `data.Provider` get_input_fn."""

    def input_fn(params=None):
      """Provides batches of CIFAR images.

      Args:
        params: A dict containing the batch_size on TPU, otherwise None.

      Returns:
        images: A `Tensor` of size [batch_size, 32, 32, 3]
        labels: A `Tensor` of size [batch_size, 1],
      """

      batch_size_ = batch_size
      if use_tpu:
        batch_size_ = params.get('batch_size', batch_size)

      training = mode == tf.estimator.ModeKeys.TRAIN
      dataset = self._cifar10_dataset(partition)
      dataset = dataset.map(
          functools.partial(
              self._preprocess_data, training=training, preprocess=preprocess))
      if training:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(
                buffer_size=500, seed=self._seed))
      return dataset.batch(
          batch_size_,
          drop_remainder=use_tpu).prefetch(tf.data.experimental.AUTOTUNE
                                          ).make_one_shot_iterator().get_next()

    return input_fn

  def get_head(self, name=None):
    """Returns a `Head` instance for multiclass CIFAR-10 with the given name."""
    return tf.contrib.estimator.multi_class_head(
        10, name=name, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

  def get_feature_columns(self):
    """Returns feature columns."""
    feature_columns = [
        tf.feature_column.numeric_column(key=FEATURES, shape=self._shape())
    ]
    return feature_columns
