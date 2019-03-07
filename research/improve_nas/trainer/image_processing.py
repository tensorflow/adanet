"""Image preprocessing and augmentation function for a single image.

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

import tensorflow as tf


class PreprocessingType(object):
  """Type of preprocessing to be applied on the image.

  * `INCEPTION`: Preprocessing used in inception.
  * `BASIC`: Minimalistic preprocessing used in NasNet for cifar.

  """
  INCEPTION = "inception"
  BASIC = "basic"


def basic_augmentation(image, image_height, image_width, seed=None):
  """Augment image according to NasNet paper (random flip + random crop)."""

  # source: https://arxiv.org/pdf/1707.07012.pdf appendix A.1
  padding = 4
  image = tf.image.random_flip_left_right(image, seed=seed)

  image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])
  image = tf.random_crop(image, [image_height, image_width, 3], seed=seed)
  return image


def resize_and_normalize(image, height, width):
  """Convert image to float, resize and normalize to zero mean and [-1, 1]."""
  if image.dtype != tf.float32:
    # Rescale pixel values to float in interval [0.0, 1.0].
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  # Resize the image to the specified height and width.
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
  image = tf.squeeze(image, [0])
  # Rescale pixels to range [-0.5, 0.5].
  image = tf.subtract(image, 0.5)
  # Rescale pixels to range [-1, 1].
  image = tf.multiply(image, 2.0)
  return image


def cutout(image, pad_size, replace=0, seed=None):
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

  Forked from learning/brain/research/meta_architect/image/image_processing.py?
    l=1172&rcl=193953073

  Args:
    image: Image `Tensor` with shape [height, width, channels].
    pad_size: The cutout shape will be at most [pad_size * 2, pad_size * 2].
    replace: Value for replacing cutout values.
    seed: Random seed.

  Returns:
    Image `Tensor` with cutout applied.
  """

  with tf.variable_scope("cutout"):
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_depth = tf.shape(image)[2]

    # Sample the location in the image where the zero mask will be applied.
    cutout_center_height = tf.random_uniform(
        shape=[], minval=0, maxval=image_height, seed=seed, dtype=tf.int32)

    cutout_center_width = tf.random_uniform(
        shape=[], minval=0, maxval=image_width, seed=seed, dtype=tf.int32)

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad)
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype),
        padding_dims,
        constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, image_depth])
    image = tf.where(
        tf.equal(mask, 0),
        tf.ones_like(image, dtype=image.dtype) * replace, image)
  return image
