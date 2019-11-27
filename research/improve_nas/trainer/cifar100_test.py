# Lint as: python3
"""Tests for cifar100 dataset.

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

from adanet.research.improve_nas.trainer import cifar100
import tensorflow as tf


class Cifar100Test(tf.test.TestCase):

  def _check_dimensions(self, partition):
    provider = cifar100.Provider(seed=4)
    input_fn = provider.get_input_fn(
        partition, tf.contrib.learn.ModeKeys.TRAIN, batch_size=3)
    data, labels = input_fn()
    self.assertIn(cifar100.FEATURES, data)
    features = data[cifar100.FEATURES]
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    with self.test_session() as sess:
      sess.run(init)
      self.assertEqual((3, 32, 32, 3), sess.run(features).shape)
      self.assertEqual((3, 1), sess.run(labels).shape)

  def test_read_cifar100(self):
    for partition in ["train", "test"]:
      self._check_dimensions(partition)

  def test_no_preprocess(self):
    provider = cifar100.Provider(seed=4)
    input_fn = provider.get_input_fn(
        "train",
        tf.contrib.learn.ModeKeys.TRAIN,
        batch_size=3,
        preprocess=False)
    data, label = input_fn()

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    with self.test_session() as sess:
      sess.run(init)
      self.assertAllEqual([220, 25, 47], sess.run(data["x"])[0][0][0])
      self.assertAllEqual([[47], [5], [52]], sess.run(label))

  def test_basic_preprocess(self):
    provider = cifar100.Provider(
        params_string="augmentation=basic", seed=4)
    input_fn = provider.get_input_fn(
        "train",
        tf.contrib.learn.ModeKeys.TRAIN,
        batch_size=3,
        preprocess=True)
    data, label = input_fn()

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    with self.test_session() as sess:
      sess.run(init)
      data_result = sess.run(data["x"])
      self.assertEqual((3, 32, 32, 3), data_result.shape)
      self.assertAllEqual([0, 0, 0], data_result[0, 0, 0])
      self.assertAlmostEqual(0.0, data_result[0, -1, 0, 0], places=3)
      self.assertAllEqual([[47], [5], [52]], sess.run(label))


if __name__ == "__main__":
  tf.test.main()
