"""Tests for AdaNet input utility functions.

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

from absl.testing import parameterized
from adanet.core.input_utils import make_placeholder_input_fn
import tensorflow as tf


class InputUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "single_label",
      "labels_fn": lambda: tf.constant([4.], name="label"),
      "label": [4.],
  }, {
      "testcase_name": "multi_label",
      # pylint: disable=g-long-lambda
      "labels_fn": lambda: {
          "label_1": tf.constant([4.], name="label_1"),
          "label_2": tf.constant([2.], name="label_2"),
      },
      "label": {
          "label_1": [4.],
          "label_2": [2.],
      },
  })
  def test_make_placeholder_input_fn(self, labels_fn, label):
    dense = [1., 2.]
    sparse = tf.SparseTensorValue(
        indices=[[0, 0], [0, 1]], values=[-1., 1.], dense_shape=[1, 2])

    def input_fn():
      features = {
          "dense":
              tf.constant(dense, name="foo"),
          "sparse":
              tf.SparseTensor(
                  indices=tf.constant(
                      sparse.indices, name="indices", dtype=tf.int64),
                  values=tf.constant(sparse.values, name="values"),
                  dense_shape=tf.constant(
                      sparse.dense_shape, name="dense_shape", dtype=tf.int64))
      }
      labels = labels_fn()
      return features, labels

    placeholder_input_fn = make_placeholder_input_fn(input_fn)
    features, labels = input_fn()
    got_features, got_labels = placeholder_input_fn()

    with self.test_session() as sess:
      self.assertAllClose(
          sess.run(features),
          sess.run(
              got_features,
              feed_dict={
                  got_features["dense"]: dense,
                  got_features["sparse"]: sparse
              }))
      if isinstance(got_labels, tf.Tensor):
        self.assertAllClose(
            sess.run(labels),
            sess.run(got_labels, feed_dict={got_labels: label}))
      else:
        self.assertAllClose(
            sess.run(labels),
            sess.run(
                got_labels,
                feed_dict={
                    got_labels["label_1"]: label["label_1"],
                    got_labels["label_2"]: label["label_2"],
                }))


if __name__ == "__main__":
  tf.test.main()
