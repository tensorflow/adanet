"""Test AdaNet mean ensemble and ensembler implementation.

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


from absl.testing import parameterized
from adanet import ensemble
from adanet import subnetwork
from adanet import tf_compat

import numpy as np
import tensorflow as tf


class MeanTest(parameterized.TestCase, tf.test.TestCase):

  def _build_subnetwork(self, multi_head=False, last_layer_dim=3):

    last_layer = tf.Variable(
        tf_compat.random_normal(shape=(2, last_layer_dim)),
        trainable=False).read_value()

    def new_logits():
      return tf_compat.v1.layers.dense(
          last_layer,
          units=1,
          kernel_initializer=tf_compat.v1.glorot_uniform_initializer())

    if multi_head:
      logits = {k: new_logits() for k in multi_head}
      last_layer = {k: last_layer for k in multi_head}
    else:
      logits = new_logits()

    return subnetwork.Subnetwork(
        last_layer=last_layer, logits=logits, complexity=2)

  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
      }, {
          'testcase_name': 'base_with_last_layer_predictions',
          'add_mean_last_layer_predictions': True
      }, {
          'testcase_name': 'base_with_last_layer_predictions_diff_shapes',
          'add_mean_last_layer_predictions': True,
          'diff_last_layer_shapes': True
      }, {
          'testcase_name': 'multi_head',
          'multi_head': ['first_head', 'second_head'],
      }, {
          'testcase_name': 'multi_head_with_last_layer_predictions',
          'multi_head': ['first_head', 'second_head'],
          'add_mean_last_layer_predictions': True
      }, {
          'testcase_name': 'multi_head_with_last_layer_predictions_diff_shapes',
          'multi_head': ['first_head', 'second_head'],
          'add_mean_last_layer_predictions': True,
          'diff_last_layer_shapes': True
      }
  )
  def test_mean_ensembler(self,
                          multi_head=False,
                          add_mean_last_layer_predictions=False,
                          diff_last_layer_shapes=False):
    ensembler = ensemble.MeanEnsembler(
        add_mean_last_layer_predictions=add_mean_last_layer_predictions)
    last_layer_dims = [3, 3]
    if diff_last_layer_shapes:
      last_layer_dims = [3, 5]
    if multi_head:
      subnetworks = [
          self._build_subnetwork(
              multi_head=multi_head, last_layer_dim=last_layer_dim)
          for last_layer_dim in last_layer_dims
      ]
    else:
      subnetworks = [
          self._build_subnetwork(last_layer_dim=last_layer_dim)
          for last_layer_dim in last_layer_dims
      ]

    if diff_last_layer_shapes:
      with self.assertRaisesRegexp(
          ValueError,
          r'Shape of \`last_layer\` tensors must be same'
      ):
        built_ensemble = ensembler.build_ensemble(
            subnetworks=subnetworks,
            previous_ensemble_subnetworks=None,
            features=None,
            labels=None,
            logits_dimension=None,
            training=None,
            iteration_step=None,
            summary=None,
            previous_ensemble=None)
      return
    built_ensemble = ensembler.build_ensemble(
        subnetworks=subnetworks,
        previous_ensemble_subnetworks=None,
        features=None,
        labels=None,
        logits_dimension=None,
        training=None,
        iteration_step=None,
        summary=None,
        previous_ensemble=None)

    with self.test_session() as sess:
      sess.run(tf_compat.v1.global_variables_initializer())
      got_logits = sess.run(built_ensemble.logits)

      if add_mean_last_layer_predictions:
        got_predictions = sess.run(built_ensemble.predictions)

      logits = sess.run([s.logits for s in subnetworks])
      last_layer = sess.run([s.last_layer for s in subnetworks])
      if not multi_head:
        expected_logits = np.mean(logits, axis=0)
        expected_predictions = {
            ensemble.MeanEnsemble.MEAN_LAST_LAYER: np.mean(last_layer, axis=0)
        }
      else:
        expected_logits = {
            head_name: np.mean([s[head_name] for s in logits
                               ], axis=0) for head_name in multi_head
        }
        expected_predictions = {
            '{}_{}'.format(ensemble.MeanEnsemble.MEAN_LAST_LAYER, head_name):
            np.mean([s[head_name] for s in last_layer], axis=0)
            for head_name in multi_head
        }

      self.assertAllClose(expected_logits, got_logits)
      if add_mean_last_layer_predictions:
        self.assertAllClose(expected_predictions, got_predictions)


if __name__ == '__main__':
  tf.test.main()
