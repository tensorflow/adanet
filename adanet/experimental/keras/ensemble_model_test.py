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
"""Tests for adanet.experimental.keras.EnsembleModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from adanet.experimental.keras import testing_utils
from adanet.experimental.keras.ensemble_model import MeanEnsemble
from adanet.experimental.keras.ensemble_model import WeightedEnsemble

import tensorflow.compat.v2 as tf


class EnsembleModelTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'mean_ensemble',
          'ensemble': MeanEnsemble,
          'want_results': [0.07671691, 0.20448962],
      }, {
          'testcase_name': 'weighted_ensemble',
          'ensemble': WeightedEnsemble,
          'output_units': 2,
          'want_results': [0.42579408, 0.53439462],
      })
  def test_lifecycle(self, ensemble, want_results, output_units=None):
    train_dataset, test_dataset = testing_utils.get_holdout_data(
        train_samples=128,
        test_samples=64,
        input_shape=(10,),
        num_classes=2,
        random_seed=42)

    # TODO: Consider performing `tf.data.Dataset` transformations
    # within get_test_data function.
    train_dataset = train_dataset.batch(32).repeat(10)
    test_dataset = test_dataset.batch(32).repeat(10)

    model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2),
    ])
    model1.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='mse')
    model1.fit(train_dataset)
    model1.trainable = False  # Since models inside ensemble should be trained.
    model1_pre_train_weights = model1.get_weights()

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2),
    ])
    model2.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='mse')
    model2.fit(train_dataset)
    model2.trainable = False  # Since models inside ensemble should be trained.
    model2_pre_train_weights = model2.get_weights()

    if output_units:
      ensemble = ensemble(submodels=[model1, model2],
                          output_units=output_units)
    else:
      ensemble = ensemble(submodels=[model1, model2])
    ensemble.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='mse',
        metrics=['mae'])

    ensemble.fit(train_dataset)

    # Make sure submodel weights were not altered during ensemble training.
    model1_post_train_weights = model1.get_weights()
    model2_post_train_weights = model2.get_weights()
    self.assertAllClose(model1_pre_train_weights, model1_post_train_weights)
    self.assertAllClose(model2_pre_train_weights, model2_post_train_weights)

    eval_results = ensemble.evaluate(test_dataset)
    self.assertAllClose(eval_results, want_results)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
