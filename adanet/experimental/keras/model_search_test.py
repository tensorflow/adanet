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
"""Tests for adanet.experimental.keras.ModelSearch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from adanet.experimental.controllers.sequential_controller import SequentialController
from adanet.experimental.keras import testing_utils
from adanet.experimental.keras.ensemble_model import MeanEnsemble
from adanet.experimental.keras.model_search import ModelSearch
from adanet.experimental.phases.train_keras_models_phase import TrainKerasModelsPhase
import tensorflow as tf


class ModelSearchTest(parameterized.TestCase, tf.test.TestCase):

  def test_end_to_end(self):
    train_dataset, test_dataset = testing_utils.get_test_data(
        train_samples=128,
        test_samples=64,
        input_shape=(10,),
        num_classes=2,
        random_seed=42)

    # TODO: Consider performing `tf.data.Dataset` transformations
    # within get_test_data function.
    train_dataset = train_dataset.batch(32)
    test_dataset = test_dataset.batch(32)

    model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    model1.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    model2.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')

    ensemble = MeanEnsemble(submodels=[model1, model2])
    ensemble.compile(
        optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae'])

    controller = SequentialController(phases=[
        TrainKerasModelsPhase([
            model1,
            model2,
        ], dataset=train_dataset),
        TrainKerasModelsPhase([ensemble], dataset=train_dataset),
    ])

    train_dataset, test_dataset = testing_utils.get_test_data(
        train_samples=128,
        test_samples=64,
        input_shape=(10,),
        num_classes=2,
        random_seed=42)

    model_search = ModelSearch(controller)
    model_search.run()

    # TODO: Test get_best_models:
    # models = model_search.get_best_models()
    # model = models[0]
    # model.evaluate(test_dataset)
    # ...


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
