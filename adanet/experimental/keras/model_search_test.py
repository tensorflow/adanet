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

import os
import shutil
import sys
import time

from absl import flags
from absl.testing import parameterized
from adanet.experimental.controllers.sequential_controller import SequentialController
from adanet.experimental.keras import testing_utils
from adanet.experimental.keras.ensemble_model import MeanEnsemble
from adanet.experimental.keras.model_search import ModelSearch
from adanet.experimental.phases.autoensemble_phase import AutoEnsemblePhase
from adanet.experimental.phases.autoensemble_phase import GrowStrategy
from adanet.experimental.phases.autoensemble_phase import MeanEnsembler
from adanet.experimental.phases.input_phase import InputPhase
from adanet.experimental.phases.keras_trainer_phase import KerasTrainerPhase
from adanet.experimental.phases.keras_tuner_phase import KerasTunerPhase
from adanet.experimental.phases.repeat_phase import RepeatPhase
from adanet.experimental.storages.in_memory_storage import InMemoryStorage
from kerastuner import tuners
import tensorflow as tf


class ModelSearchTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ModelSearchTest, self).setUp()
    # Setup and cleanup test directory.
    # Flags are not automatically parsed at this point.
    flags.FLAGS(sys.argv)
    self.test_subdirectory = os.path.join(flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.makedirs(self.test_subdirectory)

  def tearDown(self):
    super(ModelSearchTest, self).tearDown()
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)

  def test_phases_end_to_end(self):
    train_dataset, test_dataset = testing_utils.get_holdout_data(
        train_samples=128,
        test_samples=64,
        input_shape=(10,),
        num_classes=10,
        random_seed=42)

    # TODO: Consider performing `tf.data.Dataset` transformations
    # within get_holdout_data function.
    train_dataset = train_dataset.batch(32)
    test_dataset = test_dataset.batch(32)

    model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    model1.compile(
        optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae'])

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    model2.compile(
        optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae'])

    # TODO: This test could potentially have the best model be
    # a non-ensemble Keras model. Therefore, need to address this issue and
    # remove the freeze_submodels flag.
    ensemble = MeanEnsemble(submodels=[model1, model2], freeze_submodels=False)
    ensemble.compile(
        optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae'])

    controller = SequentialController(phases=[
        InputPhase(train_dataset, test_dataset),
        KerasTrainerPhase([model1, model2]),
        KerasTrainerPhase([ensemble]),
    ])

    model_search = ModelSearch(controller)
    model_search.run()
    self.assertIsInstance(
        model_search.get_best_models(num_models=1)[0], MeanEnsemble)

  def test_tuner_end_to_end(self):
    train_dataset, test_dataset = testing_utils.get_holdout_data(
        train_samples=128,
        test_samples=64,
        input_shape=(10,),
        num_classes=10,
        random_seed=42)

    # TODO: Consider performing `tf.data.Dataset` transformations
    # within get_holdout_data function.
    train_dataset = train_dataset.batch(32)
    test_dataset = test_dataset.batch(32)

    def build_model(hp):
      model = tf.keras.Sequential()
      model.add(
          tf.keras.layers.Dense(
              units=hp.Int('units', min_value=32, max_value=512, step=32),
              activation='relu'))
      model.add(tf.keras.layers.Dense(10, activation='softmax'))
      model.compile(
          optimizer=tf.keras.optimizers.Adam(
              hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
      return model

    # Define phases.
    tuner = tuners.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=3,
        executions_per_trial=1,
        directory=self.test_subdirectory,
        project_name='helloworld_tuner',
        overwrite=True)

    tuner_phase = KerasTunerPhase(tuner)

    def build_ensemble():
      ensemble = MeanEnsemble(
          submodels=tuner_phase.get_best_models(num_models=2))
      ensemble.compile(
          optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae'])
      return [ensemble]

    ensemble_phase = KerasTrainerPhase(build_ensemble)
    input_phase = InputPhase(train_dataset, test_dataset)

    controller = SequentialController(phases=[input_phase,
                                              tuner_phase,
                                              ensemble_phase])

    # Execute phases.
    model_search = ModelSearch(controller)
    model_search.run()
    self.assertIsInstance(
        model_search.get_best_models(num_models=1)[0], MeanEnsemble)

  def test_autoensemble_end_to_end(self):

    train_dataset, test_dataset = testing_utils.get_holdout_data(
        train_samples=128,
        test_samples=64,
        input_shape=(10,),
        num_classes=10,
        random_seed=42)

    # TODO: Consider performing `tf.data.Dataset` transformations
    # within get_holdout_data function.
    train_dataset = train_dataset.batch(32)
    test_dataset = test_dataset.batch(32)

    def build_model(hp):
      model = tf.keras.Sequential()
      model.add(
          tf.keras.layers.Dense(
              units=hp.Int('units', min_value=32, max_value=512, step=32),
              activation='relu'))
      model.add(tf.keras.layers.Dense(10, activation='softmax'))
      model.compile(
          optimizer=tf.keras.optimizers.Adam(
              hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
      return model

    # This allows us to have a shared storage for all the autoensemble phases
    # that occur in the repeat phase.
    autoensemble_storage = InMemoryStorage()
    input_phase = InputPhase(train_dataset, test_dataset)
    # pylint: disable=g-long-lambda
    repeat_phase = RepeatPhase(
        [
            lambda: KerasTunerPhase(
                tuners.RandomSearch(
                    build_model,
                    objective='val_accuracy',
                    max_trials=3,
                    executions_per_trial=1,
                    directory=self.test_subdirectory,
                    project_name='helloworld_' + str(int(time.time())),
                    overwrite=True)),
            lambda: AutoEnsemblePhase(
                ensemblers=[
                    MeanEnsembler('sparse_categorical_crossentropy', 'adam',
                                  ['accuracy'])
                ],
                ensemble_strategies=[GrowStrategy()],
                storage=autoensemble_storage)
        ], repetitions=3)
    # pylint: enable=g-long-lambda

    controller = SequentialController(phases=[input_phase, repeat_phase])

    model_search = ModelSearch(controller)
    model_search.run()
    self.assertIsInstance(
        model_search.get_best_models(num_models=1)[0], MeanEnsemble)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
