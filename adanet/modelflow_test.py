# Lint as: python3
# Copyright 2020 The AdaNet Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test ModelFlow imports."""

import adanet.experimental as adanet
import tensorflow as tf


class ModelFlowTest(tf.test.TestCase):

  def test_public(self):
    self.assertIsNotNone(adanet.controllers.SequentialController)
    self.assertIsNotNone(adanet.keras.EnsembleModel)
    self.assertIsNotNone(adanet.keras.MeanEnsemble)
    self.assertIsNotNone(adanet.keras.WeightedEnsemble)
    self.assertIsNotNone(adanet.keras.ModelSearch)
    self.assertIsNotNone(adanet.phases.AutoEnsemblePhase)
    self.assertIsNotNone(adanet.phases.InputPhase)
    self.assertIsNotNone(adanet.phases.KerasTrainerPhase)
    self.assertIsNotNone(adanet.phases.KerasTunerPhase)
    self.assertIsNotNone(adanet.phases.RepeatPhase)
    self.assertIsNotNone(adanet.schedulers.InProcessScheduler)
    self.assertIsNotNone(adanet.storages.InMemoryStorage)
    self.assertIsNotNone(adanet.work_units.KerasTrainerWorkUnit)
    self.assertIsNotNone(adanet.work_units.KerasTunerWorkUnit)

if __name__ == "__main__":
  tf.test.main()
