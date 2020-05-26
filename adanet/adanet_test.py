"""Test AdaNet package.

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

import adanet
from adanet.examples import simple_dnn
import tensorflow.compat.v1 as tf


class AdaNetTest(tf.test.TestCase):

  def test_public(self):
    self.assertIsNotNone(adanet.__version__)
    self.assertIsNotNone(adanet.AutoEnsembleEstimator)
    self.assertIsNotNone(adanet.AutoEnsembleSubestimator)
    self.assertIsNotNone(adanet.AutoEnsembleTPUEstimator)
    self.assertIsNotNone(adanet.distributed.PlacementStrategy)
    self.assertIsNotNone(adanet.distributed.ReplicationStrategy)
    self.assertIsNotNone(adanet.distributed.RoundRobinStrategy)
    self.assertIsNotNone(adanet.ensemble.Ensemble)
    self.assertIsNotNone(adanet.ensemble.Ensembler)
    self.assertIsNotNone(adanet.ensemble.TrainOpSpec)
    self.assertIsNotNone(adanet.ensemble.AllStrategy)
    self.assertIsNotNone(adanet.ensemble.Candidate)
    self.assertIsNotNone(adanet.ensemble.GrowStrategy)
    self.assertIsNotNone(adanet.ensemble.Strategy)
    self.assertIsNotNone(adanet.ensemble.ComplexityRegularized)
    self.assertIsNotNone(adanet.ensemble.ComplexityRegularizedEnsembler)
    self.assertIsNotNone(adanet.ensemble.MeanEnsemble)
    self.assertIsNotNone(adanet.ensemble.MeanEnsembler)
    self.assertIsNotNone(adanet.ensemble.MixtureWeightType)
    self.assertIsNotNone(adanet.ensemble.WeightedSubnetwork)
    self.assertIsNotNone(adanet.Ensemble)
    self.assertIsNotNone(adanet.Estimator)
    self.assertIsNotNone(adanet.Evaluator)
    self.assertIsNotNone(adanet.MixtureWeightType)
    self.assertIsNotNone(adanet.replay.Config)
    self.assertIsNotNone(adanet.ReportMaterializer)
    self.assertIsNotNone(adanet.Subnetwork)
    self.assertIsNotNone(adanet.subnetwork.Builder)
    self.assertIsNotNone(adanet.subnetwork.Generator)
    self.assertIsNotNone(adanet.subnetwork.Subnetwork)
    self.assertIsNotNone(adanet.subnetwork.TrainOpSpec)
    self.assertIsNotNone(adanet.Summary)
    self.assertIsNotNone(adanet.TPUEstimator)
    self.assertIsNotNone(adanet.WeightedSubnetwork)
    self.assertIsNotNone(simple_dnn.Generator)

if __name__ == "__main__":
  tf.test.main()
