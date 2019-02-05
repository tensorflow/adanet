# Copyright 2018 The AdaNet Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AdaNet: Fast and flexible AutoML with learning guarantees."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from adanet.autoensemble import AutoEnsembleEstimator
from adanet.core import ensemble
from adanet.core import Estimator
from adanet.core import Evaluator
from adanet.core import ReportMaterializer
from adanet.core import subnetwork
from adanet.core import Summary
from adanet.core import TPUEstimator
# For backwards compatibility. Previously all Ensemblers were complexity
# regularized using the AdaNet objective.
from adanet.core.ensemble import ComplexityRegularized as Ensemble
from adanet.core.ensemble import MixtureWeightType
from adanet.core.ensemble import WeightedSubnetwork
from adanet.core.subnetwork import Subnetwork

__all__ = [
    "AutoEnsembleEstimator",
    "ensemble",
    "Ensemble",
    "Estimator",
    "Evaluator",
    "ReportMaterializer",
    "subnetwork",
    "Summary",
    "TPUEstimator",
    "MixtureWeightType",
    "WeightedSubnetwork",
    "Subnetwork",
]
