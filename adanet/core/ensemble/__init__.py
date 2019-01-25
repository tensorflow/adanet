# Copyright 2019 The AdaNet Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The `adanet.ensemble` package.

This package defines built-in ensemble methods and interfaces for building
ensembles of subnetworks.

TODO: Add more details documentation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from adanet.core.ensemble.ensembler import Ensemble
from adanet.core.ensemble.ensembler import Ensembler
from adanet.core.ensemble.strategy import AllStrategy
from adanet.core.ensemble.strategy import Candidate
from adanet.core.ensemble.strategy import GrowStrategy
from adanet.core.ensemble.strategy import Strategy
from adanet.core.ensemble.weighted import ComplexityRegularized
from adanet.core.ensemble.weighted import ComplexityRegularizedEnsembler
from adanet.core.ensemble.weighted import MixtureWeightType
from adanet.core.ensemble.weighted import WeightedSubnetwork

__all__ = [
    "Ensemble",
    "Ensembler",
    "Candidate",
    "Strategy",
    "GrowStrategy",
    "AllStrategy",
    "ComplexityRegularized",
    "ComplexityRegularizedEnsembler",
    "MixtureWeightType",
    "WeightedSubnetwork",
]
