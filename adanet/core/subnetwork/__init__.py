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

"""Low-level APIs for defining custom subnetworks and search spaces."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from adanet.core.subnetwork.generator import Builder
from adanet.core.subnetwork.generator import Generator
from adanet.core.subnetwork.generator import SimpleGenerator
from adanet.core.subnetwork.generator import Subnetwork
from adanet.core.subnetwork.generator import TrainOpSpec
from adanet.core.subnetwork.report import MaterializedReport
from adanet.core.subnetwork.report import Report

__all__ = [
    "Subnetwork",
    "Builder",
    "Generator",
    "SimpleGenerator",
    "Report",
    "MaterializedReport",
]
