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
"""The `adanet.distributed` package.

This package methods for distributing computation using the TensorFlow
computation graph.
"""

# TODO: Add more details documentation.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from adanet.distributed.placement import PlacementStrategy
from adanet.distributed.placement import ReplicationStrategy
from adanet.distributed.placement import RoundRobinStrategy

__all__ = [
    "PlacementStrategy",
    "ReplicationStrategy",
    "RoundRobinStrategy",
]
