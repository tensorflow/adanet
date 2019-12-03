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
"""The AutoML controller for AdaNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import abc

from adanet.experimental.work_units.work_unit import WorkUnit
import tensorflow as tf
from typing import Iterator, Sequence


class Controller(abc.ABC):
  """Defines the machine learning workflow to produce high-quality models."""

  @abc.abstractmethod
  def work_units(self) -> Iterator[WorkUnit]:
    pass

  @abc.abstractmethod
  def get_best_models(self, num_models) -> Sequence[tf.keras.Model]:
    pass
