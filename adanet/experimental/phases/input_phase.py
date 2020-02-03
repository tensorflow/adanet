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
"""A phase that provides datasets."""

from typing import Optional
from adanet.experimental.phases.phase import DatasetProvider
from adanet.experimental.phases.phase import Phase
import tensorflow.compat.v2 as tf


class InputPhase(DatasetProvider):
  """A phase that simply relays train and eval datasets."""

  def __init__(self, train_dataset: tf.data.Dataset,
               eval_dataset: tf.data.Dataset):
    """Initializes an InputPhase.

    Args:
      train_dataset: A `tf.data.Dataset` for training.
      eval_dataset: A `tf.data.Dataset` for evaluation.
    """

    self._train_dataset = train_dataset
    self._eval_dataset = eval_dataset

  def get_train_dataset(self) -> tf.data.Dataset:
    return self._train_dataset

  def get_eval_dataset(self) -> tf.data.Dataset:
    return self._eval_dataset

  def work_units(self, previous_phase: Optional[Phase]):
    return []
