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
"""Defines mechanisms for deterministically replaying an AdaNet model search."""

# TODO: Add more detailed documentation.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow as tf


class Config(object):  # pylint: disable=g-classes-have-attributes
  # pyformat: disable
  """Defines how to deterministically replay an AdaNet model search.

  Specifically, it reconstructs the previous model and trains its components
  in the correct order without performing any search.

  Args:
    best_ensemble_indices: A list of the best ensemble indices (one per
      iteration).

  Returns:
    An :class:`adanet.replay.Config` instance.
  """
  # pyformat: enable

  def __init__(self, best_ensemble_indices=None):
    self._best_ensemble_indices = best_ensemble_indices

  @property
  def best_ensemble_indices(self):
    """The best ensemble indices per iteration."""
    return self._best_ensemble_indices

  def get_best_ensemble_index(self, iteration_number):
    """Returns the best ensemble index given an iteration number."""
    # If we are provided the list
    if (self._best_ensemble_indices
        and iteration_number < len(self._best_ensemble_indices)):
      return self._best_ensemble_indices[iteration_number]

    return None


__all__ = ["Config"]
