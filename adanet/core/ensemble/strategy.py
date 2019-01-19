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
"""Search strategy algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections


class Candidate(collections.namedtuple("Candidate", ["subnetwork_builders"])):
  """An ensemble candidate found during the search phase.

  Args:
    subnetwork_builders: Candidate`adanet.subnetwork.Builder` instances.
  """

  def __new__(cls, subnetwork_builders):
    return super(Candidate, cls).__new__(
        cls, subnetwork_builders=tuple(subnetwork_builders))


class Strategy(object):
  """An abstract ensemble strategy."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def generate_ensemble_candidates(self, subnetwork_builders):
    """Returns an iterable for `adanet.ensemble.Candidate`."""


class GrowStrategy(Strategy):
  """Greedily grows an ensemble, one subnetwork at a time."""

  def generate_ensemble_candidates(self, subnetwork_builders):
    return [
        Candidate([subnetwork_builder])
        for subnetwork_builder in subnetwork_builders
    ]


class AllStrategy(Strategy):
  """Ensembles all subnetworks from the current iteration."""

  def generate_ensemble_candidates(self, subnetwork_builders):
    return [Candidate(subnetwork_builders)]
