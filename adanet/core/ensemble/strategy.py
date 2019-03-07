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


class Candidate(
    collections.namedtuple("Candidate", [
        "name", "subnetwork_builders", "previous_ensemble_subnetwork_builders"
    ])):
  """An ensemble candidate found during the search phase.

  Args:
    name: String name of this ensemble candidate.
    subnetwork_builders: Candidate :class:`adanet.subnetwork.Builder` instances
      to include in the ensemble.
    previous_ensemble_subnetwork_builders: :class:`adanet.subnetwork.Builder`
      instances to include from the previous ensemble.
  """

  def __new__(cls, name, subnetwork_builders,
              previous_ensemble_subnetwork_builders):
    return super(Candidate, cls).__new__(
        cls,
        name=name,
        subnetwork_builders=tuple(subnetwork_builders),
        previous_ensemble_subnetwork_builders=tuple(
            previous_ensemble_subnetwork_builders or []))


class Strategy(object):
  """An abstract ensemble strategy."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def generate_ensemble_candidates(self, subnetwork_builders,
                                   previous_ensemble_subnetwork_builders):
    """Generates ensemble candidates to search over this iteration.

    Args:
      subnetwork_builders: Candidate :class:`adanet.subnetwork.Builder` instances
        for this iteration.
      previous_ensemble_subnetwork_builders: :class:`adanet.subnetwork.Builder`
        instances from the previous ensemble. Including only a subset of these
        in a returned :class:`adanet.ensemble.Candidate` is equivalent to
        pruning the previous ensemble.

    Returns:
      An iterable of :class:`adanet.ensemble.Candidate` instances to train and
      consider this iteration.
    """

    # TODO: Pruning the previous subnetwork may require more metadata such
    # as `subnetwork.Reports` and `ensemble.Reports` to make smart decisions.



class SoloStrategy(Strategy):
  """Produces a model composed of a single subnetwork.

  *An ensemble of one.*

  This is effectively the same as pruning all previous ensemble subnetworks,
  and only adding one subnetwork candidate to the ensemble.
  """

  def generate_ensemble_candidates(self, subnetwork_builders,
                                   previous_ensemble_subnetwork_builders):
    return [
        Candidate(subnetwork_builder.name, [subnetwork_builder], None)
        for subnetwork_builder in subnetwork_builders
    ]


class GrowStrategy(Strategy):
  """Greedily grows an ensemble, one subnetwork at a time."""

  def generate_ensemble_candidates(self, subnetwork_builders,
                                   previous_ensemble_subnetwork_builders):
    return [
        Candidate(subnetwork_builder.name, [subnetwork_builder],
                  previous_ensemble_subnetwork_builders)
        for subnetwork_builder in subnetwork_builders
    ]


class AllStrategy(Strategy):
  """Ensembles all subnetworks from the current iteration."""

  def generate_ensemble_candidates(self, subnetwork_builders,
                                   previous_ensemble_subnetwork_builders):
    return [
        Candidate("all", subnetwork_builders,
                  previous_ensemble_subnetwork_builders)
    ]
