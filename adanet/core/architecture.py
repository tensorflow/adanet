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
"""An internal AdaNet model architecture definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json


class _Architecture(object):
  """An AdaNet model architecture.

  This data structure is the blueprint for reconstructing an AdaNet model. It
  contains not only information about the underlying Ensemble, but also the
  `adanet.subnetwork.Builder` instances that compose the ensemble, the
  `adanet.ensemble.Ensembler` that constructed it, as well as the sequence
  of states in the search space that led to the construction of this model.
  In addition, it stores `replay_indices` A list of indices (an index per
  boosting iteration); Holding the index of the ensemble in the candidate list
  throughout the run.

  It is serializable and deserializable for persistent storage.
  """

  def __init__(self, ensemble_candidate_name, ensembler_name, global_step=None,
               replay_indices=None):
    self._ensemble_candidate_name = ensemble_candidate_name
    self._ensembler_name = ensembler_name
    self._global_step = global_step
    self._subnets = []
    self._replay_indices = replay_indices or []

  @property
  def ensemble_candidate_name(self):
    """The ensemble candidate's name.

    Returns:
      String name of the ensemble candidate with this architecture.
    """
    return self._ensemble_candidate_name

  @property
  def ensembler_name(self):
    """The ensembler's name.

    Returns:
      String name of the ensembler that constructed the ensemble.
    """
    return self._ensembler_name

  @property
  def global_step(self):
    """The global step when this architecture was serialized.

    Returns:
      Integer global step.
    """

    return self._global_step

  @property
  def subnetworks(self):
    """The component subnetworks.

    Returns:
      An Iterable of (iteration_number, builder_name) tuples.
    """

    return tuple(self._subnets)

  @property
  def replay_indices(self):
    """The list of replay indices.

    Returns:
      A list of integers (an integer per boosting iteration); Holding the index
      of the ensemble in the candidate list throughout the run
    """

    return self._replay_indices

  @property
  def subnetworks_grouped_by_iteration(self):
    """The component subnetworks grouped by iteration number.

    Returns:
      An Iterable of (iteration_number, builder_names) tuples where the builder
        names are grouped by iteration number.
    """

    subnet_by_iteration = {}
    for iteration_number, builder_name in self._subnets:
      if iteration_number not in subnet_by_iteration:
        subnet_by_iteration[iteration_number] = []
      subnet_by_iteration[iteration_number].append(builder_name)
    return tuple([
        (i, tuple(subnet_by_iteration[i])) for i in sorted(subnet_by_iteration)
    ])

  def add_subnetwork(self, iteration_number, builder_name):
    """Adds the given subnetwork metadata.

    Args:
      iteration_number: Integer iteration number when this Subnetwork was
        created.
      builder_name: String name of the `adanet.subnetwork.Builder` that produced
        this Subnetwork.
    """
    self._subnets.append((iteration_number, builder_name))

  # TODO: Remove setters and getters.
  def add_replay_index(self, index):
    self._replay_indices.append(index)

  def set_replay_indices(self, indices):
    self._replay_indices = copy.copy(indices)

  def serialize(self, iteration_number, global_step):
    """Returns a string serialization of this object."""

    # TODO: Confirm that it makes sense to have global step of 0.
    assert global_step is not None
    ensemble_arch = {
        "ensemble_candidate_name": self.ensemble_candidate_name,
        "iteration_number": iteration_number,
        "global_step": global_step,
        "ensembler_name": self.ensembler_name,
        "subnetworks": [],
        "replay_indices": self._replay_indices
    }
    for iteration_number, builder_name in self._subnets:
      subnetwork_arch = {
          "iteration_number": int(iteration_number),
          "builder_name": builder_name,
      }
      ensemble_arch["subnetworks"].append(subnetwork_arch)
    return json.dumps(ensemble_arch, sort_keys=True)

  @staticmethod
  def deserialize(serialized_architecture):
    """Deserializes a serialized architecture.

    Args:
      serialized_architecture: String representation of an `_Architecture`
        obtained by calling `serialize`.

    Returns:
      A deserialized `_Architecture` instance.
    """

    ensemble_arch = json.loads(serialized_architecture)
    architecture = _Architecture(ensemble_arch["ensemble_candidate_name"],
                                 ensemble_arch["ensembler_name"],
                                 ensemble_arch["global_step"],
                                 ensemble_arch["replay_indices"])
    for subnet in ensemble_arch["subnetworks"]:
      architecture.add_subnetwork(subnet["iteration_number"],
                                  subnet["builder_name"])
    return architecture
