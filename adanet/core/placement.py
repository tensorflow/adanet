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
"""Distributed placement strategies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class PlacementStrategy(object):
  """Abstract placement strategy for distributed training."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def should_build_ensemble(self, num_subnetworks):
    """Whether to build the ensemble on the current worker.

    Args:
      num_subnetworks: Integer number of subnetworks to train in the current
        iteration.

    Returns:
      Boolean whether to build the ensemble on the current worker.
    """

  @abc.abstractmethod
  def should_build_subnetwork(self, num_subnetworks, subnetwork_index):
    """Whether to build the given subnetwork on the current worker.

    Args:
      num_subnetworks: Integer number of subnetworks to train in the current
        iteration.
      subnetwork_index: Integer index of the subnetwork in the list of the
        current iteration's subnetworks.

    Returns:
      Boolean whether to build the given subnetwork on the current worker.
    """

  @abc.abstractmethod
  def should_train_subnetworks(self, num_subnetworks):
    """Whether to train subnetworks on the current worker.

    Args:
      num_subnetworks: Integer number of subnetworks to train in the current
        iteration.

    Returns:
      Boolean whether to train subnetworks on the current worker.
    """


class ReplicationStrategy(PlacementStrategy):
  """A simple strategy that replicates the same graph on every worker.

  This strategy does not scale well as the number of subnetworks and workers
  increases. For 'm' workers, 'n' parameter servers, and 'k' subnetworks,
  this strategy will scale with O(m) training speedup, O(m*n*k) variable
  fetches from parameter servers, and O(k) memory required per worker.
  Additionally there will be O(m) stale gradients per subnetwork when
  training with asynchronous SGD.

  Returns:
    A :class:`ReplicationStrategy` instance for the current cluster.

  """

  def should_build_ensemble(self, num_subnetworks):
    return True

  def should_build_subnetwork(self, num_subnetworks, subnetwork_index):
    return True

  def should_train_subnetworks(self, num_subnetworks):
    return True


class RoundRobinStrategy(PlacementStrategy):
  """A strategy that round-robin assigns subgraphs to specific workers.

  Specifically, it selects dedicated workers to only train ensemble variables,
  and round-robin assigns subnetworks to dedicated subnetwork-training workers.

  Unlike :class:`ReplicationStrategy`, this strategy scales better with the
  number of subnetworks, workers, and parameter servers. For 'm' workers, 'n'
  parameter servers, and 'k' subnetworks, this strategy will scale with
  O(m/k) training speedup, O(m*n/k) variable fetches from parameter servers,
  and O(1) memory required per worker. Additionally, there will only be O(m/k)
  stale gradients per subnetwork when training with asynchronous SGD, which
  improves training stability.

  When there are more workers than subnetworks, this strategy assigns
  subnetworks to workers modulo the number of subnetworks.

  Conversely, when there are more subnetworks than workers, this round robin
  assigns subnetworks modulo the number of workers. So certain workers may end
  up training more than one subnetwork.

  This strategy gracefully handles scenarios when the number of subnetworks
  does not perfectly divide the number of workers and vice-versa. It also
  supports different numbers of subnetworks at different iterations, and
  reloading training with a resized cluster.

  TODO: Implement specialized parameter server variable placement per
  subnetwork to get O(m*n/k) scaling. Currently it is O(m*n).

  TODO: Allow user to disable ensemble workers. For example, when there
  are no ensemble variables to train, such as in a uniform average ensemble,
  there is no need for a non-chief to create the full ensemble during training,
  except for the chief to initialize the ensemble's non-trainable variables.

  Args:
    num_workers: Integer number of worker replicas in the cluster.
    worker_index: Index of the current worker in the cluster.

  Returns:
    A :class:`RoundRobinStrategy` instance for the current cluster.
  """

  # TODO: Optional code organization suggestion:
  # Explicitly define what a "task" is, to make the below code clearer. One way
  # of doing this:
  #
  # def _worker_tasks(self, num_subnetworks):
  #   """Returns the set of tasks that this worker can work on.
  #
  #   Each task is represented by an integer between 0 and num_subnetworks
  #   (inclusive). 0 corresponds to the task of training the ensemble(s), 1
  #   corresponds to the task of training subnetwork 0, 2 corresponds to the
  #   task of training subnetwork 1, and so on.
  #
  #   Examples:
  #     - 1 worker, 3 subnetworks. This would return {0, 1, 2, 3} for the only
  #       worker, since the only worker would have to train the ensemble(s) and
  #       all 3 subnetworks.
  #     - 2 workers, 3 subnetworks. This would return {0} for worker 0, and
  #       {1, 2, 3} for worker 1. This means that the first worker trains the
  #       ensemble(s), while the second worker trains all three subnetworks.
  #     - 4 workers, 3 subnetworks. This would return {0} for worker 0, {1} for
  #       worker 1, {2} for worker 2, and {3} for worker 3. This means that
  #       worker 0 trains the ensemble(s) while the rest of the workers train
  #       one subnetwork each.
  #     - 5 workers, 3 subnetworks. This would return {0} for worker 0, {1} for
  #       worker 1, {2} for worker 2, {3} for worker 3, and {1} for worker 4.
  #       This is like the previous case, except that worker 4 also helps to
  #       train subnetwork 0.
  #   """
  #
  # That way, should_build_ensemble can just be:
  #
  #   return 0 in self._worker_tasks(...)
  #
  # then should_build_subnetwork can just be:
  #
  #   if (subnetwork_index in self._worker_tasks(...) or 0 in
  #       subnetwork_index in self._worker_tasks(...)):
  #     return True
  #   return False
  #
  # and should_train_subnetwork can just be:
  #
  #   return subnetwork_index in self._worker_tasks(...)

  def __init__(self, num_workers, worker_index):
    self._num_workers = num_workers
    self._worker_index = worker_index

  def _worker_task(self, num_subnetworks):
    # The first worker will always build the ensemble so we add 1.
    return self._worker_index % (num_subnetworks + 1)

  def should_build_ensemble(self, num_subnetworks):
    if num_subnetworks == 1:
      return True
    worker_task = self._worker_task(num_subnetworks)
    # The ensemble builder is always the first worker task.
    return worker_task == 0

  def should_build_subnetwork(self, num_subnetworks, subnetwork_index):
    if num_subnetworks == 1:
      return True
    worker_task = self._worker_task(num_subnetworks)
    if worker_task == 0:
      # The zeroth index worker is an ensemble worker.
      return True

    workers_per_subnetwork = self._num_workers // (num_subnetworks + 1)
    if self._num_workers % (num_subnetworks + 1) == 0:
      num_subnetwork_workers = num_subnetworks
    elif self._worker_index >= workers_per_subnetwork * (num_subnetworks + 1):
      num_subnetwork_workers = self._num_workers % (num_subnetworks + 1) - 1
    else:
      num_subnetwork_workers = num_subnetworks
    subnetwork_worker_index = worker_task - 1
    return subnetwork_worker_index == subnetwork_index % num_subnetwork_workers

  def should_train_subnetworks(self, num_subnetworks):
    if num_subnetworks == 1 or self._num_workers == 1:
      return True
    return not self.should_build_ensemble(num_subnetworks)
