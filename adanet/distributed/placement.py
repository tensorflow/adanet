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
import contextlib

from absl import logging
from adanet import tf_compat
from adanet.distributed.devices import _OpNameHashStrategy
import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class PlacementStrategy(object):
  """Abstract placement strategy for distributed training.

  Given a cluster of workers, the placement strategy determines which subgraph
  each worker constructs.
  """

  @property
  def config(self):
    """Returns this strategy's configuration.

    Returns:
      The :class:`tf.estimator.RunConfig` instance that defines the cluster.
    """

    return self._config

  @config.setter
  def config(self, config):
    """Configures the placement strategy with the given cluster description.

    Args:
      config: A :class:`tf.estimator.RunConfig` instance that defines the
        cluster.
    """

    self._config = config

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

  @abc.abstractmethod
  @contextlib.contextmanager
  def subnetwork_devices(self, num_subnetworks, subnetwork_index):
    """A context for assigning subnetwork ops to devices."""


class ReplicationStrategy(PlacementStrategy):
  # pyformat: disable
  """A simple strategy that replicates the same graph on every worker.

  This strategy does not scale well as the number of subnetworks and workers
  increases. For :math:`m` workers, :math:`n` parameter servers, and :math:`k`
  subnetworks, this strategy will scale with :math:`O(m)` training speedup,
  :math:`O(m*n*k)` variable fetches from parameter servers, and :math:`O(k)`
  memory required per worker. Additionally there will be :math:`O(m)` stale
  gradients per subnetwork when training with asynchronous SGD.

  Returns:
    A :class:`ReplicationStrategy` instance for the current cluster.
  """
  # pyformat: enable

  def should_build_ensemble(self, num_subnetworks):
    return True

  def should_build_subnetwork(self, num_subnetworks, subnetwork_index):
    return True

  def should_train_subnetworks(self, num_subnetworks):
    return True

  @contextlib.contextmanager
  def subnetwork_devices(self, num_subnetworks, subnetwork_index):
    # Use default devices.
    yield


class RoundRobinStrategy(PlacementStrategy):
  # pyformat: disable
  """A strategy that round-robin assigns subgraphs to specific workers.

  Specifically, it selects dedicated workers to only train ensemble variables,
  and round-robin assigns subnetworks to dedicated subnetwork-training workers.

  Unlike :class:`ReplicationStrategy`, this strategy scales better with the
  number of subnetworks, workers, and parameter servers. For :math:`m` workers,
  :math:`n` parameter servers, and :math:`k` subnetworks, this strategy will
  scale with :math:`O(m/k)` training speedup, :math:`O(m*n/k)` variable fetches
  from parameter servers, and :math:`O(1)` memory required per worker.
  Additionally, there will only be :math:`O(m/k)` stale gradients per subnetwork
  when training with asynchronous SGD, which reduces training instability versus
  :class:`ReplicationStrategy`.

  When there are more workers than subnetworks, this strategy assigns
  subnetworks to workers modulo the number of subnetworks.

  Conversely, when there are more subnetworks than workers, this round robin
  assigns subnetworks modulo the number of workers. So certain workers may end
  up training more than one subnetwork.

  This strategy gracefully handles scenarios when the number of subnetworks
  does not perfectly divide the number of workers and vice-versa. It also
  supports different numbers of subnetworks at different iterations, and
  reloading training with a resized cluster.

  Args:
    drop_remainder: Bool whether to drop remaining subnetworks that haven't been
      assigned to a worker in the remainder after perfect division of workers by
      the current iteration's num_subnetworks + 1. When :code:`True`, each subnetwork
      worker will only train a single subnetwork, and subnetworks that have not
      been assigned to assigned to a worker are dropped. NOTE: This can result
      in subnetworks not being assigned to any worker when
      num_workers < num_subnetworks + 1. When :code:`False`, remaining subnetworks
      during the round-robin assignment will be placed on workers that already
      have a subnetwork.

  Returns:
    A :class:`RoundRobinStrategy` instance for the current cluster.
  """
  # pyformat: enable

  # TODO: Allow user to disable ensemble workers. For example, when there
  # are no ensemble variables to train, such as in a uniform average ensemble,
  # there is no need for a non-chief to create the full ensemble during
  # training, except for the chief to initialize the ensemble's non-trainable
  # variables.

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

  def __init__(self, drop_remainder=False, dedicate_parameter_servers=True):
    self._drop_remainder = drop_remainder
    self._dedicate_parameter_servers = dedicate_parameter_servers

  @property
  def _num_workers(self):
    return self.config.num_worker_replicas

  @property
  def _worker_index(self):
    return self.config.global_id_in_cluster or 0

  def _worker_task(self, num_subnetworks):
    """Returns the worker index modulo the number of subnetworks."""

    if self._drop_remainder and self._num_workers > 1 and (num_subnetworks >
                                                           self._num_workers):
      logging.log_first_n(
          logging.WARNING,
          "With drop_remainer=True, %s workers and %s subnetworks, the last %s "
          "subnetworks will be dropped and will not be trained", 1,
          self._num_workers, num_subnetworks,
          num_subnetworks - self._num_workers - 1)
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

    subnetwork_worker_index = worker_task - 1
    if self._drop_remainder:
      return subnetwork_worker_index == subnetwork_index

    workers_per_subnetwork = self._num_workers // (num_subnetworks + 1)
    if self._num_workers % (num_subnetworks + 1) == 0:
      num_subnetwork_workers = num_subnetworks
    elif self._worker_index >= workers_per_subnetwork * (num_subnetworks + 1):
      num_subnetwork_workers = self._num_workers % (num_subnetworks + 1) - 1
    else:
      num_subnetwork_workers = num_subnetworks
    return subnetwork_worker_index == subnetwork_index % num_subnetwork_workers

  def should_train_subnetworks(self, num_subnetworks):
    if num_subnetworks == 1 or self._num_workers == 1:
      return True
    return not self.should_build_ensemble(num_subnetworks)

  @contextlib.contextmanager
  def subnetwork_devices(self, num_subnetworks, subnetwork_index):
    if not self._dedicate_parameter_servers:
      # Use default device placement.
      yield
      return

    # Each subnetwork gets its own dedicated parameter servers
    num_ps_replicas = self.config.num_ps_replicas
    ps_numbers = np.array(range(num_ps_replicas))
    subnetwork_group = subnetwork_index
    if num_ps_replicas > 0 and num_subnetworks > num_ps_replicas:
      subnetwork_group = subnetwork_index % num_ps_replicas
    ps_group = np.array_split(ps_numbers, num_subnetworks)[subnetwork_group]

    # Assign ops to parameter servers based on hashed op names.
    ps_strategy = _OpNameHashStrategy(len(ps_group))

    def device_fn(op):
      """Assigns variables to a subnetwork's dedicated parameter servers."""

      # Import here to avoid strict BUILD deps check.
      from tensorflow.core.framework import node_def_pb2  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
      node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
      from tensorflow.python.training import device_setter  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
      if num_ps_replicas > 0 and node_def.op in device_setter.STANDARD_PS_OPS:
        # ps_group lists the task ids in the group. Adding the first task id in
        # the group to the task number determined by the PS strategy gives the
        # correct parameter server assignment.
        return "/job:ps/task:{}".format(ps_group[0] + ps_strategy(op))
      return op.device

    with tf_compat.v1.device(device_fn):
      yield
