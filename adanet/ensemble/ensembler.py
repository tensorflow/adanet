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
"""Ensembler definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections


class TrainOpSpec(
    collections.namedtuple("TrainOpSpec",
                           ["train_op", "chief_hooks", "hooks"])):
  """A data structure for specifying ensembler training operations.

  Args:
    train_op: Op for the training step.
    chief_hooks: Iterable of :class:`tf.train.SessionRunHook` objects to run on
      the chief worker during training.
    hooks: Iterable of :class:`tf.train.SessionRunHook` objects to run on all
      workers during training.

  Returns:
    An :class:`adanet.ensemble.TrainOpSpec` object.
  """

  def __new__(cls, train_op, chief_hooks=None, hooks=None):
    # Make hooks immutable.
    chief_hooks = tuple(chief_hooks) if chief_hooks else ()
    hooks = tuple(hooks) if hooks else ()
    return super(TrainOpSpec, cls).__new__(cls, train_op, chief_hooks, hooks)


class Ensemble(object):  # pytype: disable=ignored-metaclass
  """An abstract ensemble of subnetworks."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def logits(self):
    """Ensemble logits :class:`tf.Tensor`."""

  @abc.abstractproperty
  def subnetworks(self):
    """Returns an ordered :class:`Iterable` of the ensemble's subnetworks."""

  @property
  def predictions(self):
    """Optional dict of Ensemble predictions to be merged in EstimatorSpec.

    These will be additional (over the default included by the head) predictions
    which will be included in the EstimatorSpec in `predictions` and
    `export_outputs` (wrapped as PredictOutput).
    """
    return None


class Ensembler(object):  # pytype: disable=ignored-metaclass
  """An abstract ensembler."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """This ensembler's unique string name."""

  @abc.abstractmethod
  def build_ensemble(self, subnetworks, previous_ensemble_subnetworks, features,
                     labels, logits_dimension, training, iteration_step,
                     summary, previous_ensemble, previous_iteration_checkpoint):
    # pyformat: disable
    """Builds an ensemble of subnetworks.

    Accessing the global step via :meth:`tf.train.get_or_create_global_step()`
    or :meth:`tf.train.get_global_step()` within this scope will return an
    incrementable iteration step since the beginning of the iteration.

    Args:
      subnetworks: Ordered iterable of :class:`adanet.subnetwork.Subnetwork`
        instances to ensemble. Must have at least one element.
      previous_ensemble_subnetworks: Ordered iterable of
        :class:`adanet.subnetwork.Subnetwork` instances present in previous
        ensemble to be used. The subnetworks from previous_ensemble not
        included in this list should be pruned. Can be set to None or empty.
      features: Input :code:`dict` of :class:`tf.Tensor` objects.
      labels: Labels :class:`tf.Tensor` or a dictionary of string label name to
        :class:`tf.Tensor` (for multi-head). Can be :code:`None`.
      logits_dimension: Size of the last dimension of the logits
        :class:`tf.Tensor`. Typically, logits have for shape `[batch_size,
        logits_dimension]`.
      training: A python boolean indicating whether the graph is in training
        mode or prediction mode.
      iteration_step: Integer :class:`tf.Tensor` representing the step since the
        beginning of the current iteration, as opposed to the global step.
      summary: An :class:`adanet.Summary` for scoping summaries to individual
        ensembles in Tensorboard. Using :meth:`tf.summary` within this scope
        will use this :class:`adanet.Summary` under the hood.
      previous_ensemble: The best :class:`adanet.Ensemble` from iteration *t-1*.
        The created subnetwork will extend the previous ensemble to form the
        :class:`adanet.Ensemble` at iteration *t*.
      previous_iteration_checkpoint: The `tf.train.Checkpoint` object associated
        with the previous iteration.

    Returns:
      An :class:`adanet.ensemble.Ensemble` subclass instance.
    """
    # pyformat: enable

  @abc.abstractmethod
  def build_train_op(self, ensemble, loss, var_list, labels, iteration_step,
                     summary, previous_ensemble):
    # pyformat: disable
    """Returns an op for training an ensemble.

    Accessing the global step via :meth:`tf.train.get_or_create_global_step`
    or :meth:`tf.train.get_global_step` within this scope will return an
    incrementable iteration step since the beginning of the iteration.

    Args:
      ensemble: The :class:`adanet.ensemble.Ensemble` subclass instance returned
        by this instance's :meth:`build_ensemble`.
      loss: A :class:`tf.Tensor` containing the ensemble's loss to minimize.
      var_list: List of ensemble :class:`tf.Variable` parameters to update as
        part of the training operation.
      labels: Labels :class:`tf.Tensor` or a dictionary of string label name to
        :class:`tf.Tensor` (for multi-head).
      iteration_step: Integer :class:`tf.Tensor` representing the step since the
        beginning of the current iteration, as opposed to the global step.
      summary: An :class:`adanet.Summary` for scoping summaries to individual
        ensembles in Tensorboard. Using :code:`tf.summary` within this scope
        will use this :class:`adanet.Summary` under the hood.
      previous_ensemble: The best :class:`adanet.ensemble.Ensemble` from the
        previous iteration.
    Returns:
      Either a train op or an :class:`adanet.ensemble.TrainOpSpec`.
    """
    # pyformat: enable
