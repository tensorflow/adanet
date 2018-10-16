"""An AdaNet subnetwork definition in Tensorflow using a single graph.

Copyright 2018 The AdaNet Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections


def _validate_nested_persisted_tensors(persisted_tensors):
  """Raises a ValueError when a nested dict is empty in persisted_tensors."""

  for key, entry in persisted_tensors.items():
    if not isinstance(entry, dict):
      continue
    if not entry:
      raise ValueError("Got empty nested dictionary for key: '{}'".format(key))
    _validate_nested_persisted_tensors(entry)


class Subnetwork(
    collections.namedtuple(
        "Subnetwork",
        ["last_layer", "logits", "complexity", "persisted_tensors"])):
  """An AdaNet subnetwork.

  In the AdaNet paper, an `adanet.Subnetwork` is are called a 'subnetwork',
  and indicated by 'h'. A collection of weighted subnetworks form an AdaNet
  ensemble.
  """

  def __new__(cls, last_layer, logits, complexity, persisted_tensors):
    """Creates a validated `Subnetwork` instance.

    Args:
      last_layer: A `Tensor` output of the last layer of the subnetwork, i.e the
        layer before the logits layer. When the mixture weight type is `MATRIX`,
        the AdaNet algorithm takes care of computing ensemble mixture weights
        matrices (one per subnetwork) that multiply the various last layers of
        the ensemble's subnetworks, and regularize them using their subnetwork's
        complexity. This field is represented by 'h' in the AdaNet paper.
      logits: logits `Tensor` for training the subnetwork. NOTE: These logits
        are not used in the ensemble's outputs if the mixture weight type is
        `MATRIX`, instead AdaNet learns its own logits (mixture weights) from
        the subnetwork's `last_layers` with complexity regularization. The
        logits are used in the ensemble only when the mixture weights type is
        `SCALAR` or `VECTOR`. Even though the logits are not used in the
        ensemble in some cases, they should always be supplied as adanet uses
        the logits to train the subnetworks.
      complexity: A scalar representing the complexity of the subnetwork's
        architecture. It is used for choosing the best subnetwork at each
        iteration, and for regularizing the weighted outputs of more complex
        subnetworks.
      persisted_tensors: Nested dictionary of string to `Tensor` to persist
        across iterations. At the end of an iteration, the `Tensors` will be
        available to subnetworks in the next iterations, whereas others that are
        not part of the `Subnetwork` will be pruned. This allows later
        `Subnetworks` to dynamically build upon arbitrary `Tensors` from
        previous `Subnetworks`.

    Returns:
      A validated `Subnetwork` object.

    Raises:
      ValueError: If last_layer is None.
      ValueError: If logits is None.
      ValueError: If complexity is None.
      ValueError: If persited_tensors is not a dictionary.
      ValueError: If persited_tensors contains an empty nested dictionary.
    """

    if last_layer is None:
      raise ValueError("last_layer not provided")
    if logits is None:
      raise ValueError("logits not provided")
    if complexity is None:
      raise ValueError("complexity not provided")
    if not isinstance(persisted_tensors, dict):
      raise ValueError("persisted_tensors must be a dict")
    _validate_nested_persisted_tensors(persisted_tensors)
    return super(Subnetwork, cls).__new__(
        cls,
        last_layer=last_layer,
        logits=logits,
        complexity=complexity,
        persisted_tensors=persisted_tensors)


class Builder(object):
  """Interface for a subnetwork builder.

  Given features, labels, and the best ensemble of subnetworks at iteration
  t-1, a `Builder` creates a `Subnetwork` to add to a candidate
  ensemble at iteration t. These candidate ensembles are evaluated against one
  another at the end of the iteration, and the best one is selected based on its
  complexity-regularized loss.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """Returns the unique name of the ensemble to contain this subnetwork."""

  @abc.abstractmethod
  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    """Returns the candidate `Subnetwork` to add to the ensemble.

    This method will be called only once, before `build_subnetwork_train_op`
    and `build_mixture_weights_train_op` are called. This method should
    construct the candidate subnetwork's graph operations and variables.

    Args:
      features: Input `dict` of `Tensor` objects.
      logits_dimension: Size of the last dimension of the logits `Tensor`.
        Typically, logits have for shape `[batch_size, logits_dimension]`.
      training: A python boolean indicating whether the graph is in training
        mode or prediction mode.
      iteration_step: Integer `Tensor` representing the step since the beginning
        of the current iteration, as opposed to the global step.
      summary: An `adanet.Summary` for scoping summaries to individual
        subnetworks in Tensorboard.
      previous_ensemble: The best `Ensemble` from iteration t-1. The created
        subnetwork will extend the previous ensemble to form the `Ensemble` at
        iteration t.

    Returns:
      A `Subnetwork` instance.
    """

  @abc.abstractmethod
  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    """Returns an op for training a new subnetwork.

    This method will be called once after `build_subnetwork`.

    NOTE: This method should _not_ increment the global step tensor.

    Args:
      subnetwork: Newest subnetwork, that is not part of the
        `previous_ensemble`.
      loss: A `Tensor` containing the subnetwork's loss to minimize.
      var_list: List of subnetwork `tf.Variable` parameters to update as part of
        the training operation.
      labels: Labels `Tensor`.
      iteration_step: Integer `Tensor` representing the step since the beginning
        of the current iteration, as opposed to the global step.
      summary: An `adanet.Summary` for scoping summaries to individual
        subnetworks in Tensorboard.
      previous_ensemble: The best `Ensemble` from iteration t-1. The created
        subnetwork will extend the previous ensemble to form the `Ensemble` at
        iteration t. Is None for iteration 0.

    Returns:
      A train op.
    """

  @abc.abstractmethod
  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    """Returns an op for training the ensemble's mixture weights.

    Allows AdaNet to learn the mixture weights of each subnetwork
    according to Equation (6).

    This method will be called once after `build_subnetwork`.

    NOTE: This method should _not_ increment the global step tensor.

    Args:
      loss: A `Tensor` containing the ensemble's loss to minimize.
      var_list: List of ensemble mixture weight `tf.Variables` to update as
        become part of the training operation.
      logits: The ensemble's logits `Tensor` from applying the mixture weights
        and bias to the ensemble's subnetworks.
      labels: Labels `Tensor`.
      iteration_step: Integer `Tensor` representing the step since the beginning
        of the current iteration, as opposed to the global step.
      summary: An `adanet.Summary` for scoping summaries to individual
        subnetworks in Tensorboard.

    Returns:
      A train op.
    """

  def build_subnetwork_report(self):
    """Returns a `subnetwork.Report` to materialize and record.

    This method will be called once after `build_subnetwork`.
    Do NOT depend on variables created in `build_subnetwork_train_op` or
    `build_mixture_weights_train_op`, because they are not called before
    `build_subnetwork_report` is called.

    If it returns None, AdaNet records the name and standard eval metrics.
    """

    return None

  def prune_previous_ensemble(self, previous_ensemble):
    """Specifies which subnetworks to include in the candidate ensemble.

    The current default implementation does not prune any subnetworks from
    the ensemble.

    Args:
      previous_ensemble: `Ensemble` object.

    Returns:
      List of integer indices of weighted_subnetworks to keep.
    """
    return range(len(previous_ensemble.weighted_subnetworks))


class Generator(object):
  """Interface for a subnetwork builder generator.

  Given the ensemble of subnetworks at iteration t-1, this object is
  responsible for generating the set of subnetworks for iteration t that
  minimize Equation (6) in the paper.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    """Generates `adanet.Builders` to train at iteration t.

    Args:
      previous_ensemble: The best `adanet.Ensemble` from iteration t-1.
        DEPRECATED. We are transitioning away from the use of previous_ensemble
        in generate_candidates. New Generators should *not* use
        previous_ensemble in their implementation of generate_candidates --
        please only use iteration_number, previous_ensemble_reports and
        all_reports.
      iteration_number: Python integer AdaNet iteration t, starting from 0.
      previous_ensemble_reports: List of `adanet.subnetwork.MaterializedReport`s
        corresponding to the Builders composing `adanet.Ensemble` from iteration
        t-1. The first element in the list corresponds to the Builder added in
        the first iteration. If `ReportMaterializer` is not supplied to the
        estimator, previous_ensemble_report is `None`.
      all_reports: List of `adanet.subnetwork.MaterializedReport`s. If
        `ReportMaterializer` is not supplied to the estimator, all_reports is
        `None`. If `ReportMaterializer` is supplied to the estimator and t=0,
        all_reports is an empty List. Otherwise, all_reports is a sequence of
        Lists. Each element of the sequence is a List containing all the
        `adanet.subnetwork.MaterializedReport`s in an AdaNet iteration, starting
        from iteration 0, and ending at iteration t-1.

    Returns:
      A list of `adanet.Builders`.
    """


class SimpleGenerator(Generator):
  """A generator that always returns the same `adanet.Builders`."""

  def __init__(self, subnetwork_builders):
    """Creates a `adanet.Subnetwork` instance.

    Args:
      subnetwork_builders: List of `adanet.Builders` to return at each iteration
        when `generate_candidates` is called.

    Returns:
      A `SimpleGenerator` instance.
    """

    self._subnetwork_builders = subnetwork_builders

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    """Returns the predefined set of `adanet.Builders`."""

    return self._subnetwork_builders
