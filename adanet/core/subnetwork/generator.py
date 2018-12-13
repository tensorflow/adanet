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

from tensorflow.python.util import deprecation


def _validate_nested_persisted_tensors(persisted_tensors):
  """Raises a ValueError when a nested dict is empty in persisted_tensors."""

  for key, entry in persisted_tensors.items():
    if not isinstance(entry, dict):
      continue
    if not entry:
      raise ValueError("Got empty nested dictionary for key: '{}'".format(key))
    _validate_nested_persisted_tensors(entry)


class TrainOpSpec(
    collections.namedtuple("TrainOpSpec",
                           ["train_op", "chief_hooks", "hooks"])):
  """A data structure for specifying training operations.

  Args:
    train_op: Op for the training step.
    chief_hooks: Iterable of :class:`tf.train.SessionRunHook` objects to run on
      the chief worker during training.
    hooks: Iterable of :class:`tf.train.SessionRunHook` objects to run on all
      workers during training.

  Returns:
    A :class:`adanet.subnetwork.TrainOpSpec` object.
  """

  def __new__(cls, train_op, chief_hooks=None, hooks=None):
    # Make hooks immutable.
    chief_hooks = tuple(chief_hooks) if chief_hooks else ()
    hooks = tuple(hooks) if hooks else ()
    return super(TrainOpSpec, cls).__new__(cls, train_op, chief_hooks, hooks)


class Subnetwork(
    collections.namedtuple(
        "Subnetwork",
        ["last_layer", "logits", "complexity", "persisted_tensors", "shared"])):
  # pyformat: disable
  """An AdaNet subnetwork.

  In the AdaNet paper, an :class:`adanet.subnetwork.Subnetwork` is are called a
  'subnetwork', and indicated by 'h'. A collection of weighted subnetworks form
  an AdaNet ensemble.

  Args:
    last_layer: :class:`tf.Tensor` output or dict of string to
      :class:`tf.Tensor` outputs (for multi-head) of the last layer of the
      subnetwork, i.e the layer before the logits layer. When the mixture weight
      type is :class:`MATRIX`, the AdaNet algorithm takes care of computing
      ensemble mixture weights matrices (one per subnetwork) that multiply the
      various last layers of the ensemble's subnetworks, and regularize them
      using their subnetwork's complexity. This field is represented by 'h' in
      the AdaNet paper.
    logits: :class:`tf.Tensor` logits or dict of string to :class:`tf.Tensor`
      logits (for multi-head) for training the subnetwork. These logits are not
      used in the ensemble's outputs if the mixture weight type is
      :class:`MATRIX`, instead AdaNet learns its own logits (mixture weights)
      from the subnetwork's `last_layers` with complexity regularization. The
      logits are used in the ensemble only when the mixture weights type is
      :class:`SCALAR` or :class:`VECTOR`. Even though the logits are not used
      in the ensemble in some cases, they should always be supplied as adanet
      uses the logits to train the subnetworks.
    complexity: A scalar :class:`tf.Tensor` representing the complexity of the
      subnetwork's architecture. It is used for choosing the best subnetwork at
      each iteration, and for regularizing the weighted outputs of more complex
      subnetworks.
    persisted_tensors: DEPRECATED. See `shared`. Optional nested dictionary of
      string to :class:`tf.Tensor` to persist across iterations. At the end of
      an iteration, the :class:`tf.Tensor` instances will be available to
      subnetworks in the next iterations, whereas others that are not part of
      the `Subnetwork` will be pruned. This allows later
      :class:`adanet.subnetwork.Subnetwork` instances to dynamically build
      upon arbitrary :class:`tf.Tensors` from previous
      :class:`adanet.subnetwork.Subnetwork` instances.
    shared: Optional Python object(s), primitive(s), or function(s) to share
      with subnetworks within the same iteration or in future iterations.

  Returns:
    A validated :class:`adanet.subnetwork.Subnetwork` object.

  Raises:
    ValueError: If last_layer is None.
    ValueError: If logits is None.
    ValueError: If logits is a dict but last_layer is not.
    ValueError: If last_layer is a dict but logits is not.
    ValueError: If complexity is None.
    ValueError: If persisted_tensors is present but not a dictionary.
    ValueError: If persisted_tensors contains an empty nested dictionary.
  """
  # pyformat: enable

  @deprecation.deprecated_args(
      None, "`persisted_tensors` is deprecated, please use `shared` instead.",
      "persisted_tensors")
  def __new__(cls,
              last_layer,
              logits,
              complexity,
              persisted_tensors=None,
              shared=None):
    if last_layer is None:
      raise ValueError("last_layer not provided")
    if logits is None:
      raise ValueError("logits not provided")
    if isinstance(logits, dict) and not isinstance(last_layer, dict):
      raise ValueError("if logits is a dict last_layer must also be a dict")
    if isinstance(last_layer, dict) and not isinstance(logits, dict):
      raise ValueError("if last_layer is a dict logits must also be a dict")
    if complexity is None:
      raise ValueError("complexity not provided")
    if persisted_tensors is not None:
      if not isinstance(persisted_tensors, dict):
        raise ValueError("persisted_tensors must be a dict")
      _validate_nested_persisted_tensors(persisted_tensors)
    return super(Subnetwork, cls).__new__(
        cls,
        last_layer=last_layer,
        logits=logits,
        complexity=complexity,
        persisted_tensors=persisted_tensors,
        shared=shared)


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
    """Returns the unique name of this subnetwork within an iteration."""

  @abc.abstractmethod
  def build_subnetwork(self,
                       features,
                       labels,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    # pyformat: disable
    """Returns the candidate `Subnetwork` to add to the ensemble.

    This method will be called only once, before
    :meth:`build_subnetwork_train_op`
    and :meth:`build_mixture_weights_train_op` are called. This method should
    construct the candidate subnetwork's graph operations and variables.

    Accessing the global step via :meth:`tf.train.get_or_create_global_step()`
    or
    :meth:`tf.train.get_global_step()` within this scope will return an
    incrementable
    iteration step since the beginning of the iteration.

    Args:
      features: Input `dict` of :class:`tf.Tensor` objects.
      labels: Labels :class:`tf.Tensor` or a dictionary of string label name to
        :class:`tf.Tensor` (for multi-head). Can be `None`.
      logits_dimension: Size of the last dimension of the logits
        :class:`tf.Tensor`. Typically, logits have for shape `[batch_size,
        logits_dimension]`.
      training: A python boolean indicating whether the graph is in training
        mode or prediction mode.
      iteration_step: Integer :class:`tf.Tensor` representing the step since the
        beginning of the current iteration, as opposed to the global step.
      summary: An :class:`adanet.Summary` for scoping summaries to individual
        subnetworks in Tensorboard. Using :meth:`tf.summary` within this scope
        will use this :class:`adanet.Summary` under the hood.
      previous_ensemble: The best :class:`adanet.Ensemble` from iteration t-1.
        The created subnetwork will extend the previous ensemble to form the
        :class:`adanet.Ensemble` at iteration t.

    Returns:
      An :class:`adanet.subnetwork.Subnetwork` instance.
    """
    # pyformat: enable

  @abc.abstractmethod
  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    """Returns an op for training a new subnetwork.

    This method will be called once after :meth:`build_subnetwork`.

    Accessing the global step via :meth:`tf.train.get_or_create_global_step()`
    or
    :meth:`tf.train.get_global_step()` within this scope will return an
    incrementable
    iteration step since the beginning of the iteration.

    Args:
      subnetwork: Newest subnetwork, that is not part of the
        `previous_ensemble`.
      loss: A :class:`tf.Tensor` containing the subnetwork's loss to minimize.
      var_list: List of subnetwork :class:`tf.Variable` parameters to update as
        part of the training operation.
      labels: Labels :class:`tf.Tensor` or a dictionary of string label name to
        :class:`tf.Tensor` (for multi-head).
      iteration_step: Integer :class:`tf.Tensor` representing the step since the
        beginning of the current iteration, as opposed to the global step.
      summary: An :class:`adanet.Summary` for scoping summaries to individual
        subnetworks in Tensorboard. Using `tf.summary` within this scope will
        use this :class:`adanet.Summary` under the hood.
      previous_ensemble: The best `Ensemble` from iteration t-1. The created
        subnetwork will extend the previous ensemble to form the `Ensemble` at
        iteration t. Is None for iteration 0.

    Returns:
      Either a train op or an :class:`adanet.subnetwork.TrainOpSpec`.
    """

  @abc.abstractmethod
  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    # pyformat: disable
    """Returns an op for training the ensemble's mixture weights.

    Allows AdaNet to learn the mixture weights of each subnetwork
    according to Equation (6).

    This method will be called once after `build_subnetwork`.

    Accessing the global step via :meth:`tf.train.get_or_create_global_step()`
    or :meth:`tf.train.get_global_step()` within this scope will return an
    incrementable iteration step since the beginning of the iteration.

    Args:
      loss: A :class:`tf.Tensor` containing the ensemble's loss to minimize.
      var_list: List of ensemble mixture weight `tf.Variables` to update as
        become part of the training operation.
      logits: The ensemble's logits :class:`tf.Tensor` from applying the mixture
        weights and bias to the ensemble's subnetworks.
      labels: Labels :class:`tf.Tensor` or a dictionary of string label name to
        :class:`tf.Tensor` (for multi-head).
      iteration_step: Integer :class:`tf.Tensor` representing the step since the
        beginning of the current iteration, as opposed to the global step.
      summary: An :class:`adanet.Summary` for scoping summaries to individual
        subnetworks in Tensorboard. Using :class:`tf.summary` within this scope
        will use this :class:`adanet.Summary` under the hood.

    Returns:
      Either a train op or an :class:`adanet.subnetwork.TrainOpSpec`.
    """
    # pyformat: enable

  def build_subnetwork_report(self):
    """Returns a `subnetwork.Report` to materialize and record.

    This method will be called once after :meth:`build_subnetwork`.
    Do NOT depend on variables created in :meth:`build_subnetwork_train_op` or
    :meth:`build_mixture_weights_train_op`, because they are not called before
    :meth:`build_subnetwork_report` is called.

    If it returns None, AdaNet records the name and standard eval metrics.
    """

    return None

  def prune_previous_ensemble(self, previous_ensemble):
    """Specifies which subnetworks from the previous ensemble to keep.

    The selected subnetworks from the previous ensemble will be kept in the
    candidate ensemble that includes this subnetwork.

    By default, none of the previous ensemble subnetworks are pruned.

    Args:
      previous_ensemble: :class:`adanet.Ensemble` object.

    Returns:
      List of integer indices of `weighted_subnetworks` to keep.
    """
    return range(len(previous_ensemble.weighted_subnetworks))


class Generator(object):
  """Interface for a candidate subnetwork generator.

  Given the ensemble of subnetworks at iteration t-1, this object is
  responsible for generating the set of candidate subnetworks for iteration t
  that minimize the objective as part of an ensemble.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    # pyformat: disable
    """Generates :class:`adanet.subnetwork.Builder` instances for an iteration.

    NOTE: Every call to :meth:`generate_candidates` must be deterministic for
    the given arguments.

    Args:
      previous_ensemble: The best :class:`adanet.Ensemble` from iteration t-1.
        DEPRECATED. We are transitioning away from the use of previous_ensemble
        in generate_candidates. New Generators should *not* use
        previous_ensemble in their implementation of generate_candidates --
        please only use iteration_number, previous_ensemble_reports and
        all_reports.
      iteration_number: Python integer AdaNet iteration t, starting from 0.
      previous_ensemble_reports: List of
        :class:`adanet.subnetwork.MaterializedReport` instances corresponding to
        the Builders composing :class:`adanet.Ensemble` from iteration t-1. The
        first element in the list corresponds to the Builder added in the
        first iteration. If a :class:`adanet.subnetwork.MaterializedReport` is
        not supplied to the estimator, previous_ensemble_report is `None`.
      all_reports: List of :class:`adanet.subnetwork.MaterializedReport`
        instances. If an :class:`adanet.subnetwork.ReportMaterializer` is not
        supplied to the estimator, `all_reports` is `None`. If
        :class:`adanet.subnetwork.ReportMaterializer` is supplied to the
        estimator and t=0, `all_reports` is an empty List. Otherwise,
        `all_reports` is a sequence of Lists. Each element of the sequence is a
        List containing all the :class:`adanet.subnetwork.MaterializedReport`
        instances in an AdaNet iteration, starting from iteration 0, and
        ending at iteration t-1.

    Returns:
      A list of :class:`adanet.subnetwork.Builder` instances.
    """
    # pyformat: enable


class SimpleGenerator(Generator):
  """Always generates the given :class:`adanet.subnetwork.Builder` instances.

  Args:
    subnetwork_builders: List of :class:`adanet.subnetwork.Builder` instances to
      return at each iteration when `generate_candidates` is called.

  Returns:
    A :class:`adanet.SimpleGenerator` instance.
  """

  def __init__(self, subnetwork_builders):
    self._subnetwork_builders = subnetwork_builders

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    return self._subnetwork_builders
