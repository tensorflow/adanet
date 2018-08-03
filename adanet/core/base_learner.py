"""An AdaNet base learner definition in Tensorflow using a single graph.

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


class BaseLearner(
    collections.namedtuple(
        "BaseLearner",
        ["last_layer", "logits", "complexity", "persisted_tensors"])):
  """An AdaNet base learner.

  In the AdaNet paper, a `adanet.BaseLearner` is are called a 'subnetwork',
  and indicated by 'h'. A collection of weighted base learners form an AdaNet
  ensemble.
  """

  def __new__(cls, last_layer, logits, complexity, persisted_tensors):
    """Creates a validated `BaseLearner` instance.

    Args:
      last_layer: A `Tensor` output of the last layer of the base learner, i.e
        the layer before the logits layer. When the mixture weight type is
        `MATRIX`, the AdaNet algorithm takes care of computing ensemble mixture
        weights matrices (one per base learner) that multiply the various
        last layers of the ensemble's base learners, and regularize them using
        their base learner's complexity. This field is represented by 'h' in the
        AdaNet paper.
      logits: logits `Tensor` for training the base learner. NOTE:
        These logits are not used in the ensemble's outputs if the mixture
        weight type is `MATRIX`, instead AdaNet learns its own logits
        (mixture weights) from the base learner's `last_layers` with complexity
        regularization. The logits are used in the ensemble only when the
        mixture weights type is `SCALAR` or `VECTOR`.
        Even though the logits are not used in the ensemble in some cases, they
        should always be supplied as adanet uses the logits to train the base
        learners.
      complexity: A scalar representing the complexity of the base learner's
        architecture. It is used for choosing the best base learner at each
        iteration, and for regularizing the weighted outputs of more complex
        base learners.
      persisted_tensors: Nested dictionary of string to `Tensor` to persist
        across iterations. At the end of an iteration, the `Tensors` will be
        available to base learners in the next iterations, whereas others that
        are not part of the `BaseLearner` will be pruned. This allows later
        `BaseLearners` to dynamically build upon arbitrary `Tensors` from
        previous `BaseLearners`.

    Returns:
      A validated `BaseLearner` object.

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
    return super(BaseLearner, cls).__new__(
        cls,
        last_layer=last_layer,
        logits=logits,
        complexity=complexity,
        persisted_tensors=persisted_tensors)


class BaseLearnerBuilder(object):
  """Interface for a base learner builder.

  Given features, labels, and the best ensemble of base learners at iteration
  t-1, a `BaseLearnerBuilder` creates a `BaseLearner` to add to a candidate
  ensemble at iteration t. These candidate ensembles are evaluated against one
  another at the end of the iteration, and the best one is selected based on its
  complexity-regularized loss.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """Returns the unique name of the ensemble to contain this base learner."""

  @abc.abstractmethod
  def build_base_learner(self,
                         features,
                         logits_dimension,
                         training,
                         iteration_step,
                         summary,
                         previous_ensemble=None):
    """Returns the candidate `BaseLearner` to add to the ensemble.

    This method will be called only once, before `build_base_learner_train_op`
    and `build_mixture_weights_train_op` are called. This method should
    construct the candidate base learner's graph operations and variables.

    Args:
      features: Input `dict` of `Tensor` objects.
      logits_dimension: Size of the last dimension of the logits `Tensor`.
        Typically, logits have for shape `[batch_size, logits_dimension]`.
      training: A python boolean indicating whether the graph is in training
        mode or prediction mode.
      iteration_step: Integer `Tensor` representing the step since the
        beginning of the current iteration, as opposed to the global step.
      summary: An `adanet.Summary` for scoping summaries to individual base
        learners in Tensorboard.
      previous_ensemble: The best `Ensemble` from iteration t-1. The created
        base learner will extend the previous ensemble to form the `Ensemble`
        at iteration t.

    Returns:
      A `BaseLearner` instance.
    """

  @abc.abstractmethod
  def build_base_learner_train_op(self, loss, var_list, labels, iteration_step,
                                  summary):
    """Returns an op for training a new base learner.

    This method will be called once after `build_base_learner`.

    NOTE: This method should _not_ increment the global step tensor.

    Args:
      loss: A `Tensor` containing the base learner's loss to minimize.
      var_list: List of base learner `tf.Variable` parameters to update as
        part of the training operation.
      labels: Labels `Tensor`.
      iteration_step: Integer `Tensor` representing the step since the
        beginning of the current iteration, as opposed to the global step.
      summary: An `adanet.Summary` for scoping summaries to individual base
        learners in Tensorboard.

    Returns:
      A train op.
    """

  @abc.abstractmethod
  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    """Returns an op for training the ensemble's mixture weights.

    Allows AdaNet to learn the mixture weights of each base learner
    according to Equation (6).

    This method will be called once after `build_base_learner`.

    NOTE: This method should _not_ increment the global step tensor.

    Args:
      loss: A `Tensor` containing the ensemble's loss to minimize.
      var_list: List of ensemble mixture weight `tf.Variables` to update as
        become part of the training operation.
      logits: The ensemble's logits `Tensor` from applying the mixture weights
        and bias to the ensemble's base learners.
      labels: Labels `Tensor`.
      iteration_step: Integer `Tensor` representing the step since the
        beginning of the current iteration, as opposed to the global step.
      summary: An `adanet.Summary` for scoping summaries to individual base
        learners in Tensorboard.

    Returns:
      A train op.
    """

  def build_base_learner_report(self):
    """Returns a `BaseLearnerReport` to materialize and record.

    This method will be called once after `build_base_learner`.
    Do NOT depend on variables created in `build_base_learner_train_op` or
    `build_mixture_weights_train_op`, because they are not called before
    `build_base_learner_report` is called.

    If it returns None, AdaNet records the name and standard eval metrics.
    """

    return None


class BaseLearnerBuilderGenerator(object):
  """Interface for a base learner builder generator.

  Given the ensemble of base learners at iteration t-1, this object is
  responsible for generating the set of base learners for iteration t that
  minimize Equation (6) in the paper.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def generate_candidates(self, previous_ensemble):
    """Generates `adanet.BaseLearnerBuilders` to train at iteration t.

    Args:
      previous_ensemble: The best `adanet.Ensemble` from iteration t-1.

    Returns:
      A list of `adanet.BaseLearnerBuilders`.
    """


class SimpleBaseLearnerBuilderGenerator(BaseLearnerBuilderGenerator):
  """A generator that always returns the same `adanet.BaseLearnerBuilders`."""

  def __init__(self, base_learner_builders):
    """Creates a `adanet.BaseLearner` instance.

    Args:
      base_learner_builders: List of `adanet.BaseLearnerBuilders` to return at
        each iteration when `generate_candidates` is called.

    Returns:
      A `SimpleBaseLearnerBuilderGenerator` instance.
    """

    self._base_learner_builders = base_learner_builders

  def generate_candidates(self, previous_ensemble):
    """Returns the predefined set of `adanet.BaseLearnerBuilders`."""

    return self._base_learner_builders
