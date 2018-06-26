"""Container for an adanet `BaseLearner`'s attributes and metrics.

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

import collections

import six
import tensorflow as tf


class BaseLearnerReport(
    collections.namedtuple("BaseLearnerReport",
                           ["hparams", "attributes", "metrics"])):
  """A container for data to be collected about a `BaseLearner`."""

  def __new__(cls, hparams, attributes, metrics):
    """Creates a validated `BaseLearnerReport` instance.

    Args:
      hparams: A dict mapping strings to python strings, ints, or floats.
        It is meant to contain the constants that define the
        `BaseLearnerBuilder`, such as dropout, number of layers, or initial
        learning rate.
      attributes: A dict mapping strings to rank 0 Tensors of dtype string,
        int32, or float32. It is meant to contain properties that may or may
        not change over the course of training the `BaseLearner`, such as the
        number of parameters, the Lipschitz constant, the L_2 norm of the
        weights, or learning rate at materialization time.
      metrics: Dict of metric results keyed by name. The values of the
        dict are the results of calling a metric function, namely a
        `(metric_tensor, update_op)` tuple. `metric_tensor` should be evaluated
        without any impact on state (typically is a pure computation results
        based on variables.). For example, it should not trigger the `update_op`
        or requires any input fetching.
        This is meant to contain metrics of interest, such as the training loss,
        complexity regularized loss, or standard deviation of the last layer
        outputs.

    Returns:
      A validated `BaseLearnerReport` object.

    Raises:
      ValueError: If validation fails.
    """

    def _is_valid_tensor(tensor):
      """Returns True only if tensor is a scalar of the right type."""

      if tensor.shape.ndims > 0:
        return False
      return tensor.dtype in (tf.int32, tf.float32, tf.string)

    # Validate hparams
    for key, value in hparams.items():
      if not (isinstance(value, int) or isinstance(value, float) or
              isinstance(value, six.string_types)):
        raise ValueError(
            "hparam '{}' refers to invalid value {}, type {}. type must be "
            "python primitive int, float, or string.".format(
                key, value, type(value)))

    # Validate attributes
    for key, value in attributes.items():
      if not isinstance(value, tf.Tensor):
        raise ValueError("attribute '{}' refers to invalid value: {}, type: {}."
                         "type must be Tensor.".format(key, value, type(value)))

      if not _is_valid_tensor(value):
        raise ValueError(
            "attribute '{}' refers to invalid tensor {}. Shape: {}".format(
                key, value, value.get_shape()))

    # Validate metrics
    for key, value in metrics.items():
      if not isinstance(value, tuple):
        raise ValueError(
            "metric '{}' has invalid type {}. Must be a tuple.".format(
                key, type(value)))

      if len(value) < 2:
        raise ValueError(
            "metric tuple '{}' has fewer than 2 elements".format(key))

      if not isinstance(value[1], tf.Tensor):
        raise ValueError(
            "second element of metric tuple '{}' has value {} and type {}. "
            "Must be a Tensor.".format(key, value, type(value[1])))

      if not _is_valid_tensor(value[1]):
        raise ValueError(
            "second element of metric '{}' refers to invalid tensor {}".format(
                key, value[1]))

    return super(BaseLearnerReport, cls).__new__(
        cls, hparams=hparams, attributes=attributes, metrics=metrics)


class MaterializedBaseLearnerReport(
    collections.namedtuple(
        "MaterializedBaseLearnerReport",
        ["hparams", "attributes", "metrics", "included_in_final_ensemble"])):
  """A container for data collected about a `BaseLearner`."""

  def __new__(cls, hparams, attributes, metrics, included_in_final_ensemble):
    """Creates a validated `MaterializedBaseLearnerReport` instance.

    Args:
      hparams: A dict mapping strings to python strings, ints, or floats.
        These are constants passed from the author of the `BaseLearnerBuilder`
        that was used to construct this `BaseLearner`. It is meant to contain
        the arguments that defined the `BaseLearnerBuilder`, such as dropout,
        number of layers, or initial learning rate.
      attributes: A dict mapping strings to python strings, ints, or floats.
        These are python primitives that come from materialized Tensors; these
        Tensors were defined by the author of the `BaseLearnerBuilder` that was
        used to construct this `BaseLearner`. It is meant to contain properties
        that may or may not change over the course of training the
        `BaseLearner`, such as the number of parameters, the Lipschitz constant,
        or the L_2 norm of the weights.
      metrics: A dict mapping strings to python strings, ints, or floats.
        These are python primitives that come from metrics that were evaluated
        on the trained `BaseLearner` over some dataset; these metrics were
        defined by the author of the `BaseLearnerBuilder` that was used to
        construct this `BaseLearner`. It is meant to contain performance metrics
        or measures that could predict generalization, such as the training
        loss, complexity regularized loss, or standard deviation of the last
        layer outputs.
      included_in_final_ensemble: A boolean denoting whether the associated
        `BaseLearner` was included in the ensemble at the end of the AdaNet
        iteration.

    Returns:
      A `MaterializedBaseLearnerReport` object.
    """

    return super(MaterializedBaseLearnerReport, cls).__new__(
        cls,
        hparams=hparams,
        attributes=attributes,
        metrics=metrics,
        included_in_final_ensemble=included_in_final_ensemble)
