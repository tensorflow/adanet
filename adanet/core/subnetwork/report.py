"""Container for an `adanet.Subnetwork`'s attributes and metrics.

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


class Report(
    collections.namedtuple("Report", ["hparams", "attributes", "metrics"])):
  """A container for data to be collected about a `Subnetwork`."""

  def __new__(cls, hparams, attributes, metrics):
    """Creates a validated `Report` instance.

    Args:
      hparams: A dict mapping strings to python strings, ints, bools, or floats.
        It is meant to contain the constants that define the `Builder`, such as
        dropout, number of layers, or initial learning rate.
      attributes: A dict mapping strings to rank 0 Tensors of dtype string,
        int32, or float32. It is meant to contain properties that may or may not
        change over the course of training the `Subnetwork`, such as the number
        of parameters, the Lipschitz constant, the L_2 norm of the weights, or
        learning rate at materialization time.
      metrics: Dict of metric results keyed by name. The values of the dict are
        the results of calling a metric function, namely a `(metric_tensor,
        update_op)` tuple. `metric_tensor` should be evaluated without any
        impact on state (typically is a pure computation results based on
        variables.). For example, it should not trigger the `update_op` or
        requires any input fetching. This is meant to contain metrics of
        interest, such as the training loss, complexity regularized loss, or
        standard deviation of the last layer outputs.

    Returns:
      A validated `Report` object.

    Raises:
      ValueError: If validation fails.
    """

    def _is_valid_tensor(tensor):
      """Returns True only if tensor is a scalar of the right type."""

      if tensor.shape.ndims > 0:
        return False
      return tensor.dtype in (tf.bool, tf.int32, tf.float32, tf.string)

    # Validate hparams
    for key, value in hparams.items():
      if not isinstance(value, (bool, int, float, six.string_types)):
        raise ValueError(
            "hparam '{}' refers to invalid value {}, type {}. type must be "
            "python primitive int, float, bool, or string.".format(
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

      if not isinstance(value[0], tf.Tensor):
        raise ValueError(
            "First element of metric tuple '{}' has value {} and type {}. "
            "Must be a Tensor.".format(key, value, type(value[0])))

      if not _is_valid_tensor(value[0]):
        raise ValueError(
            "First element of metric '{}' refers to invalid Tensor {}.".format(
                key, value[0]))

      if not (isinstance(value[1], tf.Tensor) or
              isinstance(value[1], tf.Operation)):
        raise ValueError(
            "Second element of metric tuple '{}' has value {} and type {}. "
            "Must be a Tensor or Operation.".format(key, value, type(value[1])))

    return super(Report, cls).__new__(
        cls, hparams=hparams, attributes=attributes, metrics=metrics)


class MaterializedReport(
    collections.namedtuple("MaterializedReport", [
        "iteration_number", "name", "hparams", "attributes", "metrics",
        "included_in_final_ensemble"
    ])):
  """A container for data collected about a `Subnetwork`."""

  def __new__(cls,
              iteration_number,
              name,
              hparams,
              attributes,
              metrics,
              included_in_final_ensemble=False):
    """Creates a validated `MaterializedReport` instance.

    Args:
      iteration_number: A python integer for the AdaNet iteration number,
        starting from 0.
      name: A string, which is either the name of the corresponding Builder, or
        "previous_ensemble" if it refers to the previous_ensemble.
      hparams: A dict mapping strings to python strings, ints, or floats. These
        are constants passed from the author of the `Builder` that was used to
        construct this `Subnetwork`. It is meant to contain the arguments that
        defined the `Builder`, such as dropout, number of layers, or initial
        learning rate.
      attributes: A dict mapping strings to python strings, ints, bools, or
        floats. These are python primitives that come from materialized Tensors;
        these Tensors were defined by the author of the `Builder` that was used
        to construct this `Subnetwork`. It is meant to contain properties that
        may or may not change over the course of training the `Subnetwork`, such
        as the number of parameters, the Lipschitz constant, or the L_2 norm of
        the weights.
      metrics: A dict mapping strings to python strings, ints, or floats. These
        are python primitives that come from metrics that were evaluated on the
        trained `Subnetwork` over some dataset; these metrics were defined by
        the author of the `Builder` that was used to construct this
        `Subnetwork`. It is meant to contain performance metrics or measures
        that could predict generalization, such as the training loss, complexity
        regularized loss, or standard deviation of the last layer outputs.
      included_in_final_ensemble: A boolean denoting whether the associated
        `Subnetwork` was included in the ensemble at the end of the AdaNet
        iteration.

    Returns:
      A `MaterializedReport` object.
    """

    return super(MaterializedReport, cls).__new__(
        cls,
        iteration_number=iteration_number,
        name=name,
        hparams=hparams,
        attributes=attributes,
        metrics=metrics,
        included_in_final_ensemble=included_in_final_ensemble)
