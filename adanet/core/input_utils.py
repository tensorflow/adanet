"""AdaNet input utility functions.

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

import inspect

import tensorflow as tf


def _make_any_batch_size_placeholder(name, shape, dtype, is_sparse):
  """Makes a placeholder that can accept any batch size."""

  modifiable_shape = [None]
  if shape:
    modifiable_shape += shape[1:]
  if is_sparse:
    return tf.sparse_placeholder(shape=modifiable_shape, dtype=dtype, name=name)
  return tf.placeholder(shape=modifiable_shape, dtype=dtype, name=name)


def make_placeholder_input_fn(input_fn):
  """Makes a placeholder input_fn which mimics the given input_fn.

  Requires an input function `input_fn` that returns a tuple of:

  * features: Dictionary of string feature name to `Tensor`.
  * labels: `Tensor` of labels or Dictionary of string label name to `Tensor`
  (for multi-task learning problems).

  The returned input_fn, when called, returns feature and label `Tensor`
  placeholders which have the same shape, name, and type as those returned
  by `input_fn`, but which can accept any batch size. This prevents an
  exported graph from setting a fixed batch size, so that any batch size
  can be used once imported.

  Args:
    input_fn: The input function.

  Returns:
    An input_fn that mimics `input_fn` with placeholder features and labels.
  """

  # Create temporary graph to collect `Tensor` information.
  with tf.Graph().as_default():
    # We ignore batch sizes for place_holder_input_fns so there is no need to
    # pass a `params` object to the input_fn.
    features, labels = input_fn({})
    feature_info = {}
    for name, feature in features.items():
      feature_info[name] = (feature.op.name, feature.get_shape().as_list(),
                            feature.dtype, isinstance(feature, tf.SparseTensor))
    if isinstance(labels, dict):
      label_info = {
          name: (tensor.op.name, tensor.get_shape().as_list(), tensor.dtype)
          for name, tensor in labels.items()
      }
    else:
      label_info = (labels.op.name, labels.get_shape().as_list(), labels.dtype)

  def _placeholder_input_fn():
    """Returns (features, labels) without fixed batch_size dimensions."""

    features = {}
    for name, info in feature_info.items():
      feature_op_name, feature_shape, feature_dtype, is_sparse = info
      features[name] = _make_any_batch_size_placeholder(
          feature_op_name, feature_shape, feature_dtype, is_sparse)
    if isinstance(label_info, dict):
      labels = {}
      for name, single_label_info in label_info.items():
        label_op_name, label_shape, label_dtype = single_label_info
        labels[name] = _make_any_batch_size_placeholder(
            label_op_name, label_shape, label_dtype, is_sparse=False)
    else:
      label_op_name, label_shape, label_dtype = label_info
      labels = _make_any_batch_size_placeholder(
          label_op_name, label_shape, label_dtype, is_sparse=False)
    return features, labels

  return _placeholder_input_fn


def wrap_input_fn(input_fn, use_tpu):
  """Wraps the input_fn with one which takes a `params` argument.

  This allows input_fns defined for `Estimator` to also work for `TPUEstimator`
  when not using TPUs.

  Args:
    input_fn: The input_fn to wrap. If running on TPUs or input_fn already takes
      a params argument, the original input_fn is returned.
    use_tpu: Whether TPUs are being used.

  Returns:
    An input_fn which takes a `params` argument.
  """
  has_params = "params" in inspect.getargspec(input_fn).args
  if has_params or use_tpu:
    return input_fn
  return lambda params: input_fn()
