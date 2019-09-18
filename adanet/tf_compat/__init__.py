# Copyright 2018 The AdaNet Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorFlow major version compatibility code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion
import tensorflow as tf
import tensorflow as tf_v2
# pylint: disable=unused-import
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=g-import-not-at-top
try:
  from tensorflow.python.tpu import tpu_function
except ImportError:
  from tensorflow.contrib.tpu.python.tpu import tpu_function
try:
  from tensorflow.python.keras.metrics import Metric
except ImportError:
  # When Metric is unavailable (TF < 1.13), we need to define Metric so that
  # we don't raise an exception when defining a custom metric for TF >= 1.13
  # workflows.
  Metric = object

try:
  from tensorflow.python.data.ops.dataset_ops import DatasetV1
except AttributeError:
  DatasetV1 = tf.data.Dataset
try:
  from tensorflow.python.data.ops.dataset_ops import DatasetV2
except AttributeError:
  DatasetV2 = tf.data.Dataset

from tensorflow.python import tf2
from tensorflow_estimator.python.estimator.head import regression_head
# pylint: enable=g-import-not-at-top
# pylint: enable=g-direct-tensorflow-import
# pylint: enable=unused-import

try:
  v1 = tf.compat.v1
except AttributeError:
  v1 = tf

try:
  v2 = tf.compat.v2
except AttributeError:
  v2 = tf.contrib

try:
  SessionRunHook = tf.estimator.SessionRunHook
except AttributeError:
  SessionRunHook = tf.train.SessionRunHook

try:
  SessionRunArgs = tf.estimator.SessionRunArgs
except AttributeError:
  SessionRunArgs = tf.train.SessionRunArgs

try:
  SummarySaverHook = tf.estimator.SummarySaverHook
except AttributeError:
  SummarySaverHook = tf.train.SummarySaverHook

try:
  CheckpointSaverHook = tf.estimator.CheckpointSaverHook
except AttributeError:
  CheckpointSaverHook = tf.train.CheckpointSaverHook

try:
  TPUEstimatorSpec = tf.contrib.tpu.TPUEstimatorSpec
except AttributeError:
  TPUEstimatorSpec = object

# TODO: Figure out TPU strategy for TF 2.0.
try:
  TPUEstimator = tf.contrib.tpu.TPUEstimator
except AttributeError:
  TPUEstimator = object

try:
  # Loss reduction strings change between TF 1.13 and TF 1.14, which causes
  # Heads to raise errors.
  regression_head.RegressionHead(
      loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
  SUM_OVER_BATCH_SIZE = tf.losses.Reduction.SUM_OVER_BATCH_SIZE
  SUM = tf.losses.Reduction.SUM
except ValueError:
  SUM_OVER_BATCH_SIZE = "sum_over_batch_size"
  SUM = "sum"


def tensor_name(tensor):
  """Returns the Tensor's name.

  Tensor names always have the structure <op_name>:<int>. This method
  returns the portion before the ':'.

  Args:
    tensor: Tensor.

  Returns:
    String name of the Tensor.
  """

  return tensor.name.split(":")[-2]


def version_greater_or_equal(semver):
  """Returns whether the current TF version is >= to semver string."""

  try:
    tf_version = tf.version.VERSION
  except AttributeError:
    tf_version = tf.VERSION
  return LooseVersion(tf_version) >= LooseVersion(semver)


def make_one_shot_iterator(dataset):
  """Returns a dataset's one-shot iterator."""

  try:
    return v1.data.make_one_shot_iterator(dataset)
  except AttributeError:
    return dataset.make_one_shot_iterator()


def random_normal(*args, **kwargs):
  """Returns a random normal distribution Tensor."""

  try:
    return tf.random.normal(*args, **kwargs)
  except AttributeError:
    return tf.random_normal(*args, **kwargs)


def metric_op(metric):
  """Converts Keras metrics into a metric op tuple.

  NOTE: If this method is called in for loop, the runtime is O(n^2). However
  the number of eval metrics at any given time should be small enough that
  this does not affect performance. Any impact is only during graph construction
  time, and therefore has no effect on steps/s.

  Args:
    metric: Either a `tf.keras.metric.Metric` instance or a tuple of Tensor
      value and update op.

  Returns:
    A tuple of metric Tensor value and update op.
  """

  if not isinstance(metric, tf.keras.metrics.Metric):
    return metric
  vars_to_add = set()
  vars_to_add.update(metric.variables)
  metric = (metric.result(), metric.updates[0])
  _update_variable_collection(tf.GraphKeys.LOCAL_VARIABLES, vars_to_add)
  _update_variable_collection(tf.GraphKeys.METRIC_VARIABLES, vars_to_add)
  return metric


def _update_variable_collection(collection_name, vars_to_add):
  """Add variables to collection."""
  collection = set(tf.get_collection(collection_name))
  # Skip variables that are in the collection already: O(n) runtime.
  vars_to_add = vars_to_add - collection
  for v in vars_to_add:
    tf.add_to_collection(collection_name, v)


def skip_for_tf2(f):
  """Decorator that skips tests when using TensorFlow 2."""

  def wrapper(*args, **kwargs):
    """Wraps the decorated function to determine whether to skip."""

    # Extract test case instance from args.
    self = args[0]
    try:
      # If tf.contrib doesn't exist, we are in TF 2.0.
      _ = tf.contrib
    except AttributeError:
      self.skipTest("Skipping test in TF 2.0.")
    return f(*args, **kwargs)

  return wrapper


def skip_for_tf1(f):
  """Decorator that skips tests when using TensorFlow 1."""

  def wrapper(*args, **kwargs):
    """Wraps the decorated function to determine whether to skip."""

    # Extract test case instance from args.
    self = args[0]
    try:
      # If tf.contrib doesn't exist, we are in TF 2.0.
      _ = tf_v2.contrib
    except AttributeError:
      return f(*args, **kwargs)
    self.skipTest("Skipping test in TF 1.0.")
    return f(*args, **kwargs)

  return wrapper


def is_v2_behavior_enabled():
  """Returns if user called tf.enable_v2_behavior."""

  # Since there is no actual tf.is_v2_behavior enabled, check that the
  # settings were enabled.
  return tf2.enabled()
