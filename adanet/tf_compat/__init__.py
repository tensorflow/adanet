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
  TPUEstimatorSpec = tf.contrib.tpu.TPUEstimatorSpec
except AttributeError:
  TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec


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
  except:
    tf_version = tf.VERSION
  return LooseVersion(tf_version) >= LooseVersion(semver)
