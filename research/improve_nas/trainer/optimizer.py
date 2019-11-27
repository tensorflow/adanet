# Lint as: python3
"""Definition of optimizers and learning rate schedules.

Copyright 2019 The AdaNet Authors. All Rights Reserved.

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
import functools

import tensorflow as tf


class LearningRateSchedule(object):
  """A learning rate decay schedule interface."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def apply(self, learning_rate):
    """Applies the learning rate decay schedule to the given learning rate.

    Args:
      learning_rate: Float `Tensor` learning rate.

    Returns:
      Float `Tensor` learning rate with applied decay schedule.
    """


class Constant(LearningRateSchedule):
  """A constant schedule."""

  def apply(self, learning_rate):
    """See `LearningRateSchedule`."""

    return learning_rate


class Cosine(LearningRateSchedule):
  """Cosine."""

  def __init__(self, decay_steps, alpha):
    """Returns a `Cosine` instance.

    Args:
      decay_steps: Number of steps to decay over.
      alpha: Minimum learning rate value as a fraction of learning_rate.

    Returns:
      A `Cosine` instance.
    """

    self._decay_fn = functools.partial(
        tf.train.cosine_decay, decay_steps=decay_steps, alpha=alpha)

  def apply(self, learning_rate):
    """See `LearningRateSchedule`."""

    # Start at -1 since we increment before reading.
    global_step = tf.get_variable("decay_step", initializer=-1, trainable=False)
    increment_op = tf.assign_add(global_step, 1)
    with tf.control_dependencies([increment_op]):
      learning_rate = self._decay_fn(
          learning_rate=learning_rate, global_step=global_step.read_value())
    return learning_rate


def fn_with_name(optimizer_name,
                 learning_rate_schedule="constant",
                 cosine_decay_steps=None):
  """Returns an optimizer_fn with the given name.

  Args:
    optimizer_name: Optimizer name string for identifying the optimizer. Either
      'adagrad', 'adam', 'momentum', or 'sgd'.
    learning_rate_schedule: Type of learning rate schedule to use. Opened for
      future extensions.
    cosine_decay_steps: See `Cosine`.

  Returns:
    An optimizer_fn which takes a `learning_rate` scalar `Tensor` argument and
      returns an `Optimizer` instance.

  Raises:
    ValueError: If `optimizer_name` is invalid.
  """

  optimizers = {
      "adagrad": tf.train.AdagradOptimizer,
      "adam": tf.train.AdamOptimizer,
      "lazy_adam": tf.contrib.opt.LazyAdamOptimizer,
      "momentum": functools.partial(tf.train.MomentumOptimizer, momentum=.9),
      "rmsprop": tf.train.RMSPropOptimizer,
      "sgd": tf.train.GradientDescentOptimizer,
  }
  optimizer_name = optimizer_name.lower()
  if optimizer_name not in optimizers:
    raise ValueError("Invalid optimizer '{}'".format(optimizer_name))
  optimizer_fn = optimizers[optimizer_name]
  schedules = {
      "constant":
          Constant(),
      "cosine":
          Cosine(decay_steps=cosine_decay_steps, alpha=0.0),
  }
  schedule_name = learning_rate_schedule.lower()
  if schedule_name not in schedules:
    raise ValueError(
        "Invalid learning_rate_schedule '{}'".format(schedule_name))
  schedule = schedules[schedule_name]

  def _optimizer_with_schedule(learning_rate):
    learning_rate = schedule.apply(learning_rate)
    optimizer = optimizer_fn(learning_rate)
    return optimizer, learning_rate
  return _optimizer_with_schedule
