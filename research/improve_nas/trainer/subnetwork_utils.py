# Lint as: python3
"""Definition of helpful functions to work with AdaNet subnetworks.

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

import copy
import tensorflow as tf


def capture_variables(fn):
  """Utility function that captures which tf variables were created by `fn`.

  This function encourages style that is easy to write, resonably easy to
  understand but against google codestyle.

  In general, you have an function `f` that takes some arguments (`a` and `b`)
  and returns some output. You may enclose it in lambda and get
  `fn == lambda: f(a,b)`, which is a function without arguments that does the
  same as `f`.

  This idiom makes variable management much easier and less error prone. Usable
  for prototyping or debugging.

  Args:
    fn: function with no arguments.

  Returns:
    tuple: First element of this touple is a list of tf variables created by
        fn, second is the actual output of fn

  """
  vars_before_fn = tf.trainable_variables()
  fn_return = fn()
  vars_after_fn = tf.trainable_variables()
  fn_vars = list(set(vars_after_fn) - set(vars_before_fn))
  return set(fn_vars), fn_return


def copy_update(hparams, **kwargs):
  """Deep copy hparams with values updated by kwargs.

  This enables to use hparams in an immutable manner.
  Args:
    hparams: hyperparameters.
    **kwargs: keyword arguments to change in hparams.

  Returns:
    updated hyperparameters object. Change in this object is not propagated to
    the original hparams
  """
  values = hparams.values()
  values.update(kwargs)
  values = copy.deepcopy(values)
  hp = tf.contrib.training.HParams(**values)
  return hp


def get_persisted_value_from_ensemble(ensemble, key):
  """Return constant persisted tensor values from the previous subnetwork.

  Args:
    ensemble: Previous ensemble.
    key: Name of constant to get from eprsisted tensor.

  Returns:
    int|float value of the constant.
  """
  previous_subnetwork = ensemble.weighted_subnetworks[-1].subnetwork
  persisted_tensor = previous_subnetwork.shared[key]
  return persisted_tensor
