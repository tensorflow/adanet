"""AdaNet RunConfig utility functions.

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


# TODO: raise an exception if use_tpu is True and we are passed a
# `tf.estimator.RunConfig`.
def to_tpu_run_config(config):
  """Creates a `tf.contrib.tpu.RunConfig` from a `tf.estimator.RunConfig`."""
  config = config if config else tf.contrib.tpu.RunConfig()
  if not isinstance(config, tf.contrib.tpu.RunConfig):
    # Remove the head of the args list since this is `self`.
    args = inspect.getargspec(tf.estimator.RunConfig.__init__).args[1:]
    kwargs = {
        arg: getattr(config, "_" + arg)
        for arg in args
        if hasattr(config, "_" + arg)
    }
    # tpu.RunConfig defaults evaluation_master=master if it is not explicitly
    # set. However, this breaks checks that evaluation_master matches the
    # TF_CONFIG environment variable. To avoid this, we explicitly set
    # evaluation_master.
    config = tf.contrib.tpu.RunConfig(
        evaluation_master=config.evaluation_master, **kwargs)
  return config
