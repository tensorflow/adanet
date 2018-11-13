"""Tests for AdaNet RunConfig utility functions.

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

import json
import os

from absl.testing import parameterized
from adanet.core.run_config_utils import to_tpu_run_config
import tensorflow as tf

_FAKE_CLUSTER = {
    'chief': ['host0:2222'],
    'ps': ['host1:2222', 'host2:2222'],
    'worker': ['host3:2222', 'host4:2222', 'host5:2222']
}


class RunConfigUtils(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      'testcase_name': 'chief_node',
      'task': {
          'type': 'chief',
          'index': 0,
      }
  }, {
      'testcase_name': 'worker_node',
      'task': {
          'type': 'worker',
          'index': 1
      }
  }, {
      'testcase_name': 'evaluator_node',
      'task': {
          'type': 'evaluator',
          'index': 0
      }
  })
  def test_to_tpu_config(self, task):
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': _FAKE_CLUSTER,
        'task': task
    })
    run_config = tf.estimator.RunConfig(
        model_dir='fake/model/dir',
        tf_random_seed=42,
        save_summary_steps=200,
        save_checkpoints_steps=200,
        session_config=tf.ConfigProto(),
        keep_checkpoint_max=10,
        keep_checkpoint_every_n_hours=15,
        log_step_count_steps=15,
        train_distribute=tf.contrib.distribute.DistributionStrategy(),
        device_fn=lambda op: None,
        eval_distribute=tf.contrib.distribute.DistributionStrategy())

    tpu_run_config = to_tpu_run_config(run_config)

    # Remove TPU specific vars before checking variable equality.
    tpu_run_config_vars = vars(tpu_run_config)
    del tpu_run_config_vars['_tpu_config']
    del tpu_run_config_vars['_cluster']
    self.assertEqual(vars(run_config), tpu_run_config_vars)

  def test_to_tpu_config_run_config_is_none(self):
    tpu_run_config = to_tpu_run_config(config=None)
    self.assertEqual(vars(tpu_run_config), vars(tf.contrib.tpu.RunConfig()))


if __name__ == '__main__':
  tf.test.main()
