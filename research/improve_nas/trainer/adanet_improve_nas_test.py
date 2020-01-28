# Lint as: python3
"""Tests for improve_nas.

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

import os

from absl import flags
from absl.testing import parameterized
from adanet.research.improve_nas.trainer import adanet_improve_nas
from adanet.research.improve_nas.trainer import fake_data
import tensorflow.compat.v1 as tf


class AdaNetQuetzalBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "simple_generator",
      "hparams_string": ("optimizer=sgd,boosting_iterations=2,generator=simple,"
                         "initial_learning_rate=.1,use_aux_head=False,"
                         "num_cells=3,num_conv_filters=2,use_evaluator=False"),
  }, {
      "testcase_name": "dynamic_generator",
      "hparams_string":
          ("optimizer=sgd,boosting_iterations=1,generator=dynamic,"
           "initial_learning_rate=.1,use_aux_head=False,"
           "num_cells=3,num_conv_filters=2,use_evaluator=False"),
  })
  def test_estimator(self,
                     hparams_string,
                     batch_size=1):
    """Structural test to make sure Estimator Builder works."""

    seed = 42

    # Set up and clean test directory.
    model_dir = os.path.join(flags.FLAGS.test_tmpdir,
                             "AdanetImproveNasBuilderTest")
    if tf.gfile.Exists(model_dir):
      tf.gfile.DeleteRecursively(model_dir)
    tf.gfile.MkDir(model_dir)

    data_provider = fake_data.FakeImageProvider(seed=seed)
    estimator_builder = adanet_improve_nas.Builder()
    hparams = estimator_builder.hparams(
        default_batch_size=3, hparams_string=hparams_string)
    run_config = tf.estimator.RunConfig(
        tf_random_seed=seed, model_dir=model_dir)
    _ = data_provider.get_input_fn(
        "train",
        tf.estimator.ModeKeys.TRAIN,
        batch_size=batch_size)
    test_input_fn = data_provider.get_input_fn(
        "test",
        tf.estimator.ModeKeys.EVAL,
        batch_size=batch_size)

    estimator = estimator_builder.estimator(
        data_provider=data_provider,
        run_config=run_config,
        hparams=hparams,
        train_steps=10,
        seed=seed)
    eval_metrics = estimator.evaluate(input_fn=test_input_fn, steps=1)

    self.assertGreater(eval_metrics["loss"], 0.0)


if __name__ == "__main__":
  tf.test.main()
