# Lint as: python3
"""Script to any experiment from paper.

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

from absl import flags

import tensorflow.compat.v1 as tf

# pylint: disable=g-import-not-at-top
try:
  from adanet.research.improve_nas.trainer import adanet_improve_nas
  from adanet.research.improve_nas.trainer import cifar10
  from adanet.research.improve_nas.trainer import cifar100
  from adanet.research.improve_nas.trainer import fake_data
  print("Imported from adanet.")
except ImportError as e:
  from trainer import adanet_improve_nas
  from trainer import cifar10
  from trainer import cifar100
  from trainer import fake_data
  print("Imported from trainer.")
# pylint: enable=g-import-not-at-top


FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 32,
                     "Batch size used for training, eval and inference.")
flags.DEFINE_integer("train_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer("save_summary_steps", 2000,
                     "Save summaries every this many steps.")
flags.DEFINE_string(
    "hparams", "",
    """A comma-separated list of `name=value` hyperparameter values.""")
flags.DEFINE_string(
    "dataset", "",
    "Dataset name: 'cifar10', 'cifar100' or 'fake'. 'fake' dataset is mainly "
    "for test runs.")
flags.DEFINE_integer("tf_random_seed", None,
                     "Graph level random seed for TensorFlow.")
flags.DEFINE_integer("eval_steps", None,
                     "Number of batches used for evaluation. If `None`, the "
                     "whole eval dataset is used")
flags.DEFINE_integer(
    "save_checkpoints_secs", 600, "Number of seconds between checkpoint saves. "
    "This flag is ignored when autotune is used. "
    "Cannot be used with save_checkpoints_steps -- exactly one of "
    "save_checkpoints_secs and save_checkpoints_steps must be zero, and the "
    "other must be a strictly positive integer. Defaults to 120s.")
flags.DEFINE_integer(
    "save_checkpoints_steps", 0,
    "Number of global steps between checkpoint saves."
    "This flag is ignored when autotune is used. "
    "Cannot be used with save_checkpoints_secs -- exactly one of "
    "save_checkpoints_secs and save_checkpoints_steps must be zero, and the "
    "other must be a strictly positive integer. Defaults to 0, which means "
    "save_checkpoints_steps is ignored. To use save_checkpoints_steps "
    "instead, set save_checkpoints_secs to 0 and set save_checkpoints_steps "
    "to a positive integer.")
flags.DEFINE_string(
    "data_params", "",
    """A comma-separated list of `name=value` data provider parameter values.
    This flag is used to override data provider default settings for
    preprocessing or selecting different configurations for a given data
    provider.""")
flags.DEFINE_integer(
    "keep_checkpoint_max", 5,
    "The maximum number of recent checkpoint files to keep. As new files are "
    "created, older files are deleted. If None or 0, all checkpoint files are "
    "kept. Defaults to 5 (i.e. the 5 most recent checkpoint files are kept.)")

flags.DEFINE_string(
    "job-dir", "",
    "Unused. Must be here because of ml-engine.")
flags.DEFINE_string(
    "model_dir", None, """Directory for saving models and logs.""")


def make_run_config():
  """Makes a RunConfig object with FLAGS.

  Returns:
    tf.estimator.RunConfig.
  Raises:
    ValueError: If not exactly one of `save_checkpoints_secs` and
      `save_checkpoints_steps` is specified.
  """
  save_checkpoints_secs = FLAGS.save_checkpoints_secs or None
  save_checkpoints_steps = FLAGS.save_checkpoints_steps or None
  if save_checkpoints_secs and save_checkpoints_steps:
    raise ValueError("save_checkpoints_secs and save_checkpoints_steps "
                     "cannot both be non-zero.")
  if not (save_checkpoints_secs or save_checkpoints_steps):
    raise ValueError("save_checkpoints_secs and save_checkpoints_steps "
                     "cannot both be zero.")

  # An error is thrown by absl.flags if train.sh passes tf_random_seed=None, so
  # it passes -1 instead.
  if FLAGS.tf_random_seed == -1:
    tf_random_seed = None
  else:
    tf_random_seed = FLAGS.tf_random_seed

  return tf.estimator.RunConfig(
      save_summary_steps=FLAGS.save_summary_steps,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=save_checkpoints_secs,
      save_checkpoints_steps=save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      tf_random_seed=tf_random_seed)


def main(argv):
  del argv

  run_config = make_run_config()
  estimator_builder = adanet_improve_nas.Builder()
  hparams = estimator_builder.hparams(FLAGS.batch_size, FLAGS.hparams)

  tf.logging.info("Running Experiment with HParams: %s", hparams)
  if FLAGS.dataset == "cifar10":
    data_provider = cifar10.Provider()
  elif FLAGS.dataset == "cifar100":
    data_provider = cifar100.Provider()
  elif FLAGS.dataset == "fake":
    data_provider = fake_data.FakeImageProvider(
        num_examples=10,
        num_classes=10,
        image_dim=32,
        channels=3,
        seed=42)
  else:
    raise ValueError("Invalid dataset")

  estimator = estimator_builder.estimator(
      data_provider=data_provider,
      run_config=run_config,
      hparams=hparams,
      train_steps=FLAGS.train_steps)

  train_spec = tf.estimator.TrainSpec(
      input_fn=data_provider.get_input_fn(
          partition="train",
          mode=tf.estimator.ModeKeys.TRAIN,
          batch_size=FLAGS.batch_size),
      max_steps=FLAGS.train_steps
  )

  eval_spec = tf.estimator.EvalSpec(
      input_fn=data_provider.get_input_fn(
          partition="test",
          mode=tf.estimator.ModeKeys.EVAL,
          batch_size=FLAGS.batch_size),
      steps=FLAGS.eval_steps,
      start_delay_secs=10,
      throttle_secs=1800
  )

  tf.logging.info("Training!")
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  tf.logging.info("Done training!")


if __name__ == "__main__":
  tf.app.run(main)
