# Copyright 2019 The AdaNet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Used to run estimators for distributed tests.

In distributed tests, we spawn processes to run estimator tasks like chief,
workers, parameter servers. The role of each task is determined by the TF_CONFIG
environment variable.

For more information on how tf.estimator.RunConfig uses TF_CONFIG, see
https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from adanet.autoensemble.estimator import AutoEnsembleEstimator
from adanet.core.estimator import Estimator
from adanet.core.subnetwork import Builder
from adanet.core.subnetwork import SimpleGenerator
from adanet.core.subnetwork import Subnetwork
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.training import session_manager as session_manager_lib

# Module path changed. Try importing from new and old location to maintain
# backwards compatibility.
# pylint: disable=g-import-not-at-top
try:
  from tensorflow_estimator.python.estimator import training as training_lib
except ImportError:
  from tensorflow.python.estimator import training as training_lib
# pylint: enable=g-import-not-at-top
# pylint: enable=g-direct-tensorflow-import

tf.flags.DEFINE_enum("estimator_type", "estimator", [
    "estimator",
    "autoensemble",
], "The estimator type to train.")

tf.flags.DEFINE_string("model_dir", "", "The model directory.")

FLAGS = tf.flags.FLAGS


class SessionManager(session_manager_lib.SessionManager):
  """A session manager with a shorter recovery time."""

  def __init__(self,
               local_init_op=None,
               ready_op=None,
               ready_for_local_init_op=None,
               graph=None,
               recovery_wait_secs=None,
               local_init_run_options=None):
    # Reduced wait time.
    super(SessionManager, self).__init__(
        local_init_op,
        ready_op,
        ready_for_local_init_op,
        graph,
        recovery_wait_secs=.5,
        local_init_run_options=local_init_run_options)


@contextlib.contextmanager
def _monkey_patch_distributed_training_times():
  """Monkey-patches global attributes with subnetwork-specifics ones."""

  old_delay_secs_per_worker = training_lib._DELAY_SECS_PER_WORKER  # pylint: disable=protected-access
  old_session_manager = session_manager_lib.SessionManager

  # monkey-patch global attributes.
  session_manager_lib.SessionManager = SessionManager
  # Override default delay per worker to speed up tests.
  training_lib._DELAY_SECS_PER_WORKER = .2  # pylint: disable=protected-access

  try:
    yield
  finally:
    # Revert monkey-patches.
    session_manager_lib.SessionManager = old_session_manager
    training_lib._DELAY_SECS_PER_WORKER = old_delay_secs_per_worker  # pylint: disable=protected-access


class _DNNBuilder(Builder):
  """A simple DNN subnetwork builder."""

  def __init__(self, name, config, layer_size=3, seed=13):
    self._name = name
    self._layer_size = layer_size
    self._config = config
    self._seed = seed

  @property
  def name(self):
    return self._name

  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    seed = self._seed
    if previous_ensemble:
      # Increment seed so different iterations don't learn the exact same thing.
      seed += 1
    num_ps_replicas = self._config.num_ps_replicas if self._config else 0
    partitioner = tf.min_max_variable_partitioner(
        max_partitions=num_ps_replicas)
    with tf.variable_scope("dnn", partitioner=partitioner):
      shared = {}
      with tf.variable_scope("hidden_layer"):
        w = tf.get_variable(
            shape=[2, self._layer_size],
            initializer=tf.glorot_uniform_initializer(seed=seed),
            name="weight")
        hidden_layer = tf.matmul(features["x"], w)

      if previous_ensemble:
        other_hidden_layer = previous_ensemble.weighted_subnetworks[
            -1].subnetwork.shared["hidden_layer"]
        hidden_layer = tf.concat([hidden_layer, other_hidden_layer], axis=1)

      # Use a leaky-relu activation so that gradients can flow even when
      # outputs are negative. Leaky relu has a non-zero slope when x < 0.
      # Otherwise success at learning is completely dependent on random seed.
      hidden_layer = tf.nn.leaky_relu(hidden_layer, alpha=.2)
      shared["hidden_layer"] = hidden_layer

      with tf.variable_scope("logits"):
        logits = tf.layers.dense(
            hidden_layer,
            logits_dimension,
            kernel_initializer=tf.glorot_uniform_initializer(seed=seed))

      summary.scalar("scalar", 3)

      return Subnetwork(
          last_layer=logits, logits=logits, complexity=3, shared=shared)

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
    return optimizer.minimize(loss, var_list=var_list)


def train_and_evaluate_estimator():
  """Runs Estimator distributed training."""

  # The tf.estimator.RunConfig automatically parses the TF_CONFIG environment
  # variables during construction.
  # For more information on how tf.estimator.RunConfig uses TF_CONFIG, see
  # https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig.
  config = tf.estimator.RunConfig(
      tf_random_seed=42,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          log_device_placement=False,
          # Ignore other workers; only talk to parameter servers.
          # Otherwise, when a chief/worker terminates, the others will hang.
          device_filters=["/job:ps"]))
  head = tf.contrib.estimator.regression_head(
      loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

  kwargs = {
      "max_iteration_steps": 100,
      "force_grow": True,
      "delay_secs_per_worker": .2,
      "max_worker_delay_secs": 1,
      "worker_wait_secs": .5,
      # Set low timeout to reduce wait time for failures.
      "worker_wait_timeout_secs": 60,
      "config": config
  }
  if FLAGS.estimator_type == "autoensemble":
    feature_columns = [tf.feature_column.numeric_column("x", shape=[2])]
    candidate_pool = {
        "linear":
            tf.estimator.LinearEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=tf.train.AdamOptimizer(learning_rate=.001)),
        "dnn":
            tf.estimator.DNNEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=tf.train.AdamOptimizer(learning_rate=.001),
                hidden_units=[3]),
        "dnn2":
            tf.estimator.DNNEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=tf.train.AdamOptimizer(learning_rate=.001),
                hidden_units=[5])
    }

    estimator = AutoEnsembleEstimator(
        head=head, candidate_pool=candidate_pool, **kwargs)

  elif FLAGS.estimator_type == "estimator":
    subnetwork_generator = SimpleGenerator([
        _DNNBuilder("dnn1", config, layer_size=3),
        _DNNBuilder("dnn2", config, layer_size=4),
        _DNNBuilder("dnn3", config, layer_size=5),
    ])

    estimator = Estimator(
        head=head, subnetwork_generator=subnetwork_generator, **kwargs)

  def input_fn():
    xor_features = [[1., 0.], [0., 0], [0., 1.], [1., 1.]]
    xor_labels = [[1.], [0.], [1.], [0.]]
    input_features = {"x": tf.constant(xor_features, name="x")}
    input_labels = tf.constant(xor_labels, name="y")
    return input_features, input_labels

  # Train for three iterations.
  train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=300)
  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, steps=1)

  # Calling train_and_evaluate is the official way to perform distributed
  # training with an Estimator. Calling Estimator#train directly results
  # in an error when the TF_CONFIG is setup for a cluster.
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(unused_argv):
  # Reduce hard-coded waits, delays, and timeouts for quicker tests.
  with _monkey_patch_distributed_training_times():
    train_and_evaluate_estimator()


if __name__ == "__main__":
  tf.app.run()
