"""Test AdaNet estimator cluster training support.

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

import collections
import copy
import itertools
import json
import os
import shutil
import socket
import subprocess
import time

from absl.testing import parameterized
from adanet.core.timer import _CountDownTimer
import tensorflow as tf

# Maximum number of characters to log per process.
# NOTE: The full process output is written to disk.
MAX_OUTPUT_CHARS = 15000

# A process. name is a string identifying the process in logs. stderr is a file
# object of the process's stderr.
_ProcessInfo = collections.namedtuple("_ProcessInfo",
                                      ["name", "popen", "stderr"])


def _create_task_process(task_type, task_index, estimator_type,
                         placement_strategy, tf_config, model_dir):
  """Creates a process for a single estimator task.

  Args:
    task_type: 'chief', 'worker' or 'ps'.
    task_index: The index of the task within the cluster.
    estimator_type: The estimator type to train. 'estimator' or 'autoensemble'.
    placement_strategy: The distributed placement strategy.
    tf_config: Dictionary representation of the TF_CONFIG environment variable.
      This method creates a copy as to not mutate the input dict.
    model_dir: The Estimator's model directory.

  Returns:
    A _ProcessInfo namedtuple of the running process. The stderr field of this
      tuple must be closed by the caller once the process ends.
  """

  process_name = "%s_%s" % (task_type, task_index)
  runner_binary = "bazel-bin/adanet/core/estimator_distributed_test_runner"
  args = [os.path.join(tf.flags.FLAGS.test_srcdir, runner_binary)]
  args.append("--estimator_type={}".format(estimator_type))
  args.append("--placement_strategy={}".format(placement_strategy))
  # Log everything to stderr.
  args.append("--stderrthreshold=info")
  args.append("--model_dir={}".format(model_dir))
  tf.logging.info("Spawning %s process: %s" % (process_name, " ".join(args)))
  stderr_filename = os.path.join(model_dir, "%s_stderr.txt" % process_name)
  tf.logging.info("Logging to %s", model_dir)
  stderr_file = open(stderr_filename, "w+")
  tf_config = copy.deepcopy(tf_config)
  tf_config["task"]["type"] = task_type
  tf_config["task"]["index"] = task_index
  json_tf_config = json.dumps(tf_config)
  env = os.environ.copy()
  # Allow stderr to be viewed before the process ends.
  env["PYTHONUNBUFFERED"] = "1"
  env["TF_CPP_MIN_LOG_LEVEL"] = "0"
  env["TF_CONFIG"] = json_tf_config
  # Change gRPC polling strategy to prevent blocking forever.
  # See https://github.com/tensorflow/tensorflow/issues/17852.
  env["GRPC_POLL_STRATEGY"] = "poll"
  popen = subprocess.Popen(args, stderr=stderr_file, env=env)
  return _ProcessInfo(process_name, popen, stderr_file)


def _pick_unused_port():
  """Returns a free port on localhost."""

  for family in [socket.AF_INET]:
    try:
      sock = socket.socket(family, socket.SOCK_STREAM)
      sock.bind(("", 0))  # Passing port '0' binds to a free port on localhost.
      port = sock.getsockname()[1]
      sock.close()
      return port
    except socket.error:
      continue
  raise socket.error


class EstimatorDistributedTrainingTest(parameterized.TestCase,
                                       tf.test.TestCase):
  """Tests distributed training."""

  def setUp(self):
    super(EstimatorDistributedTrainingTest, self).setUp()
    # Setup and cleanup test directory.
    self.test_subdirectory = os.path.join(tf.flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.makedirs(self.test_subdirectory)

  def _wait_for_processes(self, wait_processes, kill_processes, timeout_secs):
    """Waits until all `wait_processes` finish, then kills `kill_processes`.

    Fails an assert if a process in `wait_processes` finishes unsuccessfully.
    The processes in `kill_processes` are assumed to never finish so they are
    killed.

    Args:
      wait_processes: A list of _ProcessInfo tuples. This function will wait for
        each to finish.
      kill_processes: A list of _ProcessInfo tuples. Each will be killed once
        every process in `wait_processes` is finished.
      timeout_secs: Seconds to wait before timing out and terminating processes.

    Returns:
      A list of strings, each which is a string of the stderr of a wait process.

    Raises:
      Exception: When waiting for tasks to finish times out.
    """

    timer = _CountDownTimer(timeout_secs)
    wait_process_stderrs = [None] * len(wait_processes)
    finished_wait_processes = set()
    while len(finished_wait_processes) < len(wait_processes):
      if timer.secs_remaining() == 0:
        tf.logging.error("Timed out! Outputting logs of unfinished processes:")
        for i, wait_process in enumerate(wait_processes):
          if i in finished_wait_processes:
            continue
          wait_process.stderr.seek(0)
          wait_process_stderrs[i] = wait_process.stderr.read()
          tf.logging.info(
              "stderr for incomplete {} (last {} chars): {}\n".format(
                  wait_process.name, MAX_OUTPUT_CHARS,
                  wait_process.stderr.read()[-MAX_OUTPUT_CHARS:]))
        raise Exception("Timed out waiting for tasks to complete.")
      for i, wait_process in enumerate(wait_processes):
        if i in finished_wait_processes:
          continue
        ret_code = wait_process.popen.poll()
        if ret_code is None:
          continue
        tf.logging.info("{} finished".format(wait_process.name))
        wait_process.stderr.seek(0)
        wait_process_stderrs[i] = wait_process.stderr.read()
        tf.logging.info("stderr for {} (last {} chars): {}\n".format(
            wait_process.name, MAX_OUTPUT_CHARS,
            wait_process_stderrs[i][-MAX_OUTPUT_CHARS:]))
        self.assertEqual(0, ret_code)
        finished_wait_processes.add(i)
      for kill_process in kill_processes:
        ret_code = kill_process.popen.poll()
        # Kill processes should not end until we kill them.
        # If it returns early, note the return code.
        self.assertIsNone(ret_code)
      # Delay between polling loops.
      time.sleep(0.25)
    tf.logging.info("All wait processes finished")
    for i, kill_process in enumerate(kill_processes):
      # Kill each kill process.
      kill_process.popen.kill()
      kill_process.popen.wait()
      kill_process.stderr.seek(0)
      tf.logging.info("stderr for {} (last {} chars): {}\n".format(
          kill_process.name, MAX_OUTPUT_CHARS,
          kill_process.stderr.read()[-MAX_OUTPUT_CHARS:]))
    return wait_process_stderrs

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(
      itertools.chain(*[[
          {
              "testcase_name": "{}_one_worker".format(placement),
              "placement_strategy": placement,
              "num_workers": 1,
              "num_ps": 0,
          },
          {
              "testcase_name": "{}_one_worker_one_ps".format(placement),
              "placement_strategy": placement,
              "num_workers": 1,
              "num_ps": 1,
          },
          {
              "testcase_name": "{}_two_workers_one_ps".format(placement),
              "placement_strategy": placement,
              "num_workers": 2,
              "num_ps": 1,
          },
          {
              "testcase_name": "{}_three_workers_three_ps".format(placement),
              "placement_strategy": placement,
              "num_workers": 3,
              "num_ps": 3,
          },
          {
              "testcase_name": "{}_five_workers_three_ps".format(placement),
              "placement_strategy": placement,
              "num_workers": 5,
              "num_ps": 3,
          },
          {
              "testcase_name":
                  "autoensemble_{}_five_workers_three_ps".format(placement),
              "estimator":
                  "autoensemble",
              "placement_strategy":
                  placement,
              "num_workers":
                  5,
              "num_ps":
                  3,
          },
      ] for placement in ["replication", "round_robin"]]))
  # pylint: enable=g-complex-comprehension
  def test_distributed_training(self,
                                num_workers,
                                num_ps,
                                placement_strategy,
                                estimator="estimator"):
    """Uses multiprocessing to simulate a distributed training environment."""

    # Inspired by `tf.test.create_local_cluster`.
    worker_ports = [_pick_unused_port() for _ in range(num_workers)]
    ps_ports = [_pick_unused_port() for _ in range(num_ps)]
    ws_targets = ["localhost:%s" % port for port in worker_ports]
    ps_targets = ["localhost:%s" % port for port in ps_ports]

    # For details see:
    # https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate
    tf_config = {
        "cluster": {
            # The chief is always worker 0.
            "chief": [ws_targets[0]],
        },
        "task": {
            "type": "chief",
            "index": 0
        },
    }

    # The chief is already worker 0.
    if len(ws_targets) > 1:
      tf_config["cluster"]["worker"] = ws_targets[1:]
    if ps_targets:
      tf_config["cluster"]["ps"] = ps_targets

    worker_processes = []
    ps_processes = []

    model_dir = self.test_subdirectory

    # Chief
    worker_processes.append(
        _create_task_process("chief", 0, estimator, placement_strategy,
                             tf_config, model_dir))
    # Workers
    for i in range(len(ws_targets[1:])):
      worker_processes.append(
          _create_task_process("worker", i, estimator, placement_strategy,
                               tf_config, model_dir))
    # Parameter Servers (PS)
    for i in range(len(ps_targets)):
      ps_processes.append(
          _create_task_process("ps", i, estimator, placement_strategy,
                               tf_config, model_dir))

    # Run processes.
    try:
      # NOTE: Parameter servers do not shut down on their own.
      self._wait_for_processes(
          worker_processes, kill_processes=ps_processes, timeout_secs=600)
    finally:
      for process in worker_processes + ps_processes:
        try:
          process.popen.kill()
        except OSError:
          pass  # It's OK (and expected) if the process already exited.
        process.stderr.close()


if __name__ == "__main__":
  tf.test.main()
