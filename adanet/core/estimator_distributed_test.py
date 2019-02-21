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
import json
import os
import shutil
import socket
import subprocess
import time

from absl.testing import parameterized
from adanet.core import testing_utils as tu
from adanet.core.timer import _CountDownTimer
import tensorflow as tf

# Maximum number of characters to log per process.
# NOTE: The full process output is written to disk.
MAX_OUTPUT_CHARS = 15000

# A process. name is a string identifying the process in logs. stdout and
# stderr are file objects of the process's stdout and stderr, respectively.
_ProcessInfo = collections.namedtuple("_ProcessInfo",
                                      ["name", "popen", "stdout", "stderr"])


def _create_task_process(task_type, task_index, tf_config, output_dir,
                         model_dir):
  """Creates a process for a single estimator task.

  Args:
    task_type: 'chief', 'worker' or 'ps'.
    task_index: The index of the task within the cluster.
    tf_config: Dictionary representation of the TF_CONFIG environment variable.
      This method creates a copy as to not mutate the input dict.
    output_dir: Where to place the output files, storing the task's stdout and
      stderr.
    model_dir: The Estimator's model directory.

  Returns:
    A _ProcessInfo namedtuple of the running process. The stdout and stderr
    fields of this tuple must be closed by the caller once the process ends.
  """

  process_name = "%s_%s" % (task_type, task_index)
  args = [
      os.path.join(
          tf.flags.FLAGS.test_srcdir,
          "/adanet/core/estimator_distributed_test_runner"
      )
  ]
  # Log everything to stderr.
  args.append("--stderrthreshold=info")
  tf.logging.info("Spawning %s process: %s" % (process_name, " ".join(args)))
  stdout_filename = os.path.join(output_dir, "%s_stdout.txt" % process_name)
  stderr_filename = os.path.join(output_dir, "%s_stderr.txt" % process_name)
  tf.logging.info("Logging to %s", output_dir)
  stdout_file = open(stdout_filename, "w+")
  stderr_file = open(stderr_filename, "w+")
  tf_config = copy.deepcopy(tf_config)
  tf_config["task"]["type"] = task_type
  tf_config["task"]["index"] = task_index
  json_tf_config = json.dumps(tf_config)
  env = os.environ.copy()
  # Allow stdout to be viewed before the process ends.
  env["PYTHONUNBUFFERED"] = "1"
  env["TF_CPP_MIN_LOG_LEVEL"] = "0"
  env["TF_CONFIG"] = json_tf_config
  env["MODEL_DIR"] = model_dir
  popen = subprocess.Popen(
      args, stdout=stdout_file, stderr=stderr_file, env=env)
  return _ProcessInfo(process_name, popen, stdout_file, stderr_file)


def _pick_unused_port():
  """Returns a free port on localhost."""

  for family in (socket.AF_INET6, socket.AF_INET):
    try:
      sock = socket.socket(family, socket.SOCK_STREAM)
      sock.bind(("", 0))  # Passing port '0' binds to a free port on localhost.
      port = sock.getsockname()[1]
      sock.close()
      return port
    except socket.error:
      continue
  raise socket.error


class EstimatorDistributedTrainingTest(tu.AdanetTestCase):
  """Tests distributed training."""

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
      A list of strings, each which is a string of the stdout of a wait process.

    Raises:
      Exception: When waiting for tasks to finish times out.
    """

    timer = _CountDownTimer(timeout_secs)
    wait_process_stdouts = [None] * len(wait_processes)
    finished_wait_processes = set()
    while len(finished_wait_processes) < len(wait_processes):
      if timer.secs_remaining() == 0:
        tf.logging.error("Timed out! Outputting logs of unfinished processes:")
        for i, wait_process in enumerate(wait_processes):
          if i in finished_wait_processes:
            continue
          wait_process.stdout.seek(0)
          wait_process_stdouts[i] = wait_process.stdout.read()
          tf.logging.info(
              "stdout for incomplete {} (last {} chars): {}\n".format(
                  wait_process.name, MAX_OUTPUT_CHARS,
                  wait_process_stdouts[i][-MAX_OUTPUT_CHARS:]))
          wait_process.stderr.seek(0)
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
        wait_process.stdout.seek(0)
        wait_process_stdouts[i] = wait_process.stdout.read()
        tf.logging.info("stdout for {} (last {} chars): {}\n".format(
            wait_process.name, MAX_OUTPUT_CHARS,
            wait_process_stdouts[i][-MAX_OUTPUT_CHARS:]))
        wait_process.stderr.seek(0)
        tf.logging.info("stderr for {} (last {} chars): {}\n".format(
            wait_process.name, MAX_OUTPUT_CHARS,
            wait_process.stderr.read()[-MAX_OUTPUT_CHARS:]))
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
      kill_process.stdout.seek(0)
      tf.logging.info("stdout for {} (last {} chars): {}\n".format(
          kill_process.name, MAX_OUTPUT_CHARS,
          kill_process.stdout.read()[-MAX_OUTPUT_CHARS:]))
      kill_process.stderr.seek(0)
      tf.logging.info("stderr for {} (last {} chars): {}\n".format(
          kill_process.name, MAX_OUTPUT_CHARS,
          kill_process.stderr.read()[-MAX_OUTPUT_CHARS:]))
    return wait_process_stdouts

  @parameterized.named_parameters({
      "testcase_name": "one_worker",
      "num_workers": 1,
      "num_ps": 0,
  }, {
      "testcase_name": "one_worker_one_ps",
      "num_workers": 1,
      "num_ps": 1,
  }, {
      "testcase_name": "two_workers_one_ps",
      "num_workers": 2,
      "num_ps": 1,
  }, {
      "testcase_name": "three_workers_three_ps",
      "num_workers": 3,
      "num_ps": 3,
  })
  def test_distributed_training(self, num_workers, num_ps):
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

    output_dir = os.path.join(self.get_temp_dir(), self.id())
    tf.logging.info("Logging process outputs to %s", output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    tf.gfile.MakeDirs(output_dir)
    model_dir = self.test_subdirectory

    # Chief
    worker_processes.append(
        _create_task_process("chief", 0, tf_config, output_dir, model_dir))
    # Workers
    for i in range(len(ws_targets[1:])):
      worker_processes.append(
          _create_task_process("worker", i, tf_config, output_dir, model_dir))
    # Parameter Servers (PS)
    for i in range(len(ps_targets)):
      ps_processes.append(
          _create_task_process("ps", i, tf_config, output_dir, model_dir))

    # Run processes.
    try:
      # NOTE: Parameter servers do not shut down on their own.
      self._wait_for_processes(
          worker_processes, kill_processes=ps_processes, timeout_secs=180)
    finally:
      for process in worker_processes + ps_processes:
        try:
          process.popen.kill()
        except OSError:
          pass  # It's OK (and expected) if the process already exited.
        process.stdout.close()
        process.stderr.close()


if __name__ == "__main__":
  tf.test.main()
