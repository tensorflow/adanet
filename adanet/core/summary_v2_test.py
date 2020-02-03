"""Test AdaNet summary single graph implementation for TF 2.

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
import struct

from absl.testing import parameterized
from adanet import tf_compat
from adanet.core import testing_utils as tu
from adanet.core.summary import _ScopedSummaryV2
import tensorflow.compat.v2 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
# pylint: enable=g-direct-tensorflow-import


def simple_value(summary_value):
  """Returns the scalar parsed from the summary proto tensor_value bytes."""

  return struct.unpack("<f", summary_value.tensor.tensor_content)[0]


class ScopedSummaryV2Test(tu.AdanetTestCase):

  def read_single_event_from_eventfile(self, summary):
    dir_ = self.test_subdirectory
    if summary.namespace:
      dir_ = os.path.join(dir_, summary.namespace)
    if summary.scope:
      dir_ = os.path.join(dir_, summary.scope)
    event_files = sorted(tf.io.gfile.glob(os.path.join(dir_, "*.v2")))
    events = list(tf.compat.v1.train.summary_iterator(event_files[-1]))
    # Expect a boilerplate event for the file_version, then the summary one.
    self.assertGreaterEqual(len(events), 2)
    return events[1:]

  def write_summaries(self, summary):
    summary_ops = []
    writer = tf.summary.create_file_writer(summary.logdir)
    with writer.as_default():
      for summary_fn, tensor in summary.summary_tuples():
        summary_ops.append(summary_fn(tensor, step=10))

    writer_flush = writer.flush()
    self.evaluate([tf.compat.v1.global_variables_initializer(), writer.init()])
    self.evaluate(summary_ops)
    self.evaluate(writer_flush)

  @parameterized.named_parameters(
      {
          "testcase_name": "without_scope",
          "scope": None,
      }, {
          "testcase_name": "with_scope",
          "scope": "with_scope",
      })
  @tf_compat.skip_for_tf1
  @test_util.run_in_graph_and_eager_modes
  def test_scope(self, scope):
    scoped_summary = _ScopedSummaryV2(self.test_subdirectory, scope=scope)
    self.assertEqual(scope, scoped_summary.scope)

  @parameterized.named_parameters(
      {
          "testcase_name": "without_scope",
          "scope": None,
      }, {
          "testcase_name": "with_scope",
          "scope": "with_scope",
      }, {
          "testcase_name": "skip_summary",
          "scope": None,
          "skip_summary": True,
      })
  @tf_compat.skip_for_tf1
  @test_util.run_in_graph_and_eager_modes
  def test_scalar_summary(self, scope, skip_summary=False):
    with context.graph_mode():
      scoped_summary = _ScopedSummaryV2(
          self.test_subdirectory, scope=scope, skip_summary=skip_summary)
      i = tf.constant(3)
      with tf.name_scope("outer"):
        scoped_summary.scalar("inner", i)
      self.write_summaries(scoped_summary)
    if skip_summary:
      return
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 1)
    self.assertEqual(values[0].tag, "outer/inner")
    self.assertEqual(simple_value(values[0]), 3.0)

  @parameterized.named_parameters(
      {
          "testcase_name": "without_scope",
          "scope": None,
      }, {
          "testcase_name": "with_scope",
          "scope": "with_scope",
      })
  @tf_compat.skip_for_tf1
  @test_util.run_in_graph_and_eager_modes
  def test_summarizing_variable(self, scope):
    scoped_summary = _ScopedSummaryV2(self.test_subdirectory, scope=scope)
    c = tf.constant(42.0)
    v = tf.Variable(c)
    scoped_summary.scalar("summary", v)
    self.write_summaries(scoped_summary)
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 1)
    value = values[0]
    self.assertEqual(value.tag, "summary")
    self.assertEqual(simple_value(value), 42.0)

  @parameterized.named_parameters(
      {
          "testcase_name": "without_scope",
          "scope": None,
      }, {
          "testcase_name": "with_scope",
          "scope": "with_scope",
      }, {
          "testcase_name": "skip_summary",
          "scope": None,
          "skip_summary": True,
      })
  @tf_compat.skip_for_tf1
  @test_util.run_in_graph_and_eager_modes
  def test_image_summary(self, scope, skip_summary=False):
    with context.graph_mode():
      scoped_summary = _ScopedSummaryV2(
          self.test_subdirectory, scope=scope, skip_summary=skip_summary)
      i = tf.ones((5, 4, 4, 3))
      with tf.name_scope("outer"):
        scoped_summary.image("inner", i, max_outputs=3)
      self.write_summaries(scoped_summary)
    if skip_summary:
      return
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 1)
    self.assertEqual("outer/inner", values[0].tag)

  @parameterized.named_parameters(
      {
          "testcase_name": "without_scope",
          "scope": None,
      }, {
          "testcase_name": "with_scope",
          "scope": "with_scope",
      }, {
          "testcase_name": "skip_summary",
          "scope": None,
          "skip_summary": True,
      })
  @tf_compat.skip_for_tf1
  @test_util.run_in_graph_and_eager_modes
  def test_histogram_summary(self, scope, skip_summary=False):
    with context.graph_mode():
      scoped_summary = _ScopedSummaryV2(
          self.test_subdirectory, scope=scope, skip_summary=skip_summary)
      i = tf.ones((5, 4, 4, 3))
      with tf.name_scope("outer"):
        scoped_summary.histogram("inner", i)
      self.write_summaries(scoped_summary)
    if skip_summary:
      return
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 1)
    self.assertEqual("outer/inner", values[0].tag)

  @parameterized.named_parameters(
      {
          "testcase_name": "without_scope",
          "scope": None,
      }, {
          "testcase_name": "with_scope",
          "scope": "with_scope",
      }, {
          "testcase_name": "skip_summary",
          "scope": None,
          "skip_summary": True,
      })
  @tf_compat.skip_for_tf1
  @test_util.run_in_graph_and_eager_modes
  def test_audio_summary(self, scope, skip_summary=False):
    with context.graph_mode():
      scoped_summary = _ScopedSummaryV2(
          self.test_subdirectory, scope=scope, skip_summary=skip_summary)
      i = tf.ones((5, 3, 4))
      with tf.name_scope("outer"):
        scoped_summary.audio("inner", i, sample_rate=2, max_outputs=3)
      self.write_summaries(scoped_summary)
    if skip_summary:
      return
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 1)
    self.assertLen(values, 1)
    self.assertEqual(values[0].tag, "outer/inner")

  @parameterized.named_parameters(
      {
          "testcase_name": "without_scope",
          "scope": None,
      }, {
          "testcase_name": "with_scope",
          "scope": "with_scope",
      })
  @tf_compat.skip_for_tf1
  @test_util.run_in_graph_and_eager_modes
  def test_summary_name_conversion(self, scope):
    scoped_summary = _ScopedSummaryV2(self.test_subdirectory, scope=scope)
    c = tf.constant(3)
    scoped_summary.scalar("name with spaces", c)
    scoped_summary.scalar("name with many $#illegal^: characters!", c)
    scoped_summary.scalar("/name/with/leading/slash", c)
    self.write_summaries(scoped_summary)
    events = self.read_single_event_from_eventfile(scoped_summary)
    self.assertLen(events, 3)
    tags = [event.summary.value[0].tag for event in events]
    # Characters that were illegal in TF 1 are now valid in TF 2.
    self.assertIn("name with spaces", tags)
    self.assertIn("name with many $#illegal^: characters!", tags)
    self.assertIn("name/with/leading/slash", tags)

  @parameterized.named_parameters(
      {
          "testcase_name": "without_scope",
          "scope": None,
      }, {
          "testcase_name": "with_scope",
          "scope": "with_scope",
      })
  @tf_compat.skip_for_tf1
  @test_util.run_in_graph_and_eager_modes
  def test_current_scope(self, scope):
    with context.graph_mode():
      scoped_summary = _ScopedSummaryV2(self.test_subdirectory, scope=scope)
      i = tf.constant(3)
      with tf.compat.v1.variable_scope("outer1"):
        with tf.compat.v1.variable_scope("outer2"):
          with scoped_summary.current_scope():
            with tf.compat.v1.variable_scope("inner1"):
              scoped_summary.scalar("inner2/a/b/c", i)
      self.write_summaries(scoped_summary)
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 1)
    self.assertEqual(values[0].tag, "inner1/inner2/a/b/c")
    self.assertEqual(simple_value(values[0]), 3.0)

  @tf_compat.skip_for_tf1
  @test_util.run_in_graph_and_eager_modes
  def test_summary_args(self):
    summary = _ScopedSummaryV2(self.test_subdirectory)
    summary.scalar("scalar", 1, "family")
    summary.image("image", 1, 3, "family")
    summary.histogram("histogram", 1, "family")
    summary.audio("audio", 1, 3, 3, "family")
    self.assertLen(summary.summary_tuples(), 4)

  @tf_compat.skip_for_tf1
  @test_util.run_in_graph_and_eager_modes
  def test_summary_kwargs(self):
    summary = _ScopedSummaryV2(self.test_subdirectory)
    summary.scalar(name="scalar", tensor=1, family="family")
    summary.image(name="image", tensor=1, max_outputs=3, family="family")
    summary.histogram(name="histogram", values=1, family="family")
    summary.audio(
        name="audio", tensor=1, sample_rate=3, max_outputs=3, family="family")
    self.assertLen(summary.summary_tuples(), 4)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
