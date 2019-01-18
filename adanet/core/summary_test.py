"""Test AdaNet summary single graph implementation.

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

import os

from absl.testing import parameterized
from adanet.core import testing_utils as tu
from adanet.core.summary import _ScopedSummary
from adanet.core.summary import monkey_patched_summaries
from six.moves import range
import tensorflow as tf


def decode(proto_str):
  """Decodes a proto string."""

  return proto_str.decode("utf-8")


class ScopedSummaryTest(tu.AdanetTestCase):

  def read_single_event_from_eventfile(self, summary):
    dir_ = self.test_subdirectory
    if summary.namespace:
      dir_ = os.path.join(dir_, summary.namespace)
    if summary.scope:
      dir_ = os.path.join(dir_, summary.scope)
    event_files = sorted(tf.gfile.Glob(os.path.join(dir_, "*.v2")))
    events = list(tf.train.summary_iterator(event_files[-1]))
    # Expect a boilerplate event for the file_version, then the summary one.
    self.assertTrue(len(events) >= 2)
    return events[1:]

  @parameterized.named_parameters({
      "testcase_name": "without_scope",
      "scope": None,
  }, {
      "testcase_name": "with_scope",
      "scope": "with_scope",
  })
  def test_scope(self, scope):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory, scope=scope, global_step=10)
    self.assertEqual(scope, scoped_summary.scope)

  @parameterized.named_parameters({
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
  def test_scalar_summary(self, scope, skip_summary=False):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory,
        scope=scope,
        skip_summary=skip_summary,
        global_step=10)
    with self.test_session() as s:
      i = tf.constant(3)
      with tf.name_scope("outer"):
        scoped_summary.scalar("inner", i)
      s.run(tf.contrib.summary.summary_writer_initializer_op())
      s.run(scoped_summary.merge_all())
      s.run(scoped_summary.flush())
    if skip_summary:
      return
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 1)
    self.assertEqual(values[0].tag, "outer/inner")
    self.assertEqual(values[0].simple_value, 3.0)

  @parameterized.named_parameters({
      "testcase_name": "without_scope",
      "scope": None,
  }, {
      "testcase_name": "with_scope",
      "scope": "with_scope",
  })
  def test_scalar_summary_with_family(self, scope):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory, scope=scope, global_step=10)
    with self.test_session() as s:
      i = tf.constant(7)
      with tf.name_scope("outer"):
        scoped_summary.scalar("inner", i, family="family")
        scoped_summary.scalar("inner", i, family="family")
      s.run(tf.contrib.summary.summary_writer_initializer_op())
      s.run(scoped_summary.merge_all())
      s.run(scoped_summary.flush())
    events = self.read_single_event_from_eventfile(scoped_summary)
    self.assertLen(events[0].summary.value, 1)
    self.assertLen(events[1].summary.value, 1)

    self.assertEqual({
        "family/outer/family/inner": 7.0,
        "family/outer/family/inner_1": 7.0
    }, {
        event.summary.value[0].tag: event.summary.value[0].simple_value
        for event in events
    })

  @parameterized.named_parameters({
      "testcase_name": "without_scope",
      "scope": None,
  }, {
      "testcase_name": "with_scope",
      "scope": "with_scope",
  })
  def test_summarizing_variable(self, scope):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory, scope=scope, global_step=10)
    with self.test_session() as s:
      c = tf.constant(42.0)
      v = tf.Variable(c)
      scoped_summary.scalar("summary", v)
      init = tf.global_variables_initializer()
      s.run(init)
      s.run(tf.contrib.summary.summary_writer_initializer_op())
      s.run(scoped_summary.merge_all())
      s.run(scoped_summary.flush())
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 1)
    value = values[0]
    self.assertEqual(value.tag, "summary")
    self.assertEqual(value.simple_value, 42.0)

  @parameterized.named_parameters({
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
  def test_image_summary(self, scope, skip_summary=False):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory,
        scope=scope,
        skip_summary=skip_summary,
        global_step=10)
    with self.test_session() as s:
      i = tf.ones((5, 4, 4, 3))
      with tf.name_scope("outer"):
        scoped_summary.image("inner", i, max_outputs=3)
      s.run(tf.contrib.summary.summary_writer_initializer_op())
      s.run(scoped_summary.merge_all())
      s.run(scoped_summary.flush())
    if skip_summary:
      return
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 3)
    tags = sorted(v.tag for v in values)
    expected = sorted("outer/inner/image/{}".format(i) for i in range(3))
    self.assertEqual(tags, expected)

  @parameterized.named_parameters({
      "testcase_name": "without_scope",
      "scope": None,
  }, {
      "testcase_name": "with_scope",
      "scope": "with_scope",
  })
  def test_image_summary_with_family(self, scope):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory, scope=scope, global_step=10)
    with self.test_session() as s:
      i = tf.ones((5, 2, 3, 1))
      with tf.name_scope("outer"):
        scoped_summary.image("inner", i, max_outputs=3, family="family")
      s.run(tf.contrib.summary.summary_writer_initializer_op())
      s.run(scoped_summary.merge_all())
      s.run(scoped_summary.flush())
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 3)
    tags = sorted(v.tag for v in values)
    expected = sorted(
        "family/outer/family/inner/image/{}".format(i) for i in range(3))
    self.assertEqual(tags, expected)

  @parameterized.named_parameters({
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
  def test_histogram_summary(self, scope, skip_summary=False):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory,
        scope=scope,
        skip_summary=skip_summary,
        global_step=10)
    with self.test_session() as s:
      i = tf.ones((5, 4, 4, 3))
      with tf.name_scope("outer"):
        scoped_summary.histogram("inner", i)
      s.run(tf.contrib.summary.summary_writer_initializer_op())
      s.run(scoped_summary.merge_all())
      s.run(scoped_summary.flush())
    if skip_summary:
      return
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 1)
    self.assertEqual(values[0].tag, "outer/inner")

  @parameterized.named_parameters({
      "testcase_name": "without_scope",
      "scope": None,
  }, {
      "testcase_name": "with_scope",
      "scope": "with_scope",
  })
  def test_histogram_summary_with_family(self, scope):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory, scope=scope, global_step=10)
    with self.test_session() as s:
      i = tf.ones((5, 4, 4, 3))
      with tf.name_scope("outer"):
        scoped_summary.histogram("inner", i, family="family")
      s.run(tf.contrib.summary.summary_writer_initializer_op())
      s.run(scoped_summary.merge_all())
      s.run(scoped_summary.flush())
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 1)
    self.assertEqual(values[0].tag, "family/outer/family/inner")

  @parameterized.named_parameters({
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
  def test_audio_summary(self, scope, skip_summary=False):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory,
        scope=scope,
        skip_summary=skip_summary,
        global_step=10)
    with self.test_session() as s:
      i = tf.ones((5, 3, 4))
      with tf.name_scope("outer"):
        scoped_summary.audio("inner", i, 0.2, max_outputs=3)
      s.run(tf.contrib.summary.summary_writer_initializer_op())
      s.run(scoped_summary.merge_all())
      s.run(scoped_summary.flush())
    if skip_summary:
      return
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 3)
    tags = sorted(v.tag for v in values)
    expected = sorted("outer/inner/audio/{}".format(i) for i in range(3))
    self.assertEqual(tags, expected)

  @parameterized.named_parameters({
      "testcase_name": "without_scope",
      "scope": None,
  }, {
      "testcase_name": "with_scope",
      "scope": "with_scope",
  })
  def test_audio_summary_with_family(self, scope):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory, scope=scope, global_step=10)
    with self.test_session() as s:
      i = tf.ones((5, 3, 4))
      with tf.name_scope("outer"):
        scoped_summary.audio("inner", i, 0.2, max_outputs=3, family="family")
      s.run(tf.contrib.summary.summary_writer_initializer_op())
      s.run(scoped_summary.merge_all())
      s.run(scoped_summary.flush())
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 3)
    tags = sorted(v.tag for v in values)
    expected = sorted(
        "family/outer/family/inner/audio/{}".format(i) for i in range(3))
    self.assertEqual(tags, expected)

  @parameterized.named_parameters({
      "testcase_name": "without_scope",
      "scope": None,
  }, {
      "testcase_name": "with_scope",
      "scope": "with_scope",
  })
  def test_summary_name_conversion(self, scope):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory, scope=scope, global_step=10)
    c = tf.constant(3)
    scoped_summary.scalar("name with spaces", c)
    scoped_summary.scalar("name with many $#illegal^: characters!", c)
    scoped_summary.scalar("/name/with/leading/slash", c)
    with self.test_session() as sess:
      sess.run(tf.contrib.summary.summary_writer_initializer_op())
      sess.run(scoped_summary.merge_all())
      sess.run(scoped_summary.flush())
    events = self.read_single_event_from_eventfile(scoped_summary)
    self.assertLen(events, 3)
    tags = [event.summary.value[0].tag for event in events]
    self.assertIn("name_with_spaces", tags)
    self.assertIn("name_with_many___illegal___characters_", tags)
    self.assertIn("name/with/leading/slash", tags)

  @parameterized.named_parameters({
      "testcase_name": "single_graph",
      "nest_graph": False,
  }, {
      "testcase_name": "nested_graph",
      "nest_graph": True,
  })
  def test_merge_all(self, nest_graph):
    c0 = tf.constant(0)
    c1 = tf.constant(1)

    scoped_summary0 = _ScopedSummary(self.test_subdirectory, global_step=10)
    scoped_summary0.scalar("c0", c0)
    scoped_summary0.scalar("c1", c1)

    scoped_summary1 = _ScopedSummary(
        self.test_subdirectory, scope="scope1", global_step=10)
    scoped_summary1.scalar("c0", c0)
    scoped_summary1.scalar("c1", c1)

    scoped_summary2 = _ScopedSummary(
        self.test_subdirectory, scope="scope2", global_step=10)
    scoped_summary2.scalar("c0", c0)
    scoped_summary2.scalar("c1", c1)

    if nest_graph:
      with tf.Graph().as_default():
        scoped_summary2.scalar("c2", tf.constant(2))
        with tf.Session() as sess:
          sess.run(tf.contrib.summary.summary_writer_initializer_op())
          sess.run(scoped_summary2.merge_all())
          sess.run(scoped_summary2.flush())
          events = self.read_single_event_from_eventfile(scoped_summary2)
          values = {
              e.summary.value[0].tag: e.summary.value[0].simple_value
              for e in events
          }
          self.assertEqual({"c2": 2}, values)

    with tf.Session() as sess:
      sess.run(tf.contrib.summary.summary_writer_initializer_op())
      for scoped_summary in [scoped_summary0, scoped_summary1, scoped_summary2]:
        sess.run(scoped_summary.merge_all())
        sess.run(scoped_summary.flush())
        events = self.read_single_event_from_eventfile(scoped_summary)
        values = {
            e.summary.value[0].tag: e.summary.value[0].simple_value
            for e in events
        }
        self.assertEqual({"c0": 0, "c1": 1}, values)

  @parameterized.named_parameters({
      "testcase_name": "without_scope",
      "scope": None,
  }, {
      "testcase_name": "with_scope",
      "scope": "with_scope",
  })
  def test_current_scope(self, scope):
    scoped_summary = _ScopedSummary(
        self.test_subdirectory, scope=scope, global_step=10)
    i = tf.constant(3)
    with tf.variable_scope("outer1"):
      with tf.variable_scope("outer2"):
        with scoped_summary.current_scope():
          with tf.variable_scope("inner1"):
            scoped_summary.scalar("inner2/a/b/c", i)
    with self.test_session() as s:
      s.run(tf.contrib.summary.summary_writer_initializer_op())
      s.run(scoped_summary.merge_all())
      s.run(scoped_summary.flush())
    events = self.read_single_event_from_eventfile(scoped_summary)
    values = events[0].summary.value
    self.assertLen(values, 1)
    self.assertEqual(values[0].tag, "inner1/inner2/a/b/c")
    self.assertEqual(values[0].simple_value, 3.0)

  def test_summary_args(self):
    summary = _ScopedSummary(self.test_subdirectory, global_step=10)
    summary.scalar("scalar", 1, "family")
    summary.image("image", 1, 3, "family")
    summary.histogram("histogram", 1, "family")
    summary.audio("audio", 1, 3, 3, "family")
    self.assertLen(summary.merge_all(), 4)

  def test_summary_kwargs(self):
    summary = _ScopedSummary(self.test_subdirectory, global_step=10)
    summary.scalar(name="scalar", tensor=1, family="family")
    summary.image(name="image", tensor=1, max_outputs=3, family="family")
    summary.histogram(name="histogram", values=1, family="family")
    summary.audio(
        name="audio", tensor=1, sample_rate=3, max_outputs=3, family="family")
    self.assertLen(summary.merge_all(), 4)

  def test_monkey_patched_summaries_args(self):
    summary = _ScopedSummary(self.test_subdirectory, global_step=10)
    with monkey_patched_summaries(summary):
      tf.summary.scalar("scalar", 1, ["collection"], "family")
      tf.summary.image("image", 1, 3, ["collection"], "family")
      tf.summary.histogram("histogram", 1, ["collection"], "family")
      tf.summary.audio("audio", 1, 3, 3, ["collection"], "family")

      tf.contrib.summary.scalar("scalar_v2", 1, "family", 10)
      tf.contrib.summary.image("image_v2", 1, True, 3, "family", 10)
      tf.contrib.summary.histogram("histogram_v2", 1, "family", 10)
      tf.contrib.summary.audio("audio_v2", 1, 3, 3, "family", 10)
    self.assertLen(summary.merge_all(), 8)

  def test_monkey_patched_summaries_kwargs(self):
    summary = _ScopedSummary(self.test_subdirectory, global_step=10)
    with monkey_patched_summaries(summary):
      tf.summary.scalar(
          name="scalar", tensor=1, collections=["collection"], family="family")
      tf.summary.image(
          name="image",
          tensor=1,
          max_outputs=3,
          collections=["collection"],
          family="family")
      tf.summary.histogram(
          name="histogram",
          values=1,
          collections=["collection"],
          family="family")
      tf.summary.audio(
          name="audio",
          tensor=1,
          sample_rate=3,
          max_outputs=3,
          collections=["collection"],
          family="family")

      tf.contrib.summary.scalar(
          name="scalar_v2", tensor=1, family="family", step=10)
      tf.contrib.summary.image(
          name="image_v2",
          tensor=1,
          bad_color=True,
          max_images=3,
          family="family",
          step=10)
      tf.contrib.summary.histogram(
          name="histogram_v2", tensor=1, family="family", step=10)
      tf.contrib.summary.audio(
          name="audio_v2",
          tensor=1,
          sample_rate=3,
          max_outputs=3,
          family="family",
          step=10)
    self.assertLen(summary.merge_all(), 8)


if __name__ == "__main__":
  tf.test.main()
