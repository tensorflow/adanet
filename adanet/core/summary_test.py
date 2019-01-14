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

from absl.testing import parameterized
from adanet.core.summary import _ScopedSummary
from adanet.core.summary import monkey_patched_summaries
from six.moves import range
import tensorflow as tf


def decode(proto_str):
  """Decodes a proto string."""

  return proto_str.decode("utf-8")


class ScopedSummaryTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "without_scope",
      "scope": None,
  }, {
      "testcase_name": "with_scope",
      "scope": "with_scope",
  })
  def test_scope(self, scope):
    scoped_summary = _ScopedSummary(scope)
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
    scoped_summary = _ScopedSummary(scope, skip_summary)
    with self.test_session() as s:
      i = tf.constant(3)
      with tf.name_scope("outer"):
        im = scoped_summary.scalar("inner", i)
      summary_str = s.run(im)
    if skip_summary:
      self.assertEqual("", decode(summary_str))
      return
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
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
    scoped_summary = _ScopedSummary(scope)
    with self.test_session() as s:
      i = tf.constant(7)
      with tf.name_scope("outer"):
        im1 = scoped_summary.scalar("inner", i, family="family")
        im2 = scoped_summary.scalar("inner", i, family="family")
      sm1, sm2 = s.run([im1, im2])
    summary = tf.Summary()

    summary.ParseFromString(sm1)
    values = summary.value
    self.assertLen(values, 1)
    self.assertEqual(values[0].tag, "family/outer/family/inner")
    self.assertEqual(values[0].simple_value, 7.0)

    summary.ParseFromString(sm2)
    values = summary.value
    self.assertLen(values, 1)
    self.assertEqual(values[0].tag, "family/outer/family/inner_1")
    self.assertEqual(values[0].simple_value, 7.0)

  @parameterized.named_parameters({
      "testcase_name": "without_scope",
      "scope": None,
  }, {
      "testcase_name": "with_scope",
      "scope": "with_scope",
  })
  def test_summarizing_variable(self, scope):
    scoped_summary = _ScopedSummary(scope)
    with self.test_session() as s:
      c = tf.constant(42.0)
      v = tf.Variable(c)
      ss = scoped_summary.scalar("summary", v)
      init = tf.global_variables_initializer()
      s.run(init)
      summ_str = s.run(ss)
    summary = tf.Summary()
    summary.ParseFromString(summ_str)
    self.assertLen(summary.value, 1)
    value = summary.value[0]
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
    scoped_summary = _ScopedSummary(scope, skip_summary)
    with self.test_session() as s:
      i = tf.ones((5, 4, 4, 3))
      with tf.name_scope("outer"):
        im = scoped_summary.image("inner", i, max_outputs=3)
      summary_str = s.run(im)
    if skip_summary:
      self.assertEqual("", decode(summary_str))
      return
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
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
    scoped_summary = _ScopedSummary(scope)
    with self.test_session() as s:
      i = tf.ones((5, 2, 3, 1))
      with tf.name_scope("outer"):
        im = scoped_summary.image("inner", i, max_outputs=3, family="family")
      summary_str = s.run(im)
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
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
    scoped_summary = _ScopedSummary(scope, skip_summary)
    with self.test_session() as s:
      i = tf.ones((5, 4, 4, 3))
      with tf.name_scope("outer"):
        summ_op = scoped_summary.histogram("inner", i)
      summary_str = s.run(summ_op)
    if skip_summary:
      self.assertEqual("", decode(summary_str))
      return
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    self.assertLen(summary.value, 1)
    self.assertEqual(summary.value[0].tag, "outer/inner")

  @parameterized.named_parameters({
      "testcase_name": "without_scope",
      "scope": None,
  }, {
      "testcase_name": "with_scope",
      "scope": "with_scope",
  })
  def test_histogram_summary_with_family(self, scope):
    scoped_summary = _ScopedSummary(scope)
    with self.test_session() as s:
      i = tf.ones((5, 4, 4, 3))
      with tf.name_scope("outer"):
        summ_op = scoped_summary.histogram("inner", i, family="family")
      summary_str = s.run(summ_op)
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    self.assertLen(summary.value, 1)
    self.assertEqual(summary.value[0].tag, "family/outer/family/inner")

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
    scoped_summary = _ScopedSummary(scope, skip_summary)
    with self.test_session() as s:
      i = tf.ones((5, 3, 4))
      with tf.name_scope("outer"):
        aud = scoped_summary.audio("inner", i, 0.2, max_outputs=3)
      summary_str = s.run(aud)
    if skip_summary:
      self.assertEqual("", decode(summary_str))
      return
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
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
    scoped_summary = _ScopedSummary(scope)
    with self.test_session() as s:
      i = tf.ones((5, 3, 4))
      with tf.name_scope("outer"):
        aud = scoped_summary.audio(
            "inner", i, 0.2, max_outputs=3, family="family")
      summary_str = s.run(aud)
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
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
    scoped_summary = _ScopedSummary(scope)
    c = tf.constant(3)
    summary = tf.Summary()

    with self.test_session() as sess:
      s = scoped_summary.scalar("name with spaces", c)
      summary.ParseFromString(sess.run(s))
      self.assertEqual(summary.value[0].tag, "name_with_spaces")

      s2 = scoped_summary.scalar("name with many $#illegal^: characters!", c)
      summary.ParseFromString(sess.run(s2))
      self.assertEqual(summary.value[0].tag,
                       "name_with_many___illegal___characters_")

      s3 = scoped_summary.scalar("/name/with/leading/slash", c)
      summary.ParseFromString(sess.run(s3))
      self.assertEqual(summary.value[0].tag, "name/with/leading/slash")

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

    scoped_summary0 = _ScopedSummary()
    scoped_summary0.scalar("c0", c0)
    scoped_summary0.scalar("c1", c1)

    scoped_summary1 = _ScopedSummary("scope1")
    scoped_summary1.scalar("c0", c0)
    scoped_summary1.scalar("c1", c1)

    scoped_summary2 = _ScopedSummary("scope2")
    scoped_summary2.scalar("c0", c0)
    scoped_summary2.scalar("c1", c1)

    if nest_graph:
      with tf.Graph().as_default():
        scoped_summary2.scalar("c2", tf.constant(2))
        with tf.Session() as sess:
          summaries = scoped_summary2.merge_all()
          tf.logging.warn("summaries %s", summaries)
          summary = tf.Summary()
          summary.ParseFromString(sess.run(tf.summary.merge(summaries)))
          self.assertEqual(["c2"], [s.tag for s in summary.value])
          self.assertEqual([2], [s.simple_value for s in summary.value])

    with tf.Session() as sess:
      for scoped_summary in [scoped_summary0, scoped_summary1, scoped_summary2]:
        summaries = scoped_summary.merge_all()
        summary = tf.Summary()
        summary.ParseFromString(sess.run(tf.summary.merge(summaries)))
        self.assertEqual(["c0", "c1"], [s.tag for s in summary.value])
        self.assertEqual([0, 1], [s.simple_value for s in summary.value])

  def test_summary_args(self):
    summary = _ScopedSummary()
    summary.scalar("scalar", 1, "family")
    summary.image("image", 1, 3, "family")
    summary.histogram("histogram", 1, "family")
    summary.audio("audio", 1, 3, 3, "family")
    self.assertLen(summary.merge_all(), 4)

  def test_summary_kwargs(self):
    summary = _ScopedSummary()
    summary.scalar(name="scalar", tensor=1, family="family")
    summary.image(name="image", tensor=1, max_outputs=3, family="family")
    summary.histogram(name="histogram", values=1, family="family")
    summary.audio(
        name="audio", tensor=1, sample_rate=3, max_outputs=3, family="family")
    self.assertLen(summary.merge_all(), 4)

  def test_monkey_patched_summaries_args(self):
    summary = _ScopedSummary()
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
    summary = _ScopedSummary()
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
