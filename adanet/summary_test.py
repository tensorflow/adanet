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
from adanet.summary import _ScopedSummary
import tensorflow as tf


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
      self.assertEqual(summary_str, "")
      return
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 1)
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
    self.assertEqual(len(values), 1)
    self.assertEqual(values[0].tag, "family/outer/family/inner")
    self.assertEqual(values[0].simple_value, 7.0)

    summary.ParseFromString(sm2)
    values = summary.value
    self.assertEqual(len(values), 1)
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
    self.assertEqual(len(summary.value), 1)
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
      self.assertEqual(summary_str, "")
      return
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted("outer/inner/image/{}".format(i) for i in xrange(3))
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
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted(
        "family/outer/family/inner/image/{}".format(i) for i in xrange(3))
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
      self.assertEqual(summary_str, "")
      return
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    self.assertEqual(len(summary.value), 1)
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
    self.assertEqual(len(summary.value), 1)
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
      self.assertEqual(summary_str, "")
      return
    summary = tf.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted("outer/inner/audio/{}".format(i) for i in xrange(3))
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
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted(
        "family/outer/family/inner/audio/{}".format(i) for i in xrange(3))
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

  def test_merge_all(self):
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

    summary = tf.Summary()

    with self.test_session() as sess:
      for scoped_summary in [scoped_summary0, scoped_summary1, scoped_summary2]:
        merge_op = scoped_summary.merge_all()
        summary.ParseFromString(sess.run(merge_op))
        self.assertEqual(["c0", "c1"], [s.tag for s in summary.value])
        self.assertEqual([0, 1], [s.simple_value for s in summary.value])


if __name__ == "__main__":
  tf.test.main()
