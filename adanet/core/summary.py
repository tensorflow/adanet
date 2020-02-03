"""Tensorboard summaries for the single graph AdaNet implementation.

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

import abc
import contextlib
import os

from absl import logging
from adanet import tf_compat
import six
import tensorflow.compat.v1 as tf_v1
import tensorflow.compat.v2 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorboard import compat
from tensorflow.python.ops import summary_op_util
from tensorflow.python.summary import summary as summary_lib
# pylint: enable=g-direct-tensorflow-import

_DEFAULT_SCOPE = "default"


@six.add_metaclass(abc.ABCMeta)
class Summary(object):
  """Interface for writing summaries to Tensorboard."""

  @abc.abstractmethod
  def scalar(self, name, tensor, family=None, description=None):
    """Outputs a `tf.Summary` protocol buffer containing a single scalar value.

    The generated tf.Summary has a Tensor.proto containing the input Tensor.

    Args:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      tensor: A real numeric scalar value, convertible to a float32 Tensor.
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard. DEPRECATED
        in TF 2.
      description: Optional long-form description for this summary, as a
        constant str. Markdown is supported. Defaults to empty.

    Returns:
      A scalar `Tensor` of type `string`. Which contains a `tf.Summary`
      protobuf.

    Raises:
      ValueError: If tensor has the wrong shape or type.
    """

  @abc.abstractmethod
  def image(self, name, tensor, max_outputs=3, family=None, description=None):
    """Outputs a `tf.Summary` protocol buffer with images.

    The summary has up to `max_outputs` summary values containing images. The
    images are built from `tensor` which must be 4-D with shape `[batch_size,
    height, width, channels]` and where `channels` can be:

    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.

    The images have the same number of channels as the input tensor. For float
    input, the values are normalized one image at a time to fit in the range
    `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
    normalization algorithms:

    *  If the input values are all positive, they are rescaled so the largest
    one is 255.
    *  If any input value is negative, the values are shifted so input value 0.0
      is at 127.  They are then rescaled so that either the smallest value is 0,
      or the largest one is 255.

    The `tag` in the outputted tf.Summary.Value protobufs is generated based on
    the
    name, with a suffix depending on the max_outputs setting:

    *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
    *  If `max_outputs` is greater than 1, the summary value tags are
      generated sequentially as '*name*/image/0', '*name*/image/1', etc.

    Args:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      tensor: A Tensor representing pixel data with shape [k, h, w, c], where k
        is the number of images, h and w are the height and width of the images,
        and c is the number of channels, which should be 1, 2, 3, or 4
        (grayscale, grayscale with alpha, RGB, RGBA). Any of the dimensions may
        be statically unknown (i.e., None). Floating point data will be clipped
        to the range [0,1).
      max_outputs: Optional int or rank-0 integer Tensor. At most this many
        images will be emitted at each step. When more than max_outputs many
        images are provided, the first max_outputs many images will be used and
        the rest silently discarded.
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard. DEPRECATED
        in TF 2.
      description: Optional long-form description for this summary, as a
        constant str. Markdown is supported. Defaults to empty.

    Returns:
      A scalar `Tensor` of type `string`. The serialized `tf.Summary` protocol
      buffer.
    """

  @abc.abstractmethod
  def histogram(self,
                name,
                values,
                family=None,
                buckets=None,
                description=None):
    """Outputs a `tf.Summary` protocol buffer with a histogram.

    Adding a histogram summary makes it possible to visualize your data's
    distribution in TensorBoard. You can see a detailed explanation of the
    TensorBoard histogram dashboard
    [here](https://www.tensorflow.org/get_started/tensorboard_histograms).

    The generated [`tf.Summary`](
    tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.

    This op reports an `InvalidArgument` error if any value is not finite.

    Args:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      values: A Tensor of any shape. Must be castable to float64.
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard. DEPRECATED
        in TF 2.
      buckets: Optional positive int. The output will have this many buckets,
        except in two edge cases. If there is no data, then there are no
        buckets. If there is data but all points have the same value, then there
        is one bucket whose left and right endpoints are the same.
      description: Optional long-form description for this summary, as a
        constant str. Markdown is supported. Defaults to empty.

    Returns:
      A scalar `Tensor` of type `string`. The serialized `tf.Summary` protocol
      buffer.
    """

  @abc.abstractmethod
  def audio(self,
            name,
            tensor,
            sample_rate,
            max_outputs=3,
            family=None,
            encoding=None,
            description=None):
    """Writes an audio summary.

    Args:
      name: A name for this summary. The summary tag used for TensorBoard will
        be this name prefixed by any active name scopes.
      tensor: A Tensor representing audio data with shape [k, t, c], where k is
        the number of audio clips, t is the number of frames, and c is the
        number of channels. Elements should be floating-point values in [-1.0,
        1.0]. Any of the dimensions may be statically unknown (i.e., None).
      sample_rate: An int or rank-0 int32 Tensor that represents the sample
        rate, in Hz. Must be positive.
      max_outputs: Optional int or rank-0 integer Tensor. At most this many
        audio clips will be emitted at each step. When more than max_outputs
        many clips are provided, the first max_outputs many clips will be used
        and the rest silently discarded.
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard. DEPRECATED
        in TF 2.
      encoding: Optional constant str for the desired encoding. Only "wav" is
        currently supported, but this is not guaranteed to remain the default,
        so if you want "wav" in particular, set this explicitly.
      description: Optional long-form description for this summary, as a
        constant str. Markdown is supported. Defaults to empty.

    Returns:
      A scalar `Tensor` of type `string`. The serialized `tf.Summary` protocol
      buffer.
    """


def _strip_scope(name, scope, additional_scope):
  """Returns the name with scope stripped from it."""

  if additional_scope:
    name = name.replace("{}/".format(additional_scope), "")
  if not scope:
    scope = _DEFAULT_SCOPE
  name = name.replace("{}/".format(scope), "", 1)
  return name


class _ScopedSummary(Summary):
  """Records summaries in a given scope.

  Each scope gets assigned a different collection where summary ops gets added.

  This allows Tensorboard to display summaries with different scopes but the
  same name in the same charts.
  """

  def __init__(self, scope=None, skip_summary=False, namespace=None):
    """Initializes a `_ScopedSummary`.

    Args:
      scope: String scope name.
      skip_summary: Whether to record summary ops.
      namespace: Optional string namespace for the summary.

    Returns:
      A `_ScopedSummary` instance.
    """

    if tf_compat.tpu_function.get_tpu_context().number_of_shards:
      logging.log_first_n(
          logging.WARN,
          "Scoped summaries will be skipped since they do not support TPU", 1)
      skip_summary = True

    self._scope = scope
    self._namespace = namespace
    self._additional_scope = None
    self._skip_summary = skip_summary
    self._summary_ops = []
    self._actual_summary_scalar_fn = summary_lib.scalar
    self._actual_summary_image_fn = summary_lib.image
    self._actual_summary_histogram_fn = summary_lib.histogram
    self._actual_summary_audio_fn = summary_lib.audio

  @property
  def scope(self):
    """Returns scope string."""

    return self._scope

  @property
  def namespace(self):
    """Returns namespace string."""

    return self._namespace

  @contextlib.contextmanager
  def current_scope(self):
    """Registers the current context's scope to strip it from summary tags."""

    self._additional_scope = tf_compat.v1.get_default_graph().get_name_scope()
    yield
    self._additional_scope = None

  @contextlib.contextmanager
  def _strip_tag_scope(self):
    """Monkey patches `summary_op_util.summary_scope` to strip tag scopes."""

    original_summary_scope = summary_op_util.summary_scope

    @contextlib.contextmanager
    def strip_tag_scope_fn(name, family=None, default_name=None, values=None):
      tag, scope = (None, None)
      with original_summary_scope(name, family, default_name, values) as (t, s):
        tag = _strip_scope(t, self.scope, self._additional_scope)
        scope = s
      yield tag, scope

    summary_op_util.summary_scope = strip_tag_scope_fn
    yield
    summary_op_util.summary_scope = original_summary_scope

  def _prefix_scope(self, name):
    """Prefixes summary name with scope."""

    if self._scope:
      if name[0] == "/":
        name = name[1:]
      return "{scope}/{name}".format(scope=self._scope, name=name)
    return name

  def scalar(self, name, tensor, family=None):
    """See `Summary`."""

    if self._skip_summary:
      return tf.constant("")

    with self._strip_tag_scope():
      summary = self._actual_summary_scalar_fn(
          name=self._prefix_scope(name),
          tensor=tensor,
          family=family,
          collections=[])
    self._summary_ops.append(summary)
    return summary

  def image(self, name, tensor, max_outputs=3, family=None):
    """See `Summary`."""

    if self._skip_summary:
      return tf.constant("")

    with self._strip_tag_scope():
      summary = self._actual_summary_image_fn(
          name=self._prefix_scope(name),
          tensor=tensor,
          max_outputs=max_outputs,
          family=family,
          collections=[])
    self._summary_ops.append(summary)
    return summary

  def histogram(self, name, values, family=None):
    """See `Summary`."""

    if self._skip_summary:
      return tf.constant("")

    with self._strip_tag_scope():
      summary = self._actual_summary_histogram_fn(
          name=self._prefix_scope(name),
          values=values,
          family=family,
          collections=[])
    self._summary_ops.append(summary)
    return summary

  def audio(self, name, tensor, sample_rate, max_outputs=3, family=None):
    """See `Summary`."""

    if self._skip_summary:
      return tf.constant("")

    with self._strip_tag_scope():
      summary = self._actual_summary_audio_fn(
          name=self._prefix_scope(name),
          tensor=tensor,
          sample_rate=sample_rate,
          max_outputs=max_outputs,
          family=family,
          collections=[])
    self._summary_ops.append(summary)
    return summary

  def merge_all(self):
    """Returns the list of this graph's scoped summary ops.

    Note: this is an abuse of the tf.summary.merge_all API since it is expected
    to return a summary op with all summaries merged. However, ScopedSummary is
    only used in the internal implementation, so this should be OK.
    """

    current_graph = tf_compat.v1.get_default_graph()
    return [op for op in self._summary_ops if op.graph == current_graph]


# TODO: _ScopedSummary and _ScopedSummaryV2 share a lot of the same
# methods. Extract a base class for the two, or move shared methods into
# Summary.
class _ScopedSummaryV2(Summary):
  """Records summaries in a given scope.

  Only for TPUEstimator.

  Each scope gets assigned a different collection where summary ops gets added.

  This allows Tensorboard to display summaries with different scopes but the
  same name in the same charts.
  """

  def __init__(self, logdir, namespace=None, scope=None, skip_summary=False):
    """Initializes a `_TPUScopedSummary`.

    Args:
      logdir: String directory path for logging summaries.
      namespace: String namespace to append to the logdir. Can be shared with
        other `_ScopedSummary` objects.
      scope: String scope name.
      skip_summary: Whether to record summary ops.

    Returns:
      A `_ScopedSummary` instance.
    """

    # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    from tensorboard.plugins.audio import summary_v2 as audio_v2_lib
    from tensorboard.plugins.histogram import summary_v2 as histogram_v2_lib
    from tensorboard.plugins.image import summary_v2 as image_v2_lib
    from tensorboard.plugins.scalar import summary_v2 as scalar_v2_lib
    # pylint: enable=g-direct-tensorflow-import,g-import-not-at-top

    assert logdir

    if scope == _DEFAULT_SCOPE:
      raise ValueError("scope cannot be 'default'.")

    if namespace:
      logdir = os.path.join(logdir, namespace)
    if scope:
      logdir = os.path.join(logdir, scope)
    self._logdir = logdir
    self._namespace = namespace
    self._scope = scope
    self._additional_scope = None
    self._skip_summary = skip_summary
    self._actual_summary_scalar_fn = scalar_v2_lib.scalar
    self._actual_summary_image_fn = image_v2_lib.image
    self._actual_summary_histogram_fn = histogram_v2_lib.histogram
    self._actual_summary_audio_fn = audio_v2_lib.audio
    self._summary_tuples = []

  @property
  def namespace(self):
    """Returns namespace string."""

    return self._namespace

  @property
  def scope(self):
    """Returns scope string."""

    return self._scope

  @property
  def logdir(self):
    """Returns the logdir."""

    return self._logdir

  @property
  def writer(self):
    """Returns the file writer."""

    return self._writer

  @contextlib.contextmanager
  def current_scope(self):
    """Registers the current context's scope to strip it from summary tags."""

    self._additional_scope = tf_compat.v1.get_default_graph().get_name_scope()
    try:
      yield
    finally:
      self._additional_scope = None

  @contextlib.contextmanager
  def _strip_tag_scope(self, additional_scope):
    """Monkey patches `summary_op_util.summary_scope` to strip tag scopes."""

    # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    from tensorflow.python.ops import summary_ops_v2 as summary_v2_lib
    from tensorflow.python.ops.summary_ops_v2 import _INVALID_SCOPE_CHARACTERS
    # pylint: enable=g-direct-tensorflow-import,g-import-not-at-top

    original_summary_scope = summary_op_util.summary_scope
    original_summary_scope_v2 = getattr(summary_v2_lib, "summary_scope")

    # TF 1.
    @contextlib.contextmanager
    def strip_tag_scope_fn(name, family=None, default_name=None, values=None):
      tag, scope = (None, None)
      with original_summary_scope(name, family, default_name, values) as (t, s):
        tag = _strip_scope(t, self.scope, additional_scope)
        scope = s
      yield tag, scope

    # TF 2.
    @contextlib.contextmanager
    def monkey_patched_summary_scope_fn(name,
                                        default_name="summary",
                                        values=None):
      """Rescopes the summary tag with the ScopedSummary's scope."""

      name = name or default_name
      current_scope = tf_compat.v1.get_default_graph().get_name_scope()
      tag = current_scope + "/" + name if current_scope else name
      # Strip illegal characters from the scope name, and if that leaves
      # nothing, use None instead so we pick up the default name.
      name = _INVALID_SCOPE_CHARACTERS.sub("", name) or None
      with tf.compat.v1.name_scope(name, default_name, values) as scope:
        tag = _strip_scope(tag, self.scope, additional_scope)
        yield tag, scope

    setattr(summary_op_util, "summary_scope", strip_tag_scope_fn)
    setattr(summary_v2_lib, "summary_scope", monkey_patched_summary_scope_fn)
    setattr(compat.tf2.summary.experimental, "summary_scope",
            monkey_patched_summary_scope_fn)
    setattr(compat.tf2.summary, "summary_scope",
            monkey_patched_summary_scope_fn)
    try:
      yield
    finally:
      setattr(summary_op_util, "summary_scope", original_summary_scope)
      setattr(summary_v2_lib, "summary_scope", original_summary_scope_v2)
      setattr(compat.tf2.summary.experimental, "summary_scope",
              original_summary_scope_v2)
      setattr(compat.tf2.summary, "summary_scope", original_summary_scope_v2)

  def _prefix_scope(self, name):
    scope = self._scope
    if name[0] == "/":
      name = name[1:]
    if not scope:
      scope = _DEFAULT_SCOPE
    return "{scope}/{name}".format(scope=scope, name=name)

  def _create_summary(self, summary_fn, name, tensor):
    """Creates a summary op.

    This will create a function that takes a `Tensor` and adds it to a list with
    its matching `tensor`.

    Args:
      summary_fn: A function that takes a name string and `Tensor` and returns a
        summary op.
      name: String name of the summary.
      tensor: `Tensor` to pass to the summary.
    """
    if self._skip_summary:
      return

    # additional_scope is set with the context from `current_scope`.
    # e.g. "foo/bar".
    additional_scope = self._additional_scope
    # name_scope is from whichever scope the summary actually gets called in.
    # e.g. "foo/bar/baz"
    name_scope = tf_compat.v1.get_default_graph().get_name_scope()
    # Reuse name_scope if it exists by appending "/" to it.
    name_scope = name_scope + "/" if name_scope else name_scope

    def _summary_fn(tensor, step):
      """Creates a summary with the given `Tensor`."""

      summary_name = self._prefix_scope(name)
      # Recover the current name scope when this fn is be called, because the
      # scope may be different when fns are called.
      # e.g. "foo/bar/baz/scalar" will become "baz/scalar" when
      # additional_scope is "foo/bar".
      # TODO: Figure out a cleaner way to handle this.
      assert not tf_compat.v1.get_default_graph().get_name_scope()
      with tf_compat.v1.name_scope(name_scope):
        with self._strip_tag_scope(additional_scope):
          # TODO: Do summaries need to be reduced before writing?
          # Presumably each tensor core creates its own summary so we may be
          # writing out num_tensor_cores copies of the same value.
          return summary_fn(summary_name, tensor, step)

    self._summary_tuples.append((_summary_fn, tensor))

  def scalar(self, name, tensor, family=None, description=None):

    def _summary_fn(name, tensor, step):
      return self._actual_summary_scalar_fn(
          name=name, data=tensor, description=description, step=step)

    self._create_summary(_summary_fn, name,
                         tf.reshape(tf.convert_to_tensor(value=tensor), []))

  def image(self, name, tensor, max_outputs=3, family=None, description=None):

    def _summary_fn(name, tensor, step):
      return self._actual_summary_image_fn(
          name=name,
          data=tensor,
          max_outputs=max_outputs,
          description=description,
          step=step)

    self._create_summary(_summary_fn, name, tf.cast(tensor, tf.float32))

  def histogram(self,
                name,
                values,
                family=None,
                buckets=None,
                description=None):

    def _summary_fn(name, tensor, step):
      return self._actual_summary_histogram_fn(
          name=name,
          data=tensor,
          buckets=buckets,
          description=description,
          step=step)

    self._create_summary(_summary_fn, name, tf.convert_to_tensor(value=values))

  def audio(self,
            name,
            tensor,
            sample_rate,
            max_outputs=3,
            family=None,
            encoding=None,
            description=None):

    def _summary_fn(name, tensor, step):
      return self._actual_summary_audio_fn(
          name=name,
          data=tensor,
          sample_rate=sample_rate,
          encoding=encoding,
          description=description,
          step=step)

    self._create_summary(_summary_fn, name, tf.cast(tensor, tf.float32))

  def summary_tuples(self):
    """Returns an iterable of functions that convert a Tensor to a summary.

    Used for TPU host calls.

    Returns:
      Iterable of functions that take a single `Tensor` argument.
    """
    return tuple(self._summary_tuples)

  def clear_summary_tuples(self):
    """Clears the list of current summary tuples."""

    self._summary_tuples = []


class _TPUScopedSummary(_ScopedSummaryV2):
  """Records summaries in a given scope.

  Only for TPUEstimator.

  Each scope gets assigned a different collection where summary ops gets added.

  This allows Tensorboard to display summaries with different scopes but the
  same name in the same charts.
  """

  def __init__(self, logdir, namespace=None, scope=None, skip_summary=False):
    super(_TPUScopedSummary, self).__init__(logdir, namespace, scope,
                                            skip_summary)
    from tensorflow.python.ops import summary_ops_v2 as summary_v2_lib  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

    self._actual_summary_scalar_fn = summary_v2_lib.scalar
    self._actual_summary_image_fn = summary_v2_lib.image
    self._actual_summary_histogram_fn = summary_v2_lib.histogram
    self._actual_summary_audio_fn = summary_v2_lib.audio

  def scalar(self, name, tensor, family=None):

    def _summary_fn(name, tensor, step):
      return self._actual_summary_scalar_fn(
          name=name, tensor=tensor, family=family, step=step)

    self._create_summary(_summary_fn, name,
                         tf.reshape(tf.convert_to_tensor(value=tensor), [1]))

  def image(self, name, tensor, max_outputs=3, family=None):

    def _summary_fn(name, tensor, step):
      return self._actual_summary_image_fn(
          name=name,
          tensor=tensor,
          max_images=max_outputs,
          family=family,
          step=step)

    self._create_summary(_summary_fn, name, tf.cast(tensor, tf.float32))

  def histogram(self, name, values, family=None):

    def _summary_fn(name, tensor, step):
      return self._actual_summary_histogram_fn(
          name=name, tensor=tensor, family=family, step=step)

    self._create_summary(_summary_fn, name, tf.convert_to_tensor(value=values))

  def audio(self, name, tensor, sample_rate, max_outputs=3, family=None):

    def _summary_fn(name, tensor, step):
      return self._actual_summary_audio_fn(
          name=name,
          tensor=tensor,
          sample_rate=sample_rate,
          max_outputs=max_outputs,
          family=family,
          step=step)

    self._create_summary(_summary_fn, name, tf.cast(tensor, tf.float32))


class _SummaryWrapper(object):
  """Wraps an `adanet.Summary` to provide summary-like APIs."""

  def __init__(self, summary):
    self._summary = summary

  def scalar(self, name, tensor, collections=None, family=None):
    """See `tf.summary.scalar`."""

    if collections is not None:
      logging.warning(
          "The `collections` argument will be "
          "ignored for scalar summary: %s, %s", name, tensor)
    return self._summary.scalar(name=name, tensor=tensor, family=family)

  def image(self, name, tensor, max_outputs=3, collections=None, family=None):
    """See `tf.summary.image`."""

    if collections is not None:
      logging.warning(
          "The `collections` argument will be "
          "ignored for image summary: %s, %s", name, tensor)
    return self._summary.image(
        name=name, tensor=tensor, max_outputs=max_outputs, family=family)

  def histogram(self, name, values, collections=None, family=None):
    """See `tf.summary.histogram`."""

    if collections is not None:
      logging.warning(
          "The `collections` argument will be "
          "ignored for histogram summary: %s, %s", name, values)
    return self._summary.histogram(name=name, values=values, family=family)

  def audio(self,
            name,
            tensor,
            sample_rate,
            max_outputs=3,
            collections=None,
            family=None):
    """See `tf.summary.audio`."""

    if collections is not None:
      logging.warning(
          "The `collections` argument will be "
          "ignored for audio summary: %s, %s", name, tensor)
    return self._summary.audio(
        name=name,
        tensor=tensor,
        sample_rate=sample_rate,
        max_outputs=max_outputs,
        family=family)

  def scalar_v2(self, name, tensor, family=None, step=None):
    """See `tf.contrib.summary.scalar`."""

    if step is not None:
      logging.warning(
          "The `step` argument will be ignored to use the global step for "
          "scalar summary: %s, %s", name, tensor)
    return self._summary.scalar(name=name, tensor=tensor, family=family)

  def image_v2(self,
               name,
               tensor,
               bad_color=None,
               max_images=3,
               family=None,
               step=None):
    """See `tf.contrib.summary.image`."""

    if step is not None:
      logging.warning(
          "The `step` argument will be ignored to use the global step for "
          "image summary: %s, %s", name, tensor)
    # TODO: Add support for `bad_color` arg.
    if bad_color is not None:
      logging.warning(
          "The `bad_color` arg is not supported for image summary: %s, %s",
          name, tensor)
    return self._summary.image(
        name=name, tensor=tensor, max_outputs=max_images, family=family)

  def histogram_v2(self, name, tensor, family=None, step=None):
    """See `tf.contrib.summary.histogram`."""

    if step is not None:
      logging.warning(
          "The `step` argument will be ignored to use the global step for "
          "histogram summary: %s, %s", name, tensor)
    return self._summary.histogram(name=name, values=tensor, family=family)

  def audio_v2(self,
               name,
               tensor,
               sample_rate,
               max_outputs,
               family=None,
               step=None):
    """See `tf.contrib.summary.audio`."""

    if step is not None:
      logging.warning(
          "The `step` argument will be ignored to use the global step for "
          "audio summary: %s, %s", name, tensor)
    return self._summary.audio(
        name=name,
        tensor=tensor,
        sample_rate=sample_rate,
        max_outputs=max_outputs,
        family=family)

  def scalar_v3(self, name, data, step=None, description=None):
    """See `tf.compat.v2.summary.scalar`."""

    if step is not None:
      logging.warning(
          "The `step` argument will be ignored to use the iteration step for "
          "scalar summary: %s", name)
    return self._summary.scalar(name=name, tensor=data, description=description)

  def image_v3(self, name, data, step=None, max_outputs=3, description=None):
    """See `tf.compat.v2.summary.image`."""

    if step is not None:
      logging.warning(
          "The `step` argument will be ignored to use the iteration step for "
          "image summary: %s", name)
    return self._summary.image(
        name=name,
        tensor=data,
        max_outputs=max_outputs,
        description=description)

  def histogram_v3(self, name, data, step=None, buckets=None, description=None):
    """See `tf.compat.v2.summary.histogram`."""

    if step is not None:
      logging.warning(
          "The `step` argument will be ignored to use the global step for "
          "histogram summary: %s", name)
    return self._summary.histogram(
        name=name, tensor=data, buckets=buckets, description=description)

  def audio_v3(self,
               name,
               data,
               sample_rate,
               step=None,
               max_outputs=3,
               encoding=None,
               description=None):
    """See `tf.compat.v2.summary.audio`."""

    if step is not None:
      logging.warning(
          "The `step` argument will be ignored to use the global step for "
          "audio summary: %s", name)
    return self._summary.audio(
        name=name,
        tensor=data,
        sample_rate=sample_rate,
        max_outputs=max_outputs,
        encoding=encoding,
        description=description)


@contextlib.contextmanager
def monkey_patched_summaries(summary):
  """A context where global summary functions point to the given summary.

  Restores original summary functions upon exit.

  NOTE: This function is not thread-safe.

  Args:
    summary: An `adanet.Summary` instance.

  Yields:
    A context where summary functions are routed to the given `adanet.Summary`.
  """

  from tensorflow.python.ops import summary_ops_v2 as summary_v2_lib  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

  old_summary_scalar = summary_lib.scalar
  old_summary_image = summary_lib.image
  old_summary_histogram = summary_lib.histogram
  old_summary_audio = summary_lib.audio
  old_summary_v2_scalar = summary_v2_lib.scalar
  old_summary_v2_image = summary_v2_lib.image
  old_summary_v2_histogram = summary_v2_lib.histogram
  old_summary_v2_audio = summary_v2_lib.audio
  old_summary_compat_v2_scalar = tf_compat.v2.summary.scalar
  old_summary_compat_v2_image = tf_compat.v2.summary.image
  old_summary_compat_v2_histogram = tf_compat.v2.summary.histogram
  old_summary_compat_v2_audio = tf_compat.v2.summary.audio

  # Monkey-patch global attributes.
  wrapped_summary = _SummaryWrapper(summary)
  setattr(tf_v1.summary, "scalar", wrapped_summary.scalar)
  setattr(tf_v1.summary, "image", wrapped_summary.image)
  setattr(tf_v1.summary, "histogram", wrapped_summary.histogram)
  setattr(tf_v1.summary, "audio", wrapped_summary.audio)
  setattr(tf_compat.v1.summary, "scalar", wrapped_summary.scalar)
  setattr(tf_compat.v1.summary, "image", wrapped_summary.image)
  setattr(tf_compat.v1.summary, "histogram", wrapped_summary.histogram)
  setattr(tf_compat.v1.summary, "audio", wrapped_summary.audio)
  setattr(summary_lib, "scalar", wrapped_summary.scalar)
  setattr(summary_lib, "image", wrapped_summary.image)
  setattr(summary_lib, "histogram", wrapped_summary.histogram)
  setattr(summary_lib, "audio", wrapped_summary.audio)
  setattr(tf_compat.v2.summary, "scalar", wrapped_summary.scalar_v3)
  setattr(tf_compat.v2.summary, "image", wrapped_summary.image_v3)
  setattr(tf_compat.v2.summary, "histogram", wrapped_summary.histogram_v3)
  setattr(tf_compat.v2.summary, "audio", wrapped_summary.audio_v3)
  setattr(summary_v2_lib, "scalar", wrapped_summary.scalar_v2)
  setattr(summary_v2_lib, "image", wrapped_summary.image_v2)
  setattr(summary_v2_lib, "histogram", wrapped_summary.histogram_v2)
  setattr(summary_v2_lib, "audio", wrapped_summary.audio_v2)
  try:
    # TF 2.0 eliminates tf.contrib.
    setattr(tf_v1.contrib.summary, "scalar", wrapped_summary.scalar_v2)
    setattr(tf_v1.contrib.summary, "image", wrapped_summary.image_v2)
    setattr(tf_v1.contrib.summary, "histogram", wrapped_summary.histogram_v2)
    setattr(tf_v1.contrib.summary, "audio", wrapped_summary.audio_v2)
  except (AttributeError, ImportError):
    # TF 2.0 eliminates tf.contrib.
    # Also set the new tf.summary to be use the new summaries in TF 2.
    if tf_compat.version_greater_or_equal("2.0.0"):
      setattr(tf.summary, "scalar", wrapped_summary.scalar_v3)
      setattr(tf.summary, "image", wrapped_summary.image_v3)
      setattr(tf.summary, "histogram", wrapped_summary.histogram_v3)
      setattr(tf.summary, "audio", wrapped_summary.audio_v3)

  try:
    yield
  finally:
    # Revert monkey-patches.
    try:
      setattr(tf_v1.contrib.summary, "audio", old_summary_v2_audio)
      setattr(tf_v1.contrib.summary, "histogram", old_summary_v2_histogram)
      setattr(tf_v1.contrib.summary, "image", old_summary_v2_image)
      setattr(tf_v1.contrib.summary, "scalar", old_summary_v2_scalar)
    except (AttributeError, ImportError):
      # TF 2.0 eliminates tf.contrib.
      pass
    setattr(summary_v2_lib, "audio", old_summary_v2_audio)
    setattr(summary_v2_lib, "histogram", old_summary_v2_histogram)
    setattr(summary_v2_lib, "image", old_summary_v2_image)
    setattr(summary_v2_lib, "scalar", old_summary_v2_scalar)
    setattr(tf.summary, "audio", old_summary_compat_v2_audio)
    setattr(tf.summary, "histogram", old_summary_compat_v2_histogram)
    setattr(tf.summary, "image", old_summary_compat_v2_image)
    setattr(tf.summary, "scalar", old_summary_compat_v2_scalar)
    setattr(tf_compat.v2.summary, "audio", old_summary_compat_v2_audio)
    setattr(tf_compat.v2.summary, "histogram", old_summary_compat_v2_histogram)
    setattr(tf_compat.v2.summary, "image", old_summary_compat_v2_image)
    setattr(tf_compat.v2.summary, "scalar", old_summary_compat_v2_scalar)
    setattr(summary_lib, "audio", old_summary_audio)
    setattr(summary_lib, "histogram", old_summary_histogram)
    setattr(summary_lib, "image", old_summary_image)
    setattr(summary_lib, "scalar", old_summary_scalar)
    setattr(tf_compat.v1.summary, "audio", old_summary_audio)
    setattr(tf_compat.v1.summary, "histogram", old_summary_histogram)
    setattr(tf_compat.v1.summary, "image", old_summary_image)
    setattr(tf_compat.v1.summary, "scalar", old_summary_scalar)
    setattr(tf_v1.summary, "audio", old_summary_audio)
    setattr(tf_v1.summary, "histogram", old_summary_histogram)
    setattr(tf_v1.summary, "image", old_summary_image)
    setattr(tf_v1.summary, "scalar", old_summary_scalar)
