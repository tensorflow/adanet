"""Test utilities for AdaNet single graph implementation.

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
import shutil
import struct
import sys

from absl import flags
from absl.testing import parameterized
from adanet import tf_compat
from adanet.core.architecture import _Architecture
from adanet.core.candidate import _Candidate
from adanet.core.ensemble_builder import _EnsembleSpec
from adanet.core.ensemble_builder import _SubnetworkSpec
from adanet.core.eval_metrics import _EnsembleMetrics
from adanet.core.eval_metrics import _IterationMetrics
from adanet.core.eval_metrics import _SubnetworkMetrics
from adanet.ensemble import ComplexityRegularized
from adanet.ensemble import WeightedSubnetwork
from adanet.subnetwork import Subnetwork
import tensorflow as tf
from tensorflow_estimator.python.estimator.head import regression_head


def dummy_tensor(shape=(), random_seed=42):
  """Returns a randomly initialized tensor."""

  return tf.Variable(
      tf_compat.random_normal(shape=shape, seed=random_seed),
      trainable=False).read_value()


class ExportOutputKeys(object):
  """Different export output keys for the dummy ensemble builder."""

  CLASSIFICATION_CLASSES = "classification_classes"
  CLASSIFICATION_SCORES = "classification_scores"
  REGRESSION = "regression"
  PREDICTION = "prediction"
  INVALID = "invalid"


def dummy_ensemble_spec(name,
                        random_seed=42,
                        num_subnetworks=1,
                        bias=0.,
                        loss=None,
                        adanet_loss=None,
                        eval_metrics=None,
                        dict_predictions=False,
                        export_output_key=None,
                        subnetwork_builders=None,
                        train_op=None):
  """Creates a dummy `_EnsembleSpec` instance.

  Args:
    name: _EnsembleSpec's name.
    random_seed: A scalar random seed.
    num_subnetworks: The number of fake subnetworks in this ensemble.
    bias: Bias value.
    loss: Float loss to return. When None, it's picked from a random
      distribution.
    adanet_loss: Float AdaNet loss to return. When None, it's picked from a
      random distribution.
    eval_metrics: Optional eval metrics tuple of (metric_fn, tensor args).
    dict_predictions: Boolean whether to return predictions as a dictionary of
      `Tensor` or just a single float `Tensor`.
    export_output_key: An `ExportOutputKeys` for faking export outputs.
    subnetwork_builders: List of `adanet.subnetwork.Builder` objects.
    train_op: A train op.

  Returns:
    A dummy `_EnsembleSpec` instance.
  """

  if loss is None:
    loss = dummy_tensor([], random_seed)

  if adanet_loss is None:
    adanet_loss = dummy_tensor([], random_seed * 2)
  else:
    adanet_loss = tf.convert_to_tensor(value=adanet_loss)

  logits = dummy_tensor([], random_seed * 3)
  if dict_predictions:
    predictions = {
        "logits": logits,
        "classes": tf.cast(tf.abs(logits), dtype=tf.int64)
    }
  else:
    predictions = logits
  weighted_subnetworks = [
      WeightedSubnetwork(
          name=name,
          iteration_number=1,
          logits=dummy_tensor([2, 1], random_seed * 4),
          weight=dummy_tensor([2, 1], random_seed * 4),
          subnetwork=Subnetwork(
              last_layer=dummy_tensor([1, 2], random_seed * 4),
              logits=dummy_tensor([2, 1], random_seed * 4),
              complexity=1.,
              persisted_tensors={}))
  ]

  export_outputs = _dummy_export_outputs(export_output_key, logits, predictions)
  bias = tf.constant(bias)
  return _EnsembleSpec(
      name=name,
      ensemble=ComplexityRegularized(
          weighted_subnetworks=weighted_subnetworks * num_subnetworks,
          bias=bias,
          logits=logits,
      ),
      architecture=_Architecture("dummy_ensemble_candidate", "dummy_ensembler"),
      subnetwork_builders=subnetwork_builders,
      predictions=predictions,
      step=tf.Variable(0),
      loss=loss,
      adanet_loss=adanet_loss,
      train_op=train_op,
      eval_metrics=eval_metrics,
      export_outputs=export_outputs)


def _dummy_export_outputs(export_output_key, logits, predictions):
  """Returns a dummy export output dictionary for the given key."""

  export_outputs = None
  if export_output_key == ExportOutputKeys.CLASSIFICATION_CLASSES:
    export_outputs = {
        export_output_key:
            tf.estimator.export.ClassificationOutput(
                classes=tf.as_string(logits))
    }
  elif export_output_key == ExportOutputKeys.CLASSIFICATION_SCORES:
    export_outputs = {
        export_output_key:
            tf.estimator.export.ClassificationOutput(scores=logits)
    }
  elif export_output_key == ExportOutputKeys.REGRESSION:
    export_outputs = {
        export_output_key: tf.estimator.export.RegressionOutput(value=logits)
    }
  elif export_output_key == ExportOutputKeys.PREDICTION:
    export_outputs = {
        export_output_key:
            tf.estimator.export.PredictOutput(outputs=predictions)
    }
  elif export_output_key == ExportOutputKeys.INVALID:
    export_outputs = {export_output_key: predictions}
  return export_outputs


def dummy_estimator_spec(loss=None, random_seed=42, eval_metric_ops=None):
  """Creates a dummy `EstimatorSpec` instance.

  Args:
    loss: Float loss to return. When None, it's picked from a random
      distribution.
    random_seed: Scalar seed for random number generators.
    eval_metric_ops: Optional dictionary of metric ops.

  Returns:
    A `EstimatorSpec` instance.
  """

  if loss is None:
    loss = dummy_tensor([], random_seed)
  predictions = dummy_tensor([], random_seed * 2)
  return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.TRAIN,
      predictions=predictions,
      loss=loss,
      # Train_op cannot be tf.no_op() for Estimator, because in eager mode
      # tf.no_op() returns None.
      train_op=tf.constant(0.),
      eval_metric_ops=eval_metric_ops)


def dummy_input_fn(features, labels):
  """Returns an input_fn that returns feature and labels `Tensors`."""

  def _input_fn(params=None):
    del params  # Unused.

    input_features = {"x": tf.constant(features, name="x")}
    input_labels = tf.constant(labels, name="y")
    return input_features, input_labels

  return _input_fn


def dataset_input_fn(features=8., labels=9.):
  """Returns feature and label `Tensors` via a `Dataset`."""

  def _input_fn(params=None):
    """The `Dataset` input_fn which will be returned."""

    del params  # Unused.

    input_features = tf_compat.make_one_shot_iterator(
        tf.data.Dataset.from_tensors([features])).get_next()
    if labels is not None:
      input_labels = tf_compat.make_one_shot_iterator(
          tf.data.Dataset.from_tensors([labels])).get_next()
    else:
      input_labels = None
    return {"x": input_features}, input_labels

  return _input_fn


def head():
  return regression_head.RegressionHead(
      loss_reduction=tf_compat.SUM_OVER_BATCH_SIZE)


class ModifierSessionRunHook(tf_compat.SessionRunHook):
  """Modifies the graph by adding a variable."""

  def __init__(self, var_name="hook_created_variable"):
    self._var_name = var_name
    self._begun = False

  def begin(self):
    """Adds a variable to the graph.

    Raises:
      ValueError: If we've already begun a run.
    """

    if self._begun:
      raise ValueError("begin called twice without end.")
    self._begun = True
    _ = tf_compat.v1.get_variable(name=self._var_name, initializer="")

  def end(self, session):
    """Adds a variable to the graph.

    Args:
      session: A `tf.Session` object that can be used to run ops.

    Raises:
      ValueError: If we've not begun a run.
    """

    _ = session
    if not self._begun:
      raise ValueError("end called without begin.")
    self._begun = False


class AdanetTestCase(parameterized.TestCase, tf.test.TestCase):
  """A parameterized `TestCase` that manages a test subdirectory."""

  def setUp(self):
    super(AdanetTestCase, self).setUp()
    # Setup and cleanup test directory.
    # Flags are not automatically parsed at this point.
    flags.FLAGS(sys.argv)
    self.test_subdirectory = os.path.join(flags.FLAGS.test_tmpdir, self.id())
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.makedirs(self.test_subdirectory)

  def tearDown(self):
    super(AdanetTestCase, self).tearDown()
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)


def summary_simple_value(summary_value):
  """Returns the scalar parsed from the summary proto tensor_value bytes."""

  return struct.unpack("<f", summary_value.tensor.tensor_content)[0]


def check_eventfile_for_keyword(keyword, dir_):
  """Checks event files for the keyword."""

  tf_compat.v1.summary.FileWriterCache.clear()

  if not tf.io.gfile.exists(dir_):
    raise ValueError("Directory '{}' not found.".format(dir_))

  # Get last `Event` written.
  filenames = os.path.join(dir_, "events*")
  event_paths = tf.io.gfile.glob(filenames)
  if not event_paths:
    raise ValueError("Path '{}' not found.".format(filenames))

  for event_path in event_paths:
    for last_event in tf_compat.v1.train.summary_iterator(event_path):
      if last_event.summary is not None:
        for value in last_event.summary.value:
          if keyword == value.tag:
            if value.HasField("simple_value"):
              return value.simple_value
            if value.HasField("image"):
              return (value.image.height, value.image.width,
                      value.image.colorspace)
            if value.HasField("tensor"):
              if value.metadata.plugin_data.plugin_name == "scalars":
                return summary_simple_value(value)
              if value.metadata.plugin_data.plugin_name == "images":
                return (int(value.tensor.string_val[0]),
                        int(value.tensor.string_val[1]), 1)
              if value.tensor.string_val is not None:
                return value.tensor.string_val

  raise ValueError("Keyword '{}' not found in path '{}'.".format(
      keyword, filenames))


def create_ensemble_metrics(metric_fn,
                            use_tpu=False,
                            features=None,
                            labels=None,
                            estimator_spec=None,
                            architecture=None):
  """Creates an instance of the _EnsembleMetrics class.

  Args:
    metric_fn: A function which should obey the following signature:
    - Args: can only have following three arguments in any order:
        * predictions: Predictions `Tensor` or dict of `Tensor` created by given
          `Head`.
        * features: Input `dict` of `Tensor` objects created by `input_fn` which
          is given to `estimator.evaluate` as an argument.
        * labels:  Labels `Tensor` or dict of `Tensor` (for multi-head) created
          by `input_fn` which is given to `estimator.evaluate` as an argument.
      - Returns: Dict of metric results keyed by name. Final metrics are a union
        of this and `estimator`s existing metrics. If there is a name conflict
        between this and `estimator`s existing metrics, this will override the
        existing one. The values of the dict are the results of calling a metric
        function, namely a `(metric_tensor, update_op)` tuple.
    use_tpu: Whether to use TPU-specific variable sharing logic.
    features: Input `dict` of `Tensor` objects.
    labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
      (for multi-head).
    estimator_spec: The `EstimatorSpec` created by a `Head` instance.
    architecture: `_Architecture` object.

  Returns:
    An instance of _EnsembleMetrics.
  """

  if not estimator_spec:
    estimator_spec = tf_compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=tf.constant(2.),
        predictions=None,
        eval_metrics=None)
    if not use_tpu:
      estimator_spec = estimator_spec.as_estimator_spec()

  if not architecture:
    architecture = _Architecture(None, None)

  metrics = _EnsembleMetrics(use_tpu=use_tpu)
  metrics.create_eval_metrics(features, labels, estimator_spec, metric_fn,
                              architecture)

  return metrics


def create_subnetwork_metrics(metric_fn,
                              use_tpu=False,
                              features=None,
                              labels=None,
                              estimator_spec=None):
  """Creates an instance of the _SubnetworkMetrics class.

  Args:
    metric_fn: A function which should obey the following signature:
    - Args: can only have following three arguments in any order:
        * predictions: Predictions `Tensor` or dict of `Tensor` created by given
          `Head`.
        * features: Input `dict` of `Tensor` objects created by `input_fn` which
          is given to `estimator.evaluate` as an argument.
        * labels:  Labels `Tensor` or dict of `Tensor` (for multi-head) created
          by `input_fn` which is given to `estimator.evaluate` as an argument.
      - Returns: Dict of metric results keyed by name. Final metrics are a union
        of this and `estimator`s existing metrics. If there is a name conflict
        between this and `estimator`s existing metrics, this will override the
        existing one. The values of the dict are the results of calling a metric
        function, namely a `(metric_tensor, update_op)` tuple.
    use_tpu: Whether to use TPU-specific variable sharing logic.
    features: Input `dict` of `Tensor` objects.
    labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
      (for multi-head).
    estimator_spec: The `EstimatorSpec` created by a `Head` instance.

  Returns:
    An instance of _SubnetworkMetrics.
  """

  if not estimator_spec:
    estimator_spec = tf_compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=tf.constant(2.),
        predictions=None,
        eval_metrics=None)
    if not use_tpu:
      estimator_spec = estimator_spec.as_estimator_spec()

  metrics = _SubnetworkMetrics(use_tpu=use_tpu)
  metrics.create_eval_metrics(features, labels, estimator_spec, metric_fn)

  return metrics


def create_iteration_metrics(subnetwork_metrics=None,
                             ensemble_metrics=None,
                             use_tpu=False,
                             iteration_number=1):
  """Creates an instance of the _IterationMetrics class.

  Args:
    subnetwork_metrics: List of _SubnetworkMetrics objects.
    ensemble_metrics: List of _EnsembleMetrics objects.
    use_tpu: Whether to use TPU-specific variable sharing logic.
    iteration_number: What number iteration these metrics are for.

  Returns:
    An instance of _IterationMetrics that has been populated with the
    input metrics.
  """
  subnetwork_metrics = subnetwork_metrics or []
  ensemble_metrics = ensemble_metrics or []

  candidates = []
  for i, metric in enumerate(ensemble_metrics):
    spec = _EnsembleSpec(
        name="ensemble_{}".format(i),
        ensemble=None,
        architecture=None,
        subnetwork_builders=None,
        predictions=None,
        step=None,
        eval_metrics=metric)

    candidate = _Candidate(ensemble_spec=spec, adanet_loss=tf.constant(i))
    candidates.append(candidate)

  subnetwork_specs = []
  for i, metric in enumerate(subnetwork_metrics):
    spec = _SubnetworkSpec(
        name="subnetwork_{}".format(i),
        subnetwork=None,
        builder=None,
        predictions=None,
        step=None,
        loss=None,
        train_op=None,
        asset_dir=None,
        eval_metrics=metric)
    subnetwork_specs.append(spec)

  return _IterationMetrics(
      iteration_number,
      candidates,
      subnetwork_specs=subnetwork_specs,
      use_tpu=use_tpu)
