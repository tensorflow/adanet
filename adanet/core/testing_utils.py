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

from adanet.core.ensemble import _EnsembleSpec
from adanet.core.ensemble import Ensemble
from adanet.core.ensemble import WeightedSubnetwork
from adanet.core.subnetwork import Subnetwork
import tensorflow as tf


def dummy_tensor(shape=(), random_seed=42):
  """Returns a randomly initialized tensor."""

  return tf.Variable(
      tf.random_normal(shape=shape, seed=random_seed),
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
                        complexity_regularized_loss=None,
                        eval_metric_ops=None,
                        dict_predictions=False,
                        export_output_key=None,
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
    complexity_regularized_loss: Float complexity regularized loss to return.
      When None, it's picked from a random distribution.
    eval_metric_ops: Optional dictionary of metric ops.
    dict_predictions: Boolean whether to return predictions as a dictionary of
      `Tensor` or just a single float `Tensor`.
    export_output_key: An `ExportOutputKeys` for faking export outputs.
    train_op: A train op.

  Returns:
    A dummy `_EnsembleSpec` instance.
  """

  if loss is None:
    loss = dummy_tensor([], random_seed)
  elif not isinstance(loss, tf.Tensor):
    loss = tf.constant(loss)

  if adanet_loss is None:
    adanet_loss = dummy_tensor([], random_seed * 2)
  else:
    adanet_loss = tf.convert_to_tensor(adanet_loss)

  if complexity_regularized_loss is None:
    complexity_regularized_loss = dummy_tensor([], random_seed * 2)
  elif not isinstance(complexity_regularized_loss, tf.Tensor):
    complexity_regularized_loss = tf.constant(complexity_regularized_loss)

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
          name=tf.constant(name),
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
      ensemble=Ensemble(
          weighted_subnetworks=weighted_subnetworks * num_subnetworks,
          bias=bias,
          logits=logits,
      ),
      predictions=predictions,
      loss=loss,
      adanet_loss=adanet_loss,
      complexity_regularized_loss=complexity_regularized_loss,
      complexity_regularization=1,
      eval_metric_ops=eval_metric_ops,
      train_op=train_op,
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


def dummy_estimator_spec(loss=None,
                         random_seed=42,
                         dict_predictions=False,
                         eval_metric_ops=None):
  """Creates a dummy `EstimatorSpec` instance.

  Args:
    loss: Float loss to return. When None, it's picked from a random
      distribution.
    random_seed: Scalar seed for random number generators.
    dict_predictions: Boolean whether to return predictions as a dictionary of
      `Tensor` or just a single float `Tensor`.
    eval_metric_ops: Optional dictionary of metric ops.

  Returns:
    A `EstimatorSpec` instance.
  """

  if loss is None:
    loss = dummy_tensor([], random_seed)
  elif not isinstance(loss, tf.Tensor):
    loss = tf.constant(loss)
  predictions = dummy_tensor([], random_seed * 2)
  if dict_predictions:
    predictions = {
        "logits": predictions,
        "classes": tf.cast(tf.abs(predictions), dtype=tf.int64)
    }
  return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.TRAIN,
      predictions=predictions,
      loss=loss,
      train_op=tf.no_op(),
      eval_metric_ops=eval_metric_ops)


def dummy_input_fn(features, labels):
  """Returns an input_fn that returns feature and labels `Tensors`."""

  def _input_fn():
    input_features = {"x": tf.constant(features, name="x")}
    input_labels = tf.constant(labels, name="y")
    return input_features, input_labels

  return _input_fn


def dataset_input_fn(features=8., labels=9.):
  """Returns feature and label `Tensors` via a `Dataset`."""

  def _input_fn():
    input_features = tf.data.Dataset.from_tensors(
        [features]).make_one_shot_iterator().get_next()
    if labels is not None:
      input_labels = tf.data.Dataset.from_tensors(
          [labels]).make_one_shot_iterator().get_next()
    else:
      input_labels = None
    return {"x": input_features}, input_labels

  return _input_fn


class FakeSparseTensor(object):
  """A fake SparseTensor."""

  def __init__(self, indices, values, dense_shape):
    self.indices = indices
    self.values = values
    self.dense_shape = dense_shape


class FakePlaceholder(object):
  """A fake Placeholder."""

  def __init__(self, dtype, shape=None):
    self.dtype = dtype
    self.shape = shape


class FakeSparsePlaceholder(object):
  """A fake SparsePlaceholder."""

  def __init__(self, dtype, shape=None):
    self.dtype = dtype
    self.shape = shape


def tensor_features(features):
  """Returns features as tensors, replacing Fakes."""

  result = {}
  for key, feature in features.items():
    if isinstance(feature, FakeSparseTensor):
      feature = tf.SparseTensor(
          indices=feature.indices,
          values=feature.values,
          dense_shape=feature.dense_shape)
    elif isinstance(feature, FakeSparsePlaceholder):
      feature = tf.sparse_placeholder(dtype=feature.dtype)
    elif isinstance(feature, FakePlaceholder):
      feature = tf.placeholder(dtype=feature.dtype)
    else:
      feature = tf.convert_to_tensor(feature)
    result[key] = feature
  return result
