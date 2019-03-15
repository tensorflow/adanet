"""AdaNet metrics objects and functions.

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
import inspect

import six
import tensorflow as tf


def call_eval_metrics(eval_metrics):
  if not eval_metrics:
    return {}
  fn, kwargs = eval_metrics
  return fn(**kwargs)


def _flatten_dict(original_dict, delimiter="/"):
  """Flattens a dictionary of dictionaries by one level.

  Note that top level keys will be overridden if they collide with flat keys.
  E.g. using delimiter="/" and origial_dict={"foo/bar": 1, "foo": {"bar": 2}},
  the top level "foo/bar" key would be overwritten.

  Args:
    original_dict: The dictionary to flatten.
    delimiter: The value used to delimit the keys in the flat_dict.

  Returns:
    The falttened dictionary.
  """

  flat_dict = {}
  for outer_key, inner_dict in six.iteritems(original_dict):
    if isinstance(inner_dict, dict):
      for inner_key, value in six.iteritems(inner_dict):
        flat_dict["{}{}{}".format(outer_key, delimiter, inner_key)] = value
    else:
      flat_dict[outer_key] = inner_dict
  return flat_dict


def _unflatten_dict(flat_dict, prefixes, delimiter="/"):
  """Unflattens a dictionary into a dict of dicts by one level.

  Args:
    flat_dict: The dictionary to unflatten.
    prefixes: The string keys to use for the unflattened dictionary. Keys in the
      flat_dict which do not begin with a prefix are unmodified.
    delimiter: The value used to delmit the keys in the flat_dict.

  Returns:
    The unflattened dictionary.
  """

  unflat_dict = collections.defaultdict(dict)
  for key, value in six.iteritems(flat_dict):
    parts = key.split(delimiter)
    if len(parts) > 1:
      prefix = parts[0]
      if prefix in prefixes:
        suffix = key[len(prefix + delimiter):]
        unflat_dict[prefix][suffix] = value
      else:
        unflat_dict[key] = value
    else:
      unflat_dict[key] = value
  return unflat_dict


class _SubnetworkMetrics(object):
  """A object which creates evaluation metrics for Subnetworks."""

  class _Keys(object):
    KWARGS = "__kwargs__"
    FEATURES = "__features__"
    LABELS = "__labels__"
    PREDICTIONS = "__predictions__"

    ALL = (KWARGS, FEATURES, LABELS, PREDICTIONS)

  def __init__(self):
    self._metric_fns = []
    self._kwargs = {}

  def create_eval_metrics(self, features, labels, estimator_spec, metric_fn):
    """Creates evaluation metrics from the given arguments.

    Args:
      features: Input `dict` of `Tensor` objects.
      labels: Labels `Tensor` or a dictionary of string label name to `Tensor`
        (for multi-head).
      estimator_spec: The `EstimatorSpec` created by a `Head` instance.
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
    """

    # If estimator_spec is not a TPUEstimatorSpec we create dummy eval_metric_fn
    # and tensors.
    if isinstance(estimator_spec, tf.estimator.EstimatorSpec):
      dummy_metric_fn = lambda: estimator_spec.eval_metric_ops
      self._metric_fns.append(self._templatize_metric_fn(dummy_metric_fn))
    else:
      fn, kwargs = estimator_spec.eval_metrics
      self._metric_fns.append(self._templatize_metric_fn(fn))
      self._kwargs.update(self._flatten(kwargs, self._Keys.KWARGS, None))

    loss_fn = lambda loss: {"loss": tf.metrics.mean(loss)}
    self._metric_fns.append(self._templatize_metric_fn(loss_fn))
    self._kwargs.update({"loss": tf.reshape(estimator_spec.loss, [1])})

    # NOTE: the user supplied metrics_fn must be added last. This is because we
    # want user metrics to override AdaNet's metrics.
    if metric_fn:
      self._metric_fns.append(self._templatize_metric_fn(metric_fn))
      self._kwargs.update(
          self._flatten(features, self._Keys.FEATURES, "features"))
      self._kwargs.update(self._flatten(labels, self._Keys.LABELS, "labels"))
      self._kwargs.update(
          self._flatten(estimator_spec.predictions, self._Keys.PREDICTIONS,
                        "predictions"))

  def _templatize_metric_fn(self, metric_fn):
    """Wraps the given metric_fn with a template so it's Variables are shared.

    Hooks on TPU cannot depend on any graph Tensors. Instead the eval metrics
    returned by metric_fn are stored in Variables. These variables are later
    read from the evaluation hooks which run on the host CPU.

    Args:
      metric_fn: The function to wrap with a template.

    Returns:
      The original metric_fn wrapped with a template function.
    """

    def _metric_fn(**kwargs):
      """The wrapping function to be returned."""

      # Extract the args applicable to metric_fn from kwargs.
      argspec = inspect.getargspec(metric_fn)
      args = {k: v for k, v in six.iteritems(kwargs) if k in argspec.args}
      if argspec.keywords:
        args.update(kwargs[self._Keys.KWARGS])

      metrics = metric_fn(**args)

      wrapped_metrics = {}
      for i, key in enumerate(sorted(metrics)):
        tensor, op = metrics[key]
        # key cannot be in var name since it may contain illegal chars.
        var = tf.get_variable(
            "metric_{}".format(i),
            shape=tensor.shape,
            dtype=tensor.dtype,
            trainable=False,
            initializer=tf.zeros_initializer(),
            collections=[tf.GraphKeys.LOCAL_VARIABLES])
        if isinstance(op, tf.Operation):
          with tf.control_dependencies([op]):
            op = tf.assign(var, tensor)
        metric = (var, tf.assign(var, op))
        wrapped_metrics[key] = metric
      return wrapped_metrics

    return tf.make_template("metric_fn_template", _metric_fn)

  def _flatten(self, tensors, flat_key, default_key=None):
    """Flattens and prefixes tensors by flat_key (if dict) or default_key."""

    prefix = default_key
    if isinstance(tensors, dict):
      prefix = flat_key
      tensors_ = {}
      for key in six.iterkeys(tensors):
        # multi_head uses tuples of strings as the key.
        if isinstance(key, tuple):
          tensors_["|".join(key)] = tensors[key]
        else:
          tensors_[key] = tensors[key]
      tensors = tensors_

    tensors = {prefix: tensors}
    tensors = _flatten_dict(tensors)
    return tensors

  def eval_metrics_tuple(self):
    """Returns tuple of (metric_fn, tensors) which can be executed on TPU."""

    if not self._metric_fns:
      return None

    def _metric_fn(**kwargs):
      kwargs = self._unflatten(**kwargs)
      eval_metric_ops = {}
      for metric_fn in self._metric_fns:
        eval_metric_ops.update(metric_fn(**kwargs))
      return eval_metric_ops

    return _metric_fn, self._kwargs

  def _unflatten(self, **kwargs):
    """Unflattens kwargs and removes any key prefixes."""

    kwargs = _unflatten_dict(kwargs, prefixes=self._Keys.ALL)
    kwargs = {
        k: self._reconstruct_tuple_keys(v) for k, v in six.iteritems(kwargs)
    }
    kwargs_ = {}
    for key, value in six.iteritems(kwargs):
      if key in self._Keys.ALL and key != self._Keys.KWARGS:
        kwargs_[key.replace("_", "")] = value
      else:
        kwargs_[key] = value
    return kwargs_

  def _reconstruct_tuple_keys(self, tensors):
    """Reconstructs tuple keys from flat strings if tensors is a dict."""

    if not isinstance(tensors, dict):
      return tensors

    result = {}
    for key, value in six.iteritems(tensors):
      parts = key.split("|")
      if len(parts) > 1:
        result[tuple(parts)] = value
      else:
        result[key] = value
    return result


class _EnsembleMetrics(_SubnetworkMetrics):
  """A object which creates evaluation metrics for Ensembles."""

  def create_eval_metrics(self, features, labels, estimator_spec, metric_fn,
                          architecture):
    """Overrides parent's method to also add the ensemble's architecture."""

    super(_EnsembleMetrics, self).create_eval_metrics(features, labels,
                                                      estimator_spec, metric_fn)
    self._metric_fns.append(self._architecture_as_metric(architecture))

  def _architecture_as_metric(self, architecture):
    """Returns a representation of an ensemble's architecture as a tf.metric."""

    def _architecture_metric_fn(**kwargs):
      """Manually creates the tf.metric with a serialized tf.Summary proto."""

      del kwargs  # Unused.

      # TODO: Should architecture.subnetworks be sorted by iteration
      # number first? Or perhaps, to make this more general, to have one line
      # for each iteration, with "|" as a delimiter if there are multiple
      # subnetworks in one iteration? Something like:
      # 0 linear
      # 1 dnn_width_32_depth_1 | dnn_width_64_depth_1
      # 2
      # 3 dnn_with_32_depth_2
      # Also consider adding ensemble candidate's name, though that is already
      # included in the ensemble name.
      architecture_ = " | ".join([name for _, name in architecture.subnetworks])
      architecture_ = "| {} |".format(architecture_)
      summary_metadata = tf.SummaryMetadata(
          plugin_data=tf.SummaryMetadata.PluginData(plugin_name="text"))
      summary_proto = tf.summary.Summary()
      summary_proto.value.add(
          metadata=summary_metadata,
          tag="architecture/adanet",
          tensor=tf.make_tensor_proto(architecture_, dtype=tf.string))
      architecture_summary = tf.convert_to_tensor(
          summary_proto.SerializeToString(), name="architecture")
      return {
          "architecture/adanet/ensembles": (architecture_summary, tf.no_op())
      }

    return _architecture_metric_fn


class _IterationMetrics(object):
  """A object which creates evaluation metrics for an Iteration."""

  def __init__(self, candidates, subnetwork_specs):
    self._candidates = candidates
    self._subnetwork_specs = subnetwork_specs

  def best_eval_metric_ops(self, best_candidate_index, mode):
    """Returns best ensemble's metrics."""
    return call_eval_metrics(
        self.best_eval_metrics_tuple(best_candidate_index, mode))

  def best_eval_metrics_tuple(self, best_candidate_index, mode):
    """Returns (metric_fn, tensors) which computes the best ensemble's metrics.

    Specifically, when metric_fn(tensors) is called, it separates the metric ops
    by metric name. All candidates are not required to have the same metrics.
    When they all share a given metric, an additional metric is added which
    represents that of the best candidate.

    Args:
      best_candidate_index: `Tensor` index of the best candidate in the list.
      mode: Defines whether this is training, evaluation or inference. Eval
        metrics are only defined during evaluation. See `ModeKeys`.

    Returns:
      Dict of metric results keyed by name. The values of the dict are the
      results of calling a metric function.
    """

    if mode != tf.estimator.ModeKeys.EVAL:
      return None

    metric_fns, tensors = self._collate_metric_fns_and_tensors()
    tensors["best_candidate_index"] = tf.reshape(best_candidate_index, [1])
    tensors = _flatten_dict(tensors)

    def _best_eval_metrics_fn(**kwargs):
      """Returns the best eval metrics."""

      with tf.variable_scope("best_eval_metrics"):
        subnetwork_metric_fns = {
            k: v for k, v in metric_fns.items() if k.startswith("subnetwork_")
        }
        subnetwork_tensors = _unflatten_dict(kwargs,
                                             subnetwork_metric_fns.keys())
        subnetwork_metric_ops = self._group_metric_ops(subnetwork_metric_fns,
                                                       subnetwork_tensors)
        ensemble_metric_fns = {
            k: v for k, v in metric_fns.items() if k.startswith("ensemble_")
        }
        ensemble_tensors = _unflatten_dict(kwargs, ensemble_metric_fns.keys())
        grouped_metrics = self._group_metric_ops(ensemble_metric_fns,
                                                 ensemble_tensors)
        eval_metric_ops = {}
        for metric_name in sorted(grouped_metrics):
          metric_ops = grouped_metrics[metric_name]
          if len(metric_ops) != len(self._candidates):
            continue
          if metric_name == "loss":
            continue

          best_candidate_index = kwargs["best_candidate_index"]
          values, ops = list(six.moves.zip(*metric_ops))
          idx, idx_update_op = tf.metrics.mean(best_candidate_index)
          best_value = tf.stack(values)[tf.cast(idx, tf.int32)]
          # All tensors in this function have been outfed from the TPU, so we
          # must update them manually, otherwise the TPU will hang indefinetly
          # for the value of idx to update.
          ops = list(ops)
          ops.append(idx_update_op)
          # Bundle subnetwork eval metric ops and ensemble "loss"" ops (which
          # is a restricted Estimator keyword) into other metric ops so that
          # they are computed.
          ensemble_loss_ops = grouped_metrics.get("loss", tf.no_op())
          all_ops = tf.group(ops, ensemble_loss_ops, subnetwork_metric_ops)
          eval_metric_ops[metric_name] = (best_value, all_ops)

        # tf.estimator.Estimator does not allow a "loss" key to be present in
        # its eval_metrics.
        assert "loss" not in eval_metric_ops
        return eval_metric_ops

    return _best_eval_metrics_fn, tensors

  def _collate_metric_fns_and_tensors(self):
    """Collates all candidates' and subnetworks' eval metric fns and tesnors."""
    fns = {}
    tensors = {}
    for i, subnetwork_spec in enumerate(self._subnetwork_specs):
      if not subnetwork_spec.eval_metrics:
        continue
      metric_fn, metric_tensors = subnetwork_spec.eval_metrics
      key = "subnetwork_{}".format(i)
      fns[key] = metric_fn
      tensors[key] = metric_tensors
    for i, candidate in enumerate(self._candidates):
      ensemble_spec = candidate.ensemble_spec
      if not ensemble_spec.eval_metrics:
        continue
      metric_fn, metric_tensors = ensemble_spec.eval_metrics
      key = "ensemble_{}".format(i)
      fns[key] = metric_fn
      tensors[key] = metric_tensors
    return fns, tensors

  def _group_metric_ops(self, metric_fns, tensors):
    """Runs the metric_fns and groups the returned metric ops by name.

    Tensors will be passed as params to metric_fns which have the same key. The
    dicts of eval metrics returned by metric_fns are then reduced by key.

    Args:
      metric_fns: A dictionary of fn(tensors)->dict(metric_name, metric_ops).
      tensors: A dictionary of tensors to pass to metric_fns.

    Returns:
      The metric ops grouped by name.
    """

    grouped_metrics = {}
    for key in sorted(metric_fns):
      fn = metric_fns[key]
      args = tensors[key]
      eval_metric_ops = fn(**args)
      for metric_name in sorted(eval_metric_ops):
        metric_op = eval_metric_ops[metric_name]
        if metric_name not in grouped_metrics:
          grouped_metrics[metric_name] = []
        grouped_metrics[metric_name].append(metric_op)
    return grouped_metrics
