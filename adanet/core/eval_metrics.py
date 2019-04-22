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

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


def call_eval_metrics(eval_metrics):
  if not eval_metrics:
    return {}
  fn, args = eval_metrics
  if isinstance(args, dict):
    return fn(**args)
  else:
    return fn(*args)


class _EvalMetricsStore(object):
  """Stores and manipulate eval_metric tuples."""

  def __init__(self):
    self._metric_fns = []
    self._args = []

  def add_eval_metrics(self, metric_fn, args):
    """Adds an eval_metrics tuple to the internal store."""

    self._metric_fns.append(metric_fn)
    self._args.append(args)

  @property
  def metric_fns(self):
    return self._metric_fns

  def flatten_args(self):
    """Flattens the eval_metrics arguments to a list."""

    return nest.flatten(self._args)

  def pack_args(self, args):
    """Packs the given list of arguments into the internal args structure."""

    return nest.pack_sequence_as(self._args, args)


class _SubnetworkMetrics(object):
  """A object which creates evaluation metrics for Subnetworks."""

  def __init__(self):
    self._eval_metrics_store = _EvalMetricsStore()

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

    # If estimator_spec is not a TPUEstimatorSpec we create dummy metric_fn
    # and args.
    if isinstance(estimator_spec, tf.estimator.EstimatorSpec):
      spec_fn, spec_args = lambda: estimator_spec.eval_metric_ops, []
    else:
      spec_fn, spec_args = estimator_spec.eval_metrics
    self._eval_metrics_store.add_eval_metrics(
        self._templatize_metric_fn(spec_fn), spec_args)

    loss_fn = lambda loss: {"loss": tf.metrics.mean(loss)}
    loss_fn_args = [tf.reshape(estimator_spec.loss, [1])]
    self._eval_metrics_store.add_eval_metrics(
        self._templatize_metric_fn(loss_fn), loss_fn_args)

    # NOTE: the user supplied metrics_fn must be added last. This is because we
    # want user metrics to override AdaNet's metrics.
    if metric_fn:
      metric_fn_args = {}
      argspec = inspect.getargspec(metric_fn).args
      if "features" in argspec:
        metric_fn_args["features"] = features
      if "labels" in argspec:
        metric_fn_args["labels"] = labels
      if "predictions" in argspec:
        metric_fn_args["predictions"] = estimator_spec.predictions
      self._eval_metrics_store.add_eval_metrics(
          self._templatize_metric_fn(metric_fn), metric_fn_args)

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

    def _metric_fn(*args, **kwargs):
      """The wrapping function to be returned."""

      # We can only be passed in either a dict or a list of tensors.
      args = args if args else kwargs
      metrics = call_eval_metrics((metric_fn, args))
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
        if isinstance(op, tf.Operation) or op.shape != tensor.shape:
          with tf.control_dependencies([op]):
            op = tf.assign(var, tensor)
        metric = (var, tf.assign(var, op))
        wrapped_metrics[key] = metric
      return wrapped_metrics

    return tf.make_template("metric_fn_template", _metric_fn)

  def eval_metrics_tuple(self):
    """Returns tuple of (metric_fn, tensors) which can be executed on TPU."""

    if not self._eval_metrics_store.metric_fns:
      return None

    def _metric_fn(*args):
      metric_fns = self._eval_metrics_store.metric_fns
      metric_fn_args = self._eval_metrics_store.pack_args(args)
      eval_metric_ops = {}
      for metric_fn, args in zip(metric_fns, metric_fn_args):
        eval_metric_ops.update(call_eval_metrics((metric_fn, args)))
      return eval_metric_ops

    return _metric_fn, self._eval_metrics_store.flatten_args()


class _EnsembleMetrics(_SubnetworkMetrics):
  """A object which creates evaluation metrics for Ensembles."""

  def create_eval_metrics(self, features, labels, estimator_spec, metric_fn,
                          architecture):
    """Overrides parent's method to also add the ensemble's architecture."""

    super(_EnsembleMetrics, self).create_eval_metrics(features, labels,
                                                      estimator_spec, metric_fn)
    self._eval_metrics_store.add_eval_metrics(
        self._architecture_as_metric(architecture), [])

  def _architecture_as_metric(self, architecture):
    """Returns a representation of an ensemble's architecture as a tf.metric."""

    def _architecture_metric_fn():
      """Manually creates the tf.metric with a serialized tf.Summary proto."""

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

    self._candidates_eval_metrics_store = self._build_eval_metrics_store(
        [candidate.ensemble_spec for candidate in self._candidates])
    self._subnetworks_eval_metrics_store = self._build_eval_metrics_store(
        self._subnetwork_specs)

  def _build_eval_metrics_store(self, specs):
    """Creates an _EvalMetricsStore from Subnetwork or Ensemble specs."""

    store = _EvalMetricsStore()
    for spec in specs:
      if not spec.eval_metrics:
        continue
      metric_fn, args = spec.eval_metrics
      store.add_eval_metrics(metric_fn, args)
    return store

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

    candidate_args = self._candidates_eval_metrics_store.flatten_args()
    subnetwork_args = self._subnetworks_eval_metrics_store.flatten_args()
    args = candidate_args + subnetwork_args
    args.append(tf.reshape(best_candidate_index, [1]))

    def _best_eval_metrics_fn(*args):
      """Returns the best eval metrics."""

      with tf.variable_scope("best_eval_metrics"):
        args = list(args)
        idx, idx_update_op = tf.metrics.mean(args.pop())

        metric_fns = self._candidates_eval_metrics_store.metric_fns
        metric_fn_args = self._candidates_eval_metrics_store.pack_args(
            args[:len(candidate_args)])
        candidate_grouped_metrics = self._group_metric_ops(
            metric_fns, metric_fn_args)

        metric_fns = self._subnetworks_eval_metrics_store.metric_fns
        metric_fn_args = self._subnetworks_eval_metrics_store.pack_args(
            args[-len(subnetwork_args):])
        subnetwork_grouped_metrics = self._group_metric_ops(
            metric_fns, metric_fn_args)

        eval_metric_ops = {}
        for metric_name in sorted(candidate_grouped_metrics):
          metric_ops = candidate_grouped_metrics[metric_name]
          if len(metric_ops) != len(self._candidates):
            continue
          if metric_name == "loss":
            continue
          values, ops = list(six.moves.zip(*metric_ops))
          best_value = tf.stack(values)[tf.cast(idx, tf.int32)]
          # All tensors in this function have been outfed from the TPU, so we
          # must update them manually, otherwise the TPU will hang indefinetly
          # for the value of idx to update.
          ops = list(ops)
          ops.append(idx_update_op)
          # Bundle subnetwork eval metric ops and ensemble "loss"" ops (which
          # is a restricted Estimator keyword) into other metric ops so that
          # they are computed.
          ensemble_loss_ops = candidate_grouped_metrics.get("loss", tf.no_op())
          all_ops = tf.group(ops, ensemble_loss_ops, subnetwork_grouped_metrics)
          eval_metric_ops[metric_name] = (best_value, all_ops)

        # tf.estimator.Estimator does not allow a "loss" key to be present in
        # its eval_metrics.
        assert "loss" not in eval_metric_ops
        return eval_metric_ops

    return _best_eval_metrics_fn, args

  def _group_metric_ops(self, metric_fns, metric_fn_args):
    """Runs the metric_fns and groups the returned metric ops by name.

    Args:
      metric_fns: The eval_metrics functions to run.
      metric_fn_args: The eval_metrics function arguments.

    Returns:
      The metric ops grouped by name.
    """

    grouped_metrics = collections.defaultdict(list)
    for metric_fn, args in zip(metric_fns, metric_fn_args):
      eval_metric_ops = call_eval_metrics((metric_fn, args))
      for metric_name in sorted(eval_metric_ops):
        metric_op = eval_metric_ops[metric_name]
        grouped_metrics[metric_name].append(metric_op)
    return grouped_metrics
