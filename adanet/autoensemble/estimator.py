"""An estimator that learns to ensemble.

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

import collections
import inspect

from adanet import tf_compat
from adanet.core import Estimator
from adanet.core import TPUEstimator
from adanet.subnetwork import Builder
from adanet.subnetwork import Generator
from adanet.subnetwork import Subnetwork
from adanet.subnetwork import TrainOpSpec
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator import estimator as estimator_lib
# pylint: enable=g-direct-tensorflow-import


def _default_logits(estimator_spec):
  if isinstance(estimator_spec.predictions, dict):
    pred_keys = prediction_keys.PredictionKeys
    if pred_keys.LOGITS in estimator_spec.predictions:
      return estimator_spec.predictions[pred_keys.LOGITS]
    if pred_keys.PREDICTIONS in estimator_spec.predictions:
      return estimator_spec.predictions[pred_keys.PREDICTIONS]
  return estimator_spec.predictions


class _SecondaryTrainOpRunnerHook(tf_compat.SessionRunHook):
  """A hook for running a train op separate from the main session run call."""

  def __init__(self, train_op):
    """Initializes a `_SecondaryTrainOpRunnerHook`.

    Args:
      train_op: The secondary train op to execute before runs.
    """

    self._train_op = train_op

  def before_run(self, run_context):
    run_context.session.run(self._train_op)


class AutoEnsembleSubestimator(
    collections.namedtuple("AutoEnsembleSubestimator",
                           ["estimator", "train_input_fn"])):
  # pylint: disable=g-classes-have-attributes
  """A subestimator to train and consider for ensembling.

  Args:
    estimator: A `tf.estimator.Estimator` or `tf.estimator.tpu.TPUEstimator`
      instance to consider for ensembling.
    train_input_fn: A function that provides input data for training as
      minibatches. It can be used to implement ensemble methods like bootstrap
      aggregating (a.k.a bagging) where each subnetwork trains on different
      slices of the training data. The function should construct and return one
      of the following:
       * A `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
         `(features, labels)` with same constraints as below. NOTE: A Dataset
         must return *at least* two batches before hitting the end-of-input,
         otherwise all of training terminates.
         TODO: Figure out how to handle single-batch datasets.
       * A tuple `(features, labels)`: Where `features` is a `tf.Tensor` or a
         dictionary of string feature name to `Tensor` and `labels` is a
         `Tensor` or a dictionary of string label name to `Tensor`. Both
         `features` and `labels` are consumed by `estimator#model_fn`. They
         should satisfy the expectation of `estimator#model_fn` from inputs.

  Returns:
    An `AutoEnsembleSubestimator` instance to be auto-ensembled.
  """
  # pylint: enable=g-classes-have-attributes

  def __new__(cls, estimator, train_input_fn=None):
    return super(AutoEnsembleSubestimator, cls).__new__(cls, estimator,
                                                        train_input_fn)


class _BuilderFromSubestimator(Builder):
  """An `adanet.Builder` from a :class:`tf.estimator.Estimator`."""

  def __init__(self, name, subestimator, logits_fn, last_layer_fn, config):
    self._name = name
    self._subestimator = subestimator
    self._logits_fn = logits_fn
    self._last_layer_fn = last_layer_fn
    self._config = config

  @property
  def name(self):
    return self._name

  def _call_model_fn(self, subestimator, features, labels, mode, summary):
    with summary.current_scope():
      model_fn = subestimator.estimator.model_fn
      estimator_spec = model_fn(
          features=features, labels=labels, mode=mode, config=self._config)
      logits = self._logits_fn(estimator_spec=estimator_spec)
      last_layer = logits
      if self._last_layer_fn:
        last_layer = self._last_layer_fn(estimator_spec=estimator_spec)

      if estimator_spec.scaffold and estimator_spec.scaffold.local_init_op:
        local_init_op = estimator_spec.scaffold.local_init_op
      else:
        local_init_op = None

      train_op = TrainOpSpec(
          estimator_spec.train_op,
          chief_hooks=estimator_spec.training_chief_hooks,
          hooks=estimator_spec.training_hooks)
    return logits, last_layer, train_op, local_init_op

  def build_subnetwork(self,
                       features,
                       labels,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble,
                       config=None):
    # We don't need an EVAL mode since AdaNet takes care of evaluation for us.
    mode = tf.estimator.ModeKeys.PREDICT
    if training:
      mode = tf.estimator.ModeKeys.TRAIN

    # Call in template to ensure that variables are created once and reused.
    call_model_fn_template = tf.compat.v1.make_template("model_fn",
                                                        self._call_model_fn)
    subestimator_features, subestimator_labels = features, labels
    local_init_ops = []
    subestimator = self._subestimator(config)
    if training and subestimator.train_input_fn:
      # TODO: Consider tensorflow_estimator/python/estimator/util.py.
      inputs = subestimator.train_input_fn()
      if isinstance(inputs, (tf_compat.DatasetV1, tf_compat.DatasetV2)):
        subestimator_features, subestimator_labels = (
            tf_compat.make_one_shot_iterator(inputs).get_next())
      else:
        subestimator_features, subestimator_labels = inputs

      # Construct subnetwork graph first because of dependencies on scope.
      _, _, bagging_train_op_spec, sub_local_init_op = call_model_fn_template(
          subestimator, subestimator_features, subestimator_labels, mode,
          summary)
      # Graph for ensemble learning gets model_fn_1 for scope.
      logits, last_layer, _, ensemble_local_init_op = call_model_fn_template(
          subestimator, features, labels, mode, summary)

      if sub_local_init_op:
        local_init_ops.append(sub_local_init_op)
      if ensemble_local_init_op:
        local_init_ops.append(ensemble_local_init_op)

      # Run train op in a hook so that exceptions can be intercepted by the
      # AdaNet framework instead of the Estimator's monitored training session.
      hooks = bagging_train_op_spec.hooks + (_SecondaryTrainOpRunnerHook(
          bagging_train_op_spec.train_op),)
      train_op_spec = TrainOpSpec(
          train_op=tf.no_op(),
          chief_hooks=bagging_train_op_spec.chief_hooks,
          hooks=hooks)
    else:
      logits, last_layer, train_op_spec, local_init_op = call_model_fn_template(
          subestimator, features, labels, mode, summary)
      if local_init_op:
        local_init_ops.append(local_init_op)

    # TODO: Replace with variance complexity measure.
    complexity = tf.constant(0.)
    return Subnetwork(
        logits=logits,
        last_layer=last_layer,
        shared={"train_op": train_op_spec},
        complexity=complexity,
        local_init_ops=local_init_ops)

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    return subnetwork.shared["train_op"]


def _convert_to_subestimator(candidate):
  if callable(candidate):
    return candidate
  if isinstance(candidate, AutoEnsembleSubestimator):
    return lambda config: candidate
  if isinstance(candidate,
                (estimator_lib.Estimator, estimator_lib.EstimatorV2)):
    return lambda config: AutoEnsembleSubestimator(candidate)
  raise ValueError(
      "subestimator in candidate_pool must have type tf.estimator.Estimator or "
      "adanet.AutoEnsembleSubestimator but got {}".format(candidate.__class__))


class _GeneratorFromCandidatePool(Generator):
  """An `adanet.Generator` from a pool of `Estimator` and `Model` instances."""

  def __init__(self, candidate_pool, logits_fn, last_layer_fn):
    self._candidate_pool = candidate_pool
    if logits_fn is None:
      logits_fn = _default_logits
    self._logits_fn = logits_fn
    self._last_layer_fn = last_layer_fn

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports, config):
    assert config
    builders = []
    candidate_pool = self._maybe_call_candidate_pool(config, iteration_number)

    if isinstance(candidate_pool, dict):
      for name in sorted(candidate_pool):
        builders.append(
            _BuilderFromSubestimator(
                name,
                _convert_to_subestimator(candidate_pool[name]),
                logits_fn=self._logits_fn,
                last_layer_fn=self._last_layer_fn,
                config=config))
      return builders

    for i, estimator in enumerate(candidate_pool):
      name = "{class_name}{index}".format(
          class_name=estimator.__class__.__name__, index=i)
      builders.append(
          _BuilderFromSubestimator(
              name,
              _convert_to_subestimator(estimator),
              logits_fn=self._logits_fn,
              last_layer_fn=self._last_layer_fn,
              config=config))
    return builders

  def _maybe_call_candidate_pool(self, config, iteration_number):
    if callable(self._candidate_pool):
      # candidate_pool can be a function.
      if "iteration_number" in inspect.getargspec(self._candidate_pool).args:
        # TODO: Make the "config" argument optional using introspection.
        return self._candidate_pool(
            config=config, iteration_number=iteration_number)
      else:
        return self._candidate_pool(config=config)

    return self._candidate_pool


class AutoEnsembleEstimator(Estimator):
  # pylint: disable=g-classes-have-attributes
  # pyformat: disable
  """A :class:`tf.estimator.Estimator` that learns to ensemble models.

  Specifically, it learns to ensemble models from a candidate pool using the
  Adanet algorithm.

  .. code-block:: python

      # A simple example of learning to ensemble linear and neural network
      # models.

      import adanet
      import tensorflow as tf

      feature_columns = ...

      head = MultiClassHead(n_classes=10)

      # Learn to ensemble linear and DNN models.
      estimator = adanet.AutoEnsembleEstimator(
          head=head,
          candidate_pool=lambda config: {
              "linear":
                  tf.estimator.LinearEstimator(
                      head=head,
                      feature_columns=feature_columns,
                      config=config,
                      optimizer=...),
              "dnn":
                  tf.estimator.DNNEstimator(
                      head=head,
                      feature_columns=feature_columns,
                      config=config,
                      optimizer=...,
                      hidden_units=[1000, 500, 100])},
          max_iteration_steps=50)

      # Input builders
      def input_fn_train:
        # Returns tf.data.Dataset of (x, y) tuple where y represents label's
        # class index.
        pass
      def input_fn_eval:
        # Returns tf.data.Dataset of (x, y) tuple where y represents label's
        # class index.
        pass
      def input_fn_predict:
        # Returns tf.data.Dataset of (x, None) tuple.
        pass
      estimator.train(input_fn=input_fn_train, steps=100)
      metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
      predictions = estimator.predict(input_fn=input_fn_predict)

  Or to train candidate subestimators on different training data subsets:

  .. code-block:: python

      train_data_files = [...]

      # Learn to ensemble linear and DNN models.
      estimator = adanet.AutoEnsembleEstimator(
          head=head,
          candidate_pool=lambda config: {
              "linear":
                  adanet.AutoEnsembleSubestimator(
                      tf.estimator.LinearEstimator(
                          head=head,
                          feature_columns=feature_columns,
                          config=config,
                          optimizer=...),
                      make_train_input_fn(train_data_files[:-1])),
              "dnn":
                  adanet.AutoEnsembleSubestimator(
                      tf.estimator.DNNEstimator(
                          head=head,
                          feature_columns=feature_columns,
                          config=config,
                          optimizer=...,
                          hidden_units=[1000, 500, 100]),
                      make_train_input_fn(train_data_files[0:]))},
          max_iteration_steps=50)

      estimator.train(input_fn=make_train_input_fn(train_data_files), steps=100)


  Args:
    head: A :class:`tf.contrib.estimator.Head` instance for computing loss and
      evaluation metrics for every candidate.
    candidate_pool: List of :class:`tf.estimator.Estimator` and
      :class:`AutoEnsembleSubestimator` objects, or dict of string name to
      :class:`tf.estimator.Estimator` and :class:`AutoEnsembleSubestimator`
      objects that are candidate subestimators to ensemble at each iteration.
      The order does not directly affect which candidates will be included in
      the final ensemble, but will affect the name of the candidate. When using
      a dict, the string key becomes the candidate subestimator's name.
      Alternatively, this argument can be a function that takes a `config`
      argument and returns the aforementioned values in case the
      objects need to be re-instantiated at each adanet iteration.
    max_iteration_steps: Total number of steps for which to train candidates per
      iteration. If `OutOfRange` or `StopIteration` occurs in the middle,
      training stops before `max_iteration_steps` steps.
    logits_fn: A function for fetching the subnetwork logits from a
      :class:`tf.estimator.EstimatorSpec`, which should obey the following
      signature:
        - `Args`: Can only have following argument:
          - estimator_spec: The candidate's :class:`tf.estimator.EstimatorSpec`.
        - `Returns`: Logits :class:`tf.Tensor` or dict of string to logits
          :class:`tf.Tensor` (for multi-head) for the candidate subnetwork
          extracted from the given `estimator_spec`. When `None`, it will
          default to returning `estimator_spec.predictions` when they are a
          :class:`tf.Tensor` or the :class:`tf.Tensor` for the key 'logits' when
          they are a dict of string to :class:`tf.Tensor`.
    last_layer_fn: An optional function for fetching the subnetwork last_layer
      from a :class:`tf.estimator.EstimatorSpec`, which should obey the
      following signature:
        - `Args`: Can only have following argument:
          - estimator_spec: The candidate's :class:`tf.estimator.EstimatorSpec`.
        - `Returns`: Last layer :class:`tf.Tensor` or dict of string to last
          layer :class:`tf.Tensor` (for multi-head) for the candidate subnetwork
          extracted from the given `estimator_spec`. The last_layer can be used
          for learning ensembles or exporting them as embeddings.
      When `None`, it will default to using the logits as the last_layer.
    ensemblers: See :class:`adanet.Estimator`.
    ensemble_strategies: See :class:`adanet.Estimator`.
    evaluator:  See :class:`adanet.Estimator`.
    metric_fn:  See :class:`adanet.Estimator`.
    force_grow:  See :class:`adanet.Estimator`.
    adanet_loss_decay: See :class:`adanet.Estimator`.
    worker_wait_timeout_secs: See :class:`adanet.Estimator`.
    model_dir: See :class:`adanet.Estimator`.
    config: See :class:`adanet.Estimator`.
    debug: See :class:`adanet.Estimator`.
    enable_ensemble_summaries: See :class:`adanet.Estimator`.
    enable_subnetwork_summaries: See :class:`adanet.Estimator`.
    global_step_combiner_fn: See :class:`adanet.Estimator`.
    max_iterations: See :class:`adanet.Estimator`.
    replay_config: See :class:`adanet.Estimator`.
    **kwargs: Extra keyword args passed to the parent.

  Returns:
    An :class:`adanet.AutoEnsembleEstimator` instance.

  Raises:
    ValueError: If any of the candidates in `candidate_pool` are not
      :class:`tf.estimator.Estimator` instances.
  """
  # pyformat: enable
  # pylint: enable=g-classes-have-attributes

  def __init__(self,
               head,
               candidate_pool,
               max_iteration_steps,
               ensemblers=None,
               ensemble_strategies=None,
               logits_fn=None,
               last_layer_fn=None,
               evaluator=None,
               metric_fn=None,
               force_grow=False,
               adanet_loss_decay=.9,
               worker_wait_timeout_secs=7200,
               model_dir=None,
               config=None,
               debug=False,
               enable_ensemble_summaries=True,
               enable_subnetwork_summaries=True,
               global_step_combiner_fn=tf.math.reduce_mean,
               max_iterations=None,
               replay_config=None,
               **kwargs):
    subnetwork_generator = _GeneratorFromCandidatePool(candidate_pool,
                                                       logits_fn, last_layer_fn)
    super(AutoEnsembleEstimator, self).__init__(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        ensemblers=ensemblers,
        ensemble_strategies=ensemble_strategies,
        evaluator=evaluator,
        metric_fn=metric_fn,
        force_grow=force_grow,
        adanet_loss_decay=adanet_loss_decay,
        worker_wait_timeout_secs=worker_wait_timeout_secs,
        model_dir=model_dir,
        config=config,
        debug=debug,
        enable_ensemble_summaries=enable_ensemble_summaries,
        enable_subnetwork_summaries=enable_subnetwork_summaries,
        global_step_combiner_fn=global_step_combiner_fn,
        max_iterations=max_iterations,
        replay_config=replay_config,
        **kwargs)


class AutoEnsembleTPUEstimator(TPUEstimator):
  # pylint: disable=g-classes-have-attributes
  # pyformat: disable
  """A :class:`tf.estimator.tpu.TPUEstimator` that learns to ensemble models.

  Specifically, it learns to ensemble models from a candidate pool using the
  Adanet algorithm.

  This estimator is capable of training and evaluating on TPU. It can ensemble
  both :class:`tf.estimator.tpu.TPUEstimator` candidates as well as regular
  :class:`tf.estimator.Estimator` candidates, as long as these candidates are
  TPU compatible.

  Note the following restrictions compared to AutoEnsembleEstimator:
    * All candidates must wrap their optimizers with a
      :class:`tf.tpu.CrossShardOptimizer`.
    * The `input_fn` must expose a `params` argument.
    * The `model_fn` of :class:`tf.estimator.tpu.TPUEstimator` candidates must
      also expose a `params` argument.

  WARNING: This Estimator is a work in progress and the API could change at any
  moment. May not support all AutoEnsembleEstimator features.

    .. code-block:: python

      # A simple example of learning to ensemble linear and neural network
      # models on TPU.

      import adanet
      import tensorflow as tf

      feature_columns = ...

      head = MultiClassHead(n_classes=10)

      # Learn to ensemble linear and DNN models.
      estimator = adanet.AutoEnsembleEstimator(
          head=head,
          candidate_pool=lambda config: {
              "linear":
                  tf.estimator.LinearEstimator(
                      head=head,
                      feature_columns=feature_columns,
                      config=config,
                      optimizer=tf.tpu.CrossShardOptimizer(...)),
              "dnn":
                  tf.estimator.DNNEstimator(
                      head=head,
                      feature_columns=feature_columns,
                      config=config,
                      optimizer=tf.tpu.CrossShardOptimzier(...),
                      hidden_units=[1000, 500, 100])},
          max_iteration_steps=50)

      # Input builders
      def input_fn_train(params):
        # Returns tf.data.Dataset of (x, y) tuple where y represents label's
        # class index.
        pass
      def input_fn_eval(params):
        # Returns tf.data.Dataset of (x, y) tuple where y represents label's
        # class index.
        pass
      def input_fn_predict():
        # Returns tf.data.Dataset of (x, None) tuple.
        pass
      estimator.train(input_fn=input_fn_train, steps=100)
      metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
      predictions = estimator.predict(input_fn=input_fn_predict)

  Args:
    head: A :class:`tf.contrib.estimator.Head` instance for computing loss and
      evaluation metrics for every candidate.
    candidate_pool: List of :class:`tf.estimator.tpu.TPUEstimator` and
      :class:`AutoEnsembleSubestimator` objects, or dict of string name to
      :class:`tf.estimator.tpu.TPUEstimator` and
      :class:`AutoEnsembleSubestimator` objects that are candidate subestimators
      to ensemble at each iteration. The order does not directly affect which
      candidates will be included in the final ensemble, but will affect the
      name of the candidate. When using a dict, the string key becomes the
      candidate subestimator's name. Alternatively, this argument can be a
      function that takes a `config` argument and returns the aforementioned
      values in case the objects need to be re-instantiated at each adanet
      iteration.
    max_iteration_steps: See :class:`adanet.Estimator`.
    logits_fn: A function for fetching the subnetwork logits from a
      :class:`tf.estimator.EstimatorSpec`, which should obey the following
      signature:
        - `Args`: Can only have following argument:
          - estimator_spec: The candidate's :class:`tf.estimator.EstimatorSpec`.
        - `Returns`: Logits :class:`tf.Tensor` or dict of string to logits
          :class:`tf.Tensor` (for multi-head) for the candidate subnetwork
          extracted from the given `estimator_spec`. When `None`, it will
          default to returning `estimator_spec.predictions` when they are a
          :class:`tf.Tensor` or the :class:`tf.Tensor` for the key 'logits' when
          they are a dict of string to :class:`tf.Tensor`.
    last_layer_fn: An optional function for fetching the subnetwork last_layer
      from a :class:`tf.estimator.EstimatorSpec`, which should obey the
      following signature:
        - `Args`: Can only have following argument:
          - estimator_spec: The candidate's :class:`tf.estimator.EstimatorSpec`.
        - `Returns`: Last layer :class:`tf.Tensor` or dict of string to last
          layer :class:`tf.Tensor` (for multi-head) for the candidate subnetwork
          extracted from the given `estimator_spec`. The last_layer can be used
          for learning ensembles or exporting them as embeddings.
      When `None`, it will default to using the logits as the last_layer.
    ensemblers: See :class:`adanet.Estimator`.
    ensemble_strategies: See :class:`adanet.Estimator`.
    evaluator:  See :class:`adanet.Estimator`.
    metric_fn:  See :class:`adanet.Estimator`.
    force_grow:  See :class:`adanet.Estimator`.
    adanet_loss_decay: See :class:`adanet.Estimator`.
    model_dir: See :class:`adanet.Estimator`.
    config: See :class:`adanet.Estimator`.
    use_tpu: See :class:`adanet.Estimator`.
    eval_on_tpu: See :class:`adanet.Estimator`.
    train_batch_size: See :class:`adanet.Estimator`.
    eval_batch_size: See :class:`adanet.Estimator`.
    embedding_config_spec: See :class:`adanet.Estimator`.
    debug: See :class:`adanet.Estimator`.
    enable_ensemble_summaries: See :class:`adanet.Estimator`.
    enable_subnetwork_summaries: See :class:`adanet.Estimator`.
    global_step_combiner_fn: See :class:`adanet.Estimator`.
    max_iterations: See :class:`adanet.Estimator`.
    replay_config: See :class:`adanet.Estimator`.
    **kwargs: Extra keyword args passed to the parent.

  Returns:
    An :class:`adanet.AutoEnsembleEstimator` instance.

  Raises:
    ValueError: If any of the candidates in `candidate_pool` are not
      :class:`tf.estimator.Estimator` instances.
  """
  # pyformat: enable
  # pylint: disable=g-classes-have-attributes

  def __init__(self,
               head,
               candidate_pool,
               max_iteration_steps,
               ensemblers=None,
               ensemble_strategies=None,
               logits_fn=None,
               last_layer_fn=None,
               evaluator=None,
               metric_fn=None,
               force_grow=False,
               adanet_loss_decay=.9,
               model_dir=None,
               config=None,
               use_tpu=True,
               eval_on_tpu=True,
               train_batch_size=None,
               eval_batch_size=None,
               embedding_config_spec=None,
               debug=False,
               enable_ensemble_summaries=True,
               enable_subnetwork_summaries=True,
               global_step_combiner_fn=tf.math.reduce_mean,
               max_iterations=None,
               replay_config=None,
               **kwargs):
    subnetwork_generator = _GeneratorFromCandidatePool(candidate_pool,
                                                       logits_fn, last_layer_fn)
    super(AutoEnsembleTPUEstimator, self).__init__(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        ensemblers=ensemblers,
        ensemble_strategies=ensemble_strategies,
        evaluator=evaluator,
        metric_fn=metric_fn,
        force_grow=force_grow,
        adanet_loss_decay=adanet_loss_decay,
        model_dir=model_dir,
        config=config,
        use_tpu=use_tpu,
        eval_on_tpu=eval_on_tpu,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        embedding_config_spec=embedding_config_spec,
        debug=debug,
        enable_ensemble_summaries=enable_ensemble_summaries,
        enable_subnetwork_summaries=enable_subnetwork_summaries,
        global_step_combiner_fn=global_step_combiner_fn,
        max_iterations=max_iterations,
        replay_config=replay_config,
        **kwargs)
