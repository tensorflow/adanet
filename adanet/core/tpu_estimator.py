"""An AdaNet estimator implementation which can run on TPU.

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
import contextlib
import functools

from absl import logging
from adanet import tf_compat
from adanet.core.estimator import Estimator
import tensorflow.compat.v2 as tf


# pylint: disable=g-classes-have-attributes
class TPUEstimator(Estimator, tf.compat.v1.estimator.tpu.TPUEstimator):
  """An :class:`adanet.Estimator` capable of training and evaluating on TPU.

  Unless :code:`use_tpu=False`, training will run on TPU. However, certain parts
  of the AdaNet training loop, such as report materialization and best candidate
  selection, will still occurr on CPU. Furthermore, if using TPUEmbedding (i.e.
  :code:`embedding_config_spec` is supplied), inference will also occurr on CPU.

  TODO: Provide the missing functionality detailed below.
  N.B: Embeddings using the TPUEmbedding (i.e. :code:`embedding_config_spec`
  is provided) only support :code:`shared_embedding_columns` when running for
  multiple AdaNet iterations. Using regular :code:`embedding_columns` will cause
  iterations 2..n to fail because of mismatched embedding scopes.

  Args:
    head: See :class:`adanet.Estimator`.
    subnetwork_generator: See :class:`adanet.Estimator`.
    max_iteration_steps: See :class:`adanet.Estimator`.
    ensemblers: See :class:`adanet.Estimator`.
    ensemble_strategies: See :class:`adanet.Estimator`.
    evaluator: See :class:`adanet.Estimator`.
    report_materializer: See :class:`adanet.Estimator`.
    metric_fn: See :class:`adanet.Estimator`.
    force_grow: See :class:`adanet.Estimator`.
    replicate_ensemble_in_training: See :class:`adanet.Estimator`.
    adanet_loss_decay: See :class:`adanet.Estimator`.
    report_dir: See :class:`adanet.Estimator`.
    config: See :class:`adanet.Estimator`.
    use_tpu: Boolean to enable training on TPU. Defaults to :code:`True` and is
      only provided to allow debugging models on CPU/GPU. Use
      :class:`adanet.Estimator` instead if you do not plan to run on TPU.
    eval_on_tpu: Boolean to enable evaluating on TPU. Defaults to :code:`True`.
      Ignored if :code:`use_tpu=False`.
    export_to_tpu: See :class:`tf.compat.v1.estimator.tpu.TPUEstimator`.
    train_batch_size: See :class:`tf.compat.v1.estimator.tpu.TPUEstimator`.
      Defaults to 0 if `None`.
    eval_batch_size: See :class:`tf.compat.v1.estimator.tpu.TPUEstimator`.
      Defaults to train_batch_size if `None`.
    predict_batch_size: See :class:`tf.compat.v1.estimator.tpu.TPUEstimator`.
      Defaults to eval_batch_size if `None`.
    embedding_config_spec: See :class:`tf.compat.v1.estimator.tpu.TPUEstimator`.
      If supplied, :code:`predict` will be called on CPU and no TPU compatible
        :code:`SavedModel` will be exported.
    debug: See :class:`adanet.Estimator`.
    enable_ensemble_summaries: See :class:`adanet.Estimator`.
    enable_subnetwork_summaries: See :class:`adanet.Estimator`.
    export_subnetwork_logits: Whether to include subnetwork logits in exports.
    export_subnetwork_last_layer: Whether to include subnetwork last layer in
      exports.
    global_step_combiner_fn: See :class:`adanet.Estimator`.
    max_iterations: See :class:`adanet.Estimator`.
    replay_config: See :class:`adanet.Estimator`.
    **kwargs: Extra keyword args passed to the parent.
  """

  def __init__(self,
               head,
               subnetwork_generator,
               max_iteration_steps,
               ensemblers=None,
               ensemble_strategies=None,
               evaluator=None,
               report_materializer=None,
               metric_fn=None,
               force_grow=False,
               replicate_ensemble_in_training=False,
               adanet_loss_decay=.9,
               model_dir=None,
               report_dir=None,
               config=None,
               use_tpu=True,
               eval_on_tpu=True,
               export_to_tpu=True,
               train_batch_size=None,
               eval_batch_size=None,
               predict_batch_size=None,
               embedding_config_spec=None,
               debug=False,
               enable_ensemble_summaries=True,
               enable_subnetwork_summaries=True,
               export_subnetwork_logits=False,
               export_subnetwork_last_layer=True,
               global_step_combiner_fn=tf.math.reduce_mean,
               max_iterations=None,
               replay_config=None,
               **kwargs):
    self._use_tpu = use_tpu
    if not self._use_tpu:
      logging.warning(
          "This adanet.TPUEstimator is meant to be used for running on TPU. "
          "If you want to run on CPU/GPU, use adanet.Estimator instead.")
    # TPUEstimator modifies config under the hood. We keep track of it here so
    # we can use it from _create_temp_run_config.
    self._original_config = config or tf_compat.v1.estimator.tpu.RunConfig()
    self._eval_on_tpu = eval_on_tpu if self._use_tpu else False
    self._export_to_tpu = export_to_tpu
    self._train_batch_size = train_batch_size or 0
    self._eval_batch_size = eval_batch_size or train_batch_size or 0
    self._predict_batch_size = (
        predict_batch_size or eval_batch_size or train_batch_size or 0)
    self._embedding_config_spec = embedding_config_spec
    if self._embedding_config_spec:
      logging.warning(
          "TPU does not support inference with TPUEmbedding. Force setting "
          "`export_to_tpu=False` so no TPU SavedModel will be exported.")
      self._export_to_tpu = False

    from tensorflow_estimator.python.estimator.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    super(TPUEstimator, self).__init__(
        head=head,
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        ensemblers=ensemblers,
        ensemble_strategies=ensemble_strategies,
        evaluator=evaluator,
        report_materializer=report_materializer,
        metric_fn=metric_fn,
        force_grow=force_grow,
        replicate_ensemble_in_training=replicate_ensemble_in_training,
        adanet_loss_decay=adanet_loss_decay,
        model_dir=model_dir,
        report_dir=report_dir,
        config=self._original_config,
        use_tpu=self._use_tpu,
        eval_on_tpu=self._eval_on_tpu,
        export_to_tpu=self._export_to_tpu,
        export_saved_model_api_version=(
            tpu_estimator.ExportSavedModelApiVersion.V2),
        train_batch_size=self._train_batch_size,
        eval_batch_size=self._eval_batch_size,
        predict_batch_size=self._predict_batch_size,
        embedding_config_spec=self._embedding_config_spec,
        debug=debug,
        enable_ensemble_summaries=enable_ensemble_summaries,
        enable_subnetwork_summaries=enable_subnetwork_summaries,
        export_subnetwork_logits=export_subnetwork_logits,
        export_subnetwork_last_layer=export_subnetwork_last_layer,
        global_step_combiner_fn=global_step_combiner_fn,
        max_iterations=max_iterations,
        replay_config=replay_config,
        **kwargs)

  def predict(self,
              input_fn,
              predict_keys=None,
              hooks=None,
              checkpoint_path=None,
              yield_single_examples=True):

    use_tpu = self._use_tpu
    eval_on_tpu = self._eval_on_tpu
    if self._embedding_config_spec:
      logging.warning("TPU does not support inference with TPUEmbedding. "
                      "Falling back to CPU.")
      use_tpu = False
      eval_on_tpu = False
    if not checkpoint_path:
      checkpoint_path = tf.train.latest_checkpoint(self.model_dir)
    logging.info("Computing predictions for AdaNet model at checkpoint: %s",
                 checkpoint_path)
    params = self.params
    params.update({
        "best_ensemble_index":
            self._compute_best_ensemble_index(checkpoint_path, hooks),
        "checkpoint_path":
            checkpoint_path,
    })
    from tensorflow_estimator.python.estimator.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    # TODO: Consider extracting a common function to use here and in
    # _create_temp_estimator().
    estimator = tf_compat.v1.estimator.tpu.TPUEstimator(
        model_fn=self._create_model_fn(hooks=hooks, is_export=False),
        params=params,
        config=self._original_config,
        model_dir=self.model_dir,
        use_tpu=use_tpu,
        eval_on_tpu=eval_on_tpu,
        export_to_tpu=self._export_to_tpu,
        export_saved_model_api_version=(
            tpu_estimator.ExportSavedModelApiVersion.V2),
        train_batch_size=self._train_batch_size,
        eval_batch_size=self._eval_batch_size,
        predict_batch_size=self._predict_batch_size,
        embedding_config_spec=self._embedding_config_spec)
    return estimator.predict(
        input_fn,
        predict_keys=predict_keys,
        hooks=hooks,
        checkpoint_path=checkpoint_path,
        yield_single_examples=yield_single_examples)

  def _create_temp_run_config(self, temp_model_dir):
    """See the `Estimator` base class for details."""

    return tf_compat.v1.estimator.tpu.RunConfig(
        model_dir=temp_model_dir,
        tpu_config=self._original_config.tpu_config,
        evaluation_master=self._original_config.evaluation_master,
        master=self._original_config.master,
        cluster=self._original_config.cluster,
        tf_random_seed=self._original_config.tf_random_seed,
        session_config=self._original_config.session_config,
        protocol=self._original_config.protocol)

  def _create_temp_estimator(self, config, **create_model_fn_args):
    """See the `Estimator` base class for details."""

    from tensorflow_estimator.python.estimator.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

    temp_model_dir = config.model_dir
    return tf_compat.v1.estimator.tpu.TPUEstimator(
        model_fn=self._create_model_fn(**create_model_fn_args),
        config=config,
        model_dir=temp_model_dir,
        use_tpu=self._use_tpu,
        eval_on_tpu=self._eval_on_tpu,
        export_to_tpu=self._export_to_tpu,
        export_saved_model_api_version=(
            tpu_estimator.ExportSavedModelApiVersion.V2),
        train_batch_size=self._train_batch_size,
        eval_batch_size=self._eval_batch_size,
        predict_batch_size=self._predict_batch_size,
        embedding_config_spec=self._embedding_config_spec)

  @contextlib.contextmanager
  def _call_input_fn_in_new_graph(self, input_fn, mode, config):
    """See the `Estimator` base class for details."""

    # Bind parameters to input_fn since the parent's input_fn is not expected to
    # have any arguments.
    from tensorflow.python.util import function_utils  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    input_fn_args = function_utils.fn_args(input_fn)
    kwargs = {}
    if "mode" in input_fn_args:
      kwargs["mode"] = mode
    if "params" in input_fn_args:
      kwargs["params"] = self.params
    if "config" in input_fn_args:
      kwargs["config"] = config
    input_fn = functools.partial(input_fn, **kwargs)
    with super(TPUEstimator,
               self)._call_input_fn_in_new_graph(input_fn, mode, config) as res:
      yield res

  def _create_estimator_spec(self, current_iteration, mode,
                             iteration_number_tensor, previous_iteration_vars,
                             is_growing_phase, evaluation_name):
    """See the `Estimator` base class for details."""

    if not self._use_tpu:
      return super(TPUEstimator, self)._create_estimator_spec(
          current_iteration, mode, iteration_number_tensor,
          previous_iteration_vars, is_growing_phase, evaluation_name)

    training = mode == tf.estimator.ModeKeys.TRAIN
    iteration_estimator_spec = current_iteration.estimator_spec
    training_hooks = self._training_hooks(current_iteration, training,
                                          iteration_number_tensor,
                                          previous_iteration_vars,
                                          is_growing_phase)
    if is_growing_phase:
      training_hooks = self._process_hooks_for_growing_phase(training_hooks)
    evaluation_hooks = self._evaluation_hooks(current_iteration, training,
                                              evaluation_name)
    return tf_compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions=iteration_estimator_spec.predictions,
        loss=iteration_estimator_spec.loss,
        train_op=self._train_op(iteration_estimator_spec, is_growing_phase),
        host_call=self._create_host_call(current_iteration, training),
        eval_metrics=iteration_estimator_spec.eval_metrics,
        export_outputs=iteration_estimator_spec.export_outputs,
        # Return a constant summary_op, otherwise `Estimator` creates summary
        # ops that do not work on TPU.
        scaffold_fn=lambda: tf.compat.v1.train.Scaffold(  # pylint: disable=g-long-lambda
            summary_op=tf.constant("")),
        training_hooks=training_hooks,
        evaluation_hooks=evaluation_hooks)

  def _training_hooks(self, current_iteration, training,
                      iteration_number_tensor, previous_iteration_vars,
                      is_growing_phase):
    """See the `Estimator` base class for details."""

    training_hooks = super(TPUEstimator,
                           self)._training_hooks(current_iteration, training,
                                                 iteration_number_tensor,
                                                 previous_iteration_vars,
                                                 is_growing_phase)
    if self._use_tpu:
      # Remove summary hooks on TPU since summaries are saved via host_call.
      training_hooks = [
          hook for hook in training_hooks
          if not isinstance(hook, tf.compat.v1.train.SummarySaverHook)
      ]

    return training_hooks

  def _create_host_call(self, current_iteration, training):
    """Construct a host_call writing scalar summaries.

    Args:
      current_iteration: The current `_Iteration`.
      training: Boolean indicating whether in training mode.

    Returns:
      (fn, args) Pair to be called by TPUEstimator as the host_call.
    """

    if not training:
      return lambda **kwargs: [tf.no_op()], {}

    # Collect and flatten summary functions and arguments.
    summary_kwargs = collections.OrderedDict()
    gs_t = tf.reshape(tf.cast(tf.train.get_global_step(), dtype=tf.int32), [1])
    summary_kwargs["global_step"] = gs_t

    summary_fns = collections.defaultdict(list)
    for i, summary in enumerate(current_iteration.summaries):
      for j, (summary_fn, tensor) in enumerate(summary.summary_tuples()):
        summary_fns[i].append(summary_fn)
        summary_kwargs["summary_{}_{}".format(i, j)] = tensor

    def _host_call_fn(**kwargs):
      """Training host call.

      Creates summaries for training metrics.

      Args:
        **kwargs: Dict of {str: Tensor} , with `Tensor` of shape `[batch]`. Must
          contain key "global_step" with value of current global_step Tensor.

      Returns:
        List of summary ops to run on the CPU host.
      """

      from tensorflow.python.ops import summary_ops_v2  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

      gs = tf.cast(kwargs.pop("global_step")[0], dtype=tf.int64)
      for i, summary in enumerate(current_iteration.summaries):
        with summary_ops_v2.create_file_writer(summary.logdir).as_default():
          with summary_ops_v2.record_summaries_every_n_global_steps(
              n=self.config.save_summary_steps, global_step=gs):
            for j, summary_fn in enumerate(summary_fns[i]):
              tensor = kwargs["summary_{}_{}".format(i, j)]
              summary_fn(tensor, step=gs)
        summary.clear_summary_tuples()
      return tf.compat.v1.summary.all_v2_summary_ops()

    return _host_call_fn, summary_kwargs

  def _create_model_fn(self,
                       is_growing_phase=False,
                       is_inside_training_loop=False,
                       is_export=False,
                       evaluation_name=None,
                       best_ensemble_index=None,
                       checkpoint_path=None,
                       hooks=None):
    """See the `Estimator` base class for details."""

    from tensorflow_estimator.python.estimator.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

    adanet_model_fn = super(TPUEstimator, self)._create_model_fn(
        is_growing_phase, is_inside_training_loop, is_export, evaluation_name,
        best_ensemble_index, checkpoint_path, hooks)

    def _model_fn(features, labels, mode, params, config):
      """The model_fn to return which supports exporting on TPU."""

      if (is_export and params["use_tpu"] and
          mode == tf.estimator.ModeKeys.PREDICT):
        batch_config = tpu_estimator.BatchConfig(
            # Set num_batch_threads to the number of TPU cores on Servomatic.
            num_batch_threads=2,
            max_batch_size=self._predict_batch_size,
            # TODO: Magic number. Investigate whether there is a better
            # way to set this, or have the user pass it in.
            batch_timeout_micros=60 * 1000,
            allowed_batch_sizes=[self._predict_batch_size])
        return tpu_estimator.model_fn_inference_on_tpu(
            adanet_model_fn,
            features=features,
            labels=labels,
            config=config,
            params=params,
            batch_config=batch_config)

      return adanet_model_fn(features, labels, mode, params, config)

    return _model_fn
