"""An AdaNet Keras model implementation.

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

from absl import logging
from adanet import core

import tensorflow as tf


class _KerasHead(object):
  """A `tf.estimator.Head`-like alternative for usage within AdaNet."""

  def __init__(self, logits_dimension, loss, metrics):
    """Initialize a _KerasHead object.

    Args:
      logits_dimension: The dimension of the final layer of any subnetworks.
      loss: A `tf.keras.losses.Loss`. Note: must set `from_logits` to True if
        the loss is a non-regression loss.
      metrics: List of lambdas that return `tf.keras.metric.Metric` objects.
        Each metric object must have `name` set to some string and `from_logits`
        set to True if it is a non-regression metric.

    Raises:
      ValueError: If `from_logits` isn't `True` for a non-regression `loss`.
    """

    self.logits_dimension = logits_dimension
    self.metrics = metrics

    if hasattr(loss, "from_logits") and not loss.from_logits:
      raise ValueError("from_logits must be True for non-regression losses.")
    self.loss = loss

  def create_estimator_spec(self, features, mode, logits, labels, train_op_fn):
    """Returns EstimatorSpec that a `model_fn` can return."""

    del features, train_op_fn  # unused

    eval_metric_ops = None
    export_outputs = None
    loss = None
    train_op = None
    # TODO: Currently the predictions are the raw logits which
    # means that the predictions will not be correct for anything other than
    # regression. Should look into how Keras handles this.
    predictions = {"predictions": logits}

    if mode == tf.estimator.ModeKeys.PREDICT:
      # TODO: Populate export_outputs for SavedModel.
      export_outputs = {}
    elif mode == tf.estimator.ModeKeys.EVAL:
      eval_results = {}
      for metric in self.metrics:
        # We wrap the metric within a function since Estimator subnetworks
        # need to have this created within their graphs.
        metric = metric()
        metric.update_state(y_true=labels, y_pred=logits)
        eval_results[metric.name] = metric
      eval_metric_ops = eval_results
      loss = tf.math.reduce_mean(self.loss(y_true=labels, y_pred=logits))
    elif mode == tf.estimator.ModeKeys.TRAIN:
      loss = tf.math.reduce_mean(self.loss(y_true=labels, y_pred=logits))
      train_op = tf.no_op()

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs,
        train_op=train_op)


class Model(object):
  """A `tf.keras.Model`-like object for training, evaluation, and serving."""

  def __init__(self,
               subnetwork_generator,
               max_iteration_steps,
               logits_dimension=1,
               ensemblers=None,
               ensemble_strategies=None,
               evaluator=None,
               adanet_loss_decay=.9,
               filepath=None):
    # pyformat: disable
    """Initializes an `adanet.keras.Model`.

    Args:
      subnetwork_generator: The :class:`adanet.subnetwork.Generator` which
        defines the candidate subnetworks to train and evaluate at every AdaNet
        iteration.
      max_iteration_steps: Total number of steps for which to train candidates
        per iteration. If :class:`OutOfRange` or :class:`StopIteration` occurs
        in the middle, training stops before `max_iteration_steps` steps. When
        :code:`None`, it will train the current iteration forever.
      logits_dimension: The dimension of the final layer of any subnetworks.
      ensemblers: An iterable of :class:`adanet.ensemble.Ensembler` objects that
        define how to ensemble a group of subnetworks. If there are multiple,
        each should have a different `name` property.
      ensemble_strategies: An iterable of :class:`adanet.ensemble.Strategy`
        objects that define the candidate ensembles of subnetworks to explore at
        each iteration.
      evaluator: An :class:`adanet.Evaluator` for candidate selection after all
        subnetworks are done training. When :code:`None`, candidate selection
        uses a moving average of their :class:`adanet.Ensemble` AdaNet loss
        during training instead. In order to use the *AdaNet algorithm* as
        described in [Cortes et al., '17], the given :class:`adanet.Evaluator`
        must be created with the same dataset partition used during training.
        Otherwise, this framework will perform *AdaNet.HoldOut* which uses a
        holdout set for candidate selection, but does not benefit from learning
        guarantees.
      adanet_loss_decay: Float decay for the exponential-moving-average of the
        AdaNet objective throughout training. This moving average is a data-
        driven way tracking the best candidate with only the training set.
      filepath: Directory to save model parameters, graph and etc. This can also
        be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
    """
    # pyformat: enable

    logging.warning("""The AdaNet Keras API is currently experimental.""")

    self._subnetwork_generator = subnetwork_generator
    self._max_iteration_steps = max_iteration_steps
    self._logits_dimension = logits_dimension
    self._ensemblers = ensemblers
    self._ensemble_strategies = ensemble_strategies
    self._evaluator = evaluator
    self._adanet_loss_decay = adanet_loss_decay
    self._filepath = filepath
    self._model = None
    self._metrics_names = ["loss"]

  @property
  def metrics_names(self):
    return self._metrics_names

  def fit(self, x, epochs=1, steps_per_epoch=None, callbacks=None):
    """Trains the model for a fixed number of epochs.

    Args:
      x: A function that returns a `tf.data` dataset.
      epochs: Number of epochs to train the model.
      steps_per_epoch: Integer or None. Total number of steps (batches of
        samples) before declaring one epoch finished and starting the next
        epoch. When training with input tensors such as TensorFlow data tensors,
        the default None is equal to the number of samples in your dataset
        divided by the batch size, or 1 if that cannot be determined. If
        'steps_per_epoch' is 'None', the epoch will run until the input dataset
        is exhausted.
      callbacks: List of `keras.callbacks.Callback` instances. List of callbacks
        to apply during evaluation.

    Raises:
      RuntimeError: If the model was never compiled.
    """
    if callbacks is not None:
      logging.warning("Callbacks are currently not supported.")

    if self._model is not None:
      for _ in range(epochs):
        self._model.train(input_fn=x, steps=steps_per_epoch)
    else:
      raise RuntimeError(
          "You must compile your model before training. Use `model.compile(loss)`."
      )

  def evaluate(self, x, steps=None, callbacks=None):
    """Returns the loss value & metrics values for the model in test mode.

    Args:
      x: A function that returns a `tf.data` dataset.
      steps: Integer or `None`. Total number of steps (batches of samples)
        before declaring the evaluation round finished. Ignored with the default
        value of `None`. If `steps` is None, 'evaluate' will run until the
        dataset is exhausted.
      callbacks: List of `keras.callbacks.Callback` instances. List of callbacks
        to apply during evaluation.

    Returns:
      A list of scalars for loss and metrics. The attribute model.metrics_names
      will give you the display labels for the scalar outputs.

    Raises:
      RuntimeError: If the model was never compiled.
    """

    if callbacks is not None:
      logging.warning("Callbacks are currently not supported.")

    if self._model is not None:
      results = self._model.evaluate(input_fn=x, steps=steps)
      return [results[result] for result in self._metrics_names]
    else:
      raise RuntimeError(
          "You must compile your model before testing. Use `model.compile(loss)`."
      )

  def predict(
      self,
      x,
      # `steps` unused by Estimators, but usable by Keras models later.
      steps=None,  # pylint: disable=unused-argument
      callbacks=None):
    """Generates output predictions for the input samples.

    Args:
      x: A function that returns a `tf.data` Dataset.
      steps: Total number of steps (batches of samples) before declaring the
        prediction round finished. Ignored with the default value of `None`. If
        `None`, `predict` will run until the input dataset is exhausted.
      callbacks: List of `keras.callbacks.Callback` instances. List of callbacks
        to apply during prediction.

    Returns:
      Numpy array(s) of predictions.

    Raises:
      RuntimeError: If the model was never compiled.
    """
    if steps is not None:
      logging.warning("Steps is currently unused.")

    if callbacks is not None:
      logging.warning("Callbacks are currently not supported.")

    if self._model is not None:
      results = self._model.predict(x, yield_single_examples=False)
      # TODO: Make predictions match the format of the task class.
      logging.warning("Prediction results are in raw logit form.")
      # Convert the generator object returned by Estimator's predict method to a
      # numpy array of all the predictions.
      return next(results)["predictions"]
    else:
      raise RuntimeError(
          "You must compile your model before prediction. Use `model.compile(loss)`."
      )

  def compile(self, loss, metrics=None):
    """Configures the model for training.

    Args:
      loss: A `tf.keras.losses.Loss`. Note: must set `from_logits` to True if
        the loss is a non-regression loss.
      metrics: List of lambdas that return `tf.keras.metric.Metric` objects.
        Each metric object must have `name` set to some string and
        `from_logits` set to True if it is a non-regression metric.

    Raises:
      ValueError: If a metric does not have a name.
    """

    if metrics is None:
      metrics = []
    else:
      # TODO: Assure `from_logits=True` for every metric.
      logging.warning(
          "Assure non-regression metrics initialized with `from_logits=True`.")

    for metric in metrics:
      metric = metric()
      if metric.name is None:
        raise ValueError("Metrics must have names.")
      self._metrics_names.append(metric.name)

    keras_head = _KerasHead(
        logits_dimension=self._logits_dimension, loss=loss, metrics=metrics)

    self._model = core.Estimator(
        head=keras_head,
        subnetwork_generator=self._subnetwork_generator,
        max_iteration_steps=self._max_iteration_steps,
        ensemblers=self._ensemblers,
        ensemble_strategies=self._ensemble_strategies,
        evaluator=self._evaluator,
        adanet_loss_decay=self._adanet_loss_decay,
        model_dir=self._filepath)

  # TODO: Implement `adanet.Model#save.`
  def save(self):
    raise NotImplementedError("Saving is currently not supported.")
