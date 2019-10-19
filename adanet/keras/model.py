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
from adanet.core import Estimator
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import multi_class_head
from tensorflow_estimator.python.estimator.head import regression_head


def _dataset_to_input_fn(dataset):
  """Converts a `tf.data.Dataset` to an input_fn."""

  def input_fn(params=None):
    del params  # unused
    return dataset()

  return input_fn


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
    # Use lambdas to defer initialization of Head.
    self._loss_head_map = {
        "binary_crossentropy":
            lambda: binary_class_head.BinaryClassHead(),  # pylint: disable=unnecessary-lambda
        "mse":
            lambda: regression_head.RegressionHead(self._logits_dimension),
        "mean_squared_error":
            lambda: regression_head.RegressionHead(self._logits_dimension),
        "sparse_categorical_crossentropy":
            lambda: multi_class_head.MultiClassHead(self._logits_dimension),
    }

  @property
  def metrics_names(self):
    return ["loss"]

  def fit(self,
          x,
          epochs=1,
          steps_per_epoch=None,
          callbacks=None):
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
      callbacks: List of `keras.callbacks.Callback` instances.
        List of callbacks to apply during evaluation.

    Raises:
      RuntimeError: If the model was never compiled.
    """
    if callbacks is not None:
      logging.warning("Callbacks are currently not supported.")

    if self._model is not None:
      for _ in range(epochs):
        self._model.train(
            input_fn=_dataset_to_input_fn(x), steps=steps_per_epoch)
    else:
      raise RuntimeError(
          "You must compile your model before training. Use `model.compile(loss)`."
      )

  def evaluate(self,
               x,
               steps=None,
               callbacks=None):
    """Returns the loss value & metrics values for the model in test mode.

    Args:
      x: A function that returns a `tf.data` dataset.
      steps: Integer or `None`.
        Total number of steps (batches of samples) before declaring the
        evaluation round finished. Ignored with the default value of `None`.
        If `steps` is None, 'evaluate' will run until the dataset is exhausted.
      callbacks: List of `keras.callbacks.Callback` instances.
        List of callbacks to apply during evaluation.

    Returns:
      Scalar test loss (if the model has a single output and no metrics) or list
      of scalars (if the model has multiple outputs and/or metrics). The
      attribute model.metrics_names will give you the display labels for the
      scalar outputs.

    Raises:
      RuntimeError: If the model was never compiled.
    """

    if callbacks is not None:
      logging.warning("Callbacks are currently not supported.")

    if self._model is not None:
      results = self._model.evaluate(
          input_fn=_dataset_to_input_fn(x), steps=steps)
      return results
    else:
      raise RuntimeError(
          "You must compile your model before testing. Use `model.compile(loss)`."
      )

  def predict(self,
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
      callbacks: List of `keras.callbacks.Callback` instances.
        List of callbacks to apply during prediction.

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
      results = self._model.predict(_dataset_to_input_fn(x),
                                    yield_single_examples=False)
      # Convert the generator object returned by Estimator's predict method to a
      # numpy array of all the predictions.
      return next(results)["predictions"]
    else:
      raise RuntimeError(
          "You must compile your model before prediction. Use `model.compile(loss)`."
      )

  # TODO: Support passed in metrics.
  def compile(self, loss, metrics=None):
    """Configures the model for training.

    Args:
      loss: String of a built in `tf.keras.Loss` function.
      metrics: List of metrics to be evaluated by the model during training
          and testing. Typically you will use `metrics=['accuracy']`.
          To specify different metrics for different outputs of a
          multi-output model, you could also pass a dictionary, such as
          `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
          You can also pass a list (len = len(outputs)) of lists of metrics
          such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or
          `metrics=['accuracy', ['accuracy', 'mse']]`.
    """
    if metrics is not None:
      logging.warning("Passing in metrics is currently not supported.")

    head = self._loss_head_map.get(loss, None)
    if head is not None:
      self._model = Estimator(
          head=head(),
          max_iteration_steps=self._max_iteration_steps,
          ensemblers=self._ensemblers,
          ensemble_strategies=self._ensemble_strategies,
          evaluator=self._evaluator,
          adanet_loss_decay=self._adanet_loss_decay,
          model_dir=self._filepath,
          subnetwork_generator=self._subnetwork_generator)
    else:
      raise ValueError(
          "'{}' is not a currently supported loss. Currently supported losses are: {}."
          .format(loss, [
              "binary_crossentropy", "mean_squared_error",
              "sparse_categorical_crossentropy"
          ]))

  def save(self):
    raise NotImplementedError("Saving is currently not supported.")
