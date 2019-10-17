"""A Keras model that learns to ensemble.

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
from adanet.autoensemble.estimator import _GeneratorFromCandidatePool
from adanet.keras.model import Model


class AutoEnsemble(Model):
  """A :class:`tf.keras.Model` that learns to ensemble models."""

  def __init__(self,
               candidate_pool,
               max_iteration_steps,
               logits_dimension=1,
               ensemblers=None,
               ensemble_strategies=None,
               evaluator=None,
               adanet_loss_decay=.9,
               filepath=None,
               logits_fn=None,
               last_layer_fn=None):
    """Instantiates an `adanet.AutoEnsemble`.

    Args:
      candidate_pool: List of :class:`tf.estimator.Estimator` and
        :class:`AutoEnsembleSubestimator` objects, or dict of string name to
        :class:`tf.estimator.Estimator` and :class:`AutoEnsembleSubestimator`
        objects that are candidate subestimators to ensemble at each iteration.
        The order does not directly affect which candidates will be included in
        the final ensemble, but will affect the name of the candidate. When
        using a dict, the string key becomes the candidate subestimator's name.
        Alternatively, this argument can be a function that takes a `config`
        argument and returns the aforementioned values in case the
        objects need to be re-instantiated at each adanet iteration.
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
      logits_fn: A function for fetching the subnetwork logits from a
        :class:`tf.estimator.EstimatorSpec`, which should obey the
        following signature:
          - `Args`: Can only have following argument:
            - estimator_spec: The candidate's
              :class:`tf.estimator.EstimatorSpec`.
          - `Returns`: Logits :class:`tf.Tensor` or dict of string to logits
            :class:`tf.Tensor` (for multi-head) for the candidate subnetwork
            extracted from the given `estimator_spec`. When `None`, it will
            default to returning `estimator_spec.predictions` when they are a
            :class:`tf.Tensor` or the :class:`tf.Tensor` for the key 'logits'
            when they are a dict of string to :class:`tf.Tensor`.
      last_layer_fn: An optional function for fetching the subnetwork last_layer
        from a :class:`tf.estimator.EstimatorSpec`, which should obey the
        following signature:
          - `Args`: Can only have following argument:
            - estimator_spec: The candidate's
              :class:`tf.estimator.EstimatorSpec`.
          - `Returns`: Last layer :class:`tf.Tensor` or dict of string to last
            layer :class:`tf.Tensor` (for multi-head) for the candidate
            subnetwork extracted from the given `estimator_spec`. The last_layer
            can be used for learning ensembles or exporting them as embeddings.
        When `None`, it will default to using the logits as the last_layer.
    """

    logging.warning("The AdaNet AutoEnsemble API is currently experimental.")

    subnetwork_generator = _GeneratorFromCandidatePool(candidate_pool,
                                                       logits_fn, last_layer_fn)

    super(AutoEnsemble, self).__init__(
        subnetwork_generator=subnetwork_generator,
        max_iteration_steps=max_iteration_steps,
        logits_dimension=logits_dimension,
        ensemblers=ensemblers,
        ensemble_strategies=ensemble_strategies,
        evaluator=evaluator,
        adanet_loss_decay=adanet_loss_decay,
        filepath=filepath)




