<!-- Copyright 2018 The AdaNet Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================-->

# Current version (0.8.0.dev)
 * Under development.
 * TODO: Add official Keras Model support, including Keras layers, Sequential, and Model subclasses for defining subnetworks.
 * **BREAKING CHANGE**: AdaNet now supports arbitrary metrics when choosing the best ensemble. To achieve this, the interface of `adanet.Evaluator` is changing. The `Evaluator.evaluate_adanet_losses(sess, adanet_losses)` function is being replaced with `Evaluator.evaluate(sess, ensemble_metrics)`. The `ensemble_metrics` parameter contains all computed metrics for each candidate ensemble as well as the `adanet_loss`. Code which overrides `evaluate_adanet_losses` must migrate over to use the new `evaluate` method (we suspect that such cases are very rare).
 * Allow user to specify a maximum number of AdaNet iterations.
 * **BREAKING CHANGE**: When supplied, run the `adanet.Evaluator` before `Estimator#evaluate`, `Estimator#predict`, and `Estimator#export_saved_model`. This can have the effect of changing the best candidate chosen at the final round. When the user passes an Evaluator, we run it to establish the best candidate during evaluation, predict, and export_saved_model. Previously they used the adanet_loss moving average collected during training. While the previous ensemble would have been established by the Evaluator, the current set of candidate ensembles that were not done training would be considered according to the adanet_loss. Now when a user passes an Evaluator that, for example, uses a hold-out set, AdaNet runs it before making predictions or exporting a SavedModel to use the best new candidate according to the hold-out set.
 * Support `tf.keras.metrics.Metrics` during evaluation.
 * Stop individual subnetwork training on `OutOfRangeError` raised during bagging.
 * Gracefully handle NaN losses from ensembles during training. When an ensemble or subnetwork has a NaN loss during training, its training is marked as terminated. As long as one ensemble (and therefore underlying subnetworks) does not have a NaN loss, training will continue.
 * Train forever if `max_steps` and `steps` are both `None`.

# Release 0.7.0
 * Add embeddings support on TPU via `TPUEmbedding`.
 * Train the current iteration forever when `max_iteration_steps=None`.
 * Introduce `adanet.AutoEnsembleSubestimator` for training subestimators on different training data partitions and implement ensemble methods like bootstrap aggregating (a.k.a bagging).
 * Fix bug when using Gradient Boosted Decision Tree Estimators with `AutoEnsembleEstimator` during distributed training.
 * Allow `AutoEnsembleEstimator's` `candidate_pool` argument to be a `lambda` in order to create `Estimators` lazily.
 * Remove `adanet.subnetwork.Builder#prune_previous_ensemble` for abstract class. This behavior is now specified using `adanet.ensemble.Strategy` subclasses.
 * **BREAKING CHANGE**: Only support TensorFlow >= 1.14 to better support TensorFlow 2.0. Drop support for versions < 1.14.
 * Correct eval metric computations on CPU and GPU.

# Release 0.6.2
 * Fix n+1 global-step increment bug in `adanet.AutoEnsembleEstimator`. This bug incremented the global_step by n+1 for n canned `Estimators` like `DNNEstimator`.

# Release 0.6.1
 * Maintain compatibility with TensorFlow versions >=1.9.

# Release 0.6.0
 * Officially support AdaNet on TPU using `adanet.TPUEstimator` with `adanet.Estimator` feature parity.
 * Support dictionary candidate pools in `adanet.AutoEnsembleEstimator` constructor to specify human-readable candidate names.
 * Improve AutoEnsembleEstimator ability to handling custom `tf.estimator.Estimator` subclasses.
 * Introduce `adanet.ensemble` which contains interfaces and examples of ways to learn ensembles using AdaNet. Users can now extend AdaNet to use custom ensemble-learning methods.
 * Record TensorBoard `scalar`, `image`, `histogram`, and `audio` summaries on TPU during training.
 * Add debug mode to help detect NaNs and Infs during training.
 * Improve subnetwork `tf.train.SessionRunHook` support to handle more edge cases.
 * ~~Maintain compatibility with TensorFlow versions 1.9 thru 1.13~~ Only works for TensorFlow version >=1.13.
 * Improve documentation including adding 'Getting Started' documentation to [adanet.readthedocs.io](http://adanet.readthedocs.io).
 * **BREAKING CHANGE**: Importing the `adanet.subnetwork` package using `from adanet.core import subnetwork` will no longer work, because the package was moved to the `adanet/subnetwork` directory. Most users should already be using `adanet.subnetwork` or `from adanet import subnetwork`, and should not be affected.

# Release 0.5.0
 * Support training on TPU using `adanet.TPUEstimator`.
 * Allow subnetworks to specify `tf.train.SessionRunHook` instances for training with `adanet.subnetwork.TrainOpSpec`.
 * Add API documentation generation with Sphinx.
 * Fix bug preventing subnetworks with Resource variables from working beyond the first iteration.

# Release 0.4.0
 * Add `shared` field to `adanet.Subnetwork` to deprecate, replace, and be more flexible than `persisted_tensors`.
 * Officially support multi-head learning with or without dict labels.
 * Rebuild the ensemble across iterations in Python without a frozen graph. This allows users to share more than `Tensors` between iterations including Python primitives, objects, and lambdas for greater flexibility. Eliminating reliance on a `MetaGraphDef` proto also eliminates I/O allowing for faster training, and better future-proofing.
 * Allow users to pass custom eval metrics when constructing an `adanet.Estimator`.
 * Add `adanet.AutoEnsembleEstimator` for learning to ensemble `tf.estimator.Estimator` instances.
 * Pass labels to `adanet.subnetwork.Builder`'s `build_subnetwork` method.
 * The TRAINABLE_VARIABLES collection will only contain variables relevant to the current `adanet.subnetwork.Builder`, so not passing `var_list` to the `optimizer.minimize` will lead to the same behavior as passing it in by default.
 * Using `tf.summary` inside `adanet.subnetwork.Builder` is now equivalent to using the `adanet.Summary` object.
 * Accessing the `global_step` from within an `adanet.subnetwork.Builder` will return the `iteration_step` variable instead, so that the step starts at zero at the beginning of each iteration. One subnetwork incrementing the step will not affect other subnetworks.
 * Summaries will automatically scope themselves to the current subnetwork's scope. Similar summaries will now be correctly grouped together correctly across subnetworks in TensorBoard. This eliminates the need for the `tf.name_scope("")` hack.
 * Provide an override to force the AdaNet ensemble to grow at the end of each iteration.
 * Correctly seed TensorFlow graph between iterations. This breaks some tests that check the outputs of `adanet.Estimator` models.

# Release 0.3.0
 * Add official support for `tf.keras.layers`.
 * Fix bug that incorrectly pruned colocation constraints between iterations.

# Release 0.2.0
 * Estimator no longer creates eval metric ops in train mode.
 * Freezer no longer converts Variables to constants, allowing AdaNet to handle Variables larger than 2GB.
 * Fixes some errors with Python 3.

# Release 0.1.0
 * Initial AdaNet release.

## Requirements
 * tf-nightly>=1.9.0.dev20180601 || tensorflow>=1.9.0rc0
