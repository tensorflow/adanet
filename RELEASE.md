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

# Current version (0.6.0-dev)
 * Under development.
 * Introduce `adanet.ensemble` which contains interfaces and examples of ways to learn ensembles using AdaNet. Users can now extend AdaNet to use custom ensemble-learning methods.
 * Add support for evaluation on TPU using `adanet.TPUEstimator`.

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
