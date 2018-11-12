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

# Current version (0.4.0-dev)
 * Under development.
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
 * tf-nightly>=1.7.0.dev20180308 || tensorflow>=1.7.0rc0
