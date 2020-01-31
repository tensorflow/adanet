#!/usr/bin/env bash
# Copyright 2018 The AdaNet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

: "${TF_VERSION:?}"

if [[ "$TF_VERSION" == "tf-nightly"  ]]; then
  pip install tf-nightly;
elif [[ "$TF_VERSION" == "tf-nightly-2.0-preview"  ]]; then
  pip install tf-nightly-2.0-preview;
else
  pip install -q "tensorflow==$TF_VERSION"
fi

# Build adanet pip packaging script
bazel build -c opt //... --local_resources 2048,.5,1.0 --force_python=PY3
