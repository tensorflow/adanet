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

if [[ "$TF_VERSION" == "tf-nightly"  ]]
then
  pip install tf-nightly;
else
  pip install -q "tensorflow==$TF_VERSION"
fi

# Make sure we have the latest version of numpy - avoid problems we were
# seeing with Python 3
pip install -q -U numpy

# Install Bazel for tests.
# Step 1: Install the JDK
sudo apt-get install openjdk-8-jdk

# Step 2: Add Bazel distribution URI as a package source
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

# Step 3: Install and update Bazel
sudo apt-get update && sudo apt-get install bazel

# Build adanet pip packaging script
bazel build //adanet/pip_package:build_pip_package

# Create the adanet pip package
bazel-bin/adanet/pip_package/build_pip_package /tmp/adanet_pkg

# Install and test the pip package
pip install /tmp/adanet_pkg/*.whl

# Finally try importing `adanet` in Python outside the cloned directory:
cd ..
python -c "import adanet"
