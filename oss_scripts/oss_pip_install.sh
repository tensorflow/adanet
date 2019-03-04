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

BAZEL_VERSION=0.20.0

: "${TF_VERSION:?}"

# if [[ "$TF_VERSION" == "tf-nightly"  ]]
# then
#   pip install tf-nightly;
# else
#   pip install -q "tensorflow==$TF_VERSION"
# fi

# Make sure we have the latest version of numpy - avoid problems we were
# seeing with Python 3
# pip install -q -U numpy

# Install Bazel for tests.
# Step 1: Install required packages
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python

# Step 2: Download Bazel binary installer
wget https://github.com/bazelbuild/bazel/releases/download/"$BAZEL_VERSION"/bazel-"$BAZEL_VERSION"-installer-linux-x86_64.sh

# Step 3: Install Bazel
chmod +x bazel-"$BAZEL_VERSION"-installer-linux-x86_64.sh
./bazel-"$BAZEL_VERSION"-installer-linux-x86_64.sh --user
export PATH="$PATH:$HOME/bin"

# Build adanet pip packaging script
bazel build -c opt //... --local_resources 2048,.5,1.0

# Create the adanet pip package
bazel-bin/adanet/pip_package/build_pip_package /tmp/adanet_pkg

# Install and test the pip package
pip install /tmp/adanet_pkg/*.whl --user

# cp -R bazel-bin/adanet/core/estimator_distributed_test_runner* adanet/core
cp bazel-genfiles/adanet/core/architecture_pb2.py adanet/core
cp bazel-genfiles/adanet/core/report_pb2.py adanet/core

# Finally try importing `adanet` in Python outside the cloned directory:
cd ..
python -c "import adanet"
cd adanet
