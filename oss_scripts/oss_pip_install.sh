#!/bin/bash

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

# First ensure that the base dependencies are sufficient for a full import
# pip install -q -e .

# Build adanet pip packaging script
bazel build //adanet/pip_package:build_pip_package

# Create the adanet pip package
bazel-bin/adanet/pip_package/build_pip_package /tmp/adanet_pkg

# Install and test the pip package
pip install /tmp/adanet_pkg/*.whl

python -c "import adanet"

# Then install the test dependencies
# pip install -q -e .[tests]
