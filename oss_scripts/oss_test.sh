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

# Install coverage for collecting coverage reports.
pip install coverage

# Install nose for running the test suite with coverage.
pip install nose rednose

# Build with Bazel.
bazel build -c opt //...

# Copy Bazel generated code for report proto.
cp bazel-genfiles/adanet/core/report_pb2.py adanet/core

# Run test suite and collect coverage (see setup.cfg in root).
nosetests
