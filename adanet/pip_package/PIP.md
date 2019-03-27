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
=============================================================================-->
# Creating the adanet pip package using Linux

This requires Python, Bazel and Git. (And TensorFlow for testing the package.)

### Activate virtualenv

Install virtualenv if it's not installed already:

```shell
~$ sudo apt-get install python-virtualenv
```

Create a virtual environment for the package creation:

```shell
~$ virtualenv --system-site-packages adanet_env
```

And activate it:

```shell
~$ source ~/adanet_env/bin/activate  # bash, sh, ksh, or zsh
~$ source ~/adanet_env/bin/activate.csh  # csh or tcsh
```

### Clone the adanet repository.

```shell
(adanet_env)~$ git clone https://github.com/tensorflow/adanet && cd adanet
```

### Build adanet pip packaging script

To build a pip package for adanet:

```shell
(adanet_env)~/adanet$ bazel build //adanet/pip_package:build_pip_package
```

### Create the adanet pip package

```shell
(adanet_env)~/adanet$ bazel-bin/adanet/pip_package/build_pip_package /tmp/adanet_pkg
```

### Install and test the pip package (optional)

Run the following command to install the pip package:

```shell
(adanet_env)~/adanet$ pip install /tmp/adanet_pkg/*.whl
```

Finally try importing `adanet` in Python outside the cloned directory:

```shell
(adanet_env)~/adanet$ cd ~
(adanet_env)~$ python -c "import adanet"
```

### De-activate the virtualenv

```shell
(adanet_env)~/$ deactivate
```
