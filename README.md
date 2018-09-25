# AdaNet

**adanet** is a lightweight and scalable TensorFlow framework for training and deploying adaptive neural networks using the AdaNet algorithm [[Cortes et al. ICML 2017](https://arxiv.org/abs/1607.01097)].

This is not an official Google product.

## Getting Started

To get you started:

- [Tutorials: for understanding the AdaNet algorithm and learning to use this package](./examples/tutorials)

## Requirements

`adanet` depends on bug fixes and enhancements not present in TensorFlow releases prior to 1.7. You must install or upgrade your TensorFlow package to at least 1.7:

```shell
$ pip install "tensorflow>=1.7.0"
```

## Installing from source

To install from source first you'll need to install `bazel` following their [installation instructions](https://docs.bazel.build/versions/master/install.html).

Next clone `adanet`:

```shell
$ git clone git@github.com:tensorflow/adanet.git
```

From the `adanet` root directory run the tests:

```shell
$ cd adanet
$ bazel test -c opt //...
```

Once you have verified that everything works well, install `adanet` as a pip package.

TODO: Add installation instructions for pip.
