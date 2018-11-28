# AdaNet

[![PyPI version](https://badge.fury.io/py/adanet.svg)](https://badge.fury.io/py/adanet)
[![Travis](https://travis-ci.org/tensorflow/adanet.svg?branch=master)](https://travis-ci.org/tensorflow/adanet)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/tensorflow/adanet/blob/master/LICENSE)

<div align="center">
  <img src="https://tensorflow.github.io/adanet/images/adanet_tangram_logo.png" alt="adanet_tangram_logo"><br><br>
</div>

**AdaNet** is a lightweight TensorFlow-based framework for automatically learning high-quality models with minimal expert intervention. AdaNet builds on recent AutoML efforts to be fast and flexible while providing learning guarantees. Importantly, AdaNet provides a general framework for not only learning a neural network architecture, but also for learning to ensemble to obtain even better models.

This project is based on the _AdaNet algorithm_, presented in “[AdaNet: Adaptive Structural Learning of Artificial Neural Networks](http://proceedings.mlr.press/v70/cortes17a.html)” at [ICML 2017](https://icml.cc/Conferences/2017), for learning the structure of a neural network as an ensemble of subnetworks.

AdaNet has the following goals:

* _Ease of use_: Provide familiar APIs for easily training, evaluating, and serving models.
* _Speed_: Scale with available compute with distributed training on CPU and GPU (TPU support coming soon).
* _Flexibility_: Provide an extensible API for defining new subnetworks for different tasks.
* _Learning guarantees_: Optimize an objective that offers theoretical learning guarantees.

The following animation shows AdaNet adaptively growing an ensemble of neural networks. At each iteration, it measures the ensemble loss for each candidate, and selects the best one to move onto the next iteration. At subsequent iterations, the blue subnetworks are frozen, and only yellow subnetworks are trained:

<div align="center" style="max-width: 450px; display: block; margin: 0 auto;">
  <img src="https://tensorflow.github.io/adanet/images/adanet_animation.gif" alt="adanet_tangram_logo"><br><br>
</div>

AdaNet was first announced on the Google AI research blog: "[Introducing AdaNet: Fast and Flexible AutoML with Learning Guarantees](https://ai.googleblog.com/2018/10/introducing-adanet-fast-and-flexible.html)".

This is not an official Google product.

## Example

A simple example of learning to ensemble linear and neural network models:

```python
import adanet
import tensorflow

# Define the model head for computing loss and evaluation metrics.
head = tf.contrib.estimator.multi_class_head(n_classes=10)

# Feature columns define how to process examples.
feature_columns = ...

# Learn to ensemble linear and neural network models.
estimator = adanet.AutoEnsembleEstimator(
    head=head,
    candidate_pool=[
        tf.estimator.LinearEstimator(
            head=head,
            feature_columns=feature_columns,
            optimizer=tf.train.FtrlOptimizer(...)),
        tf.estimator.DNNEstimator(
            head=head,
            feature_columns=feature_columns,
            optimizer=tf.train.ProximalAdagradOptimizer(...),
            hidden_units=[1000, 500, 100])],
    max_iteration_steps=50)

estimator.train(input_fn=train_input_fn, steps=100)
metrics = estimator.evaluate(input_fn=eval_input_fn)
predictions = estimator.predict(input_fn=predict_input_fn)
```

## Getting Started

To get you started:

- [Tutorials: for understanding the AdaNet algorithm and learning to use this package](./adanet/examples/tutorials)

## Requirements

Requires [Python](https://www.python.org/) 2.7, 3.4, 3.5, or 3.6.

`adanet` depends on bug fixes and enhancements not present in TensorFlow releases prior to 1.9. You must install or upgrade your TensorFlow package to at least 1.9:

```shell
$ pip install "tensorflow>=1.9.0"
```

## Installing with Pip

You can use the [pip package manager](https://pip.pypa.io/en/stable/installing/) to install the official `adanet` package from [PyPi](https://pypi.org/project/adanet/):

```shell
$ pip install adanet
```

## Installing from source

To install from source first you'll need to install `bazel` following their [installation instructions](https://docs.bazel.build/versions/master/install.html).

Next clone `adanet` and `cd` into its root directory:

```shell
$ git clone https://github.com/tensorflow/adanet && cd adanet
```

From the `adanet` root directory run the tests:

```shell
$ cd adanet
$ bazel test -c opt //...
```

Once you have verified that everything works well, install `adanet` as a [ pip package ](./adanet/pip_package/PIP.md).

You are now ready to experiment with `adanet`.

```python
import adanet
```

## License

AdaNet is released under the [Apache License 2.0](LICENSE).
