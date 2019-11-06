# AdaNet

<div align="center">
  <img src="https://tensorflow.github.io/adanet/images/adanet_tangram_logo.png" alt="adanet_tangram_logo"><br><br>
</div>

[![Documentation Status](https://readthedocs.org/projects/adanet/badge)](https://adanet.readthedocs.io)
[![PyPI version](https://badge.fury.io/py/adanet.svg)](https://badge.fury.io/py/adanet)
[![Travis](https://travis-ci.org/tensorflow/adanet.svg?branch=master)](https://travis-ci.org/tensorflow/adanet)
[![codecov](https://codecov.io/gh/tensorflow/adanet/branch/master/graph/badge.svg)](https://codecov.io/gh/tensorflow/adanet)
[![Gitter](https://badges.gitter.im/tensorflow/adanet.svg)](https://gitter.im/tensorflow/adanet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Downloads](https://pepy.tech/badge/adanet)](https://pepy.tech/project/adanet)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/tensorflow/adanet/blob/master/LICENSE)

**AdaNet** is a lightweight TensorFlow-based framework for automatically learning high-quality models with minimal expert intervention. AdaNet builds on recent AutoML efforts to be fast and flexible while providing learning guarantees. Importantly, AdaNet provides a general framework for not only learning a neural network architecture, but also for learning to ensemble to obtain even better models.

This project is based on the _AdaNet algorithm_, presented in “[AdaNet: Adaptive Structural Learning of Artificial Neural Networks](http://proceedings.mlr.press/v70/cortes17a.html)” at [ICML 2017](https://icml.cc/Conferences/2017), for learning the structure of a neural network as an ensemble of subnetworks.

AdaNet has the following goals:

* _Ease of use_: Provide familiar APIs (e.g. Keras, Estimator) for training, evaluating, and serving models.
* _Speed_: Scale with available compute and quickly produce high quality models.
* _Flexibility_: Allow researchers and practitioners to extend AdaNet to novel subnetwork architectures, search spaces, and tasks.
* _Learning guarantees_: Optimize an objective that offers theoretical learning guarantees.

The following animation shows AdaNet adaptively growing an ensemble of neural networks. At each iteration, it measures the ensemble loss for each candidate, and selects the best one to move onto the next iteration. At subsequent iterations, the blue subnetworks are frozen, and only yellow subnetworks are trained:

<div align="center" style="max-width: 450px; display: block; margin: 0 auto;">
  <img src="https://tensorflow.github.io/adanet/images/adanet_animation.gif" alt="adanet_tangram_logo"><br><br>
</div>

AdaNet was first announced on the Google AI research blog: "[Introducing AdaNet: Fast and Flexible AutoML with Learning Guarantees](https://ai.googleblog.com/2018/10/introducing-adanet-fast-and-flexible.html)".

This is not an official Google product.

## Features

AdaNet provides the following AutoML features:

 * Adaptive neural architecture search and ensemble learning in a single train call.
 * Regression, binary and multi-class classification, and multi-head task support.
 * A [`tf.estimator.Estimator`](https://www.tensorflow.org/guide/estimators) API for training, evaluation, prediction, and serving models.
 * The [`adanet.AutoEnsembleEstimator`](https://github.com/tensorflow/adanet/blob/master/adanet/autoensemble/estimator.py) for learning to ensemble user-defined `tf.estimator.Estimators`.
 * The ability to define subnetworks that change structure over time using [`tf.layers`](https://www.tensorflow.org/api_docs/python/tf/layers) via the [`adanet.subnetwork` API](https://github.com/tensorflow/adanet/blob/master/adanet/subnetwork/generator.py).
 * CPU, GPU, and TPU support.
 * [Distributed multi-server training](https://cloud.google.com/blog/products/gcp/easy-distributed-training-with-tensorflow-using-tfestimatortrain-and-evaluate-on-cloud-ml-engine).
 * TensorBoard integration.

## Example

A simple example of learning to ensemble linear and neural network models:

```python
import adanet
import tensorflow as tf

# Define the model head for computing loss and evaluation metrics.
head = MultiClassHead(n_classes=10)

# Feature columns define how to process examples.
feature_columns = ...

# Learn to ensemble linear and neural network models.
estimator = adanet.AutoEnsembleEstimator(
    head=head,
    candidate_pool={
        "linear":
            tf.estimator.LinearEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=...),
        "dnn":
            tf.estimator.DNNEstimator(
                head=head,
                feature_columns=feature_columns,
                optimizer=...,
                hidden_units=[1000, 500, 100])},
    max_iteration_steps=50)

estimator.train(input_fn=train_input_fn, steps=100)
metrics = estimator.evaluate(input_fn=eval_input_fn)
predictions = estimator.predict(input_fn=predict_input_fn)
```

## Getting Started

To get you started:

- [API Documentation](https://adanet.readthedocs.io)
- [Tutorials: for understanding the AdaNet algorithm and learning to use this package](./adanet/examples/tutorials)

## Requirements

Requires [Python](https://www.python.org/) 2.7, 3.4, 3.5, 3.6, or 3.7.

`adanet` supports both TensorFlow 2.0 and TensorFlow >=1.14. It depends on bug fixes and enhancements not present in TensorFlow releases prior to 1.14. You must install or upgrade your TensorFlow package to at least 1.14:

```shell
$ pip install "tensorflow>=1.14,<2.1"
```

## Installing with Pip

You can use the [pip package manager](https://pip.pypa.io/en/stable/installing/) to install the official `adanet` package from [PyPi](https://pypi.org/project/adanet/):

```shell
$ pip install adanet
```

## Installing from Source

To install from source first you'll need to install `bazel` following their [installation instructions](https://docs.bazel.build/versions/master/install.html).

Next clone the `adanet` repository:

```shell
$ git clone https://github.com/tensorflow/adanet
$ cd adanet
```

From the `adanet` root directory run the tests:

```shell
$ bazel build -c opt //...
# Run tests with nosetests, but skip example tests.
$ NOSE_EXCLUDE='.*nasnet.*.py.*' python3 -m nose
```

Once you have verified that the tests have passed, install `adanet` from source as a [ pip package ](./adanet/pip_package/PIP.md).

You are now ready to experiment with `adanet`.

```python
import adanet
```

## Citing this Work

If you use this AdaNet library for academic research, you are encouraged to cite the following paper from the [ICML 2019 AutoML Workshop](https://arxiv.org/abs/1905.00080):

    @misc{weill2019adanet,
        title={AdaNet: A Scalable and Flexible Framework for Automatically Learning Ensembles},
        author={Charles Weill and Javier Gonzalvo and Vitaly Kuznetsov and Scott Yang and Scott Yak and Hanna Mazzawi and Eugen Hotaj and Ghassen Jerfel and Vladimir Macko and Ben Adlam and Mehryar Mohri and Corinna Cortes},
        year={2019},
        eprint={1905.00080},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

## License

AdaNet is released under the [Apache License 2.0](LICENSE).
