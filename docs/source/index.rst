.. AdaNet documentation master file, created by
   sphinx-quickstart on Fri Nov 30 18:10:08 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/tensorflow/adanet

AdaNet documentation
==================================

AdaNet is a TensorFlow framework for fast and flexible AutoML with learning guarantees.

.. raw:: html

    <div align="center">
      <img src="https://raw.githubusercontent.com/tensorflow/adanet/master/images/adanet_tangram_logo.png" alt="adanet_tangram_logo"><br><br>
    </div>

**AdaNet** is a lightweight TensorFlow-based framework for automatically learning high-quality models with minimal expert intervention. AdaNet builds on recent AutoML efforts to be fast and flexible while providing learning guarantees. Importantly, AdaNet provides a general framework for not only learning a neural network architecture, but also for learning to ensemble to obtain even better models.

This project is based on the `AdaNet algorithm`, presented in “`AdaNet: Adaptive Structural Learning of Artificial Neural Networks <http://proceedings.mlr.press/v70/cortes17a.html>`_” at `ICML 2017 <https://icml.cc/Conferences/2017>`_, for learning the structure of a neural network as an ensemble of subnetworks.

AdaNet has the following goals:

* **Ease of use**: Provide familiar APIs (e.g. Keras, Estimator) for training, evaluating, and serving models.
* **Speed**: Scale with available compute and quickly produce high quality models.
* **Flexibility**: Allow researchers and practitioners to extend AdaNet to novel subnetwork architectures, search spaces, and tasks.
* **Learning guarantees**: Optimize an objective that offers theoretical learning guarantees.

The following animation shows AdaNet adaptively growing an ensemble of neural networks. At each iteration, it measures the ensemble loss for each candidate, and selects the best one to move onto the next iteration. At subsequent iterations, the blue subnetworks are frozen, and only yellow subnetworks are trained:

.. raw:: html

    <div align="center" style="max-width: 450px; display: block; margin: 0 auto;">
      <img src="https://raw.githubusercontent.com/tensorflow/adanet/master/images/adanet_animation.gif" alt="adanet_animation"><br><br>
    </div>

AdaNet was first announced on the Google AI research blog: "`Introducing AdaNet: Fast and Flexible AutoML with Learning Guarantees <https://ai.googleblog.com/2018/10/introducing-adanet-fast-and-flexible.html>`_".

This is not an official Google product.

.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Getting Started

  overview
  quick_start
  tutorials
  tensorboard
  distributed
  tpu

.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Research

  algorithm
  theory


.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Package Reference

  adanet
  adanet.ensemble
  adanet.keras
  adanet.subnetwork
  adanet.distributed

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
