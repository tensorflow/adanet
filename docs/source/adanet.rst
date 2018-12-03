.. role:: hidden
    :class: hidden-section

adanet
==============

.. automodule:: adanet
.. currentmodule:: adanet

Estimators
---------------

High-level APIs for training, evaluating, predicting, and serving AdaNet model.

:hidden:`AutoEnsembleEstimator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AutoEnsembleEstimator
    :members:
    :show-inheritance:
    :inherited-members:

:hidden:`Estimator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Estimator
    :members:
    :show-inheritance:
    :inherited-members:

:hidden:`TPUEstimator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TPUEstimator
    :members:
    :show-inheritance:
    :inherited-members:

Ensembles
---------------

Collections representing learned combinations of subnetworks.

:hidden:`MixtureWeightType`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MixtureWeightType
    :members:

:hidden:`WeightedSubnetwork`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: WeightedSubnetwork
    :members:

:hidden:`Ensemble`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Ensemble
    :members:

Evaluator
---------------

Measures :class:`adanet.Ensemble` performance on a given dataset.

:hidden:`Evaluator`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Evaluator
    :members:

Summary
---------------

Extends :mod:`tf.summary` to power AdaNet's TensorBoard integration.

:hidden:`Summary`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Summary
    :members:

ReportMaterializer
---------------

:hidden:`ReportMaterializer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ReportMaterializer
    :members:
