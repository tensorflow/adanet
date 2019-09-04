# Quick start



If you are already using
[`tf.estimator.Estimator`](https://www.tensorflow.org/guide/estimators), the
fastest way to get up and running with AdaNet is to use the
[`adanet.AutoEnsembleEstimator`](https://adanet.readthedocs.io/en/latest/adanet.html#autoensembleestimator).
This estimator will automatically convert a list of estimators into subnetworks,
and learn to ensemble them for you.

## Import AdaNet

The first step is to import the `adanet` package:

```python
import adanet
```


## AutoEnsembleEstimator

Next you will want to define which estimators you want to ensemble. For example,
if you don't know if the best model a linear model, or a neural network, or some
combination, then you can try using `tf.estimator.LinearEstimator` and
`tf.estimator.DNNEstimator` as subnetworks:

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
    candidate_pool=lambda config: {
        "linear":
            tf.estimator.LinearEstimator(
                head=head,
                feature_columns=feature_columns,
                config=config,
                optimizer=...),
        "dnn":
            tf.estimator.DNNEstimator(
                head=head,
                feature_columns=feature_columns,
                config=config,
                optimizer=...,
                hidden_units=[1000, 500, 100])},
    max_iteration_steps=50)

estimator.train(input_fn=train_input_fn, steps=100)
metrics = estimator.evaluate(input_fn=eval_input_fn)
predictions = estimator.predict(input_fn=predict_input_fn)
```

The above code will train both the `linear` and `dnn` subnetworks in parallel,
and will average their predictions. After `max_iteration_steps=100` steps, the
best subnetwork will compose the ensemble according to its performance on the
*training set*.

## Ensemble strategies

The way AdaNet chooses which subnetworks to include in a candidate ensemble is
via **ensemble strategies**.

### Grow strategy

The default ensemble strategy is `adanet.ensemble.GrowStrategy` which will only
select the subnetwork that most improved the ensemble's performance. The
remaining subnetworks will be pruned from the graph.

### All strategy

Suppose instead of only selecting the *single best* subnetwork, you want to
ensemble *all* of the subnetworks, regardless of their individual performance.
You can pass an instance of the `adanet.ensemble.AllStrategy` to the
`adanet.AutoEnsembleEstimator` constructor:

```python
estimator = adanet.AutoEnsembleEstimator(
    [...]
    ensemble_strategies=[adanet.ensemble.AllStrategy()]
    candidate_pool={
        "linear": ...,
        "dnn": ...,
    },
    [...])
```

<!-- TODO: Evaluators, ensemblers, custom subnetworks. -->

## Tutorials

To play with AdaNet in Colab notebooks, and learn about more advanced features
like customizing AdaNet and training on TPU, see our
[tutorials section](./tutorials).

