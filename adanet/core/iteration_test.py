"""Test AdaNet iteration single graph implementation.

Copyright 2018 The AdaNet Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from adanet.core.base_learner import BaseLearner
from adanet.core.base_learner import BaseLearnerBuilder
from adanet.core.base_learner_report import BaseLearnerReport
from adanet.core.candidate import _Candidate
from adanet.core.ensemble import Ensemble
from adanet.core.iteration import _Iteration
from adanet.core.iteration import _IterationBuilder
import adanet.core.testing_utils as tu
import tensorflow as tf


def _dummy_candidate():
  """Returns a dummy `_Candidate` instance."""

  return _Candidate(
      ensemble=tu.dummy_ensemble("foo"), adanet_loss=1., is_training=True)


class IterationTest(parameterized.TestCase, tf.test.TestCase):

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters({
      "testcase_name": "single_candidate",
      "number": 0,
      "candidates": [_dummy_candidate()],
      "estimator_spec": tu.dummy_estimator_spec(),
      "best_candidate_index": 0,
      "is_over": True,
  }, {
      "testcase_name": "two_candidates",
      "number": 0,
      "candidates": [_dummy_candidate(),
                     _dummy_candidate()],
      "estimator_spec": tu.dummy_estimator_spec(),
      "best_candidate_index": 0,
      "is_over": True,
  }, {
      "testcase_name": "positive_number",
      "number": 1,
      "candidates": [_dummy_candidate()],
      "estimator_spec": tu.dummy_estimator_spec(),
      "best_candidate_index": 0,
      "is_over": True,
  }, {
      "testcase_name": "false_is_over",
      "number": 1,
      "candidates": [_dummy_candidate()],
      "estimator_spec": tu.dummy_estimator_spec(),
      "best_candidate_index": 0,
      "is_over": False,
  }, {
      "testcase_name": "zero_best_predictions",
      "number": 1,
      "candidates": [_dummy_candidate()],
      "estimator_spec": tu.dummy_estimator_spec(),
      "best_candidate_index": 0,
      "is_over": True,
  }, {
      "testcase_name": "zero_best_loss",
      "number": 1,
      "candidates": [_dummy_candidate()],
      "estimator_spec": tu.dummy_estimator_spec(),
      "best_candidate_index": 0,
      "is_over": True,
  }, {
      "testcase_name":
          "pass_base_learner_report",
      "number":
          1,
      "candidates": [_dummy_candidate()],
      "estimator_spec":
          tu.dummy_estimator_spec(),
      "best_candidate_index":
          0,
      "is_over":
          True,
      "base_learner_reports_fn": lambda: {
          "foo": BaseLearnerReport(
              hparams={"dropout": 1.0},
              attributes={"aoo": tf.constant("aoo")},
              metrics={"moo": (tf.constant("moo1"), tf.constant("moo2"))})
      },
  })
  def test_new(self,
               number,
               candidates,
               estimator_spec,
               best_candidate_index,
               is_over,
               base_learner_reports_fn=None,
               step=0):
    if base_learner_reports_fn is None:
      base_learner_reports = {}
    else:
      base_learner_reports = base_learner_reports_fn()
    with self.test_session():
      iteration = _Iteration(
          number=number,
          candidates=candidates,
          estimator_spec=estimator_spec,
          best_candidate_index=best_candidate_index,
          summaries=[],
          is_over=is_over,
          base_learner_reports=base_learner_reports,
          step=step)
      self.assertEqual(iteration.number, number)
      self.assertEqual(iteration.candidates, candidates)
      self.assertEqual(iteration.estimator_spec, estimator_spec)
      self.assertEqual(iteration.best_candidate_index, best_candidate_index)
      self.assertEqual(iteration.is_over, is_over)
      self.assertEqual(iteration.base_learner_reports, base_learner_reports)
      self.assertEqual(iteration.step, step)

  @parameterized.named_parameters({
      "testcase_name": "negative_number",
      "number": -1,
  }, {
      "testcase_name": "float_number",
      "number": 1.213,
  }, {
      "testcase_name": "none_number",
      "number": None,
  }, {
      "testcase_name": "empty_candidates",
      "candidates": lambda: [],
  }, {
      "testcase_name": "none_candidates",
      "candidates": lambda: None,
  }, {
      "testcase_name": "non_list_candidates",
      "candidates": lambda: {"foo": _dummy_candidate()},
  }, {
      "testcase_name": "none_estimator_spec",
      "estimator_spec": None,
  }, {
      "testcase_name": "none_best_candidate_index",
      "best_candidate_index": None,
  }, {
      "testcase_name": "none_base_learner_reports",
      "base_learner_reports": lambda: None,
  }, {
      "testcase_name": "none_step",
      "step": None,
  })
  def test_new_errors(self,
                      number=0,
                      candidates=lambda: [_dummy_candidate()],
                      estimator_spec=tu.dummy_estimator_spec(),
                      best_candidate_index=0,
                      is_over=True,
                      base_learner_reports=lambda: [],
                      step=0):
    with self.test_session():
      with self.assertRaises(ValueError):
        _Iteration(
            number=number,
            candidates=candidates(),
            estimator_spec=estimator_spec,
            best_candidate_index=best_candidate_index,
            summaries=[],
            is_over=is_over,
            base_learner_reports=base_learner_reports(),
            step=step)


class _FakeBaseLearnerBuilder(BaseLearnerBuilder):

  def __init__(self, name, random_seed=11):
    self._name = name
    self._random_seed = random_seed

  @property
  def name(self):
    return self._name

  @property
  def seed(self):
    return self._random_seed

  def build_base_learner(self,
                         features,
                         logits_dimension,
                         training,
                         iteration_step,
                         summary,
                         previous_ensemble=None):
    base_learner = BaseLearner(
        last_layer=tu.dummy_tensor(),
        logits=tu.dummy_tensor([2, logits_dimension]),
        complexity=tu.dummy_tensor(),
        persisted_tensors={"random_seed": self._random_seed})
    return base_learner

  def build_base_learner_train_op(self, loss, var_list, labels, iteration_step,
                                  summary, previous_ensemble):
    return None

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    return None


class _DNNBaseLearnerBuilder(BaseLearnerBuilder):
  """A simple DNN base learner builder."""

  def __init__(self, name, train_op_fn, layer_size=1, num_layers=1, seed=42):
    self._name = name
    self._layer_size = layer_size
    self._num_layers = num_layers
    self._seed = seed
    self._train_op_fn = train_op_fn

  @property
  def seed(self):
    return self._seed

  @property
  def name(self):
    return self._name

  def build_base_learner(self,
                         features,
                         logits_dimension,
                         training,
                         iteration_step,
                         summary,
                         previous_ensemble=None):
    seed = self._seed
    with tf.variable_scope("dnn"):
      persisted_tensors = {}
      prev_layer_size = 2
      prev_layer = features["x"]
      for i in range(self._num_layers):
        with tf.variable_scope("hidden_layer_{}".format(i)):
          w = tf.get_variable(
              shape=[prev_layer_size, self._layer_size],
              initializer=tf.glorot_uniform_initializer(seed=seed),
              name="weight")
          hidden_layer = tf.matmul(prev_layer, w)
          persisted_tensors["hidden_layer_{}".format(i)] = hidden_layer

        hidden_layer = tf.nn.relu(hidden_layer)
        prev_layer = hidden_layer
        prev_layer_size = self._layer_size

      with tf.variable_scope("logits"):
        logits = tf.layers.dense(
            prev_layer,
            units=logits_dimension,
            kernel_initializer=tf.glorot_uniform_initializer(seed=seed))

    return BaseLearner(
        last_layer=prev_layer,
        logits=logits,
        complexity=3,
        persisted_tensors=persisted_tensors,
    )

  def train_base_learner(self, loss, var_list, logits, labels, iteration_step,
                         summary):
    return self._train_op_fn(loss, var_list)

  def train_mixture_weights(self, loss, var_list, logits, labels,
                            iteration_step, summary):
    return self._train_op_fn(loss, var_list)


class _FakeEnsembleBuilder(object):

  def __init__(self,
               dict_predictions=False,
               eval_metric_ops_fn=None,
               export_output_key=None):
    self._dict_predictions = dict_predictions
    self._eval_metric_ops_fn = lambda: None
    self._export_output_key = export_output_key
    if eval_metric_ops_fn:
      self._eval_metric_ops_fn = eval_metric_ops_fn

  def append_new_base_learner(self, ensemble, base_learner_builder,
                              iteration_step, summary, features, mode, labels):
    del summary
    del mode
    del features
    del labels
    del iteration_step

    num_base_learners = 0
    if ensemble:
      num_base_learners += 1

    return tu.dummy_ensemble(
        name=base_learner_builder.name,
        num_base_learners=num_base_learners,
        random_seed=base_learner_builder.seed,
        dict_predictions=self._dict_predictions,
        eval_metric_ops=self._eval_metric_ops_fn(),
        export_output_key=self._export_output_key)


class _FakeCandidateBuilder(object):

  def build_candidate(self,
                      ensemble,
                      training,
                      iteration_step,
                      summary,
                      previous_ensemble=None,
                      is_previous_best=False):
    del training  # Unused
    del iteration_step  # Unused
    del summary  # Unused
    del previous_ensemble  # Unused
    return _Candidate(
        ensemble=ensemble,
        adanet_loss=ensemble.adanet_loss,
        is_training="training" in ensemble.name,
        is_previous_best=is_previous_best)


def _export_output_tensors(export_outputs):
  """Returns a dict of `Tensor`, tuple of `Tensor`, or dict of `Tensor`."""

  outputs = {}
  for key, export_output in export_outputs.items():
    if isinstance(export_output, tf.estimator.export.ClassificationOutput):
      result = ()
      if export_output.classes is not None:
        result += (tf.string_to_number(export_output.classes),)
      if export_output.scores is not None:
        result += (export_output.scores,)
      outputs[key] = result
    elif isinstance(export_output, tf.estimator.export.RegressionOutput):
      outputs[key] = export_output.value
    elif isinstance(export_output, tf.estimator.export.PredictOutput):
      outputs[key] = export_output.outputs
  return outputs


class IterationBuilderTest(parameterized.TestCase, tf.test.TestCase):

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      {
          "testcase_name": "single_base_learner_fn",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "base_learner_builders": [_FakeBaseLearnerBuilder("training")],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "want_loss": 1.403943,
          "want_predictions": 2.129,
          "want_best_candidate_index": 0,
      }, {
          "testcase_name":
              "single_base_learner_with_eval_metrics",
          "ensemble_builder":
              _FakeEnsembleBuilder(eval_metric_ops_fn=lambda: {
                  "a": (tf.constant(1), tf.constant(2))
              }),
          "base_learner_builders": [_FakeBaseLearnerBuilder("training",),],
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_loss":
              1.403943,
          "want_predictions":
              2.129,
          "want_eval_metric_ops": ["a"],
          "want_best_candidate_index":
              0,
      }, {
          "testcase_name":
              "single_base_learner_with_non_tensor_eval_metric_op",
          "ensemble_builder":
              _FakeEnsembleBuilder(
                  eval_metric_ops_fn=lambda: {
                      "a": (tf.constant(1), tf.no_op())
                  }
              ),
          "base_learner_builders": [_FakeBaseLearnerBuilder("training",),],
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_loss":
              1.403943,
          "want_predictions":
              2.129,
          "want_eval_metric_ops": ["a"],
          "want_best_candidate_index":
              0,
      }, {
          "testcase_name": "single_base_learner_done_training_fn",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "base_learner_builders": [_FakeBaseLearnerBuilder("done")],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "want_loss": 1.403943,
          "want_predictions": 2.129,
          "want_best_candidate_index": 0,
          "want_is_over": True,
      }, {
          "testcase_name": "single_dict_predictions_base_learner_fn",
          "ensemble_builder": _FakeEnsembleBuilder(dict_predictions=True),
          "base_learner_builders": [_FakeBaseLearnerBuilder("training")],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "want_loss": 1.403943,
          "want_predictions": {
              "classes": 2,
              "logits": 2.129
          },
          "want_best_candidate_index": 0,
      }, {
          "testcase_name": "previous_ensemble",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "base_learner_builders": [_FakeBaseLearnerBuilder("training")],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "previous_ensemble": lambda: tu.dummy_ensemble("old"),
          "want_loss": 1.403943,
          "want_predictions": 2.129,
          "want_best_candidate_index": 1,
      }, {
          "testcase_name":
              "previous_ensemble_is_best",
          "ensemble_builder":
              _FakeEnsembleBuilder(),
          "base_learner_builders": [_FakeBaseLearnerBuilder("training")],
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "previous_ensemble":
              lambda: tu.dummy_ensemble("old", random_seed=12),
          "want_loss":
              -.437,
          "want_predictions":
              .688,
          "want_best_candidate_index":
              0,
      }, {
          "testcase_name":
              "previous_ensemble_and_eval_metrics",
          "ensemble_builder":
              _FakeEnsembleBuilder(eval_metric_ops_fn=lambda: {
                  "a": (tf.constant(1), tf.constant(2))
              }),
          "base_learner_builders": [_FakeBaseLearnerBuilder("training")],
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "previous_ensemble":
              lambda: tu.dummy_ensemble("old", eval_metric_ops={
                  "a": (tf.constant(1), tf.constant(2))
              }),
          "want_loss":
              1.403943,
          "want_predictions":
              2.129,
          "want_eval_metric_ops": ["a"],
          "want_best_candidate_index":
              1,
      }, {
          "testcase_name":
              "two_base_learner_fns",
          "ensemble_builder":
              _FakeEnsembleBuilder(),
          "base_learner_builders": [
              _FakeBaseLearnerBuilder("training"),
              _FakeBaseLearnerBuilder("training2", random_seed=7)
          ],
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_loss":
              1.40394,
          "want_predictions":
              2.129,
          "want_best_candidate_index":
              0,
      }, {
          "testcase_name":
              "two_base_learner_fns_other_best",
          "ensemble_builder":
              _FakeEnsembleBuilder(),
          "base_learner_builders": [
              _FakeBaseLearnerBuilder("training"),
              _FakeBaseLearnerBuilder("training2", random_seed=12)
          ],
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_loss":
              -.437,
          "want_predictions":
              .688,
          "want_best_candidate_index":
              1,
      }, {
          "testcase_name":
              "two_base_learner_one_training_fns",
          "ensemble_builder":
              _FakeEnsembleBuilder(),
          "base_learner_builders": [
              _FakeBaseLearnerBuilder("training"),
              _FakeBaseLearnerBuilder("done", random_seed=7)
          ],
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_loss":
              1.403943,
          "want_predictions":
              2.129,
          "want_best_candidate_index":
              0,
      }, {
          "testcase_name":
              "two_base_learner_done_training_fns",
          "ensemble_builder":
              _FakeEnsembleBuilder(),
          "base_learner_builders": [
              _FakeBaseLearnerBuilder("done"),
              _FakeBaseLearnerBuilder("done1", random_seed=7)
          ],
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_loss":
              1.403943,
          "want_predictions":
              2.129,
          "want_best_candidate_index":
              0,
          "want_is_over":
              True,
      }, {
          "testcase_name":
              "two_dict_predictions_base_learner_fns",
          "ensemble_builder":
              _FakeEnsembleBuilder(dict_predictions=True),
          "base_learner_builders": [
              _FakeBaseLearnerBuilder("training"),
              _FakeBaseLearnerBuilder("training2", random_seed=7)
          ],
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_loss":
              1.404,
          "want_predictions": {
              "classes": 2,
              "logits": 2.129
          },
          "want_best_candidate_index":
              0,
      }, {
          "testcase_name":
              "two_dict_predictions_base_learner_fns_predict_classes",
          "ensemble_builder":
              _FakeEnsembleBuilder(
                  dict_predictions=True,
                  export_output_key=tu.ExportOutputKeys.CLASSIFICATION_CLASSES),
          "base_learner_builders": [
              _FakeBaseLearnerBuilder("training"),
              _FakeBaseLearnerBuilder("training2", random_seed=7)
          ],
          "mode":
              tf.estimator.ModeKeys.PREDICT,
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_loss":
              1.404,
          "want_predictions": {
              "classes": 2,
              "logits": 2.129
          },
          "want_best_candidate_index":
              0,
          "want_export_outputs": {
              tu.ExportOutputKeys.CLASSIFICATION_CLASSES: [2.129],
              "serving_default": [2.129],
          },
      }, {
          "testcase_name":
              "two_dict_predictions_base_learner_fns_predict_scores",
          "ensemble_builder":
              _FakeEnsembleBuilder(
                  dict_predictions=True,
                  export_output_key=tu.ExportOutputKeys.CLASSIFICATION_SCORES),
          "base_learner_builders": [
              _FakeBaseLearnerBuilder("training"),
              _FakeBaseLearnerBuilder("training2", random_seed=7)
          ],
          "mode":
              tf.estimator.ModeKeys.PREDICT,
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_loss":
              1.404,
          "want_predictions": {
              "classes": 2,
              "logits": 2.129
          },
          "want_best_candidate_index":
              0,
          "want_export_outputs": {
              tu.ExportOutputKeys.CLASSIFICATION_SCORES: [2.129],
              "serving_default": [2.129],
          },
      }, {
          "testcase_name":
              "two_dict_predictions_base_learner_fns_predict_regression",
          "ensemble_builder":
              _FakeEnsembleBuilder(
                  dict_predictions=True,
                  export_output_key=tu.ExportOutputKeys.REGRESSION),
          "base_learner_builders": [
              _FakeBaseLearnerBuilder("training"),
              _FakeBaseLearnerBuilder("training2", random_seed=7)
          ],
          "mode":
              tf.estimator.ModeKeys.PREDICT,
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_predictions": {
              "classes": 2,
              "logits": 2.129
          },
          "want_best_candidate_index":
              0,
          "want_export_outputs": {
              tu.ExportOutputKeys.REGRESSION: 2.129,
              "serving_default": 2.129,
          },
      }, {
          "testcase_name":
              "two_dict_predictions_base_learner_fns_predict_prediction",
          "ensemble_builder":
              _FakeEnsembleBuilder(
                  dict_predictions=True,
                  export_output_key=tu.ExportOutputKeys.PREDICTION),
          "base_learner_builders": [
              _FakeBaseLearnerBuilder("training"),
              _FakeBaseLearnerBuilder("training2", random_seed=7)
          ],
          "mode":
              tf.estimator.ModeKeys.PREDICT,
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_predictions": {
              "classes": 2,
              "logits": 2.129
          },
          "want_best_candidate_index":
              0,
          "want_export_outputs": {
              tu.ExportOutputKeys.PREDICTION: {
                  "classes": 2,
                  "logits": 2.129
              },
              "serving_default": {
                  "classes": 2,
                  "logits": 2.129
              },
          },
      })
  def test_build_iteration(self,
                           ensemble_builder,
                           base_learner_builders,
                           features,
                           labels,
                           want_predictions,
                           want_best_candidate_index,
                           want_eval_metric_ops=(),
                           want_is_over=False,
                           previous_ensemble=lambda: None,
                           want_loss=None,
                           want_export_outputs=None,
                           mode=tf.estimator.ModeKeys.TRAIN):
    global_step = tf.train.create_global_step()
    builder = _IterationBuilder(_FakeCandidateBuilder(), ensemble_builder)
    iteration = builder.build_iteration(
        iteration_number=0,
        base_learner_builders=base_learner_builders,
        features=features(),
        labels=labels(),
        mode=mode,
        previous_ensemble=previous_ensemble())
    with self.test_session() as sess:
      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())
      sess.run(init)
      estimator_spec = iteration.estimator_spec
      self.assertAllClose(
          want_predictions, sess.run(estimator_spec.predictions), atol=1e-3)
      self.assertEqual(
          set(want_eval_metric_ops), set(estimator_spec.eval_metric_ops.keys()))
      self.assertEqual(want_best_candidate_index,
                       sess.run(iteration.best_candidate_index))
      self.assertEqual(want_is_over, sess.run(iteration.is_over))

      if mode == tf.estimator.ModeKeys.PREDICT:
        self.assertIsNotNone(estimator_spec.export_outputs)
        self.assertAllClose(
            want_export_outputs,
            sess.run(_export_output_tensors(estimator_spec.export_outputs)),
            atol=1e-3)
        self.assertIsNone(iteration.estimator_spec.train_op)
        self.assertIsNone(iteration.estimator_spec.loss)
        self.assertIsNotNone(want_export_outputs)
        return

      self.assertAlmostEqual(
          want_loss, sess.run(iteration.estimator_spec.loss), places=3)
      self.assertIsNone(iteration.estimator_spec.export_outputs)
      if mode == tf.estimator.ModeKeys.TRAIN:
        sess.run(iteration.estimator_spec.train_op)
        self.assertEqual(1, sess.run(global_step))
        self.assertEqual(1, sess.run(iteration.step))

  @parameterized.named_parameters({
      "testcase_name": "empty_base_learner_builders",
      "ensemble_builder": _FakeEnsembleBuilder(),
      "base_learner_builders": [],
      "want_raises": ValueError,
  }, {
      "testcase_name":
          "same_base_learner_builder_names",
      "ensemble_builder":
          _FakeEnsembleBuilder(),
      "base_learner_builders": [
          _FakeBaseLearnerBuilder("same_name"),
          _FakeBaseLearnerBuilder("same_name")
      ],
      "want_raises":
          ValueError,
  }, {
      "testcase_name": "same_name_as_previous_ensemble",
      "ensemble_builder": _FakeEnsembleBuilder(),
      "previous_ensemble_fn": lambda: tu.dummy_ensemble("same_name"),
      "base_learner_builders": [_FakeBaseLearnerBuilder("same_name"),],
      "want_raises": ValueError,
  }, {
      "testcase_name":
          "predict_invalid",
      "ensemble_builder":
          _FakeEnsembleBuilder(
              dict_predictions=True,
              export_output_key=tu.ExportOutputKeys.INVALID),
      "base_learner_builders": [
          _FakeBaseLearnerBuilder("training"),
          _FakeBaseLearnerBuilder("training2", random_seed=7)
      ],
      "mode":
          tf.estimator.ModeKeys.PREDICT,
      "want_raises":
          TypeError,
  })
  def test_build_iteration_error(self,
                                 ensemble_builder,
                                 base_learner_builders,
                                 want_raises,
                                 previous_ensemble_fn=lambda: None,
                                 mode=tf.estimator.ModeKeys.TRAIN):
    builder = _IterationBuilder(_FakeCandidateBuilder(), ensemble_builder)
    features = [[1., -1., 0.]]
    labels = [1]
    with self.test_session():
      with self.assertRaises(want_raises):
        builder.build_iteration(
            iteration_number=0,
            base_learner_builders=base_learner_builders,
            features=features,
            labels=labels,
            mode=mode,
            previous_ensemble=previous_ensemble_fn())


class _HeadEnsembleBuilder(object):

  def __init__(self, head):
    self._head = head

  def append_new_base_learner(self, ensemble, base_learner_builder,
                              iteration_step, summary, features, mode, labels):
    del ensemble
    del base_learner_builder
    del iteration_step
    del summary

    logits = [[.5]]

    estimator_spec = self._head.create_estimator_spec(
        features=features, mode=mode, labels=labels, logits=logits)
    return Ensemble(
        name="test",
        weighted_base_learners=None,
        bias=None,
        logits=None,
        predictions=estimator_spec.predictions,
        loss=None,
        adanet_loss=.1,
        complexity_regularized_loss=None,
        train_op=None,
        complexity_regularization=None,
        eval_metric_ops=None,
        export_outputs=estimator_spec.export_outputs)


class ExportOutputsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      "testcase_name": "regression_head",
      "head": tf.contrib.estimator.regression_head(),
  }, {
      "testcase_name": "binary_classification_head",
      "head": tf.contrib.estimator.binary_classification_head(),
  })
  def test_head_export_outputs(self, head):
    ensemble_builder = _HeadEnsembleBuilder(head)
    builder = _IterationBuilder(_FakeCandidateBuilder(), ensemble_builder)
    features = [[1., -1., 0.]]
    labels = [1]
    mode = tf.estimator.ModeKeys.PREDICT
    iteration = builder.build_iteration(
        iteration_number=0,
        base_learner_builders=[_FakeBaseLearnerBuilder("test")],
        features=features,
        labels=labels,
        mode=mode)

    # Compare iteration outputs with default head outputs.
    spec = head.create_estimator_spec(
        features=features, labels=labels, mode=mode, logits=[[.5]])
    self.assertEqual(
        len(spec.export_outputs), len(iteration.estimator_spec.export_outputs))
    with self.test_session() as sess:
      for key in spec.export_outputs:
        if isinstance(spec.export_outputs[key],
                      tf.estimator.export.RegressionOutput):
          self.assertAlmostEqual(
              sess.run(spec.export_outputs[key].value),
              sess.run(iteration.estimator_spec.export_outputs[key].value))
          continue
        if isinstance(spec.export_outputs[key],
                      tf.estimator.export.ClassificationOutput):
          self.assertAllClose(
              sess.run(spec.export_outputs[key].scores),
              sess.run(iteration.estimator_spec.export_outputs[key].scores))
          self.assertAllEqual(
              sess.run(spec.export_outputs[key].classes),
              sess.run(iteration.estimator_spec.export_outputs[key].classes))
          continue
        if isinstance(spec.export_outputs[key],
                      tf.estimator.export.PredictOutput):
          if "classes" in spec.export_outputs[key].outputs:
            # Verify string Tensor outputs separately.
            self.assertAllEqual(
                sess.run(spec.export_outputs[key].outputs["classes"]),
                sess.run(iteration.estimator_spec.export_outputs[key].outputs[
                    "classes"]))
            del spec.export_outputs[key].outputs["classes"]
            del iteration.estimator_spec.export_outputs[key].outputs["classes"]
          self.assertAllClose(
              sess.run(spec.export_outputs[key].outputs),
              sess.run(iteration.estimator_spec.export_outputs[key].outputs))
          continue
        self.fail("Invalid export_output for {}.".format(key))


if __name__ == "__main__":
  tf.test.main()
