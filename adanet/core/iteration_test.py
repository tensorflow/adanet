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

import functools

from absl.testing import parameterized
from adanet import tf_compat
from adanet.core.architecture import _Architecture
from adanet.core.candidate import _Candidate
from adanet.core.ensemble_builder import _EnsembleSpec
from adanet.core.ensemble_builder import _SubnetworkSpec
from adanet.core.iteration import _Iteration
from adanet.core.iteration import _IterationBuilder
from adanet.core.iteration import _TrainManager
from adanet.core.summary import _ScopedSummary
from adanet.core.summary import _TPUScopedSummary
import adanet.core.testing_utils as tu
from adanet.ensemble import Candidate as EnsembleCandidate
from adanet.subnetwork import Builder as SubnetworkBuilder
from adanet.subnetwork import Report as SubnetworkReport
from adanet.subnetwork import Subnetwork
from adanet.subnetwork import TrainOpSpec
import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
# pylint: enable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import regression_head


def _dummy_candidate():
  """Returns a dummy `_Candidate` instance."""

  return _Candidate(ensemble_spec=tu.dummy_ensemble_spec("foo"), adanet_loss=1.)


class IterationTest(tu.AdanetTestCase):

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      {
          "testcase_name": "single_candidate",
          "number": 0,
          "candidates": [_dummy_candidate()],
          "estimator_spec": tu.dummy_estimator_spec(),
          "best_candidate_index": 0,
      }, {
          "testcase_name": "two_candidates",
          "number": 0,
          "candidates": [_dummy_candidate(),
                         _dummy_candidate()],
          "estimator_spec": tu.dummy_estimator_spec(),
          "best_candidate_index": 0,
      }, {
          "testcase_name": "positive_number",
          "number": 1,
          "candidates": [_dummy_candidate()],
          "estimator_spec": tu.dummy_estimator_spec(),
          "best_candidate_index": 0,
      }, {
          "testcase_name": "zero_best_predictions",
          "number": 1,
          "candidates": [_dummy_candidate()],
          "estimator_spec": tu.dummy_estimator_spec(),
          "best_candidate_index": 0,
      }, {
          "testcase_name": "zero_best_loss",
          "number": 1,
          "candidates": [_dummy_candidate()],
          "estimator_spec": tu.dummy_estimator_spec(),
          "best_candidate_index": 0,
      }, {
          "testcase_name":
              "pass_subnetwork_report",
          "number":
              1,
          "candidates": [_dummy_candidate()],
          "estimator_spec":
              tu.dummy_estimator_spec(),
          "best_candidate_index":
              0,
          "subnetwork_reports_fn":
              lambda: {
                  "foo":
                      SubnetworkReport(
                          hparams={"dropout": 1.0},
                          attributes={"aoo": tf.constant("aoo")},
                          metrics={
                              "moo": (tf.constant("moo1"), tf.constant("moo2"))
                          })
              },
      })
  @test_util.run_in_graph_and_eager_modes
  def test_new(self,
               number,
               candidates,
               estimator_spec,
               best_candidate_index,
               subnetwork_reports_fn=None):
    if subnetwork_reports_fn is None:
      subnetwork_reports = {}
    else:
      subnetwork_reports = subnetwork_reports_fn()
    iteration = _Iteration(
        number=number,
        candidates=candidates,
        subnetwork_specs=None,
        estimator_spec=estimator_spec,
        best_candidate_index=best_candidate_index,
        summaries=[],
        subnetwork_reports=subnetwork_reports,
        train_manager=_TrainManager([], [],
                                    self.test_subdirectory,
                                    is_chief=True))
    self.assertEqual(iteration.number, number)
    self.assertEqual(iteration.candidates, candidates)
    self.assertEqual(iteration.estimator_spec, estimator_spec)
    self.assertEqual(iteration.best_candidate_index, best_candidate_index)
    self.assertEqual(iteration.subnetwork_reports, subnetwork_reports)

  @parameterized.named_parameters(
      {
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
          "candidates": lambda: {
              "foo": _dummy_candidate()
          },
      }, {
          "testcase_name": "none_estimator_spec",
          "estimator_spec": None,
      }, {
          "testcase_name": "none_best_candidate_index",
          "best_candidate_index": None,
      }, {
          "testcase_name": "none_subnetwork_reports",
          "subnetwork_reports": lambda: None,
      })
  @test_util.run_in_graph_and_eager_modes
  def test_new_errors(self,
                      number=0,
                      candidates=lambda: [_dummy_candidate()],
                      estimator_spec=tu.dummy_estimator_spec(),
                      best_candidate_index=0,
                      subnetwork_reports=lambda: []):
    with self.assertRaises(ValueError):
      _Iteration(
          number=number,
          candidates=candidates(),
          subnetwork_specs=None,
          estimator_spec=estimator_spec,
          best_candidate_index=best_candidate_index,
          summaries=[],
          subnetwork_reports=subnetwork_reports(),
          train_manager=_TrainManager([], [],
                                      self.test_subdirectory,
                                      is_chief=True))


class _FakeBuilder(SubnetworkBuilder):

  def __init__(self, name, random_seed=11, chief_hook=None):
    self._name = name
    self._random_seed = random_seed
    self._chief_hook = chief_hook

  @property
  def name(self):
    return self._name

  @property
  def seed(self):
    return self._random_seed

  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    return Subnetwork(
        last_layer=tu.dummy_tensor(),
        logits=tu.dummy_tensor([2, logits_dimension]),
        complexity=tu.dummy_tensor(),
        persisted_tensors={"random_seed": self._random_seed})

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    if self._chief_hook:
      return TrainOpSpec(
          train_op=tf.no_op(), chief_hooks=[self._chief_hook], hooks=None)
    return None

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    return None


class _FakeEnsembleBuilder(object):

  def __init__(self,
               dict_predictions=False,
               eval_metric_ops_fn=None,
               export_output_key=None):
    self._dict_predictions = dict_predictions
    if not eval_metric_ops_fn:
      eval_metric_ops_fn = lambda: {"a": (tf.constant(1), tf.constant(1))}
    self._eval_metric_ops_fn = eval_metric_ops_fn
    self._export_output_key = export_output_key

  def build_ensemble_spec(self,
                          name,
                          candidate,
                          ensembler,
                          subnetwork_specs,
                          summary,
                          features,
                          mode,
                          iteration_number,
                          labels=None,
                          previous_ensemble_spec=None,
                          my_ensemble_index=None,
                          params=None):
    del ensembler
    del subnetwork_specs
    del summary
    del features
    del mode
    del labels
    del iteration_number
    del params
    del my_ensemble_index

    num_subnetworks = 0
    if previous_ensemble_spec:
      num_subnetworks += 1

    return tu.dummy_ensemble_spec(
        name=name,
        num_subnetworks=num_subnetworks,
        random_seed=candidate.subnetwork_builders[0].seed,
        subnetwork_builders=candidate.subnetwork_builders,
        dict_predictions=self._dict_predictions,
        eval_metrics=tu.create_ensemble_metrics(
            metric_fn=self._eval_metric_ops_fn),
        export_output_key=self._export_output_key)


class _FakeSubnetworkManager(object):

  def build_subnetwork_spec(self,
                            name,
                            subnetwork_builder,
                            summary,
                            features,
                            mode,
                            labels=None,
                            previous_ensemble=None,
                            config=None,
                            params=None):
    del summary
    del features
    del mode
    del labels
    del previous_ensemble
    del params
    del config

    return _SubnetworkSpec(
        name=name,
        subnetwork=None,
        builder=subnetwork_builder,
        step=tf.Variable(0.),
        predictions=None,
        loss=None,
        train_op=subnetwork_builder.build_subnetwork_train_op(
            *[None for _ in range(7)]),
        eval_metrics=tu.create_subnetwork_metrics(
            metric_fn=lambda: {"a": (tf.constant(1), tf.constant(1))}))


class _FakeCandidateBuilder(object):

  def build_candidate(self,
                      ensemble_spec,
                      training,
                      summary,
                      previous_ensemble_spec=None):
    del training  # Unused
    del summary  # Unused
    del previous_ensemble_spec  # Unused

    return _Candidate(
        ensemble_spec=ensemble_spec, adanet_loss=ensemble_spec.adanet_loss)


def _export_output_tensors(export_outputs):
  """Returns a dict of `Tensor`, tuple of `Tensor`, or dict of `Tensor`."""

  outputs = {}
  for key, export_output in export_outputs.items():
    if isinstance(export_output, tf.estimator.export.ClassificationOutput):
      result = ()
      if export_output.classes is not None:
        result += (tf.strings.to_number(export_output.classes),)
      if export_output.scores is not None:
        result += (export_output.scores,)
      outputs[key] = result
    elif isinstance(export_output, tf.estimator.export.RegressionOutput):
      outputs[key] = export_output.value
    elif isinstance(export_output, tf.estimator.export.PredictOutput):
      outputs[key] = export_output.outputs
  return outputs


class _FakeEnsembler(object):

  @property
  def name(self):
    return "fake_ensembler"


class IterationBuilderTest(tu.AdanetTestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "negative_max_steps",
          "max_steps": -1,
      }, {
          "testcase_name": "zero_max_steps",
          "max_steps": 0,
      })
  @test_util.run_in_graph_and_eager_modes
  def test_init_errors(self, max_steps):
    with self.assertRaises(ValueError):
      _IterationBuilder(
          _FakeCandidateBuilder(),
          _FakeSubnetworkManager(),
          _FakeEnsembleBuilder(),
          summary_maker=_ScopedSummary,
          ensemblers=[_FakeEnsembler()],
          max_steps=max_steps)

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters(
      {
          "testcase_name": "single_subnetwork_fn",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "subnetwork_builders": [_FakeBuilder("training")],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "want_loss": 1.403943,
          "want_predictions": 2.129,
          "want_best_candidate_index": 0,
      }, {
          "testcase_name":
              "single_subnetwork_fn_mock_summary",
          "ensemble_builder":
              _FakeEnsembleBuilder(),
          "subnetwork_builders": [_FakeBuilder("training")],
          "summary_maker":
              functools.partial(_TPUScopedSummary, logdir="/tmp/fakedir"),
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
              "single_subnetwork_with_eval_metrics",
          "ensemble_builder":
              _FakeEnsembleBuilder(eval_metric_ops_fn=lambda:
                                   {"a": (tf.constant(1), tf.constant(2))}),
          "subnetwork_builders": [_FakeBuilder("training",),],
          "mode":
              tf.estimator.ModeKeys.EVAL,
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_loss":
              1.403943,
          "want_predictions":
              2.129,
          "want_eval_metric_ops": ["a", "iteration"],
          "want_best_candidate_index":
              0,
      }, {
          "testcase_name":
              "single_subnetwork_with_non_tensor_eval_metric_op",
          "ensemble_builder":
              _FakeEnsembleBuilder(eval_metric_ops_fn=lambda:
                                   {"a": (tf.constant(1), tf.no_op())}),
          "subnetwork_builders": [_FakeBuilder("training",),],
          "mode":
              tf.estimator.ModeKeys.EVAL,
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "want_loss":
              1.403943,
          "want_predictions":
              2.129,
          "want_eval_metric_ops": ["a", "iteration"],
          "want_best_candidate_index":
              0,
      }, {
          "testcase_name": "single_subnetwork_done_training_fn",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "subnetwork_builders": [_FakeBuilder("done")],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "want_loss": 1.403943,
          "want_predictions": 2.129,
          "want_best_candidate_index": 0,
      }, {
          "testcase_name": "single_dict_predictions_subnetwork_fn",
          "ensemble_builder": _FakeEnsembleBuilder(dict_predictions=True),
          "subnetwork_builders": [_FakeBuilder("training")],
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
          "subnetwork_builders": [_FakeBuilder("training")],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "previous_ensemble_spec": lambda: tu.dummy_ensemble_spec("old"),
          "want_loss": 1.403943,
          "want_predictions": 2.129,
          "want_best_candidate_index": 1,
      }, {
          "testcase_name":
              "previous_ensemble_is_best",
          "ensemble_builder":
              _FakeEnsembleBuilder(),
          "subnetwork_builders": [_FakeBuilder("training")],
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "previous_ensemble_spec":
              lambda: tu.dummy_ensemble_spec("old", random_seed=12),
          "want_loss":
              -.437,
          "want_predictions":
              .688,
          "want_best_candidate_index":
              0,
      }, {
          "testcase_name":
              "previous_ensemble_spec_and_eval_metrics",
          "ensemble_builder":
              _FakeEnsembleBuilder(eval_metric_ops_fn=lambda:
                                   {"a": (tf.constant(1), tf.constant(2))}),
          "subnetwork_builders": [_FakeBuilder("training")],
          "mode":
              tf.estimator.ModeKeys.EVAL,
          "features":
              lambda: [[1., -1., 0.]],
          "labels":
              lambda: [1],
          "previous_ensemble_spec":
              lambda: tu.dummy_ensemble_spec(
                  "old",
                  eval_metrics=tu.create_ensemble_metrics(metric_fn=lambda: {
                      "a": (tf.constant(1), tf.constant(2))
                  })),
          "want_loss":
              1.403943,
          "want_predictions":
              2.129,
          "want_eval_metric_ops": ["a", "iteration"],
          "want_best_candidate_index":
              1,
      }, {
          "testcase_name": "two_subnetwork_fns",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "subnetwork_builders": [
              _FakeBuilder("training"),
              _FakeBuilder("training2", random_seed=7)
          ],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "want_loss": 1.40394,
          "want_predictions": 2.129,
          "want_best_candidate_index": 0,
      }, {
          "testcase_name": "two_subnetwork_fns_other_best",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "subnetwork_builders": [
              _FakeBuilder("training"),
              _FakeBuilder("training2", random_seed=12)
          ],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "want_loss": -.437,
          "want_predictions": .688,
          "want_best_candidate_index": 1,
      }, {
          "testcase_name": "two_subnetwork_one_training_fns",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "subnetwork_builders":
              [_FakeBuilder("training"),
               _FakeBuilder("done", random_seed=7)],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "want_loss": 1.403943,
          "want_predictions": 2.129,
          "want_best_candidate_index": 0,
      }, {
          "testcase_name": "two_subnetwork_done_training_fns",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "subnetwork_builders":
              [_FakeBuilder("done"),
               _FakeBuilder("done1", random_seed=7)],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "want_loss": 1.403943,
          "want_predictions": 2.129,
          "want_best_candidate_index": 0,
      }, {
          "testcase_name": "two_dict_predictions_subnetwork_fns",
          "ensemble_builder": _FakeEnsembleBuilder(dict_predictions=True),
          "subnetwork_builders": [
              _FakeBuilder("training"),
              _FakeBuilder("training2", random_seed=7)
          ],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "want_loss": 1.404,
          "want_predictions": {
              "classes": 2,
              "logits": 2.129
          },
          "want_best_candidate_index": 0,
      }, {
          "testcase_name":
              "two_dict_predictions_subnetwork_fns_predict_classes",
          "ensemble_builder":
              _FakeEnsembleBuilder(
                  dict_predictions=True,
                  export_output_key=tu.ExportOutputKeys.CLASSIFICATION_CLASSES),
          "subnetwork_builders": [
              _FakeBuilder("training"),
              _FakeBuilder("training2", random_seed=7)
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
              "two_dict_predictions_subnetwork_fns_predict_scores",
          "ensemble_builder":
              _FakeEnsembleBuilder(
                  dict_predictions=True,
                  export_output_key=tu.ExportOutputKeys.CLASSIFICATION_SCORES),
          "subnetwork_builders": [
              _FakeBuilder("training"),
              _FakeBuilder("training2", random_seed=7)
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
              "two_dict_predictions_subnetwork_fns_predict_regression",
          "ensemble_builder":
              _FakeEnsembleBuilder(
                  dict_predictions=True,
                  export_output_key=tu.ExportOutputKeys.REGRESSION),
          "subnetwork_builders": [
              _FakeBuilder("training"),
              _FakeBuilder("training2", random_seed=7)
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
              "two_dict_predictions_subnetwork_fns_predict_prediction",
          "ensemble_builder":
              _FakeEnsembleBuilder(
                  dict_predictions=True,
                  export_output_key=tu.ExportOutputKeys.PREDICTION),
          "subnetwork_builders": [
              _FakeBuilder("training"),
              _FakeBuilder("training2", random_seed=7)
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
      }, {
          "testcase_name": "chief_session_run_hook",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "subnetwork_builders": [
              _FakeBuilder("training", chief_hook=tu.ModifierSessionRunHook())
          ],
          "features": lambda: [[1., -1., 0.]],
          "labels": lambda: [1],
          "want_loss": 1.403943,
          "want_predictions": 2.129,
          "want_best_candidate_index": 0,
          "want_chief_hooks": True,
      })
  @test_util.run_in_graph_and_eager_modes
  def test_build_iteration(self,
                           ensemble_builder,
                           subnetwork_builders,
                           features,
                           labels,
                           want_predictions,
                           want_best_candidate_index,
                           want_eval_metric_ops=(),
                           previous_ensemble_spec=lambda: None,
                           want_loss=None,
                           want_export_outputs=None,
                           mode=tf.estimator.ModeKeys.TRAIN,
                           summary_maker=_ScopedSummary,
                           want_chief_hooks=False):
    with context.graph_mode():
      tf_compat.v1.train.create_global_step()
      builder = _IterationBuilder(
          _FakeCandidateBuilder(),
          _FakeSubnetworkManager(),
          ensemble_builder,
          summary_maker=summary_maker,
          ensemblers=[_FakeEnsembler()],
          max_steps=1)
      iteration = builder.build_iteration(
          base_global_step=0,
          iteration_number=0,
          ensemble_candidates=[
              EnsembleCandidate(b.name, [b], None) for b in subnetwork_builders
          ],
          subnetwork_builders=subnetwork_builders,
          features=features(),
          labels=labels(),
          mode=mode,
          config=tf.estimator.RunConfig(model_dir=self.test_subdirectory),
          previous_ensemble_spec=previous_ensemble_spec())
      init = tf.group(tf_compat.v1.global_variables_initializer(),
                      tf_compat.v1.local_variables_initializer())
      self.evaluate(init)
      estimator_spec = iteration.estimator_spec
      if want_chief_hooks:
        self.assertNotEmpty(iteration.estimator_spec.training_chief_hooks)
      self.assertAllClose(
          want_predictions,
          self.evaluate(estimator_spec.predictions),
          atol=1e-3)

      # A default architecture metric is always included, even if we don't
      # specify one.
      eval_metric_ops = estimator_spec.eval_metric_ops
      if "architecture/adanet/ensembles" in eval_metric_ops:
        del eval_metric_ops["architecture/adanet/ensembles"]
      self.assertEqual(set(want_eval_metric_ops), set(eval_metric_ops.keys()))

      self.assertEqual(want_best_candidate_index,
                       self.evaluate(iteration.best_candidate_index))

      if mode == tf.estimator.ModeKeys.PREDICT:
        self.assertIsNotNone(estimator_spec.export_outputs)
        self.assertAllClose(
            want_export_outputs,
            self.evaluate(
                _export_output_tensors(estimator_spec.export_outputs)),
            atol=1e-3)
        self.assertEqual(iteration.estimator_spec.train_op.type,
                         tf.no_op().type)
        self.assertIsNone(iteration.estimator_spec.loss)
        self.assertIsNotNone(want_export_outputs)
        return

      self.assertAlmostEqual(
          want_loss, self.evaluate(iteration.estimator_spec.loss), places=3)
      self.assertIsNone(iteration.estimator_spec.export_outputs)
      if mode == tf.estimator.ModeKeys.TRAIN:
        self.evaluate(iteration.estimator_spec.train_op)

  @parameterized.named_parameters(
      {
          "testcase_name": "empty_subnetwork_builders",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "subnetwork_builders": [],
          "want_raises": ValueError,
      }, {
          "testcase_name": "same_subnetwork_builder_names",
          "ensemble_builder": _FakeEnsembleBuilder(),
          "subnetwork_builders":
              [_FakeBuilder("same_name"),
               _FakeBuilder("same_name")],
          "want_raises": ValueError,
      }, {
          "testcase_name":
              "same_name_as_previous_ensemble_spec",
          "ensemble_builder":
              _FakeEnsembleBuilder(),
          "previous_ensemble_spec_fn":
              lambda: tu.dummy_ensemble_spec("same_name"),
          "subnetwork_builders": [_FakeBuilder("same_name"),],
          "want_raises":
              ValueError,
      }, {
          "testcase_name":
              "predict_invalid",
          "ensemble_builder":
              _FakeEnsembleBuilder(
                  dict_predictions=True,
                  export_output_key=tu.ExportOutputKeys.INVALID),
          "subnetwork_builders": [
              _FakeBuilder("training"),
              _FakeBuilder("training2", random_seed=7)
          ],
          "mode":
              tf.estimator.ModeKeys.PREDICT,
          "want_raises":
              TypeError,
      })
  @test_util.run_in_graph_and_eager_modes
  def test_build_iteration_error(self,
                                 ensemble_builder,
                                 subnetwork_builders,
                                 want_raises,
                                 previous_ensemble_spec_fn=lambda: None,
                                 mode=tf.estimator.ModeKeys.TRAIN,
                                 summary_maker=_ScopedSummary):
    with context.graph_mode():
      builder = _IterationBuilder(
          _FakeCandidateBuilder(),
          _FakeSubnetworkManager(),
          ensemble_builder,
          summary_maker=summary_maker,
          ensemblers=[_FakeEnsembler()],
          max_steps=100)
      features = [[1., -1., 0.]]
      labels = [1]
      with self.assertRaises(want_raises):
        builder.build_iteration(
            base_global_step=0,
            iteration_number=0,
            ensemble_candidates=[
                EnsembleCandidate("test", subnetwork_builders, None)
            ],
            subnetwork_builders=subnetwork_builders,
            features=features,
            labels=labels,
            mode=mode,
            config=tf.estimator.RunConfig(model_dir=self.test_subdirectory),
            previous_ensemble_spec=previous_ensemble_spec_fn())


class _HeadEnsembleBuilder(object):

  def __init__(self, head):
    self._head = head

  def build_ensemble_spec(self,
                          name,
                          candidate,
                          ensembler,
                          subnetwork_specs,
                          summary,
                          features,
                          mode,
                          iteration_number,
                          labels=None,
                          previous_ensemble_spec=None,
                          my_ensemble_index=None,
                          params=None):
    del ensembler
    del subnetwork_specs
    del summary
    del iteration_number
    del previous_ensemble_spec
    del my_ensemble_index
    del params

    logits = [[.5]]

    estimator_spec = self._head.create_estimator_spec(
        features=features, mode=mode, labels=labels, logits=logits)
    return _EnsembleSpec(
        name=name,
        ensemble=None,
        architecture=_Architecture("foo", "bar"),
        subnetwork_builders=candidate.subnetwork_builders,
        predictions=estimator_spec.predictions,
        step=tf.Variable(0),
        loss=None,
        adanet_loss=.1,
        train_op=None,
        eval_metrics=None,
        export_outputs=estimator_spec.export_outputs)


class IterationExportOutputsTest(tu.AdanetTestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "regression_head",
          "head": regression_head.RegressionHead(),
      }, {
          "testcase_name": "binary_classification_head",
          "head": binary_class_head.BinaryClassHead(),
      })
  @test_util.run_in_graph_and_eager_modes
  def test_head_export_outputs(self, head):
    with context.graph_mode():
      ensemble_builder = _HeadEnsembleBuilder(head)
      builder = _IterationBuilder(
          _FakeCandidateBuilder(),
          _FakeSubnetworkManager(),
          ensemble_builder,
          summary_maker=_ScopedSummary,
          ensemblers=[_FakeEnsembler()],
          max_steps=10)
      features = [[1., -1., 0.]]
      labels = [1]
      mode = tf.estimator.ModeKeys.PREDICT
      subnetwork_builders = [_FakeBuilder("test")]
      iteration = builder.build_iteration(
          base_global_step=0,
          iteration_number=0,
          ensemble_candidates=[
              EnsembleCandidate("test", subnetwork_builders, None)
          ],
          subnetwork_builders=subnetwork_builders,
          features=features,
          labels=labels,
          config=tf.estimator.RunConfig(model_dir=self.test_subdirectory),
          mode=mode)

      # Compare iteration outputs with default head outputs.
      spec = head.create_estimator_spec(
          features=features, labels=labels, mode=mode, logits=[[.5]])
      self.assertEqual(
          len(spec.export_outputs),
          len(iteration.estimator_spec.export_outputs))
      for key in spec.export_outputs:
        if isinstance(spec.export_outputs[key],
                      tf.estimator.export.RegressionOutput):
          self.assertAlmostEqual(
              self.evaluate(spec.export_outputs[key].value),
              self.evaluate(iteration.estimator_spec.export_outputs[key].value))
          continue
        if isinstance(spec.export_outputs[key],
                      tf.estimator.export.ClassificationOutput):
          self.assertAllClose(
              self.evaluate(spec.export_outputs[key].scores),
              self.evaluate(
                  iteration.estimator_spec.export_outputs[key].scores))
          self.assertAllEqual(
              self.evaluate(spec.export_outputs[key].classes),
              self.evaluate(
                  iteration.estimator_spec.export_outputs[key].classes))
          continue
        if isinstance(spec.export_outputs[key],
                      tf.estimator.export.PredictOutput):
          if "classes" in spec.export_outputs[key].outputs:
            # Verify string Tensor outputs separately.
            self.assertAllEqual(
                self.evaluate(spec.export_outputs[key].outputs["classes"]),
                self.evaluate(iteration.estimator_spec.export_outputs[key]
                              .outputs["classes"]))
            del spec.export_outputs[key].outputs["classes"]
            del iteration.estimator_spec.export_outputs[key].outputs["classes"]
          if "all_classes" in spec.export_outputs[key].outputs:
            # Verify string Tensor outputs separately.
            self.assertAllEqual(
                self.evaluate(spec.export_outputs[key].outputs["all_classes"]),
                self.evaluate(iteration.estimator_spec.export_outputs[key]
                              .outputs["all_classes"]))
            del spec.export_outputs[key].outputs["all_classes"]
            del iteration.estimator_spec.export_outputs[key].outputs[
                "all_classes"]
          self.assertAllClose(
              self.evaluate(spec.export_outputs[key].outputs),
              self.evaluate(
                  iteration.estimator_spec.export_outputs[key].outputs))
          continue
        self.fail("Invalid export_output for {}.".format(key))


if __name__ == "__main__":
  tf.test.main()
