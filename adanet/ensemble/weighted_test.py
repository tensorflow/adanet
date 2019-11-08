"""Test AdaNet single weighted subnetwork and ensembler implementation.

Copyright 2019 The AdaNet Authors. All Rights Reserved.

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

import collections
import contextlib

from absl.testing import parameterized
from adanet import ensemble
from adanet import subnetwork
from adanet import tf_compat
from adanet.core.summary import Summary
import mock
import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
# pylint: enable=g-direct-tensorflow-import


class _FakeSummary(Summary):
  """A fake adanet.Summary."""
  scalars = collections.defaultdict(list)

  def scalar(self, name, tensor, family=None):
    self.scalars[name].append(tensor)
    return 'fake_scalar'

  def image(self, name, tensor, max_outputs=3, family=None):
    return 'fake_image'

  def histogram(self, name, values, family=None):
    return 'fake_histogram'

  def audio(self, name, tensor, sample_rate, max_outputs=3, family=None):
    return 'fake_audio'

  def clear_scalars(self):
    self.scalars.clear()

  @contextlib.contextmanager
  def current_scope(self):
    yield


def _get_norm_summary_key(subnetwork_index):
  return ('mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_%s' %
          subnetwork_index)


def _get_fractions_summary_key(subnetwork_index):
  return (
      'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_%s' %
      subnetwork_index)


def _get_complexity_regularization_summary_key():
  return 'complexity_regularization/adanet/adanet_weighted_ensemble'


class ComplexityRegularizedEnsemblerTest(parameterized.TestCase,
                                         tf.test.TestCase):

  def setUp(self):
    super(ComplexityRegularizedEnsemblerTest, self).setUp()

    self._optimizer = tf_compat.v1.train.GradientDescentOptimizer(
        learning_rate=.1)
    self.easy_ensembler = ensemble.ComplexityRegularizedEnsembler(
        optimizer=self._optimizer)

    mock.patch.object(tf.train, 'load_variable', autospec=False).start()
    mock.patch.object(
        tf.compat.v1.train, 'load_variable', autospec=False).start()
    mock.patch.object(
        tf.compat.v2.train, 'load_variable', autospec=False).start()

    def load_variable(checkpoint_dir, name):
      self.assertEqual(checkpoint_dir, 'fake_checkpoint_dir')
      var = tf_compat.v1.get_variable(
          name='fake_loaded_variable_' + name, initializer=1.)
      with self.test_session() as sess:
        sess.run(var.initializer)
        return var

    tf.train.load_variable.side_effect = load_variable

    self.summary = _FakeSummary()

  def _build_easy_ensemble(self, subnetworks):
    return self.easy_ensembler.build_ensemble(
        subnetworks=subnetworks,
        previous_ensemble_subnetworks=None,
        features=None,
        labels=None,
        logits_dimension=None,
        training=None,
        iteration_step=None,
        summary=self.summary,
        previous_ensemble=None)

  def _build_subnetwork(self, multi_head=False):

    last_layer = tf.Variable(
        tf_compat.random_normal(shape=(2, 3)), trainable=False).read_value()

    def new_logits():
      return tf_compat.v1.layers.dense(
          last_layer,
          units=1,
          kernel_initializer=tf_compat.v1.glorot_uniform_initializer())

    if multi_head:
      logits = {k: new_logits() for k in multi_head}
      last_layer = {k: last_layer for k in multi_head}
    else:
      logits = new_logits()

    return subnetwork.Subnetwork(last_layer=logits, logits=logits, complexity=2)

  @parameterized.named_parameters(
      {
          'testcase_name': 'default',
          'expected_summary_scalars': {
              _get_norm_summary_key(0): [1],
              _get_fractions_summary_key(0): [1],
              _get_complexity_regularization_summary_key(): [0.],
          },
          'expected_complexity_regularization': 0.,
      }, {
          'testcase_name': 'one_previous_network',
          'num_previous_ensemble_subnetworks': 1,
          'expected_summary_scalars': {
              _get_norm_summary_key(0): [0.5],
              _get_norm_summary_key(1): [0.5],
              _get_fractions_summary_key(0): [0.5],
              _get_fractions_summary_key(1): [0.5],
              _get_complexity_regularization_summary_key(): [0.],
          },
          'expected_complexity_regularization': 0.,
      }, {
          'testcase_name': 'one_previous_network_with_lambda',
          'adanet_lambda': 0.1,
          'num_previous_ensemble_subnetworks': 1,
          'expected_summary_scalars': {
              _get_norm_summary_key(0): [0.5],
              _get_norm_summary_key(1): [0.5],
              _get_fractions_summary_key(0): [0.5],
              _get_fractions_summary_key(1): [0.5],
              _get_complexity_regularization_summary_key(): [0.2],
          },
          'expected_complexity_regularization': 0.2,
      }, {
          'testcase_name': 'two_subnetworks_one_previous_network_with_lambda',
          'adanet_lambda': 0.1,
          'num_previous_ensemble_subnetworks': 1,
          'expected_summary_scalars': {
              _get_norm_summary_key(0): [0.5],
              _get_norm_summary_key(1): [0.5],
              _get_fractions_summary_key(0): [0.5],
              _get_fractions_summary_key(1): [0.5],
              _get_complexity_regularization_summary_key(): [0.2],
          },
          'expected_complexity_regularization': 0.2,
      }, {
          'testcase_name': 'all_previous_networks_with_lambda',
          'adanet_lambda': 0.1,
          'num_previous_ensemble_subnetworks': 2,
          'expected_summary_scalars': {
              _get_norm_summary_key(0): [1 / 3.],
              _get_norm_summary_key(1): [1 / 3.],
              _get_norm_summary_key(2): [1 / 3.],
              _get_fractions_summary_key(0): [1 / 3.],
              _get_fractions_summary_key(1): [1 / 3.],
              _get_fractions_summary_key(2): [1 / 3.],
              _get_complexity_regularization_summary_key(): [1 / 5.],
          },
          'expected_complexity_regularization': 1 / 5.,
      }, {
          'testcase_name': 'all_previous_networks_and_two_subnetworks',
          'num_subnetworks': 2,
          'adanet_lambda': 0.1,
          'num_previous_ensemble_subnetworks': 2,
          'expected_summary_scalars': {
              _get_norm_summary_key(0): [1 / 4.],
              _get_norm_summary_key(1): [1 / 4.],
              _get_norm_summary_key(2): [1 / 4.],
              _get_norm_summary_key(3): [1 / 4.],
              _get_fractions_summary_key(0): [1 / 4.],
              _get_fractions_summary_key(1): [1 / 4.],
              _get_fractions_summary_key(2): [1 / 4.],
              _get_fractions_summary_key(3): [1 / 4.],
              _get_complexity_regularization_summary_key(): [1 / 5.],
          },
          'expected_complexity_regularization': 1 / 5.,
      }, {
          'testcase_name': 'all_nets_and_string_multihead',
          'num_subnetworks': 2,
          'adanet_lambda': 0.1,
          'multi_head': ['head1', 'head2'],
          'use_bias': True,
          'num_previous_ensemble_subnetworks': 2,
          'expected_summary_scalars': {
              'complexity_regularization/adanet/adanet_weighted_ensemble_head2':
                  [0.2],
              'complexity_regularization/adanet/adanet_weighted_ensemble_head1':
                  [0.2],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head1_1':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head1_0':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head1_2':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head1_3':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head2_0':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head2_1':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head2_2':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head2_3':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head1_3':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head1_2':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head1_1':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head1_0':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head2_0':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head2_1':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head2_2':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head2_3':
                  [0.25]
          },
          'expected_complexity_regularization': 2 / 5.,
      }, {
          'testcase_name': 'all_nets_and_string_tuple_multihead',
          'num_subnetworks': 2,
          'adanet_lambda': 0.1,
          'multi_head': [('bar', 'baz'), ('foo', 'bar')],
          'use_bias': True,
          'num_previous_ensemble_subnetworks': 2,
          'expected_summary_scalars': {
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_bar_baz_3':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_bar_baz_2':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_bar_baz_1':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_bar_baz_0':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_foo_bar_1':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_bar_baz_3':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_bar_baz_2':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_bar_baz_1':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_bar_baz_0':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_foo_bar_1':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_foo_bar_2':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_foo_bar_0':
                  [0.25],
              'complexity_regularization/adanet/adanet_weighted_ensemble_foo_bar':
                  [0.2],
              'complexity_regularization/adanet/adanet_weighted_ensemble_bar_baz':
                  [0.2],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_foo_bar_2':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_foo_bar_3':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_foo_bar_3':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_foo_bar_0':
                  [0.25]
          },
          'expected_complexity_regularization': 2 / 5.,
      }, {
          'testcase_name': 'all_nets_and_tuple_multihead',
          'num_subnetworks': 2,
          'adanet_lambda': 0.1,
          'multi_head': [('bar', 0), ('foo', 1)],
          'use_bias': True,
          'num_previous_ensemble_subnetworks': 2,
          'expected_summary_scalars': {
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_bar_0_2':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_foo_1_3':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_foo_1_2':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_foo_1_1':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_foo_1_0':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_bar_0_1':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_bar_0_0':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_bar_0_3':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_bar_0_2':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_bar_0_1':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_foo_1_3':
                  [0.25],
              'complexity_regularization/adanet/adanet_weighted_ensemble_bar_0':
                  [0.2],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_foo_1_1':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_foo_1_0':
                  [0.25],
              'complexity_regularization/adanet/adanet_weighted_ensemble_foo_1':
                  [0.2],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_bar_0_0':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_bar_0_3':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_foo_1_2':
                  [0.25]
          },
          'expected_complexity_regularization': 2 / 5.,
      }, {
          'testcase_name': 'all_nets_and_number_multihead',
          'num_subnetworks': 2,
          'adanet_lambda': 0.1,
          'multi_head': [0, 1],
          'use_bias': True,
          'num_previous_ensemble_subnetworks': 2,
          'expected_summary_scalars': {
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_1_2':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_1_3':
                  [0.25],
              'complexity_regularization/adanet/adanet_weighted_ensemble_1':
                  [0.2],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_1_1':
                  [0.25],
              'complexity_regularization/adanet/adanet_weighted_ensemble':
                  [0.2],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_1_3':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_0':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_1':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_2':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_3':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_0':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_1':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_2':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_3':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_1_2':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_1_0':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_1_0':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_1_1':
                  [0.25]
          },
          'expected_complexity_regularization': 2 / 5.,
      }, {
          'testcase_name': 'all_nets_with_warm_start',
          'num_subnetworks': 2,
          'adanet_lambda': 0.1,
          'warm_start_mixture_weights': True,
          'num_previous_ensemble_subnetworks': 2,
          'expected_summary_scalars': {
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_2':
                  [0.1],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_3':
                  [0.1],
              'complexity_regularization/adanet/adanet_weighted_ensemble':
                  [0.5],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_1':
                  [0.4],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_2':
                  [0.25],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_0':
                  [0.4],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_0':
                  [1.0],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_1':
                  [1.0],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_3':
                  [0.25]
          },
          'expected_complexity_regularization': 1 / 2.,
      }, {
          'testcase_name': 'all_nets_with_warm_start_and_multihead',
          'num_subnetworks': 2,
          'adanet_lambda': 0.1,
          'multi_head': ['head1', 'head2'],
          'use_bias': True,
          'warm_start_mixture_weights': True,
          'num_previous_ensemble_subnetworks': 2,
          'expected_summary_scalars': {
              'complexity_regularization/adanet/adanet_weighted_ensemble_head2':
                  [0.5],
              'complexity_regularization/adanet/adanet_weighted_ensemble_head1':
                  [0.5],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head2_1':
                  [1.0],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head2_0':
                  [1.0],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head2_3':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head2_2':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head1_2':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head1_3':
                  [0.25],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head1_0':
                  [1.0],
              'mixture_weight_norms/adanet/adanet_weighted_ensemble/subnetwork_head1_1':
                  [1.0],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head1_2':
                  [0.1],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head1_3':
                  [0.1],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head1_0':
                  [0.4],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head1_1':
                  [0.4],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head2_1':
                  [0.4],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head2_0':
                  [0.4],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head2_3':
                  [0.1],
              'mixture_weight_fractions/adanet/adanet_weighted_ensemble/subnetwork_head2_2':
                  [0.1]
          },
          'expected_complexity_regularization': 1.,
          'name': 'with_bias',
      })
  @test_util.run_in_graph_and_eager_modes
  def test_build_ensemble(self,
                          mixture_weight_type=ensemble.MixtureWeightType.SCALAR,
                          mixture_weight_initializer=None,
                          warm_start_mixture_weights=False,
                          adanet_lambda=0.,
                          adanet_beta=0.,
                          multi_head=None,
                          use_bias=False,
                          num_subnetworks=1,
                          num_previous_ensemble_subnetworks=0,
                          expected_complexity_regularization=0.,
                          expected_summary_scalars=None,
                          name=None):
    with context.graph_mode():
      model_dir = None
      if warm_start_mixture_weights:
        model_dir = 'fake_checkpoint_dir'
      ensembler = ensemble.ComplexityRegularizedEnsembler(
          optimizer=self._optimizer,
          mixture_weight_type=mixture_weight_type,
          mixture_weight_initializer=mixture_weight_initializer,
          warm_start_mixture_weights=warm_start_mixture_weights,
          model_dir=model_dir,
          adanet_lambda=adanet_lambda,
          adanet_beta=adanet_beta,
          use_bias=use_bias,
          name=name)

      if name:
        self.assertEqual(ensembler.name, name)
      else:
        self.assertEqual(ensembler.name, 'complexity_regularized')

      with tf_compat.v1.variable_scope('dummy_adanet_scope_iteration_0'):
        previous_ensemble_subnetworks_all = [
            self._build_subnetwork(multi_head),
            self._build_subnetwork(multi_head)
        ]

        previous_ensemble = self._build_easy_ensemble(
            previous_ensemble_subnetworks_all)

      with tf_compat.v1.variable_scope('dummy_adanet_scope_iteration_1'):
        subnetworks_pool = [
            self._build_subnetwork(multi_head),
            self._build_subnetwork(multi_head),
        ]

        subnetworks = subnetworks_pool[:num_subnetworks]

        previous_ensemble_subnetworks = previous_ensemble_subnetworks_all[:(
            num_previous_ensemble_subnetworks)]

        self.summary.clear_scalars()

        built_ensemble = ensembler.build_ensemble(
            subnetworks=subnetworks,
            previous_ensemble_subnetworks=previous_ensemble_subnetworks,
            features=None,
            labels=None,
            logits_dimension=None,
            training=None,
            iteration_step=None,
            summary=self.summary,
            previous_ensemble=previous_ensemble)

        with self.test_session() as sess:
          sess.run(tf_compat.v1.global_variables_initializer())

        summary_scalars, complexity_regularization = sess.run(
            (self.summary.scalars, built_ensemble.complexity_regularization))

        if expected_summary_scalars:
          for key in expected_summary_scalars.keys():
            print(summary_scalars)
            self.assertAllClose(expected_summary_scalars[key],
                                summary_scalars[key])

        self.assertEqual(
            [l.subnetwork for l in built_ensemble.weighted_subnetworks],
            previous_ensemble_subnetworks + subnetworks)

        self.assertAllClose(expected_complexity_regularization,
                            complexity_regularization)
        self.assertIsNotNone(sess.run(built_ensemble.logits))

  @test_util.run_in_graph_and_eager_modes
  def test_build_ensemble_subnetwork_has_scalar_logits(self):
    with context.graph_mode():
      logits = tf.ones(shape=(100,))
      ensemble_spec = self._build_easy_ensemble([
          subnetwork.Subnetwork(
              last_layer=logits, logits=logits, complexity=0.)
      ])
      self.assertEqual([1], ensemble_spec.bias.shape.as_list())

  @test_util.run_in_graph_and_eager_modes
  def test_build_train_op_no_op(self):
    with context.graph_mode():
      train_op = ensemble.ComplexityRegularizedEnsembler().build_train_op(
          *[None] * 7)  # arguments unused
      self.assertEqual(train_op.type, tf.no_op().type)

  @test_util.run_in_graph_and_eager_modes
  def test_build_train_op_callable_optimizer(self):
    with context.graph_mode():
      dummy_weight = tf.Variable(0., name='dummy_weight')
      dummy_loss = dummy_weight * 2.
      ensembler = ensemble.ComplexityRegularizedEnsembler(
          optimizer=lambda: tf_compat.v1.train.GradientDescentOptimizer(.1))
      train_op = ensembler.build_train_op(
          self._build_easy_ensemble([self._build_subnetwork()]),
          dummy_loss, [dummy_weight],
          labels=None,
          iteration_step=None,
          summary=None,
          previous_ensemble=None)
      config = tf.compat.v1.ConfigProto(
          gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
      with tf_compat.v1.Session(config=config) as sess:
        sess.run(tf_compat.v1.global_variables_initializer())
        sess.run(train_op)
        self.assertAllClose(-.2, sess.run(dummy_weight))

  @test_util.run_in_graph_and_eager_modes
  def test_build_train_op(self):
    with context.graph_mode():
      dummy_weight = tf.Variable(0., name='dummy_weight')
      dummy_loss = dummy_weight * 2.
      ensembler = ensemble.ComplexityRegularizedEnsembler(
          optimizer=tf_compat.v1.train.GradientDescentOptimizer(.1))
      train_op = ensembler.build_train_op(
          self._build_easy_ensemble([self._build_subnetwork()]),
          dummy_loss, [dummy_weight],
          labels=None,
          iteration_step=None,
          summary=None,
          previous_ensemble=None)
      config = tf.compat.v1.ConfigProto(
          gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
      with tf_compat.v1.Session(config=config) as sess:
        sess.run(tf_compat.v1.global_variables_initializer())
        sess.run(train_op)
        self.assertAllClose(-.2, sess.run(dummy_weight))

  def tearDown(self):
    self.summary.clear_scalars()
    mock.patch.stopall()
    tf_compat.v1.reset_default_graph()
    super(ComplexityRegularizedEnsemblerTest, self).tearDown()


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
