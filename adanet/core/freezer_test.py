"""Test AdaNet freezer single graph implementation.

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

import os
import shutil

from absl.testing import parameterized
from adanet.core.ensemble import WeightedSubnetwork
from adanet.core.freezer import _EnsembleFreezer
from adanet.core.subnetwork import Subnetwork
import adanet.core.testing_utils as tu
import tensorflow as tf


def _extract_feature(features):
  """Returns the single feature `Tensor` to use from the given features."""

  unused_key = "unused"
  features = features.copy()
  if unused_key in features:
    del features[unused_key]
  assert len(features) == 1
  feature = list(features.values())[0]
  if isinstance(feature, tf.SparseTensor):
    return tf.sparse_tensor_to_dense(feature)
  return feature


def _simple_subnetwork_fn(feature_columns,
                          keep_persisted_tensors=False,
                          seed=42):
  """A simple subnetwork."""

  def _simple(features):
    inputs = tf.feature_column.input_layer(
        features=features, feature_columns=feature_columns)
    with tf.variable_scope("simple"):
      with tf.variable_scope("logits"):
        w = tf.Variable(tf.random_normal([2, 2], seed=seed), name="weight")
        b = tf.Variable(tf.random_normal([1], seed=seed), name="bias")
        predictions = tf.matmul(inputs, w) + b

      some_persisted_tensor_constant = tf.constant(
          seed, name="some_persisted_tensor_constant")
      persisted_tensors = {}
      if keep_persisted_tensors:
        persisted_tensors = {
            "some_persisted_tensor_constant": some_persisted_tensor_constant,
        }
      complexity = tf.constant(3, name="complexity")
      subnetwork = Subnetwork(
          last_layer=predictions,
          logits=predictions,
          complexity=complexity,
          persisted_tensors=persisted_tensors)
      return WeightedSubnetwork(
          name=tf.constant("simple", name="name"),
          logits=predictions,
          weight=w,
          subnetwork=subnetwork)

  return _simple


def _linear_subnetwork_fn(keep_persisted_tensors=False, seed=42):
  """A linear subnetwork."""

  def _linear(features):
    inputs = _extract_feature(features)
    with tf.variable_scope("linear"):
      with tf.variable_scope("logits"):
        w = tf.Variable(tf.random_normal([2, 1], seed=seed), name="weight")
        b = tf.Variable(tf.random_normal([1], seed=seed), name="bias")
        predictions = tf.matmul(inputs, w) + b

      some_persisted_tensor_constant = tf.constant(
          seed, name="some_persisted_tensor_constant")
      nested_persisted_tensor_constant = tf.constant(
          seed, name="nested_persisted_tensor_constant")
      persisted_tensors = {}
      if keep_persisted_tensors:
        persisted_tensors = {
            "some_persisted_tensor_constant": some_persisted_tensor_constant,
            "nested": {
                "nested": {
                    "value": nested_persisted_tensor_constant,
                    "separated/by/slash": nested_persisted_tensor_constant,
                },
                "value": some_persisted_tensor_constant,
            }
        }
      complexity = tf.constant(3, name="complexity")
      subnetwork = Subnetwork(
          last_layer=inputs,
          logits=predictions,
          complexity=complexity,
          persisted_tensors=persisted_tensors)
      return WeightedSubnetwork(
          name=tf.constant("linear", name="name"),
          logits=predictions,
          weight=w,
          subnetwork=subnetwork)

  return _linear


def _dnn_subnetwork_fn(keep_persisted_tensors=False, seed=42):
  """A single layer neural network subnetwork."""

  def _dnn(features):
    inputs = _extract_feature(features)
    layer_size = 10
    with tf.variable_scope("dnn"):
      with tf.variable_scope("hidden_layer"):
        w = tf.Variable(
            tf.random_normal([2, layer_size], seed=seed), name="weight")
        b = tf.Variable(tf.random_normal([layer_size], seed=seed), name="bias")
        hidden_layer = tf.matmul(inputs, w) + b
      with tf.variable_scope("logits"):
        w = tf.Variable(
            tf.random_normal([layer_size, 1], seed=seed), name="weight")
        b = tf.Variable(tf.random_normal([1], seed=seed), name="bias")
        predictions = tf.matmul(hidden_layer, w) + b

      some_persisted_tensor_constant = tf.constant(
          seed, name="some_persisted_tensor_constant")
      persisted_tensors = {}
      if keep_persisted_tensors:
        persisted_tensors = {
            "some_persisted_tensor_constant": some_persisted_tensor_constant,
        }
      complexity = tf.constant(6, name="complexity")
      subnetwork = Subnetwork(
          last_layer=hidden_layer,
          logits=predictions,
          complexity=complexity,
          persisted_tensors=persisted_tensors)
      return WeightedSubnetwork(
          name=tf.constant("dnn", name="name"),
          logits=predictions,
          weight=w,
          subnetwork=subnetwork)

  return _dnn


class EnsembleFreezerTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    # Setup and cleanup test directory.
    self.test_subdirectory = os.path.join(tf.flags.FLAGS.test_tmpdir,
                                          "EnsembleFreezerTest")
    shutil.rmtree(self.test_subdirectory, ignore_errors=True)
    os.mkdir(self.test_subdirectory)

    self.maxDiff = None

  def test_wrapped_features_none(self):
    freezer = _EnsembleFreezer()
    got = freezer.wrapped_features(None)
    self.assertIsNone(got)

  def test_wrapped_features_tensors(self):
    freezer = _EnsembleFreezer()
    features = {"x": tf.constant([1, 2], name="foo")}
    got = freezer.wrapped_features(features)
    with self.test_session() as sess:
      self.assertAllClose(sess.run(features), sess.run(got))

  def test_wrapped_features_sparse_tensors(self):
    freezer = _EnsembleFreezer()
    features = {
        "x":
            tf.SparseTensor(
                indices=[[0, 0], [0, 1]], values=[-1., 1.], dense_shape=[1, 2])
    }
    got = freezer.wrapped_features(features)
    with self.test_session() as sess:
      self.assertAllClose(sess.run(features), sess.run(got))

  def test_wrapped_features_placeholder(self):
    freezer = _EnsembleFreezer()
    features = {"x": tf.placeholder(dtype=tf.float32, shape=[], name="foo")}
    got = freezer.wrapped_features(features)
    with self.test_session() as sess:
      self.assertAllClose(
          sess.run(features, feed_dict={features["x"]: 1.}),
          sess.run(got, feed_dict={got["x"]: 1.}))

  def test_wrapped_features_sparse_placeholder(self):
    freezer = _EnsembleFreezer()
    features = {
        "x":
            tf.sparse_placeholder(
                dtype=tf.float32, shape=[None, 2], name="foo")
    }
    got = freezer.wrapped_features(features)
    value = tf.SparseTensorValue(
        indices=[[0, 0], [0, 1]], values=[-1., 1.], dense_shape=[1, 2])
    with self.test_session() as sess:
      self.assertAllClose(
          sess.run(features, feed_dict={features["x"]: value}),
          sess.run(got, feed_dict={got["x"]: value}))

  @parameterized.named_parameters({
      "testcase_name":
          "dnn_no_persisted_tensors",
      "subnetwork_fns": [_dnn_subnetwork_fn()],
      "features": {
          "feature": [[4., 3.]]
      },
      "want_nodes": [
          u"feature",
          u"dnn/hidden_layer/weight",
          u"dnn/hidden_layer/weight/read",
          u"dnn/hidden_layer/bias",
          u"dnn/hidden_layer/bias/read",
          u"dnn/hidden_layer/MatMul",
          u"dnn/hidden_layer/add",
          u"dnn/logits/weight",
          u"dnn/logits/weight/read",
          u"dnn/logits/bias",
          u"dnn/logits/bias/read",
          u"dnn/logits/MatMul",
          u"dnn/logits/add",
          u"dnn/complexity",
          u"dnn/name",
          u"bias",
      ],
      "want_consts": [
          u"feature",
          u"dnn/hidden_layer/weight",
          u"dnn/hidden_layer/bias",
          u"dnn/logits/weight",
          u"dnn/logits/bias",
          u"dnn/complexity",
          u"dnn/name",
          u"bias",
      ],
  }, {
      "testcase_name":
          "dnn_unused_feature",
      "subnetwork_fns": [_dnn_subnetwork_fn()],
      "features": {
          "feature": [[4., 3.]],
          "unused": [[1., 2.]]
      },
      "want_nodes": [
          u"feature",
          u"unused",
          u"dnn/hidden_layer/weight",
          u"dnn/hidden_layer/weight/read",
          u"dnn/hidden_layer/bias",
          u"dnn/hidden_layer/bias/read",
          u"dnn/hidden_layer/MatMul",
          u"dnn/hidden_layer/add",
          u"dnn/logits/weight",
          u"dnn/logits/weight/read",
          u"dnn/logits/bias",
          u"dnn/logits/bias/read",
          u"dnn/logits/MatMul",
          u"dnn/logits/add",
          u"dnn/complexity",
          u"dnn/name",
          u"bias",
      ],
      "want_consts": [
          u"feature",
          u"unused",
          u"dnn/hidden_layer/weight",
          u"dnn/hidden_layer/bias",
          u"dnn/logits/weight",
          u"dnn/logits/bias",
          u"dnn/complexity",
          u"dnn/name",
          u"bias",
      ],
  }, {
      "testcase_name":
          "dnn_with_persisted_tensors",
      "subnetwork_fns": [_dnn_subnetwork_fn(keep_persisted_tensors=True)],
      "features": {
          "feature": [[4., 3.]]
      },
      "want_nodes": [
          u"feature",
          u"dnn/hidden_layer/weight",
          u"dnn/hidden_layer/weight/read",
          u"dnn/hidden_layer/bias",
          u"dnn/hidden_layer/bias/read",
          u"dnn/hidden_layer/MatMul",
          u"dnn/hidden_layer/add",
          u"dnn/logits/weight",
          u"dnn/logits/weight/read",
          u"dnn/logits/bias",
          u"dnn/logits/bias/read",
          u"dnn/logits/MatMul",
          u"dnn/logits/add",
          u"dnn/some_persisted_tensor_constant",
          u"dnn/complexity",
          u"dnn/name",
          u"bias",
      ],
      "want_consts": [
          u"feature",
          u"dnn/hidden_layer/weight",
          u"dnn/hidden_layer/bias",
          u"dnn/logits/weight",
          u"dnn/logits/bias",
          u"dnn/some_persisted_tensor_constant",
          u"dnn/complexity",
          u"dnn/name",
          u"bias",
      ],
  }, {
      "testcase_name":
          "linear_and_linear",
      "subnetwork_fns": [_linear_subnetwork_fn(),
                         _linear_subnetwork_fn()],
      "features": {
          "feature": [[4., 3.]]
      },
      "want_nodes": [
          u"feature",
          u"linear/logits/weight",
          u"linear/logits/weight/read",
          u"linear/logits/bias",
          u"linear/logits/bias/read",
          u"linear/logits/MatMul",
          u"linear/logits/add",
          u"linear/complexity",
          u"linear/name",
          u"linear_1/logits/weight",
          u"linear_1/logits/weight/read",
          u"linear_1/logits/bias",
          u"linear_1/logits/bias/read",
          u"linear_1/logits/MatMul",
          u"linear_1/logits/add",
          u"linear_1/complexity",
          u"linear_1/name",
          u"bias",
      ],
      "want_consts": [
          u"feature",
          u"linear/logits/weight",
          u"linear/logits/bias",
          u"linear/complexity",
          u"linear/name",
          u"linear_1/logits/weight",
          u"linear_1/logits/bias",
          u"linear_1/complexity",
          u"linear_1/name",
          u"bias",
      ],
  }, {
      "testcase_name":
          "simple_with_feature_column",
      "subnetwork_fns": [
          _simple_subnetwork_fn(
              tf.feature_column.indicator_column(
                  categorical_column=(
                      tf.feature_column.categorical_column_with_vocabulary_list(
                          key="human_names", vocabulary_list=["alice", "bob"])
                  )))
      ],
      "features": {
          "human_names": [["alice"], ["bob"]]
      },
      "want_nodes": [
          u"human_names",
          u"input_layer/human_names_indicator/to_sparse_input/ignore_value"
          "/x",
          u"input_layer/human_names_indicator/to_sparse_input/NotEqual",
          u"input_layer/human_names_indicator/to_sparse_input/indices",
          u"input_layer/human_names_indicator/to_sparse_input/values",
          u"input_layer/human_names_indicator/to_sparse_input/dense_shape",
          u"input_layer/human_names_indicator/human_names_lookup/Const",
          u"input_layer/human_names_indicator/human_names_lookup/Size",
          u"input_layer/human_names_indicator/human_names_lookup/range"
          "/start",
          u"input_layer/human_names_indicator/human_names_lookup/range"
          "/delta",
          u"input_layer/human_names_indicator/human_names_lookup/range",
          u"input_layer/human_names_indicator/human_names_lookup/ToInt64",
          u"input_layer/human_names_indicator/human_names_lookup"
          "/hash_table",
          u"input_layer/human_names_indicator/human_names_lookup"
          "/hash_table/Const",
          u"input_layer/human_names_indicator/human_names_lookup"
          "/hash_table/table_init",
          u"input_layer/human_names_indicator/hash_table_Lookup",
          u"input_layer/human_names_indicator/SparseToDense/default_value",
          u"input_layer/human_names_indicator/SparseToDense",
          u"input_layer/human_names_indicator/one_hot/depth",
          u"input_layer/human_names_indicator/one_hot/on_value",
          u"input_layer/human_names_indicator/one_hot/off_value",
          u"input_layer/human_names_indicator/one_hot",
          u"input_layer/human_names_indicator/Sum/reduction_indices",
          u"input_layer/human_names_indicator/Sum",
          u"input_layer/human_names_indicator/Shape",
          u"input_layer/human_names_indicator/strided_slice/stack",
          u"input_layer/human_names_indicator/strided_slice/stack_1",
          u"input_layer/human_names_indicator/strided_slice/stack_2",
          u"input_layer/human_names_indicator/strided_slice",
          u"input_layer/human_names_indicator/Reshape/shape/1",
          u"input_layer/human_names_indicator/Reshape/shape",
          u"input_layer/human_names_indicator/Reshape",
          u"input_layer/concat",
          u"simple/logits/weight",
          u"simple/logits/weight/read",
          u"simple/logits/bias",
          u"simple/logits/bias/read",
          u"simple/logits/MatMul",
          u"simple/logits/add",
          u"simple/complexity",
          u"simple/name",
          u"bias",
      ],
      "want_consts": [
          u"human_names",
          u"input_layer/human_names_indicator/to_sparse_input/ignore_value"
          "/x",
          u"input_layer/human_names_indicator/to_sparse_input/dense_shape",
          u"input_layer/human_names_indicator/human_names_lookup/Const",
          u"input_layer/human_names_indicator/human_names_lookup/Size",
          u"input_layer/human_names_indicator/human_names_lookup/range"
          "/start",
          u"input_layer/human_names_indicator/human_names_lookup/range"
          "/delta",
          u"input_layer/human_names_indicator/human_names_lookup/hash_table"
          "/Const",
          u"input_layer/human_names_indicator/SparseToDense/default_value",
          u"input_layer/human_names_indicator/one_hot/depth",
          u"input_layer/human_names_indicator/one_hot/on_value",
          u"input_layer/human_names_indicator/one_hot/off_value",
          u"input_layer/human_names_indicator/Sum/reduction_indices",
          u"input_layer/human_names_indicator/Shape",
          u"input_layer/human_names_indicator/strided_slice/stack",
          u"input_layer/human_names_indicator/strided_slice/stack_1",
          u"input_layer/human_names_indicator/strided_slice/stack_2",
          u"input_layer/human_names_indicator/Reshape/shape/1",
          u"simple/logits/weight",
          u"simple/logits/bias",
          u"simple/complexity",
          u"simple/name",
          u"bias",
      ],
  }, {
      "testcase_name":
          "linear_and_dnn",
      "subnetwork_fns": [_dnn_subnetwork_fn(),
                         _linear_subnetwork_fn()],
      "features": {
          "feature": [[4., 3.]]
      },
      "want_nodes": [
          u"feature",
          u"dnn/hidden_layer/weight",
          u"dnn/hidden_layer/weight/read",
          u"dnn/hidden_layer/bias",
          u"dnn/hidden_layer/bias/read",
          u"dnn/hidden_layer/MatMul",
          u"dnn/hidden_layer/add",
          u"dnn/logits/weight",
          u"dnn/logits/weight/read",
          u"dnn/logits/bias",
          u"dnn/logits/bias/read",
          u"dnn/logits/MatMul",
          u"dnn/logits/add",
          u"dnn/complexity",
          u"dnn/name",
          u"linear/logits/weight",
          u"linear/logits/weight/read",
          u"linear/logits/bias",
          u"linear/logits/bias/read",
          u"linear/logits/MatMul",
          u"linear/logits/add",
          u"linear/complexity",
          u"linear/name",
          u"bias",
      ],
      "want_consts": [
          u"feature",
          u"dnn/hidden_layer/weight",
          u"dnn/hidden_layer/bias",
          u"dnn/logits/weight",
          u"dnn/logits/bias",
          u"dnn/complexity",
          u"dnn/name",
          u"linear/logits/weight",
          u"linear/logits/bias",
          u"linear/complexity",
          u"linear/name",
          u"bias",
      ],
  }, {
      "testcase_name":
          "linear_and_dnn_with_persisted_tensors",
      "subnetwork_fns": [
          _dnn_subnetwork_fn(keep_persisted_tensors=True),
          _linear_subnetwork_fn(keep_persisted_tensors=True)
      ],
      "features": {
          "feature": [[4., 3.]]
      },
      "want_nodes": [
          u"feature",
          u"dnn/hidden_layer/weight",
          u"dnn/hidden_layer/weight/read",
          u"dnn/hidden_layer/bias",
          u"dnn/hidden_layer/bias/read",
          u"dnn/hidden_layer/MatMul",
          u"dnn/hidden_layer/add",
          u"dnn/logits/weight",
          u"dnn/logits/weight/read",
          u"dnn/logits/bias",
          u"dnn/logits/bias/read",
          u"dnn/logits/MatMul",
          u"dnn/logits/add",
          u"dnn/some_persisted_tensor_constant",
          u"dnn/complexity",
          u"dnn/name",
          u"linear/logits/weight",
          u"linear/logits/weight/read",
          u"linear/logits/bias",
          u"linear/logits/bias/read",
          u"linear/logits/MatMul",
          u"linear/logits/add",
          u"linear/some_persisted_tensor_constant",
          u"linear/nested_persisted_tensor_constant",
          u"linear/complexity",
          u"linear/name",
          u"bias",
      ],
      "want_consts": [
          u"feature",
          u"dnn/hidden_layer/weight",
          u"dnn/hidden_layer/bias",
          u"dnn/logits/weight",
          u"dnn/logits/bias",
          u"dnn/some_persisted_tensor_constant",
          u"dnn/complexity",
          u"dnn/name",
          u"linear/logits/weight",
          u"linear/logits/bias",
          u"linear/some_persisted_tensor_constant",
          u"linear/nested_persisted_tensor_constant",
          u"linear/complexity",
          u"linear/name",
          u"bias",
      ],
  })
  def test_freeze_ensemble(self,
                           subnetwork_fns,
                           features,
                           want_nodes,
                           want_consts,
                           bias=0):
    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      freezer = _EnsembleFreezer()
      filename = os.path.join(self.test_subdirectory, "frozen.pbtxt")
      features = {
          k: tf.constant(features[k], name=k) for k in sorted(features.keys())
      }
      weighted_subnetworks = [fn(features) for fn in subnetwork_fns]
      bias = tf.constant(bias, name="bias")
      sess.run(tf.global_variables_initializer())
      freezer.freeze_ensemble(
          sess=sess,
          filename=filename,
          weighted_subnetworks=weighted_subnetworks,
          bias=bias,
          features=features)
      with tf.gfile.FastGFile(filename, "rb") as f:
        meta_graph_def = tf.MetaGraphDef()
        meta_graph_def.ParseFromString(f.read())
      nodes = []
      consts = []
      for node_def in meta_graph_def.graph_def.node:
        nodes.append(node_def.name)
        if node_def.op == "Const":
          consts.append(node_def.name)
      self.assertEqual(want_nodes, nodes)
      self.assertEqual(want_consts, consts)

  @parameterized.named_parameters({
      "testcase_name": "persisted_tensor_with separator",
      "subnetwork_fns": [_dnn_subnetwork_fn(keep_persisted_tensors=True)],
      "bad_persisted_tensors": {
          "foo|bar": .1
      },
  })
  def test_freeze_ensemble_error(self, subnetwork_fns, bad_persisted_tensors):
    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      freezer = _EnsembleFreezer()
      filename = os.path.join(self.test_subdirectory, "frozen.pbtxt")
      features = {"x": tf.constant([[-1., 1.]], name="features")}
      weighted_subnetworks = [fn(features) for fn in subnetwork_fns]
      for wwl in weighted_subnetworks:
        wwl.subnetwork.persisted_tensors.update(bad_persisted_tensors)
      bias = tf.constant(0, name="bias")
      with self.assertRaises(ValueError):
        sess.run(tf.global_variables_initializer())
        freezer.freeze_ensemble(
            sess=sess,
            filename=filename,
            weighted_subnetworks=weighted_subnetworks,
            bias=bias,
            features=features)

  @parameterized.named_parameters({
      "testcase_name": "linear_no_persisted_tensors",
      "subnetwork_fns": [_linear_subnetwork_fn("linear_subnetwork")],
      "features_to_freeze": {
          "x": [[-1., 1.]]
      },
      "features_to_load": {
          "x": [[-1., 1.]]
      },
      "want_logits": [[-.137752]],
  }, {
      "testcase_name": "linear_bias",
      "subnetwork_fns": [_linear_subnetwork_fn("linear_subnetwork")],
      "bias": 3.,
      "features_to_freeze": {
          "x": [[-1., 1.]]
      },
      "features_to_load": {
          "x": [[-1., 1.]]
      },
      "want_logits": [[-.137752]],
  }, {
      "testcase_name": "linear_unused_feature",
      "subnetwork_fns": [_linear_subnetwork_fn("linear_subnetwork")],
      "features_to_freeze": {
          "x": [[-1., 1.]],
          "unused": [[0., 2.]]
      },
      "features_to_load": {
          "x": [[-1., 1.]],
          "unused": [[0., 2.]]
      },
      "want_logits": [[-.137752]],
  }, {
      "testcase_name": "linear_sparse_feature",
      "subnetwork_fns": [_linear_subnetwork_fn("linear_subnetwork")],
      "features_to_freeze": {
          "x":
              tu.FakeSparseTensor(
                  indices=[[0, 0], [0, 1]],
                  values=[-1., 1.],
                  dense_shape=[1, 2]),
      },
      "features_to_load": {
          "x":
              tu.FakeSparseTensor(
                  indices=[[0, 0], [0, 1]],
                  values=[-1., 1.],
                  dense_shape=[1, 2]),
      },
      "want_logits": [[-.137752]],
  }, {
      "testcase_name": "linear_unused_sparse_feature",
      "subnetwork_fns": [_linear_subnetwork_fn("linear_subnetwork")],
      "features_to_freeze": {
          "x":
              tu.FakeSparseTensor(
                  indices=[[0, 0], [0, 1]],
                  values=[-1., 1.],
                  dense_shape=[1, 2]),
          "unused":
              tu.FakeSparseTensor(
                  indices=[[0, 0], [0, 1]],
                  values=[-1., 1.],
                  dense_shape=[1, 2]),
      },
      "features_to_load": {
          "x":
              tu.FakeSparseTensor(
                  indices=[[0, 0], [0, 1]],
                  values=[-1., 1.],
                  dense_shape=[1, 2]),
          "unused":
              tu.FakeSparseTensor(
                  indices=[[0, 0], [0, 1]],
                  values=[-1., 1.],
                  dense_shape=[1, 2]),
      },
      "want_logits": [[-.137752]],
  }, {
      "testcase_name": "dnn_no_persisted_tensors",
      "subnetwork_fns": [_dnn_subnetwork_fn()],
      "features_to_freeze": {
          "x": [[-1., 1.]]
      },
      "features_to_load": {
          "x": [[-1., 1.]]
      },
      "want_logits": [[-2.857512]],
  }, {
      "testcase_name": "dnn_with_persisted_tensors",
      "subnetwork_fns": [_dnn_subnetwork_fn()],
      "features_to_freeze": {
          "x": [[-1., 1.]]
      },
      "features_to_load": {
          "x": [[-1., 1.]]
      },
      "want_logits": [[-2.857512]],
  }, {
      "testcase_name":
          "linear_and_linear",
      "subnetwork_fns": [
          _linear_subnetwork_fn(keep_persisted_tensors=True),
          _linear_subnetwork_fn(keep_persisted_tensors=True, seed=99),
      ],
      "features_to_freeze": {
          "x": [[-1., 1.]]
      },
      "features_to_load": {
          "x": [[-1., 1.]]
      },
      "want_logits": [[.733523]],
  }, {
      "testcase_name":
          "linear_and_dnn",
      "subnetwork_fns": [
          _dnn_subnetwork_fn(keep_persisted_tensors=True),
          _linear_subnetwork_fn(keep_persisted_tensors=True)
      ],
      "features_to_freeze": {
          "x": [[-1., 1.]]
      },
      "features_to_load": {
          "x": [[-1., 1.]]
      },
      "want_logits": [[-.137752]],
  }, {
      "testcase_name": "dnn_with_different_inputs",
      "subnetwork_fns": [_dnn_subnetwork_fn()],
      "features_to_freeze": {
          "x": [[-1., 1.]]
      },
      "features_to_load": {
          "x": [[2., 3]]
      },
      "want_logits": [[14.742406]],
  }, {
      "testcase_name":
          "linear_and_dnn_with_different_inputs",
      "subnetwork_fns": [
          _dnn_subnetwork_fn(keep_persisted_tensors=True),
          _linear_subnetwork_fn(keep_persisted_tensors=True)
      ],
      "features_to_freeze": {
          "x": [[-1., 1.]]
      },
      "features_to_load": {
          "x": [[2., 3]]
      },
      "want_logits": [[-1.255581]],
  }, {
      "testcase_name":
          "linear_and_dnn_with_placeholder",
      "subnetwork_fns": [
          _dnn_subnetwork_fn(keep_persisted_tensors=True),
          _linear_subnetwork_fn(keep_persisted_tensors=True)
      ],
      "features_placeholder": {
          "x": [None, 2]
      },
      "features_to_freeze":
          None,
      "features_to_load": {
          "x": [[2., 3], [4., -5]]
      },
      "want_logits": [[-1.255581], [-.715115]],
  })
  def test_load_frozen_ensemble(self,
                                subnetwork_fns,
                                features_to_freeze,
                                features_to_load,
                                want_logits,
                                bias=0,
                                features_placeholder=None):
    freezer = _EnsembleFreezer()
    filename = os.path.join(self.test_subdirectory, "frozen.pbtxt")
    bias_value = bias

    # First freeze ensemble.
    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      bias = tf.constant(bias, name="bias")

      if features_to_freeze is not None:
        features_to_freeze = tu.tensor_features(features_to_freeze)
      elif features_placeholder is not None:
        features_to_freeze = {
            k: tf.placeholder(dtype=tf.float32, name=k, shape=shape)
            for k, shape in features_placeholder.items()
        }
      features_to_freeze = freezer.wrapped_features(features_to_freeze)
      weighted_subnetworks = [fn(features_to_freeze) for fn in subnetwork_fns]
      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())
      sess.run(init)
      freezer.freeze_ensemble(
          sess=sess,
          filename=filename,
          weighted_subnetworks=weighted_subnetworks,
          bias=bias,
          features=features_to_freeze)

    # Load frozen ensemble into a new graph with potentially different
    # features than those used when saving.
    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      features_to_load = tu.tensor_features(features_to_load)
      features_to_load = freezer.wrapped_features(features_to_load)
      frozen_ensemble, frozen_bias = freezer.load_frozen_ensemble(
          filename=filename, features=features_to_load)

      want_ensemble = [fn(features_to_load) for fn in subnetwork_fns]
      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())
      sess.run(init)
      want_ensemble = sess.run(want_ensemble)
      frozen_ensemble = sess.run(frozen_ensemble)
      self.assertEqual([w.name for w in want_ensemble],
                       [w.name for w in frozen_ensemble])
      self.assertAllClose(
          [(w.logits, w.weight, w.subnetwork) for w in want_ensemble],
          [(w.logits, w.weight, w.subnetwork) for w in frozen_ensemble])
      self.assertAllEqual(bias_value, sess.run(frozen_bias))

  @parameterized.named_parameters({
      "testcase_name": "features",
      "features_to_freeze": {
          "x": tu.FakePlaceholder(dtype=tf.float32),
      },
      "features_to_load": {
          "x": [[-1., 1.]],
      },
  }, {
      "testcase_name": "sparse_features",
      "features_to_freeze": {
          "x": tu.FakeSparsePlaceholder(dtype=tf.float32),
      },
      "features_to_load": {
          "x":
              tu.FakeSparseTensor(
                  indices=[[0, 0], [0, 1]],
                  values=[-1., 1.],
                  dense_shape=[1, 2]),
      },
  })
  def test_load_frozen_ensemble_colocation_bug(self, features_to_freeze,
                                               features_to_load):
    """Test colocation bug b/74595432."""

    freezer = _EnsembleFreezer()
    filename = os.path.join(self.test_subdirectory, "frozen.pbtxt")

    # First freeze ensemble.
    with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
      features_to_freeze = tu.tensor_features(features_to_freeze)
      features_to_freeze = freezer.wrapped_features(features_to_freeze)

      for k, feature in features_to_freeze.items():

        with tf.colocate_with(feature):
          if isinstance(feature, tf.SparseTensor):
            feature = tf.SparseTensor(
                tf.identity(feature.indices, name="colocated_indices"),
                tf.identity(feature.values, name="colocated_values"),
                tf.identity(feature.dense_shape, name="colocated_dense_shape"))
          else:
            feature = tf.identity(feature, name="colocated")
        features_to_freeze[k] = feature

      weighted_subnetworks = [_dnn_subnetwork_fn()(features_to_freeze)]
      init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())
      sess.run(init)
      freezer.freeze_ensemble(
          sess=sess,
          filename=filename,
          weighted_subnetworks=weighted_subnetworks,
          bias=tf.constant(0, name="bias"),
          features=features_to_freeze)

    # Verify that repeatedly freezing and reloading frozen ensembles works.
    features_to_load_copy = features_to_load.copy()
    for _ in range(5):
      with tf.Graph().as_default() as g, self.test_session(graph=g) as sess:
        features_to_load = tu.tensor_features(features_to_load_copy)
        features_to_load = freezer.wrapped_features(features_to_load)
        frozen_ensemble, frozen_bias = freezer.load_frozen_ensemble(
            filename=filename, features=features_to_load)
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess.run(init)
        freezer.freeze_ensemble(
            sess=sess,
            filename=filename,
            weighted_subnetworks=frozen_ensemble,
            bias=frozen_bias,
            features=features_to_freeze)


if __name__ == "__main__":
  tf.test.main()
