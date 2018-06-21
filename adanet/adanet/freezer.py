"""An AdaNet ensemble freezer in Tensorflow.

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

from adanet.adanet.base_learner import BaseLearner
from adanet.adanet.ensemble import WeightedBaseLearner
import tensorflow as tf


class _EnsembleFreezer(object):
  """Stateless object that saves, and loads a frozen AdaNet ensemble."""

  class Keys(object):
    """Collection keys for freezing ensemble fields.

    The following frozen keys are defined:

    * `BIAS`: Ensemble bias term `Tensor`.
    * `COMPLEXITY`: base learner complexity `Tensor`.
    * `LAST_LAYER`: base learner's last layer `Tensor`.
    * `LOGITS`: base learner's mixture-weighted logits `Tensor`.
    * `PERSISTED_TENSORS`: base learner persisted `Tensors`.
    * `PERSISTED_TENSORS_SEPARATOR`: Separator symbol for persisted tensor keys.
    * `WEIGHT`: the mixture weight `Tensor` of each base learner.
    """

    BIAS = "bias"
    COMPLEXITY = "complexity"
    LAST_LAYER = "last_layer"
    LOGITS = "logits"
    PERSISTED_TENSORS = "persisted_tensors"
    PERSISTED_TENSORS_SEPARATOR = "|"
    WEIGHT = "weight"

  def wrapped_features(self, features):
    """Wraps each feature `Tensor` in one with a similar name as its key.

    For `SparseTensor` features, it replaces the feature with a new
    `SparseTensor` composed of the original's wrapped indices, values, and
    dense_shape `Tensor`s.

    Args:
      features: Dictionary of wrapped `Tensor` objects keyed by feature name.

    Returns:
      A dictionary of wrapped feature `Tensor`s.
    """

    if features is None:
      return features

    result = {}
    for key, feature in features.items():
      if isinstance(feature, tf.SparseTensor):
        feature = tf.SparseTensor(
            indices=self._wrapped_tensor(
                feature.indices, name="{}_indices".format(key)),
            values=self._wrapped_tensor(
                feature.values, name="{}_values".format(key)),
            dense_shape=self._wrapped_tensor(
                feature.dense_shape, name="{}_dense_shape".format(key)))
      else:
        feature = self._wrapped_tensor(feature, key)
      result[key] = feature
    return result

  def _wrapped_tensor(self, tensor, name):
    """Doubly wraps the given tensor with the given name."""

    wrapped_name = self._wrapped_name(name)
    unwrapped_name = self._unwrapped_name(wrapped_name)
    tensor = tf.identity(tensor, name=unwrapped_name)
    return tf.identity(tensor, name=wrapped_name)

  def _wrapped_name(self, name):
    """Returns the wrapped name."""

    return "wrapped_{}".format(name)

  def _unwrapped_name(self, wrapped_name):
    """Returns the unwrapped name."""

    return "un" + wrapped_name

  def freeze_ensemble(self, sess, filename, weighted_base_learners, bias,
                      features):
    """Freezes an ensemble of base learners' weights and persists its subgraph.

    Specifically, this method prunes all nodes from the current graph definition
    unrelated to the ensemble's `WeightedBaseLearner`s' subgraphs. A subgraph is
    defined as any ops that contribute to their outputs, complexity, and side
    inputs.

    These tensors-to-keep are added to named graph collections, so that the
    `WeightedBaseLearners` can be easily restored with only the information
    stored in the frozen graph. Next this method freezes the pruned subgraphs'
    non-local variables by converting them to constants. The final pruned graph
    is serialized and written to disk as a `MetaGraphDef` proto.

    This method should only be called up to once per graph.

    Args:
      sess: `Session` instance with most recent variable values loaded.
      filename: String filename for the `MetaGraphDef` proto to be written.
      weighted_base_learners: List of `WeightedBaseLearner` instances to freeze.
      bias: The ensemble's `Tensor` bias vector.
      features: Dictionary of wrapped `Tensor` objects keyed by feature name.
        Ops for unused features will not be pruned.

    Returns:
      A `MetaGraphDef` proto of the frozen ensemble.
    """

    # A destination node is a node in the output DAG that will have only
    # incoming edges. Marking these nodes to keep will cause all upstream nodes
    # that are connected by some path of directed edges to the destination node
    # to be marked to keep.
    destination_nodes = set()
    collection_set = set(
        [tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.TABLE_INITIALIZERS])

    for index, weighted_base_learner in enumerate(weighted_base_learners):
      self._freeze_weighted_base_learner(
          weighted_base_learner=weighted_base_learner,
          index=index,
          collection_set=collection_set,
          destination_nodes=destination_nodes)

    self._freeze_bias(bias, collection_set, destination_nodes)

    # Save feature `Tensor`s so that they can be input-mapped upon loading in
    # case the ensemble doesn't yet use them in its sub-graph.
    for feature in features.values():
      if isinstance(feature, tf.SparseTensor):
        destination_nodes.add(feature.indices.op.name)
        destination_nodes.add(feature.values.op.name)
        destination_nodes.add(feature.dense_shape.op.name)
      else:
        destination_nodes.add(feature.op.name)

    # We need to add the variable initializers to the destination nodes, or they
    # will not be initializable upon loading.
    for local_var in tf.local_variables():
      destination_nodes.add(local_var.op.name)
      destination_nodes.add(local_var.op.name + "/read")
      destination_nodes.add(local_var.initializer.name)
      destination_nodes.add(local_var.initial_value.op.name)

    # We need to add the table initialization ops to the destination nodes, or
    # they will not be initializable upon loading.
    for table_init_op in tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS):
      destination_nodes.add(table_init_op.name)

    # Convert all non-local variables to constants. Local variables are those
    # are unrelated to training and do not need to be checkpointed, such as
    # metric variables.
    variables_blacklist = [v.op.name for v in tf.local_variables()]
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=tf.get_default_graph().as_graph_def(),
        output_node_names=list(destination_nodes),
        variable_names_blacklist=variables_blacklist)
    return tf.train.export_meta_graph(
        filename=filename,
        graph_def=frozen_graph_def,
        collection_list=list(collection_set),
        clear_devices=True,
        clear_extraneous_savers=True,
        as_text=False)

  def _freeze_bias(self, bias, collection_set, destination_nodes):
    """Freezes the bias `Tensor`.

    Args:
      bias: The ensemble's `Tensor` bias vector.
      collection_set: Set of string names of collections to keep.
      destination_nodes: Set of string names of ops to keep.
    """

    self._clear_collection(self.Keys.BIAS)
    destination_nodes.add(bias.op.name)
    tf.add_to_collection(self.Keys.BIAS, bias)
    collection_set.add(self.Keys.BIAS)

  def _freeze_weighted_base_learner(self, index, weighted_base_learner,
                                    collection_set, destination_nodes):
    """Freezes a `WeightedBaseLearner`.

    Converts a `WeightedBaseLearner`'s variables to constants and stores its
    ops in collections so that they easily be restored into a
    `WeightedBaseLearner` instance.

    Args:
      index: Index of the `WeightedBaseLearner` in the `Ensemble`.
      weighted_base_learner: The `WeightedBaseLearner` to freeze.
      collection_set: Set of string names of collections to keep.
      destination_nodes: Set of string names of ops to keep.
    """

    tensors_to_persist = {
        self.Keys.WEIGHT: weighted_base_learner.weight,
        self.Keys.LOGITS: weighted_base_learner.logits,
    }
    for field, tensor in tensors_to_persist.items():
      collection_key = self._weighted_base_learner_collection_key(index, field)
      self._clear_collection(collection_key)
      self._keep_tensors(
          collection_set=collection_set,
          destination_nodes=destination_nodes,
          collection_key=collection_key,
          tensor=tensor)
    self._freeze_base_learner(
        base_learner=weighted_base_learner.base_learner,
        index=index,
        collection_set=collection_set,
        destination_nodes=destination_nodes)

  def _freeze_base_learner(self, index, base_learner, collection_set,
                           destination_nodes):
    """Freezes a `BaseLearner`.

    Converts a `BaseLearner`'s variables to constants and stores its ops in
    collections so that they easily be restored into a `BaseLearner` instance.

    Args:
      index: Index of the `BaseLearner` in the `Ensemble`.
      base_learner: The `BaseLearner` to freeze.
      collection_set: Set of string names of collections to keep.
      destination_nodes: Set of string names of ops to keep.
    """

    tensors_to_persist = {
        self.Keys.COMPLEXITY: base_learner.complexity,
        self.Keys.LAST_LAYER: base_learner.last_layer,
        self.Keys.LOGITS: base_learner.logits,
    }
    tensors_to_persist = self._persist_persisted_tensors(
        prefix=self.Keys.PERSISTED_TENSORS,
        persisted_tensors=base_learner.persisted_tensors,
        tensors_to_persist=tensors_to_persist)

    for field, tensor in tensors_to_persist.items():
      collection_key = self._base_learner_collection_key(index, field)
      self._clear_collection(collection_key)
      self._keep_tensors(
          collection_set=collection_set,
          destination_nodes=destination_nodes,
          collection_key=collection_key,
          tensor=tensor)

  def _clear_collection(self, collection_key):
    """Empties the collection with the given key."""

    collection = tf.get_collection_ref(collection_key)
    del collection[:]

  def _keep_tensors(self, collection_set, destination_nodes, collection_key,
                    tensor):
    """Marks a `Tensor` to be kept.

    This `Tensor` is added to the appropriate lists and collection so that it
    and its subgraph are not pruned before freezing.

    Args:
      collection_set: Set of string names of collections to keep.
      destination_nodes: Set of string names of ops to keep.
      collection_key: String key of the collection to add tensor.
      tensor: `Tensor` to keep. Its name is added to the destination_nodes
        list and it is added to the collection identitified by collection_key.
    """

    tensor = tf.convert_to_tensor(tensor)
    destination_nodes.add(tensor.op.name)
    tf.add_to_collection(collection_key, tensor)
    collection_set.add(collection_key)

  def _persist_persisted_tensors(self, prefix, persisted_tensors,
                                 tensors_to_persist):
    """Flattens nested persisted_tensors entries into the tensors_to_persist.

    Recursively calls itself for each nested persisted tensor entry.

    Args:
      prefix: String prefix to prepend to each persisted tensor key.
      persisted_tensors: Dictionary of tensors to persist.
      tensors_to_persist: Flat dictionary of string key to `Tensor` that will be
        persisted.

    Returns:
      Dictionary copy of tensors_to_persist with persisted tensors included.

    Raises:
      ValueError: If persisted tensor keys include the persisted tensor
      separator symbol.
    """

    tensors_to_persist = tensors_to_persist.copy()
    for key, value in persisted_tensors.items():
      if self.Keys.PERSISTED_TENSORS_SEPARATOR in key:
        raise ValueError("Persisted tensor keys cannot contain '{}'.".format(
            self.Keys.PERSISTED_TENSORS_SEPARATOR))
      persisted_key = "{prefix}{separator}{key}".format(
          prefix=prefix,
          separator=self.Keys.PERSISTED_TENSORS_SEPARATOR,
          key=key)
      if isinstance(value, dict):
        tensors_to_persist = self._persist_persisted_tensors(
            persisted_key, value, tensors_to_persist)
        continue
      tensors_to_persist[persisted_key] = value
    return tensors_to_persist

  def load_frozen_ensemble(self, filename, features):
    """Loads ensemble `WeightedBaseLearners` and bias from a `MetaGraphDef`.

    This methods imports the graph of a frozen ensemble into the default graph
    and reconstructs it `WeightedBaseLearners` and bias. The frozen features
    are replaced with those given in the arguments.

    This method should only be called up to once per graph.

    Args:
      filename: String filename of the serialized `MetaGraphDef`.
      features: Dictionary of wrapped `Tensor` objects keyed by feature name.

    Returns:
      A two-tuple of a list of frozen `WeightedBaseLearners` instances and a
        bias term `Tensor`.
    """

    # Wrapped features need to be unwrapped so that the inner `Tensor` can be
    # replaced with the new feature `Tensors`. Due to b/74595432, the wrapper
    # cannot be replaced itself.
    input_map = {}
    for feature in features.values():
      if isinstance(feature, tf.SparseTensor):
        input_map[self._unwrapped_name(feature.indices.name)] = feature.indices
        input_map[self._unwrapped_name(feature.values.name)] = feature.values
        input_map[self._unwrapped_name(
            feature.dense_shape.name)] = feature.dense_shape
      else:
        input_map[self._unwrapped_name(feature.name)] = feature

    # Import base learner's meta graph into default graph. Since there are no
    # variables to restore, import_meta_graph does not create a `Saver`.
    tf.train.import_meta_graph(
        meta_graph_or_file=filename, input_map=input_map, clear_devices=True)

    weighted_base_learners = []
    index = 0
    while True:
      weighted_base_learner = self._reconstruct_weighted_base_learner(index)
      if weighted_base_learner is None:
        break
      weighted_base_learners.append(weighted_base_learner)
      index += 1

    bias_collection = tf.get_collection(self.Keys.BIAS)
    assert len(bias_collection) == 1
    bias_tensor = bias_collection[-1]
    return weighted_base_learners, bias_tensor

  def _reconstruct_weighted_base_learner(self, index):
    """Reconstructs a `WeightedBaseLearner` from the graph's collections.

    Args:
      index: Integer index of the base learner in a list of base learners.

    Returns:
      A frozen `WeightedBaseLearner` instance or `None` if there is no
        `WeightedBaseLearner` frozen at index.
    """

    weight = None
    logits = None
    for key in tf.get_default_graph().get_all_collection_keys():
      prefix = self._weighted_base_learner_collection_key(index, "")
      if prefix not in key:
        continue

      # Verify that each frozen collection is of size one, as each collection
      # should have been cleared before adding a tensor to freeze.
      frozen_collection = tf.get_collection(key)
      assert len(frozen_collection) == 1
      frozen_tensor = frozen_collection[-1]

      field = self._weighted_base_learner_collection_key_field(key, index)
      if field is None:
        continue
      if field == self.Keys.LOGITS:
        logits = frozen_tensor
        continue
      if field == self.Keys.WEIGHT:
        weight = frozen_tensor
        continue

    # No weighted base learner found at given index.
    if weight is None and logits is None:
      return None

    base_learner = self._reconstruct_base_learner(index)
    return WeightedBaseLearner(
        logits=logits, weight=weight, base_learner=base_learner)

  def _reconstruct_base_learner(self, index):
    """Reconstructs a `BaseLearner` from the graph's collections.

    Args:
      index: Integer index of the base learner in a list of base learners.

    Returns:
      A frozen `BaseLearner` instance.

    Raises:
      ValueError: If a field in the frozen collection does not belong to a base
        learner. This should not happen if the collection was created by
        `freeze_ensemble`.
    """

    last_layer = None
    logits = None
    complexity = None
    persisted_tensors = {}
    for key in tf.get_default_graph().get_all_collection_keys():
      prefix = self._base_learner_collection_key(index, "")
      if prefix not in key:
        continue

      # Verify that each frozen collection is of size one, as each collection
      # should have been cleared before adding a tensor to freeze.
      frozen_collection = tf.get_collection(key)
      assert len(frozen_collection) == 1
      frozen_tensor = frozen_collection[-1]

      field = self._base_learner_collection_key_field(key, index)
      if field is None:
        continue
      if field == self.Keys.LAST_LAYER:
        last_layer = frozen_tensor
        continue
      if field == self.Keys.LOGITS:
        logits = frozen_tensor
        continue
      if field == self.Keys.COMPLEXITY:
        complexity = frozen_tensor
        continue
      if field.startswith(self.Keys.PERSISTED_TENSORS):
        # Remove persisted tensors prefix plus separator.
        prefix_length = len(self.Keys.PERSISTED_TENSORS)
        prefix_length += len(self.Keys.PERSISTED_TENSORS_SEPARATOR)
        field = field[prefix_length:]
        persisted_tensors = self._reconstruct_persisted_tensor(
            field, frozen_tensor, persisted_tensors)
        continue

      # This line should not be hit if the frozen graph was created with
      # freeze_ensemble.
      raise ValueError("'{}' in not a valid field.".format(field))

    return BaseLearner(
        last_layer=last_layer,
        logits=logits,
        complexity=complexity,
        persisted_tensors=persisted_tensors)

  def _reconstruct_persisted_tensor(self, field, frozen_tensor,
                                    persisted_tensors):
    """Reconstructs a flattened persisted tensor from its field.

    Args:
      field: String field name. Nested fields are separated with the side inputs
        separator symbol.
      frozen_tensor: The frozen tensor to add to the persisted tensors.
      persisted_tensors: Dictionary of string keys to persisted tensor
        `Tensors`.

    Returns:
      A copy of persisted tensors with the frozen tensor.
    """

    persisted_tensors = persisted_tensors.copy()
    nested_persisted_tensors = persisted_tensors
    fields = field.split(self.Keys.PERSISTED_TENSORS_SEPARATOR)
    for i in range(len(fields) - 1):
      key = fields[i]
      if key not in nested_persisted_tensors:
        nested_persisted_tensors[key] = {}
      nested_persisted_tensors = nested_persisted_tensors[key]
    nested_persisted_tensors[fields[-1]] = frozen_tensor
    return persisted_tensors

  def _weighted_base_learner_collection_key(self, index, field):
    """Returns the collection key for the given arguments.

    Args:
      index: Integer index of the weighted base learner in a list.
      field: String name of one of the weighted base learner's fields.

    Returns:
      String collection key.
    """

    return "{}/weighted_base_learner/{}".format(index, field)

  def _weighted_base_learner_collection_key_field(self, collection_key, index):
    """Returns a weighted base learner's field name from the given arguments.

    Args:
      collection_key: String name of the collection where the field `Tensor`
        is stored during freezing.
      index: Integer index of the weighted base learner in a list.

    Returns:
      String name of one of the weighted base learner's fields.
    """

    prefix = "{}/weighted_base_learner/".format(index)
    if not collection_key.startswith(prefix):
      return None
    return collection_key.replace(prefix, "")

  def _base_learner_collection_key(self, index, field):
    """Returns the collection key for the given arguments.

    Args:
      index: Integer index of the base learner in a list of base learners.
      field: String name of one of the base learner's fields.

    Returns:
      String collection key.
    """

    return "{}/weighted_base_learner/base_learner/{}".format(index, field)

  def _base_learner_collection_key_field(self, collection_key, index):
    """Returns a base learner's field name from the given arguments.

    Args:
      collection_key: String name of the collection where the field `Tensor`
        is stored during freezing.
      index: Integer index of the base learner in a list of base learners.

    Returns:
      String name of one of the base learner's fields.
    """

    prefix = "{}/weighted_base_learner/base_learner/".format(index)
    if not collection_key.startswith(prefix):
      return None
    return collection_key.replace(prefix, "")
