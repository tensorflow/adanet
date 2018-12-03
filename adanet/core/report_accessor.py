"""Store and retrieve adanet.IterationReport protos.

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

from adanet.core import subnetwork
import six
import tensorflow as tf

# Required for Sphinx to generate docs from source-only, because report_pb2
# will not be found without building with Bazel.
# pylint: disable=g-import-not-at-top
try:
  from adanet.core import report_pb2 as report_proto
except ImportError:
  tf.logging.warn(
      "Failed to import report_pb2. ReportMaterializer will not work.")
# pylint: enable=g-import-not-at-top


def _parse_iteration_report_proto(iteration_pb_string):
  """Parses a serialized adanet.Iteration proto and returns the proto object."""

  iteration_report_pb = report_proto.IterationReport()
  iteration_report_pb.ParseFromString(iteration_pb_string)
  return iteration_report_pb


def _iteration_report_pb_to_subnetwork_reports(iteration_report_pb):
  """Converts IterationReport proto to a `MaterializedReport` list."""

  def _proto_map_to_dict(proto, field_name):
    """Converts map field of a proto to a dict.

    Args:
      proto: the proto to read from.
      field_name: name of the map field in the proto.

    Returns:
      Dict with the keys and values in proto.field_name.

    Raises:
      ValueError: if proto.field_name has a value that's not an int_value,
        float_value, bool_value, bytes_value, or string_value.
    """

    dictionary = {}
    proto_field = getattr(proto, field_name)
    for key in proto_field:
      if proto_field[key].HasField("int_value"):
        value = proto_field[key].int_value
      elif proto_field[key].HasField("float_value"):
        value = proto_field[key].float_value
      elif proto_field[key].HasField("bytes_value"):
        value = proto_field[key].bytes_value
      elif proto_field[key].HasField("string_value"):
        value = proto_field[key].string_value
      elif proto_field[key].HasField("bool_value"):
        value = proto_field[key].bool_value
      else:
        raise ValueError("{} map in subnetwork_report_pb has invalid field. "
                         "key: {} value: {} type: {}".format(
                             field_name, key, proto_field[key],
                             type(proto_field[key])))
      dictionary[key] = value

    return dictionary

  return [
      subnetwork.MaterializedReport(
          iteration_number=iteration_report_pb.iteration_number,
          name=subnetwork_report_pb.name,
          hparams=_proto_map_to_dict(
              proto=subnetwork_report_pb, field_name="hparams"),
          attributes=_proto_map_to_dict(
              proto=subnetwork_report_pb, field_name="attributes"),
          metrics=_proto_map_to_dict(
              proto=subnetwork_report_pb, field_name="metrics"),
          included_in_final_ensemble=(
              subnetwork_report_pb.included_in_final_ensemble))
      for subnetwork_report_pb in iteration_report_pb.subnetwork_reports
  ]


def _create_subnetwork_report_proto(materialized_subnetwork_report):
  """Creates a Subnetwork proto."""

  def _update_proto_map_from_dict(proto, field_name, dictionary):
    """Updates map field of proto with key-values in dictionary.

    Args:
      proto: the proto to be updated in place.
      field_name: name of the map field in the proto.
      dictionary: dict where the keys and values come from.

    Raises:
      ValueError: if value in dictionary is not a binary type
        (str in python 2; bytes in python 3), text type (unicode in python 2;
        str in python 3), int, bool, or float.
    """

    for key, value in dictionary.items():
      field = getattr(proto, field_name)
      if isinstance(value, bool):
        field[key].bool_value = value
      elif isinstance(value, six.binary_type):
        field[key].bytes_value = value
      elif isinstance(value, six.text_type):
        field[key].string_value = value
      elif isinstance(value, int):
        field[key].int_value = value
      elif isinstance(value, float):
        field[key].float_value = value
      else:
        raise ValueError("{} {}'s value must be a binary type "
                         "(str in python 2; bytes in python 3), "
                         "a text type (unicode in python 2; str in python 3), "
                         "int, bool, or float, but its type is {}.".format(
                             field_name, key, type(value)))

  subnetwork_report_pb = report_proto.SubnetworkReport()
  subnetwork_report_pb.name = materialized_subnetwork_report.name
  _update_proto_map_from_dict(
      proto=subnetwork_report_pb,
      field_name="hparams",
      dictionary=materialized_subnetwork_report.hparams)
  _update_proto_map_from_dict(
      proto=subnetwork_report_pb,
      field_name="attributes",
      dictionary=materialized_subnetwork_report.attributes)
  _update_proto_map_from_dict(
      proto=subnetwork_report_pb,
      field_name="metrics",
      dictionary=materialized_subnetwork_report.metrics)
  subnetwork_report_pb.included_in_final_ensemble = (
      materialized_subnetwork_report.included_in_final_ensemble)

  return subnetwork_report_pb


def _create_iteration_report_pb(iteration_number, subnetwork_report_pb_list):
  """Creates an IterationReport proto."""

  iteration_report_pb = report_proto.IterationReport()

  iteration_report_pb.iteration_number = iteration_number
  iteration_report_pb.subnetwork_reports.extend(subnetwork_report_pb_list)

  return iteration_report_pb


class _ReportAccessor(object):
  """Store and retrieve adanet.IterationReport protos."""

  def __init__(self, report_dir, filename="iteration_reports.tfrecord"):
    """Creates a `_ReportAccessor` instance.

    Args:
      report_dir: Directory to store the report.
      filename: Name of the file.

    Returns:
      A `_ReportAccessor` instance.
    """

    tf.gfile.MakeDirs(report_dir)
    self._full_filepath = os.path.join(report_dir, filename)

  def write_iteration_report(self, iteration_number, materialized_reports):
    """Writes an iteration's `MaterializedReports` to a `Report` proto.

    TODO: Remove iteration_number from the argument of this method.

    Note that even materialized_reports also contain iteration
    number, those are ignored -- only the iteration_number that is passed into
    this method would be written to the proto.

    Args:
      iteration_number: Int for the iteration number.
      materialized_reports: A list of `adanet.subnetwork.MaterializedReport`
        objects.
    """

    iteration_report_pb = _create_iteration_report_pb(
        iteration_number=iteration_number,
        subnetwork_report_pb_list=map(_create_subnetwork_report_proto,
                                      materialized_reports),
    )
    self._append_iteration_report_pb(iteration_report_pb)
    tf.logging.info("Wrote IterationReport for iteration %s to %s",
                    iteration_number, self._full_filepath)

  def _append_iteration_report_pb(self, iteration_pb):
    """Appends an adanet.IterationReport proto to the end of the file."""

    # As of Tensorflow 1.7, TFRecordWriter does not support appending,
    # which is why this method has to read the whole file, overwrite the same
    # file with the records it had just read, then write the new record.
    #
    # https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/python/lib/io/tf_record.py#L96
    previous_iterations = []
    if tf.gfile.Exists(self._full_filepath):
      try:
        tf_record_iterator = tf.compat.v1.io.tf_record_iterator
      except AttributeError:
        tf_record_iterator = tf.python_io.tf_record_iterator

      previous_iterations = list(tf_record_iterator(self._full_filepath))
    with tf.python_io.TFRecordWriter(self._full_filepath) as writer:
      for prev_iteration_pb_string in previous_iterations:
        writer.write(prev_iteration_pb_string)
      writer.write(iteration_pb.SerializeToString())

  def read_iteration_reports(self):
    """Reads all iterations of the Report.

    Each `adanet.subnetwork.MaterializedReport` list is one AdaNet iteration.
    The first list in the sequence is iteration 0, followed by iteration 1, and
    so on.

    Returns:
      Iterable of lists of `adanet.subnetwork.MaterializedReport`s.
    """

    iteration_pb_seq = self._read_iteration_report_pb_list()
    return map(_iteration_report_pb_to_subnetwork_reports, iteration_pb_seq)

  def _read_iteration_report_pb_list(self):
    """Returns an Iterable of adanet.IterationReport protos."""

    if tf.gfile.Exists(self._full_filepath):
      try:
        tf_record_iterator = tf.compat.v1.io.tf_record_iterator
      except AttributeError:
        tf_record_iterator = tf.python_io.tf_record_iterator
      return map(_parse_iteration_report_proto,
                 tf_record_iterator(self._full_filepath))
    return []
