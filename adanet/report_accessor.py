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

from adanet.base_learner_report import MaterializedBaseLearnerReport
import adanet.report_pb2 as report_proto
import six
import tensorflow as tf


def _parse_iteration_report_proto(iteration_pb_string):
  """Parses a serialized adanet.Iteration proto and returns the proto object."""

  iteration_report_pb = report_proto.IterationReport()
  iteration_report_pb.ParseFromString(iteration_pb_string)
  return iteration_report_pb


def _iteration_report_pb_to_base_learner_reports(iteration_report_pb):
  """Converts IterationReport proto to `MaterializedBaseLearnerReport` list."""

  def _proto_map_to_dict(proto, field_name):
    """Converts map field of a proto to a dict.

    Args:
      proto: the proto to read from.
      field_name: name of the map field in the proto.

    Returns:
      dict with the keys and values in proto.field_name.

    Raises:
      ValueError: if proto.field_name has a value that's not an int_value,
        float_value, bool_value, or string_value.
    """

    dictionary = {}
    proto_field = getattr(proto, field_name)
    for key in proto_field:
      if proto_field[key].HasField("int_value"):
        value = proto_field[key].int_value
      elif proto_field[key].HasField("float_value"):
        value = proto_field[key].float_value
      elif proto_field[key].HasField("string_value"):
        value = proto_field[key].string_value
      elif proto_field[key].HasField("bool_value"):
        value = proto_field[key].bool_value
      else:
        raise ValueError("{} map in base_learner_report_pb has invalid field. "
                         "key: {} value: {} type: {}".format(
                             field_name, key, proto_field[key],
                             type(proto_field[key])))
      dictionary[key] = value

    return dictionary

  return [
      MaterializedBaseLearnerReport(
          hparams=_proto_map_to_dict(
              proto=base_learner_report_pb, field_name="hparams"),
          attributes=_proto_map_to_dict(
              proto=base_learner_report_pb, field_name="attributes"),
          metrics=_proto_map_to_dict(
              proto=base_learner_report_pb, field_name="metrics"),
          included_in_final_ensemble=(
              base_learner_report_pb.included_in_final_ensemble))
      for base_learner_report_pb in iteration_report_pb.base_learner_reports
  ]


def _create_base_learner_report_proto(materialized_base_learner_report):
  """Creates a BaseLearner proto."""

  def _update_proto_map_from_dict(proto, field_name, dictionary):
    """Updates map field of proto with key-values in dictionary.

    Args:
      proto: the proto to be updated in place.
      field_name: name of the map field in the proto.
      dictionary: dict where the keys and values come from.

    Raises:
      ValueError: if value in dictionary is not an instance of string, int,
        or float.
    """

    for key, value in dictionary.items():
      field = getattr(proto, field_name)
      if isinstance(value, bool):
        field[key].bool_value = value
      elif isinstance(value, six.string_types):
        field[key].string_value = value
      elif isinstance(value, int):
        field[key].int_value = value
      elif isinstance(value, float):
        field[key].float_value = value
      else:
        raise ValueError("{} {}'s value must be an instance of string, int, "
                         "bool, or float, but its type is {}.".format(
                             field_name, key, type(value)))

  base_learner_report_pb = report_proto.BaseLearnerReport()
  _update_proto_map_from_dict(
      proto=base_learner_report_pb,
      field_name="hparams",
      dictionary=materialized_base_learner_report.hparams)
  _update_proto_map_from_dict(
      proto=base_learner_report_pb,
      field_name="attributes",
      dictionary=materialized_base_learner_report.attributes)
  _update_proto_map_from_dict(
      proto=base_learner_report_pb,
      field_name="metrics",
      dictionary=materialized_base_learner_report.metrics)
  base_learner_report_pb.included_in_final_ensemble = (
      materialized_base_learner_report.included_in_final_ensemble)

  return base_learner_report_pb


def _create_iteration_report_pb(iteration_number,
                                base_learner_report_pb_list):
  """Creates an IterationReport proto."""

  iteration_report_pb = report_proto.IterationReport()

  iteration_report_pb.iteration_number = iteration_number
  iteration_report_pb.base_learner_reports.extend(base_learner_report_pb_list)

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

  def write_iteration_report(self, iteration_number,
                             materialized_base_learner_reports):
    """Writes the `MaterializedBaseLearnerReports` at an iteration to Report.

    Args:
      iteration_number: int for the iteration number.
      materialized_base_learner_reports: `MaterializedBaseLearnerReport` list.
    """

    iteration_report_pb = _create_iteration_report_pb(
        iteration_number=iteration_number,
        base_learner_report_pb_list=map(_create_base_learner_report_proto,
                                        materialized_base_learner_reports),
    )
    self._append_iteration_report_pb(iteration_report_pb)

  def _append_iteration_report_pb(self, iteration_pb):
    """Appends an adanet.IterationReport proto to the end of the file."""

    # As of Tensorflow 1.7, TFRecordWriter does not support appending,
    # which is why this method has to read the whole file, overwrite the same
    # file with the records it had just read, then write the new record.
    #
    # https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/python/lib/io/tf_record.py#L96
    previous_iterations = []
    if tf.gfile.Exists(self._full_filepath):
      previous_iterations = list(
          tf.python_io.tf_record_iterator(self._full_filepath))
    with tf.python_io.TFRecordWriter(self._full_filepath) as writer:
      for prev_iteration_pb_string in previous_iterations:
        writer.write(prev_iteration_pb_string)
      writer.write(iteration_pb.SerializeToString())

  def read_iteration_reports(self):
    """Reads all iterations of the Report.

    Each `MaterializedBaseLearnerReport` list is one AdaNet iteration. The first
    list in the sequence is iteration 0, followed by iteration 1, and so on.

    Returns: Iterable of lists of `MaterializedBaseLearnerReport`s.
    """

    iteration_pb_seq = self._read_iteration_report_pb_list()
    return map(_iteration_report_pb_to_base_learner_reports, iteration_pb_seq)

  def _read_iteration_report_pb_list(self):
    """Returns an Iterable of adanet.IterationReport protos."""

    if tf.gfile.Exists(self._full_filepath):
      return map(_parse_iteration_report_proto,
                 tf.python_io.tf_record_iterator(self._full_filepath))
    return []
