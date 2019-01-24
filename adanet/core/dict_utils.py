"""AdaNet dictionary utilities.

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

import collections

import six


def flatten_dict(original_dict, delimiter="/"):
  """Flattens a dictionary of dictionaries by one level.

  Note that top level keys will be overridden if they collide with flat keys.
  E.g. using delimiter="/" and origial_dict={"foo/bar": 1, "foo": {"bar": 2}},
  the top level "foo/bar" key would be overwritten.

  Args:
    original_dict: The dictionary to flatten.
    delimiter: The value used to delimit the keys in the flat_dict.

  Returns:
    The falttened dictionary.
  """

  flat_dict = {}
  for outer_key, inner_dict in six.iteritems(original_dict):
    if isinstance(inner_dict, dict):
      for inner_key, value in six.iteritems(inner_dict):
        flat_dict["{}{}{}".format(outer_key, delimiter, inner_key)] = value
    else:
      flat_dict[outer_key] = inner_dict
  return flat_dict


def unflatten_dict(flat_dict, prefixes, delimiter="/"):
  """Unflattens a dictionary into a dict of dicts by one level.

  Args:
    flat_dict: The dictionary to unflatten.
    prefixes: The string keys to use for the unflattened dictionary. Keys in the
      flat_dict which do not begin with a prefix are unmodified.
    delimiter: The value used to delmit the keys in the flat_dict.

  Returns:
    The unflattened dictionary.
  """

  unflat_dict = collections.defaultdict(dict)
  for key, value in six.iteritems(flat_dict):
    parts = key.split(delimiter)
    if len(parts) > 1:
      prefix = parts[0]
      if prefix in prefixes:
        suffix = key[len(prefix + delimiter):]
        unflat_dict[prefix][suffix] = value
      else:
        unflat_dict[key] = value
    else:
      unflat_dict[key] = value
  return unflat_dict
