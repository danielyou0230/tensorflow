# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reverse xxd dump from *.cc source file"""
import argparse
import os
import re


def generate_default_output(filename, postfix=None, extension=None):
  """Generate output filename given the filename and extension

  Args:
    filename
    postfix
    extension

  Return:
    string for the output filename given input args
  """
  name, ext = os.path.splitext(filename)

  if extension is not None:
    if not extension.startswith("."):
      extension = "." + extension

    ext = extension

  if postfix is None:
    postfix = ""

  output = "{}{}{}".format(name, postfix, ext)

  return output


def reverse_dump(filename, output=None, extension=".tflite"):
  """Reverse dump the tensorflow model weight from C++ array source array

  Args:
    filename: string of the filename (the input *.cc file)
    output: string of output filename, default to be same as input file
      but with different extension, default extension is *.tflite
  """
  if output is None:
    output = generate_default_output(filename, extension=extension)

  # Pattern to match with hexadecimal value in the array
  pattern = re.compile(r"\W*(0x[0-9a-fA-F,x ]+).*")

  array = bytearray()
  with open(filename) as f:
    for line in f:
      values_match = pattern.match(line)

      if values_match is None:
        continue

      # Match in the parentheses (hex array only)
      list_text = values_match.group(1)
      # Extract hex values (text)
      values_text = filter(None, list_text.split(","))
      # Convert to hex
      values = [int(x, base=16) for x in values_text]

      array.extend(values)

  with open(output, 'wb') as f:
    f.write(array)

  print("Byte data written to `{}`".format(output))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "source",
    type=str,
    help="C/C++ source file dumped from `xxd -i [HEX_FILE]`")
  parser.add_argument("-o", "--output", type=str, help="Output filename")

  args = parser.parse_args()

  reverse_dump(args.source, args.output)
