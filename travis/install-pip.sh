#!/usr/bin/env bash

set -e

echo "First install pybind11, so that nmslib build can succeed"
pip3 install pybind11

echo "pip installing required python packages"
pip3 install -r requirements.txt

python --version

# echo "pip installing test-related packages (coveralls, etc.)"
# pip install -r requirements-tests.txt
