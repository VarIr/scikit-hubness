#!/usr/bin/env bash

set -e

echo "pip installing required python packages"
pip install -r requirements.txt

python --version

# echo "pip installing test-related packages (coveralls, etc.)"
# pip install -r requirements-tests.txt
