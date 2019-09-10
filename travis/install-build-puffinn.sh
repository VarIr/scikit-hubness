#!/usr/bin/env bash
# Build external dependencies that cannot successfully install via pip or conda

set -e

# Check for the operating system and install puffinn
if [[ $(uname) == "Darwin" ]]; then
  echo "Running under Mac OS X and CPU..."
  git clone https://github.com/puffinn/puffinn.git
  cd puffinn
  python3 setup.py build
  pip3 install .

elif [[ $(uname -s) == Linux* ]]; then
  echo "Running under Linux on CPU..."
  git clone https://github.com/puffinn/puffinn.git
  cd puffinn
  python3 setup.py build
  pip3 install .

elif [[ $(uname -s) == MINGW32_NT* ]]; then
  echo "Running under Win x86-32"
  echo "Nothing to build."

elif [[ $(uname -s) == MINGW64_NT* ]]; then
  echo "Running under Win x86-64"
  echo "Nothing to build."

elif [[ $(uname -s) == CYGWIN* ]]; then
  echo "Running under Cygwin"
  echo "Nothing to build."

fi
