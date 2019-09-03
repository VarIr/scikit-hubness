#!/usr/bin/env bash
# Build external dependencies that cannot successfully install via pip or conda

set -e

# Check for the operating system and install NGT (C++ lib)
if [[ $(uname) == "Darwin" ]]; then
  echo "Running under Mac OS X"

  # Setup environment
  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  brew install cmake
  brew install gcc@9
  ln -s ./gcc-9 /usr/local/bin/gcc
  ln -s ./g++-9 /usr/local/bin/g++
  export CXX=g++
  export CC=gcc

  # Find the latest release
  FILE=$(curl -s https://api.github.com/repos/yahoojapan/NGT/releases/latest | grep zipball_url | cut -d '"' -f 4)
  wget "$FILE"
  BNAME=$(basename "$FILE")

  # Install NGT C++
  rm -rf yahoojapan-NGT-*
  unzip "$BNAME"
  cd yahoojapan-NGT-*
  mkdir build
  cd build
  cmake ..
  make
  make install

  # Install NGT Python bindings
  cd ../python
  rm -rf dist
  python3 setup.py sdist
  pip3 install dist/ngt-*.tar.gz

elif [[ $(uname -s) == Linux* ]]; then
  echo "Running under Linux"

  # Find the latest release
  FILE=$(curl -s https://api.github.com/repos/yahoojapan/NGT/releases/latest | grep zipball_url | cut -d '"' -f 4)
  wget "$FILE"
  BNAME=$(basename "$FILE")

  # Install NGT
  rm -rf yahoojapan-NGT-*
  unzip "$BNAME"
  cd yahoojapan-NGT-*
  mkdir build
  cd build
  cmake ..
  make
  make install

  # Install NGT Python bindings
  cd ../python
  rm -rf dist
  python3 setup.py sdist
  pip3 install dist/ngt-*.tar.gz

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
