#!/usr/bin/env bash
# Build external dependencies that cannot successfully install via pip or conda

set -e

# Check for the operating system and install NGT (C++ lib)
if [[ $(uname) == "Darwin" ]]; then
  echo "Running under Mac OS X"

  # Setup environment
  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  if brew ls --versions cmake > /dev/null; then
    echo "cmake already installed"
  else
    brew install cmake
  fi
  if brew ls --versions gcc@9 > /dev/null; then
    echo "gcc@9 already installed"
  else
    brew install gcc@9
  fi
  ln -s ./gcc-9 /usr/local/bin/gcc
  ln -s ./g++-9 /usr/local/bin/g++
  export CXX=g++
  export CC=gcc

  # Find the latest release
  FILE=$(curl -s https://api.github.com/repos/yahoojapan/NGT/releases/latest | grep zipball_url | cut -d '"' -f 4)
  if [ -z "${FILE}" ]; then
    FILE="https://github.com/yahoojapan/NGT/archive/v1.7.9.zip"
    echo "Could not fetch latest release, will use predefined one.";
  else
    echo "Latest release is '$FILE'";
  fi
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
  sudo make install

  #  # Install NGT Python bindings
  #  cd ../python
  #  rm -rf dist
  #  python setup.py sdist
  #  pip install dist/ngt-*.tar.gz
  pip install ngt --upgrade

elif [[ $(uname -s) == Linux* ]]; then
  echo "Running under Linux"

  # Find the latest release
  FILE=$(curl -s https://api.github.com/repos/yahoojapan/NGT/releases/latest | grep zipball_url | cut -d '"' -f 4)
  if [ -z "${FILE}" ]; then
    FILE="https://github.com/yahoojapan/NGT/archive/v1.7.9.zip"
    echo "Could not fetch latest release, will use predefined one.";
  else
    echo "Latest release is '$FILE'";
  fi
  echo "Downloading $FILE"
  wget "$FILE"
  BNAME=$(basename "$FILE")
  echo "Release is $BNAME"

  # Install NGT
  rm -rf yahoojapan-NGT-*
  unzip "$BNAME"
  cd yahoojapan-NGT-*
  mkdir build
  cd build
  cmake ..
  make
  sudo make install

  #  # Install NGT Python bindings
  #  cd ../python
  #  rm -rf dist
  #  python setup.py sdist
  #  pip install dist/ngt-*.tar.gz
  pip install ngt --upgrade

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
