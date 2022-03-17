#!/usr/bin/env bash
# Build external dependencies that cannot successfully install via pip or conda
# If you use this file as template, don't forget to `chmod a+x newfile`

set -e

# Check for the operating system and install NGT
if [[ $(uname) == "Darwin" ]]; then
  if [[ $(command ngt > /dev/null 2>&1) && $(command ngtq > /dev/null 2>&1) && $(command ngtqg > /dev/null 2>&1) ]]; then
    # This only checks for available ngt commands. Does not currently check the version.
    # To update NGT, this must be adapted.
    echo "NGT already installed"
  else
    echo "Installing NGT under Mac OS X..."
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    brew install cmake
    brew install gcc@9
    export CXX=/usr/local/bin/g++-9
    export CC=/usr/local/bin/gcc-9
    pushd /tmp/
    git clone https://github.com/yahoojapan/NGT
    cd NGT/
    mkdir build
    cd build/
    cmake ..
    make
    sudo make install
  fi

elif [[ $(uname -s) == Linux* ]]; then
  if [[ $(command ngt > /dev/null 2>&1) && $(command ngtq > /dev/null 2>&1) && $(command ngtqg > /dev/null 2>&1) ]]; then
    # This only checks for available ngt commands. Does not currently check the version.
    # To update NGT, this must be adapted.
    echo "NGT already installed"
  else
    echo "Installing NGT under Linux..."
    pushd /tmp/
    git clone https://github.com/yahoojapan/NGT
    cd NGT/
    mkdir build
    cd build/
    cmake ..
    make
    sudo make install
    sudo ldconfig /usr/local/lib/
    popd
    rm -r /tmp/NGT
  fi

elif [[ $(uname -s) == MINGW32_NT* ]]; then
  echo "NGT not available under Win x86-32"

elif [[ $(uname -s) == MINGW64_NT* ]]; then
  echo "NGT not available under Win x86-64"

elif [[ $(uname -s) == CYGWIN* ]]; then
  echo "NGT not available under Cygwin"

fi
