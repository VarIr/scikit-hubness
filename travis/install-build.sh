#!/usr/bin/env bash
# Build external dependencies that cannot successfully install via pip or conda

set -e

# Check for the operating system and install NGT (C++ lib)
if [[ $(uname) == "Darwin" ]]; then
  echo "Running under Mac OS X and CPU..."
  sysctl machdep.cpu.brand_string

  # Setup environment
  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  echo "brew update && brew upgrade"
  if brew ls --versions cmake > /dev/null; then
    echo "cmake already installed"
  else
    brew install cmake
  fi
  brew update && brew upgrade
  if brew ls --versions gcc@9 > /dev/null; then
    echo "gcc@9 already installed"
  else
    brew install gcc@9
  fi
  ln -s ./gcc-9 /usr/local/bin/gcc
  # ln -s /usr/local/Cellar/gcc/9.2.0/bin/gcc-9/gcc-9 /usr/local/bin/gcc  # TODO don't hardcode version number
  ln -s ./g++-9 /usr/local/bin/g++
  # ln -s /usr/local/Cellar/gcc/9.2.0/bin/gcc-9/g++-9 /usr/local/bin/g++
  echo "What is in /usr/local/Cellar/gcc/9.2.0/bin/gcc-9?"
  ls /usr/local/Cellar/gcc/9.2.0/bin/gcc-9
  echo "What is in /usr/local/bin?"
  ls /usr/local/bin
  echo "Prepend /usr/local/bin to PATH"
  export PATH=/usr/local/bin:$PATH
#  brew unlink gcc
#  brew cleanup
#  brew link gcc
  export CXX=g++
  export CC=gcc
  #  alias gcc='gcc-9'
  #  alias cc='gcc-9'
  #  alias g++='g++-9'
  #  alias c++='c++-9'
  #  export CXX=g++
  #  export CC=gcc

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
  rm -rf ./*NGT*
  unzip "$BNAME"
  cd ./*NGT*  # could be NGT-v.x.x.x, or yahoojapan-NGT-v.x.x.x
  mkdir build
  cd build
  which gcc
  echo "$PATH"
  cmake ..
  # make SDKROOT="$(xcrun --show-sdk-path)" MACOSX_DEPLOYMENT_TARGET=
  make
  sudo make install

  # make library available
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib64:/usr/local/lib"
  sudo ldconfig

  # Install NGT Python bindings
  cd ../python
  rm -rf dist
  python setup.py sdist
  pip install dist/ngt-*.tar.gz
  # pip install ngt --upgrade

elif [[ $(uname -s) == Linux* ]]; then
  echo "Running under Linux on CPU..."
  cat /proc/cpuinfo

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
  rm -rf ./*NGT*
  unzip "$BNAME"
  cd ./*NGT*  # could be NGT-v.x.x.x, or yahoojapan-NGT-v.x.x.x
  mkdir build
  cd build
  which gcc
  echo "$PATH"
  cmake ..
  make
  sudo make install

  # make library available
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib64:/usr/local/lib"
  sudo ldconfig

  # Install NGT Python bindings
  cd ../python
  rm -rf dist
  python setup.py sdist
  pip install dist/ngt-*.tar.gz
  # pip install ngt --upgrade

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
