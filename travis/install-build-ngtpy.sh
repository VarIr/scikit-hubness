#!/usr/bin/env bash
# Build external dependencies that cannot successfully install via pip or conda
# If you use this file as template, don't forget to `chmod a+x newfile`

set -e

# Check for the operating system and install NGT (C++ lib) and others
if [[ $(uname) == "Darwin" ]]; then
  echo "Running under Mac OS X and CPU..."
  sysctl machdep.cpu.brand_string

  if pip install ngt; then
    exit 0
  fi

  # Setup environment
  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
  echo "brew update && brew upgrade"
  if brew ls --versions cmake > /dev/null; then
    echo "cmake already installed"
  else
    brew install cmake
  fi
  xcode-select --install || true
  echo "Install MacOS SDK header for 10.14..."
  open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg || true

  if brew ls --versions gcc@9 > /dev/null; then
    echo "gcc@9 already installed, upgrading"
    brew upgrade gcc@9
  else
    brew install gcc@9
  fi
  ln -s ./gcc-9 /usr/local/bin/gcc
  ln -s ./g++-9 /usr/local/bin/g++
  echo "Prepend /usr/local/bin to PATH"
  export PATH=/usr/local/bin:$PATH
  export CXX=g++
  export CC=gcc

  # Find the latest release of NGT
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
  # TODO work-around for https://github.com/yahoojapan/NGT/issues/34
  # enable AVX when bug is fixed
  cmake -DNGT_AVX_DISABLED=ON ..
  CXXFLAGS='-fpermissive' make
  sudo make install

  # make library available
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib64:/usr/local/lib"

  # Install NGT Python bindings
  cd ../python
  rm -rf dist
  python3 setup.py sdist  # python somehow maps to python2...
  pip3 install dist/ngt-*.tar.gz

elif [[ $(uname -s) == Linux* ]]; then
  echo "Running under Linux on CPU..."
  cat /proc/cpuinfo

  if pip install ngt; then
    exit 0
  fi

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
