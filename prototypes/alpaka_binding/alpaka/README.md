# About

The prototype uses pybind11 to create a Python 3 binding from Alpaka C++ code. The goal of the implementation is to share memory between the Python and Alpaka C++ code and to implement an algorithm in CUDA.

This is the last step in providing a Python binding of an algorithm written with the alpaka library.

# Requirements

* pybind11
* numpy
* cmake >= 3.15
* cuda >= 10.0
* boost >= 1.67.0

## Linux

### conda

```bash
  # use CUDA of the host system
  # use Boost of the host system
  # alternative: conda install -c anaconda boost
  conda install -c conda-forge cmake=3.16 pybind11
  conda install -c anaconda numpy

  #cd ptychography/prototypes/alpaka_binding/
  git https://github.com/alpaka-group/alpaka.git
  mkdir alpaka/build && cd alpaka/build
  cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -Dalpaka_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
  make -j install
```

### pip

``` bash
  # use CUDA of the host system
  # use Boost of the host system
  pip install cmake
  pip install numpy

  # unfortunately, the pybind11 pip package has a bug at provding the cmake files
  # install from source
  PYBIND11_VERSION=2.5.0
  wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz
  tar -xf v${PYBIND11_VERSION}.tar.gz && cd pybind11-${PYBIND11_VERSION}
  mkdir build && cd build && cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} ..
  make install

  #cd ptychography/prototypes/alpaka_binding/
  git https://github.com/alpaka-group/alpaka.git
  mkdir alpaka/build && cd alpaka/build
  cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -Dalpaka_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
  make -j install
```

## Windows

# Install

```bash
  mkdir build
  cd build
  # maybe you need to set: -DPYTHON_EXECUTABLE:FILEPATH=
  cmake ..
  make
```

# Using

```bash
  python test.py
  # test, if it is a cuda application
  nvprof python test.py
```
