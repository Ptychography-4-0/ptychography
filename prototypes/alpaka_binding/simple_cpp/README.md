# About

The prototype uses pybind11 to create a Python 3 binding from C++ code. The goal of the implementation is to share memory between the Python and C++ code and to implement an algorithm in C++.

This is the first step in providing a Python binding of an algorithm written with the alpaka library.

# Requirements

* cmake >= 3.15
* pybind11
* numpy

## Linux

### conda

``` bash
  conda install -c conda-forge cmake=3.16 pybind11
  conda install -c anaconda numpy
```

### pip

``` bash
  pip install cmake
  pip install numpy

  # unfortunately, the pybind11 pip package has a bug at provding the cmake files
  # install from source
  PYBIND11_VERSION=2.5.0
  wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz
  tar -xf v${PYBIND11_VERSION}.tar.gz && cd pybind11-${PYBIND11_VERSION}
  mkdir build && cd build && cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} ..
  make install
```

## Windows

# Install

```bash
  mkdir build
  cd build
  cmake ..
  make
```

# Using

```bash
  python test.py
```
