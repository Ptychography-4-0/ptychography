# About

The prototype uses pybind11 to create a Python 3 binding from CUDA C++ code. The goal of the implementation is to share memory between the Python and CUDA C++ code and to implement an algorithm in CUDA.

This is the second step in providing a Python binding of an algorithm written with the alpaka library.

# Requirements

* pybind11
* numpy
* cmake > 3.14
* cuda > 9.0

Available via pip or conda.

```bash
  conda install -c conda-forge cppyy
  conda install -c anaconda numpy
  conda install -c anaconda cmake
```


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
