# About

The prototype uses pybind11 to create a Python 3 binding from C++ code. The goal of the implementation is to share memory between the Python and C++ code and to implement an algorithm in C++.

This is the first step in providing a Python binding of an algorithm written with the alpaka library.

# Requirements

* pybind11
* numpy

Available via pip or conda.

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
