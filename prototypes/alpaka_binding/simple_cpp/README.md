# About

The prototype uses cppyy to create a Python 3 binding from C++ code. The goal of the implementation is to share memory between the Python and C++ code and to implement an algorithm in C++.

This is the first step in providing a Python binding of an algorithm written with the alpaka library.

# Requirements

* cppyy
* numpy

```bash
  conda install -c conda-forge cppyy
  conda install -c anaconda numpy
```

# Install

```bash
  mkdir build
  cd build
  cmake ..
  make
```

# Using

```bash
  # to run a test
  python algo.py
  # use interactive
  python
  >>> import algo
  >>> size = 12
  >>> a = algo.AlgoFloat(size)
```

# Technical Details

There are two wrappers. The `algo.py` is a wrapper for a nice Python interface. The `wrapper.hpp` and `wrapper.cpp` are necessary for us to compile a template class to a shared library. Actually `cppyy` is able to do the JIT compilation of the template class `algo`. But if we use `CUDA` or `Alpaka`, `cppyy` may not be able to JIT compile the code because `Cling` does not fully support CUDA at this time. With the `Wrapper` files we can decouple the algorithm code from the compilation capabilities of `Cling`.
