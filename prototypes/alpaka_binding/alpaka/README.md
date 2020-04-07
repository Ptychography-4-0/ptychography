# About

The prototype uses pybind11 to create a Python 3 binding from Alpaka C++ code. The goal of the implementation is to share memory between the Python and Alpaka C++ code and to implement an algorithm in CUDA.

This is the last step in providing a Python binding of an algorithm written with the alpaka library.

# Requirements

* pybind11
* numpy
* cmake > 3.14
* cuda > 9.0

Available via pip or conda.

```bash
  # use CUDA of the host system
  conda install -c conda-forge cmake=3.16 pybind11
  conda install -c anaconda numpy boost
```

Install Alpaka

```bash
#cd ptychography/prototypes/alpaka_binding/
git https://github.com/ComputationalRadiationPhysics/alpaka.git
mkdir alpaka/build && cd alpaka/build
cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -Dalpaka_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
make -j install
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
