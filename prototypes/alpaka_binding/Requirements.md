# About

Each project extends the requirements of its predecessor project. The order is simple\_cpp, simple\_cuda and alpaka. This means that the simple\_cuda project needs the dependencies of simple\_cpp and the alpaka project needs the dependencies of the simple\_cuda project.

# Linux

## simple_cpp

git

``` bash
  conda install -c conda-forge cmake=3.16
  conda install -c anaconda numpy
  # or
  # pip install cmake numpy
```

``` bash
  # there is a bugfix (https://github.com/pybind/pybind11/pull/2240), which is not in a release yet
  git clone https://github.com/pybind/pybind11.git
  cd pybind11
  git checkout c776e9e
  mkdir build && cd build && cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} ..
  cmake --install .
```

## simple_cuda

CUDA > 9.0

## alpaka

boost >= 1.67.0

``` bash
  git clone --depth=1 --branch release-0.5.0 https://github.com/alpaka-group/alpaka.git
  mkdir alpaka/build && cd alpaka/build
  cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
  cmake --install .
```

# Windows

Instructions are for the Powershell.

## simple_cpp

git

``` bash
  conda install -c conda-forge cmake=3.16
  conda install -c anaconda numpy
  # or
  # pip install cmake numpy
```

``` bash
  # there is a bugfix (https://github.com/pybind/pybind11/pull/2240), which is not in a release yet
  git clone https://github.com/pybind/pybind11.git
  cd pybind11
  git checkout c776e9e
  mkdir build ; cd build ;
  cmake -DPYBIND11_TEST=off -G"Visual Studio 15 2017 Win64" -DCMAKE_PREFIX_PATH="${ENV:CONDA_PREFIX}" -DCMAKE_INSTALL_PREFIX="${ENV:CONDA_PREFIX}" ..
  cmake --install .
```


## simple_cuda

CUDA > 9.0

## alpaka

boost >= 1.67.0

``` bash
  git clone --depth=1 --branch release-0.5.0 https://github.com/alpaka-group/alpaka.git
  mkdir alpaka/build ; cd alpaka/build
  cmake -G"Visual Studio 15 2017 Win64" -DCMAKE_PREFIX_PATH="${ENV:CONDA_PREFIX}" -DCMAKE_INSTALL_PREFIX="${ENV:CONDA_PREFIX}" -DBoost_INCLUDE_DIR=C:\Path\to\boost\ ..
  cmake --install .
```
