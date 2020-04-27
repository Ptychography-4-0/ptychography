# About

Each project extends the requirements of its predecessor project. The order is simple\_cpp, simple\_cuda and alpaka. This means that the simple\_cuda project needs the dependencies of simple\_cpp and the alpaka project needs the dependencies of the simple\_cuda project.

# Linux

## simple_cpp

``` bash
  conda install -c conda-forge cmake=3.16 pybind11
  conda install -c anaconda numpy
```

or

``` bash
  pip install cmake numpy

  # unfortunately, the pybind11 pip package has a bug at provding the cmake files
  # install from source
  PYBIND11_VERSION=2.5.0
  wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz
  tar -xf v${PYBIND11_VERSION}.tar.gz && cd pybind11-${PYBIND11_VERSION}
  mkdir build && cd build && cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} ..
  cmake --install .
```

## simple_cuda

CUDA > 9.0

## alpaka

boost >= 1.67.0

``` bash
  git clone https://github.com/alpaka-group/alpaka.git
  mkdir alpaka/build && cd alpaka/build
  cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -Dalpaka_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
  cmake --install .
```

# Windows

Instructions are for the Powershell.

## simple_cpp

``` bash
  conda install -c conda-forge cmake=3.16 pybind11
  conda install -c anaconda numpy
```

or

``` bash
  pip install cmake numpy

  # unfortunately, the pybind11 pip package has a bug at provding the cmake files
  # install from source
  wget https://github.com/pybind/pybind11/archive/v2.5.0.zip -OutFile v2.5.0.zip
  Expand-Archive .\v2.5.0.zip -DestinationPath . ; cd pybind11-2.5.0
  mkdir build ; cd build ;
  cmake -DPYBIND11_TEST=off -G"Visual Studio 15 2017 Win64" -DCMAKE_PREFIX_PATH="${ENV:CONDA_PREFIX}" -DCMAKE_INSTALL_PREFIX="${ENV:CONDA_PREFIX}" ..
  cmake --install .
```


## simple_cuda

CUDA > 9.0

**Important:** Pybind11 has a bug, which avoids to compile pybind11 at Windows with the nvcc: https://github.com/pybind/pybind11/issues/2180

To workaround the bug, change the line `explicit operator type&() { return *(this->value); }` in `${CONDA_PREFIX}\library\include\pybind11\cast.h:1495` or `${CONDA_PREFIX}\include\pybind11\cast.h:1495` to `explicit operator type&() { return *(static_cast<type *>(this->value)); }`.

## alpaka

boost >= 1.67.0

``` bash
  git clone https://github.com/alpaka-group/alpaka.git
  mkdir alpaka/build ; cd alpaka/build
  cmake -G"Visual Studio 15 2017 Win64" -DCMAKE_PREFIX_PATH="${ENV:CONDA_PREFIX}" -DCMAKE_INSTALL_PREFIX="${ENV:CONDA_PREFIX}" -DBoost_INCLUDE_DIR=C:\Path\to\boost\ -Dalpaka_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF ..
  cmake --install .
```
