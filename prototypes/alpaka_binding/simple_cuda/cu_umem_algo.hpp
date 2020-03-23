#pragma once

#include "algo.hpp"

#include <pybind11/numpy.h>

#include <cmath>

namespace py = pybind11;

namespace CuUmem {

template <typename DATA_T, // type of the data
          typename SIZE_T> // type of the data size
__global__ void
kernel(DATA_T *input, DATA_T *output, SIZE_T size) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    output[id] = input[id] + static_cast<DATA_T>(id % 3);
  }
}

} // namespace CuUmem

template <typename DATA_T, // type of the data
          typename SIZE_T> // type of the data size
class CuUmemAlgo : public Algo<DATA_T, SIZE_T> {
private:
  // points to memory that is automatically shared between CPU and GPU
  DATA_T *m_input = nullptr;
  DATA_T *m_output = nullptr;

  bool m_init = false;

public:
  CuUmemAlgo(SIZE_T lenght) : Algo<DATA_T, SIZE_T>(lenght) {}

  ~CuUmemAlgo() { deinit(); }

  bool init() override {
    if (!m_init) {
      cudaError_t err;
      const SIZE_T size = Algo<DATA_T, SIZE_T>::get_size();

      err = cudaMallocManaged(&m_input, sizeof(DATA_T) * size);
      if (err) {
        std::cerr << "Error at allocate GPU memory for the input:" << std::endl
                  << cudaGetErrorString(err) << std::endl;
        return false;
      }

      err = cudaMallocManaged(&m_output, sizeof(DATA_T) * size);
      if (err) {
        std::cerr << "Error at allocate GPU memory for the output:" << std::endl
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(m_input);
        return false;
      }

      m_init = true;
      return true;
    } else {
      return false;
    }
  }

  bool deinit() override {
    if (m_init) {
      cudaFree(m_input);
      cudaFree(m_output);
      m_init = false;
      return true;
    } else {
      return false;
    }
  }

  py::array get_input_view() override {
    if (!m_init)
      init();
    return py::array(Algo<DATA_T, SIZE_T>::get_size(), m_input,
                     py::cast(*this));
  }

  py::array get_output_view() override {
    if (!m_init)
      init();
    return py::array(Algo<DATA_T, SIZE_T>::get_size(), m_output,
                     py::cast(*this));
  }

  virtual bool compute() {
    if (!m_init)
      init();

    cudaError_t err;
    const SIZE_T size = Algo<DATA_T, SIZE_T>::get_size();

    int threads = 32;
    int blocks = std::ceil(size / static_cast<double>(threads));

    CuUmem::kernel<DATA_T, SIZE_T>
        <<<blocks, threads>>>(m_input, m_output, size);
    err = cudaGetLastError();
    if (err) {
      std::cerr << "Kernel error:" << std::endl
                << cudaGetErrorString(err) << std::endl;
      return false;
    }

    cudaDeviceSynchronize();

    return true;
  }
};
