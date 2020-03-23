#pragma once

#include "algo.hpp"

#include <pybind11/numpy.h>

#include <cmath>

namespace py = pybind11;

namespace Cu {

template <typename DATA_T, // type of the data
          typename SIZE_T> // type of the data size
__global__ void
kernel(DATA_T *input, DATA_T *output, SIZE_T size) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    output[id] = input[id] + static_cast<DATA_T>(id % 3);
  }
}

} // namespace Cu

template <typename DATA_T, // type of the data
          typename SIZE_T> // type of the data size
class CuAlgo : public Algo<DATA_T, SIZE_T> {
private:
  // allocate memory on CPU and GPU and copy it manual
  DATA_T *m_input = nullptr;
  DATA_T *m_output = nullptr;
  DATA_T *m_input_device = nullptr;
  DATA_T *m_output_device = nullptr;

  bool m_init = false;

public:
  CuAlgo(SIZE_T lenght) : Algo<DATA_T, SIZE_T>(lenght) {}

  ~CuAlgo() { deinit(); }

  bool init() override {
    if (!m_init) {
      cudaError_t err;
      const SIZE_T size = Algo<DATA_T, SIZE_T>::get_size();

      m_input = new DATA_T[size];
      m_output = new DATA_T[size];

      err = cudaMalloc((void **)&m_input_device, sizeof(DATA_T) * size);
      if (err) {
        std::cerr << "Error at allocate GPU memory for the input:" << std::endl
                  << cudaGetErrorString(err) << std::endl;
        delete[] m_input;
        delete[] m_output;
        return false;
      }

      err = cudaMalloc((void **)&m_output_device, sizeof(DATA_T) * size);
      if (err) {
        std::cerr << "Error at allocate GPU memory for the output:" << std::endl
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(m_input_device);
        delete[] m_input;
        delete[] m_output;
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
      delete[] m_input;
      delete[] m_output;

      cudaFree(m_input_device);
      cudaFree(m_output_device);
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

    err = cudaMemcpy(m_input_device, m_input, sizeof(DATA_T) * size,
                     cudaMemcpyHostToDevice);
    if (err) {
      std::cerr << "Cannot copy data from host to device:" << std::endl
                << cudaGetErrorString(err) << std::endl;
      return false;
    }

    int threads = 32;
    int blocks = std::ceil(size / static_cast<double>(threads));

    Cu::kernel<DATA_T, SIZE_T>
        <<<blocks, threads>>>(m_input_device, m_output_device, size);
    err = cudaGetLastError();
    if (err) {
      std::cerr << "Kernel error:" << std::endl
                << cudaGetErrorString(err) << std::endl;
      return false;
    }

    err = cudaMemcpy(m_output, m_output_device, sizeof(DATA_T) * size,
                     cudaMemcpyDeviceToHost);
    if (err) {
      std::cerr << "Cannot copy data from device to host:" << std::endl
                << cudaGetErrorString(err) << std::endl;
      return false;
    }

    return true;
  }
};
