#pragma once

#include <alpaka/alpaka.hpp>
#include <pybind11/numpy.h>

#include <cmath>
#include <iostream>
#include <memory>

namespace py = pybind11;

struct ComputeKernel {

  template <typename TAcc, typename TData, typename TExtends>
  ALPAKA_FN_ACC void operator()(TAcc const &acc, TData *const input,
                                TData *const output,
                                TExtends const &extends) const {
    auto const globalThreadIdx =
        alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    auto const globalThreadExtends =
        alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

    auto const linearizedGlobalThreadIdx =
        alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtends);

    for (size_t i = linearizedGlobalThreadIdx[0]; i < extends.prod();
         i += globalThreadExtends.prod()) {
      output[i] = input[i] + static_cast<TData>(i % 3);
    }
  }
};

template <typename DATA_T, // type of the data
          typename SIZE_T> // type of the data size
class Algo {
public:
  using Dim = alpaka::dim::DimInt<1u>;
  using Vec = alpaka::vec::Vec<Dim, SIZE_T>;

  using HostAcc = alpaka::acc::AccCpuSerial<Dim, SIZE_T>;
  using HostQueueProperty = alpaka::queue::Blocking;
  using HostQueue = alpaka::queue::Queue<HostAcc, HostQueueProperty>;
  using HostDev = alpaka::dev::Dev<HostAcc>;
  using HostPltf = alpaka::pltf::Pltf<HostDev>;

  using DevAcc = alpaka::acc::AccGpuCudaRt<Dim, SIZE_T>;
  using DevQueueProperty = alpaka::queue::Blocking;
  using DevQueue = alpaka::queue::Queue<DevAcc, DevQueueProperty>;
  using DevDev = alpaka::dev::Dev<DevAcc>;
  using DevPltf = alpaka::pltf::Pltf<DevDev>;

private:
  std::unique_ptr<HostDev> hostDev;
  std::unique_ptr<HostQueue> hostQueue;

  DevDev const devDev;
  DevQueue devQueue;

  using HostBuf = alpaka::mem::buf::Buf<HostDev, DATA_T, Dim, SIZE_T>;
  std::unique_ptr<HostBuf> host_input;
  std::unique_ptr<HostBuf> host_output;

  using DevBuf = alpaka::mem::buf::Buf<DevDev, DATA_T, Dim, SIZE_T>;
  std::unique_ptr<DevBuf> device_input;
  std::unique_ptr<DevBuf> device_output;

  const SIZE_T m_size;
  const Vec extends;
  DATA_T *m_input = nullptr;
  DATA_T *m_output = nullptr;
  DATA_T *m_input_dev = nullptr;
  DATA_T *m_output_dev = nullptr;

  bool m_init = false;

public:
  Algo(SIZE_T size, unsigned int device_id = 0)
      : m_size(size), devDev(alpaka::pltf::getDevByIdx<DevPltf>(device_id)),
        devQueue(devDev), extends(Vec::all(static_cast<SIZE_T>(m_size))) {

    hostDev =
        std::make_unique<HostDev>(alpaka::pltf::getDevByIdx<HostPltf>(0u));
    hostQueue = std::make_unique<HostQueue>(*hostDev);
  }

  ~Algo() { deinit(); }

  // allocate memory
  virtual bool init() {
    if (m_init)
      return false;

    host_input = std::make_unique<HostBuf>(
        alpaka::mem::buf::alloc<DATA_T, SIZE_T>(*hostDev, extends));
    host_output = std::make_unique<HostBuf>(
        alpaka::mem::buf::alloc<DATA_T, SIZE_T>(*hostDev, extends));

    m_input = alpaka::mem::view::getPtrNative(*host_input);
    m_output = alpaka::mem::view::getPtrNative(*host_output);

    device_input = std::make_unique<DevBuf>(
        alpaka::mem::buf::alloc<DATA_T, SIZE_T>(devDev, extends));
    device_output = std::make_unique<DevBuf>(
        alpaka::mem::buf::alloc<DATA_T, SIZE_T>(devDev, extends));

    m_input_dev = alpaka::mem::view::getPtrNative(*device_input);
    m_output_dev = alpaka::mem::view::getPtrNative(*device_output);

    m_init = true;

    return true;
  }

  // deallocate memory
  virtual bool deinit() {
    if (!m_init)
      return false;

    host_input.reset(nullptr);
    host_output.reset(nullptr);
    device_input.reset(nullptr);
    device_output.reset(nullptr);

    m_input = nullptr;
    m_output = nullptr;
    m_input_dev = nullptr;
    m_output_dev = nullptr;

    m_init = false;

    return true;
  }

  SIZE_T get_size() const { return m_size; }

  virtual py::array get_input_view() {
    if (!m_init)
      init();
    return py::array(Algo<DATA_T, SIZE_T>::get_size(), m_input,
                     py::cast(*this));
  }

  virtual py::array get_output_view() {
    if (!m_init)
      init();
    return py::array(Algo<DATA_T, SIZE_T>::get_size(), m_output,
                     py::cast(*this));
  }

  // example algorithm out[i] = in[i] + (i%3)
  virtual bool compute() {
    // copy to GPU
    alpaka::mem::view::copy(devQueue, *device_input, *host_input, extends);

    // run kernel
    Vec const elementsPerThread(Vec::all(static_cast<SIZE_T>(1)));
    Vec const threadsPerBlock(Vec::all(static_cast<SIZE_T>(32)));
    Vec const blocksPerGrid(Vec::all(static_cast<SIZE_T>(
        std::ceil(m_size / static_cast<double>(threadsPerBlock[0])))));

    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, SIZE_T>;
    WorkDiv const workdiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

    ComputeKernel computeKernel;
    alpaka::kernel::exec<DevAcc>(devQueue, workdiv, computeKernel, m_input_dev,
                                 m_output_dev, extends);
    // copy to CPU
    alpaka::mem::view::copy(devQueue, *host_output, *device_output, extends);
    alpaka::wait::wait(devQueue);
    return true;
  }
};
