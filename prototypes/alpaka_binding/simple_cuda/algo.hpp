#pragma once

#include <pybind11/numpy.h>

#include <iostream>

namespace py = pybind11;

template <typename DATA_T, // type of the data
          typename SIZE_T> // type of the data size
class Algo {
private:
  const SIZE_T m_size;
  DATA_T *m_input = nullptr;
  DATA_T *m_output = nullptr;
  bool m_init = false;

public:
  Algo(SIZE_T length) : m_size(length) {}

  // allocate memory
  virtual bool init() = 0;

  // deallocate memory
  virtual bool deinit() = 0;

  SIZE_T get_size() const { return m_size; }

  virtual py::array get_input_view() = 0;

  virtual py::array get_output_view() = 0;

  // example algorithm out[i] = in[i] + (i%3)
  virtual bool compute() = 0;
};
