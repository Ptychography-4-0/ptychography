#pragma once

#include <pybind11/numpy.h>

#include <iostream>

namespace py = pybind11;

template <typename DATA_T, // type of the data
          typename SIZE_T> // type of the data size
class Algo {
private:
  SIZE_T m_size;
  DATA_T *m_input = nullptr;
  DATA_T *m_output = nullptr;
  bool m_init = false;

public:
  Algo(SIZE_T length) : m_size(length) {}

  // allocate memory
  bool init() {
    if (!m_init) {
      m_input = new DATA_T[m_size];
      m_output = new DATA_T[m_size];
      m_init = true;
      return true;
    } else {
      return false;
    }
  }

  // deallocate memory
  bool deinit() {
    if (m_init) {
      delete[] m_input;
      delete[] m_output;
      m_init = false;
      return true;
    } else {
      return false;
    }
  }

  SIZE_T get_size() const { return m_size; }

  py::array get_input_view() {
    if (!m_init)
      init();
    return py::array(m_size, m_input, py::cast(*this));
  }

  py::array get_output_view() {
    if (!m_init)
      init();
    return py::array(m_size, m_output, py::cast(*this));
  }

  void compute() {
    if (!m_init)
      init();
    for (SIZE_T i = 0; i < m_size; ++i)
      m_output[i] = m_input[i] + static_cast<DATA_T>(i % 3);
  }

  ~Algo() { deinit(); }
};
