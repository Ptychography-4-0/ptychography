#pragma once

#include <iostream>

template<typename DATA_T, // type of the data
	 typename SIZE_T> // type of the data size
class Algo {
public:
  // the IOBuffer memory object is necessary for the numpy binding
  struct IOBuffer {
  private:
    SIZE_T m_size;
    DATA_T * m_data;

  public:
    IOBuffer(SIZE_T size) : m_size(size), m_data(new DATA_T[m_size]) {}

    DATA_T* data() { return m_data; }
    SIZE_T get_size() const { return m_size; }

    ~IOBuffer(){
      delete[] m_data;
    }

    DATA_T& operator[](SIZE_T i){
      return m_data[i];
    }
  };

private:
  SIZE_T m_size;
  bool m_init = false;

public:
  // the test algorithm use an input buffer as source and an output buffer as destination
  IOBuffer * m_input = nullptr;
  IOBuffer * m_output = nullptr;

  Algo(SIZE_T length) : m_size(length) {}

  // allocate memory
  bool init() {
    if (!m_init){
      m_input = new IOBuffer(m_size);
      m_output = new IOBuffer(m_size);
      m_init = true;
      return true;
    } else {
      return false;
    }
  }

  // deallocate memory
  bool deinit() {
    if (m_init) {
      delete m_input;
      delete m_output;
      m_init = false;
      return true;
    } else {
      return false;
    }
  }

  SIZE_T get_size() const { return m_size; }

  IOBuffer * get_input(){
    if (!m_init)
      init();
    return m_input;
  }

  IOBuffer * get_output(){
    if (!m_init)
      init();
    return m_output;
  }

  void compute(){
    if (!m_init)
      init();
    for(SIZE_T i = 0; i < m_size; ++i)
      (*m_output)[i] = (*m_input)[i] + static_cast<DATA_T>(i%3);
  }

  ~Algo(){
    deinit();
  }
};
