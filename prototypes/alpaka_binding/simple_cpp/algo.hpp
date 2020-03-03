#pragma once

template <typename DATA_T, // type of the data
	  typename SIZE_T> // type of the data size
class Algo {
private:
  bool m_init = false;
  SIZE_T m_size;

  // the test algorithm use an input buffer as source and an output buffer as destination
  DATA_T *m_input_data = nullptr;
  DATA_T *m_output_data = nullptr;

public:
  Algo(SIZE_T size) : m_size(size) {}

  ~Algo(){
    if(m_init)
      deinit();
  }

  // allocate memory
  bool init() {
    if(m_init) {
      return false;
    } else {
      m_input_data = new DATA_T[m_size];
      m_output_data = new DATA_T[m_size];
      m_init = true;
      return true;
    }
  }

  // deallocate memory
  bool deinit() {
    if(m_init) {
      delete[] m_input_data;
      delete[] m_output_data;
      m_init = true;
      return true;
    } else {
      return false;
    }
  }

  // return the size of the 1D arrays
  SIZE_T get_size() {
    return m_size;
  }

  // get the input buffer
  DATA_T * get_input_memory(){
    if(!m_init)
      init();
    return m_input_data;
  }

  // get the output buffer
  DATA_T * get_output_memory(){
    if(!m_init)
      init();
    return m_output_data;
  }

  // run a test algorithm
  bool algo() {
    if(!m_init)
      return false;

    for(SIZE_T i = 0; i < m_size; ++i){
      m_output_data[i] = m_input_data[i] + static_cast<DATA_T>(i%3);
    }
    return true;
  }
};
