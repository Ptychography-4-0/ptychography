cmake_minimum_required(VERSION 3.3)
project(cppbind)

add_library(wrapper SHARED wrapper.cpp)
# add flag -fPIC
set_property(TARGET wrapper PROPERTY POSITION_INDEPENDENT_CODE)

# the header files are necessary for the cppyy JIT
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/algo.hpp
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/wrapper.hpp
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# python wrapper class
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/algo.py
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
