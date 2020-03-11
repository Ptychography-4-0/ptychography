cmake_minimum_required(VERSION 3.3)
project(cppBinding)

find_package(pybind11 REQUIRED)

pybind11_add_module(cppBinding binding.cpp)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test.py
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
