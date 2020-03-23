cmake_minimum_required(VERSION 3.14)
project(cuBinding LANGUAGES CXX CUDA)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

add_library(cuBinding SHARED binding.cu)
set_property(TARGET cuBinding PROPERTY CUDA_STANDARD 11)
target_link_libraries(cuBinding
  ${PYTHON_LIBRARIES}
  cudart)
set_target_properties(cuBinding PROPERTIES PREFIX "")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test.py
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
