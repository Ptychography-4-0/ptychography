cmake_minimum_required(VERSION 3.14)
project(cuBinding LANGUAGES CXX CUDA)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})

add_library(cuBinding SHARED binding.cu)
target_link_libraries(cuBinding
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(cuBinding PROPERTIES CUDA_STANDARD 11)
set_target_properties(cuBinding PROPERTIES PREFIX "")

if(MSVC)
  set_target_properties(cuBinding PROPERTIES SUFFIX ".pyd")
endif()

# on windows, the build can be done with the Visual Studio or ninja generator
# this configuration is Visual Studio specific
if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
  set_target_properties(cuBinding PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
  set_target_properties(cuBinding PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR} )
  set_target_properties(cuBinding PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR} )
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test.py
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
