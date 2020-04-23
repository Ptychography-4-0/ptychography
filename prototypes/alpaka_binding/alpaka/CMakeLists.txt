cmake_minimum_required(VERSION 3.14)
project(alpakaBinding LANGUAGES CXX)

set(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE_DEFAULT ON)
set(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE_DEFAULT OFF)
set(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE_DEFAULT OFF)
set(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE_DEFAULT OFF)
set(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE_DEFAULT OFF)
set(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE_DEFAULT OFF)
set(ALPAKA_ACC_CPU_BT_OMP4_ENABLE_DEFAULT OFF)
set(ALPAKA_ACC_GPU_CUDA_ENABLE ON)
set(ALPAKA_ACC_GPU_HIP_ENABLE OFF)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(alpaka REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})

alpaka_add_library(alpakaBinding SHARED binding.cpp)
target_link_libraries(alpakaBinding
  PRIVATE ${PYTHON_LIBRARIES}
  PUBLIC alpaka::alpaka
  )

set_target_properties(alpakaBinding PROPERTIES PREFIX "")

if(MSVC)
  set_target_properties(alpakaBinding PROPERTIES SUFFIX ".pyd")
endif()

# on windows, the build can be done with the Visual Studio or ninja generator
# this configuration is Visual Studio specific
if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
  set_target_properties(alpakaBinding PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
  set_target_properties(alpakaBinding PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR} )
  set_target_properties(alpakaBinding PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR} )
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test.py
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
