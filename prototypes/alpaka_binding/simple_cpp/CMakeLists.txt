cmake_minimum_required(VERSION 3.3)
project(cppBinding)

find_package(pybind11 REQUIRED)

pybind11_add_module(cppBinding binding.cpp)

# on windows, the build can be done with the Visual Studio or ninja generator
# this configuration is Visual Studio specific
if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
  set_target_properties(cppBinding PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
  set_target_properties(cppBinding PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR} )
  set_target_properties(cppBinding PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR} )
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test.py
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
