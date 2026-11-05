cmake_minimum_required(VERSION 3.22)
project(ctcs_stc VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Aligator: point CMAKE_PREFIX_PATH at its install prefix (and at the prefix of its
# dependencies, e.g. a conda env). See README.md.
find_package(aligator REQUIRED)

# Header-only interface: target_link_libraries(<target> PRIVATE ctcs_stc::ctcs_stc)
add_library(ctcs_stc INTERFACE)
add_library(ctcs_stc::ctcs_stc ALIAS ctcs_stc)
target_include_directories(ctcs_stc INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(ctcs_stc INTERFACE cxx_std_17)
target_link_libraries(ctcs_stc INTERFACE aligator::aligator)

option(CTCS_STC_BUILD_EXAMPLES "Build the examples" ${PROJECT_IS_TOP_LEVEL})
option(CTCS_STC_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})

if(CTCS_STC_BUILD_EXAMPLES)
  foreach(example pointmass_sphere stc_landing)
    add_executable(${example} examples/${example}.cpp)
    target_include_directories(${example} PRIVATE examples)
    target_link_libraries(${example} PRIVATE ctcs_stc::ctcs_stc)
  endforeach()
endif()

if(CTCS_STC_BUILD_TESTS)
  enable_testing()
  add_executable(test_spec tests/test_spec.cpp)
  target_include_directories(test_spec PRIVATE examples)
  target_link_libraries(test_spec PRIVATE ctcs_stc::ctcs_stc)
  add_test(NAME test_spec COMMAND test_spec)
endif()
