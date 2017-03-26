project( DEPCCG )
cmake_minimum_required( VERSION 2.8.5 )
# set(CMAKE_VERBOSE_MAKEFILE 1)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3 -g -fpic -march=native")

find_package( OpenMP )
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include( UseCython )
find_package(NumPy REQUIRED)
include_directories( ${PYTHON_NUMPY_INCLUDE_DIR} )
include_directories( ${DEPCCG_SOURCE_DIR}/src )

add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/src
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )


add_subdirectory( src )
