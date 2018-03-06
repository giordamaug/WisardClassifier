cmake_minimum_required(VERSION 2.8)
# Change compiler to your preferred compiler
set( CMAKE_C_COMPILER /opt/local/bin/gcc-mp-5 )
set( CMAKE_CXX_COMPILER /opt/local/bin/g++-mp-5 )

project( wisard-cxx_static_3.0 )


# Compile Library
set(LIBRARY_NAME wisard-cxx_static_3.0)
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY .. )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(${LIBRARY_NAME} SHARED wisard.cpp)
get_target_property(sf ${LIBRARY_NAME} IMPORT_SUFFIX)
install(FILES libwisard-cxx_static_3.0${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries( ${LIBRARY_NAME})

# Compile test
add_executable(test test.cpp )
target_link_libraries (test ${LIBRARY_NAME})

