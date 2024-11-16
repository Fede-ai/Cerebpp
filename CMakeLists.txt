cmake_minimum_required(VERSION 3.10)
project(Cerebpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING "Restricted build types" FORCE)

add_subdirectory(Cerebpp)
add_subdirectory(Test)
