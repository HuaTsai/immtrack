cmake_minimum_required(VERSION 3.15)
project(immtrack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.4 CONFIG REQUIRED)

# Header-only core library — used by both the pybind module and tests.
add_library(immtrack_core INTERFACE)
target_include_directories(immtrack_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
target_link_libraries(immtrack_core INTERFACE Eigen3::Eigen)
target_compile_features(immtrack_core INTERFACE cxx_std_20)

pybind11_add_module(_core bindings/_core.cc)
target_link_libraries(_core PRIVATE immtrack_core)

install(TARGETS _core DESTINATION immtrack)

option(IMMTRACK_BUILD_TESTS "Build C++ unit tests" OFF)
if(IMMTRACK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
