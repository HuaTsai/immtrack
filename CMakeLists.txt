cmake_minimum_required(VERSION 3.15)
project(immtrack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.4 CONFIG REQUIRED)

pybind11_add_module(_core bindings/_core.cc)
target_include_directories(_core PRIVATE cpp/include)
target_link_libraries(_core PRIVATE Eigen3::Eigen)

install(TARGETS _core DESTINATION immtrack)
