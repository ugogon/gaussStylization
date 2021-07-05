cmake_minimum_required(VERSION 3.4...3.18)
project(gauss_stylization)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE release)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(LIBIGL REQUIRED)

# Add your project files
include_directories("../utils_gauss/" "../utils_cubic/")

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

file(GLOB SRCFILES *.cpp ../utils_gauss/*.cpp ../utils_cubic/*.cpp)


find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} ${SRCFILES})

target_link_libraries(${PROJECT_NAME} PRIVATE igl::core)
