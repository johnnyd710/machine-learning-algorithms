cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(xlm)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(xlm xlm.cpp)
target_link_libraries(xlm "${TORCH_LIBRARIES}")
set_property(TARGET xlm PROPERTY CXX_STANDARD 14)
