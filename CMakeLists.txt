cmake_minimum_required(VERSION 3.10)
project(Connect4)

set(CMAKE_CXX_STANDARD 17)

# SFML
set(SFML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SFML-2.5.1-windows-gcc-7.3.0-mingw-64-bit")
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

# Executable
add_executable(Connect4 connect4_gui.cpp connect4.cpp)

# Link libraries
target_link_libraries(Connect4 PUBLIC
        sfml-graphics
        sfml-window
        sfml-system
        )

if(OpenMP_CXX_FOUND)
    target_link_libraries(Connect4 PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(Connect4 PUBLIC -fopenmp)
endif()
