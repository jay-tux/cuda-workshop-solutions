cmake_minimum_required(VERSION 3.28)
set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES conan_provider.cmake)
project(cuda-solutions LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0 -march=native -rdc=true")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -march=native -rdc=true")
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_compile_definitions("DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

find_package(stb REQUIRED)
find_package(Curses REQUIRED)
find_package(ASSIMP REQUIRED)
find_package(inipp REQUIRED)

add_subdirectory(0a_vec_add)
add_subdirectory(0b_grayscale)
add_subdirectory(1_matmul)
add_subdirectory(2_convolution)
add_subdirectory(3_game_of_life)
add_subdirectory(4_raytrace)