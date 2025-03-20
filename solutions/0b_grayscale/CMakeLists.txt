cmake_minimum_required(VERSION 3.28)

add_executable(0b_grayscale main.cu)
target_link_libraries(0b_grayscale stb::stb)