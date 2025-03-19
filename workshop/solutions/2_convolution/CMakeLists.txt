cmake_minimum_required(VERSION 3.28)

add_executable(2_convolution main.cu kernel.cu)
target_link_libraries(2_convolution stb::stb)