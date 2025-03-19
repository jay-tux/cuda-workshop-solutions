cmake_minimum_required(VERSION 3.28)

add_executable(4_raytrace main.cu loader.cu scene.cu kernel.cu)
target_link_libraries(4_raytrace stb::stb assimp::assimp inipp::inipp)