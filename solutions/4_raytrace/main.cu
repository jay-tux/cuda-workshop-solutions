//
// Created by jay on 3/18/25.
//

#include <cuda_wrapper.hpp>
#include <iostream>
#include <string>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "loader.hpp"
#include "kernel.hpp"
#include "scene.hpp"

int main(const int argc, const char **argv) {
  if (argc != 4 || argv[1] == std::string("-h")) {
    std::cout << "Usage: " << argv[0] << " <input scene file> <output png file> <mode>\n";
    std::cout << "  Mode is either 'id', 'dist', or 'color'\n";
    return 0;
  }

  bool ok;
  const auto scene = loader::load(argv[1], ok);

  if (!ok) {
    std::cerr << "Errors in scene file.\n";
    return 1;
  }

  scene.dump();

  image out { .data = nullptr, .width = 1920, .height = 1080 };
  cuCheck(cudaMallocManaged(&out.data, out.width * out.height * 3 * sizeof(byte)));
  auto scene_gpu = scene.to_gpu();
  scene_gpu.dump();

  if (argv[3] == std::string("id")) {
    kernel::render(scene_gpu, out, kernel::mode::ID);
  }
  else if (argv[3] == std::string("dist")) {
    kernel::render(scene_gpu, out, kernel::mode::DIST);
  }
  else if (argv[3] == std::string("color")) {
    kernel::render(scene_gpu, out, kernel::mode::COLOR);
  }
  else {
    std::cout << "Invalid mode: " << argv[3] << "\n";
    return 1;
  }

  stbi_write_png(argv[2], out.width, out.height, 3, out.data, out.width * 3);
  out.cleanup();
  scene_gpu.cleanup();
}