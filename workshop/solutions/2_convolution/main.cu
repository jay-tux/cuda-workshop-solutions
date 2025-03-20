//
// Created by jay on 3/12/25.
//

#include <fstream>
#include <string>
#include <iostream>
#include <optional>
#include <sstream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "cuda_wrapper.hpp"
#include "kernel.hpp"

using namespace c1_convolution;

std::optional<conv_kernel> ld_kernel(const std::string &file) {
  std::ifstream ifs(file);
  if (!ifs) {
    std::cerr << "Failed to open kernel file " << file << std::endl;
    return std::nullopt;
  }

  std::string line;
  std::getline(ifs, line);
  std::stringstream ss(line);

  int w, h;
  ss >> w >> h;

  // ReSharper disable once CppDFAMemoryLeak // we are returning the buffer
  auto *buffer = new float[w * h];
  for (int i = 0; i < h; i++) {
    if (!ifs) {
      std::cerr << "Unexpected EOF\n";
      delete[] buffer;
      return std::nullopt;
    }

    std::getline(ifs, line);
    ss = std::stringstream(line);
    for (int j = 0; j < w; j++) {
      if (!ss) {
        std::cerr << "Unexpected EOL\n";
        delete[] buffer;
        return std::nullopt;
      }

      ss >> buffer[i * w + j];
    }
  }
  // ReSharper disable once CppDFAMemoryLeak // we are returning the buffer
  return conv_kernel{.buffer = buffer, .w = w, .h = h};
}

__global__ void convolution_inline(const byte *image, const float *matrix, const int i_w, const int i_h, const int m_w, const int m_h, byte *out) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= i_w || y >= i_h) return;

  byte r = 0, g = 0, b = 0;
  for (int kx = -m_w / 2; kx < m_w / 2; kx++) {
    const int mx = kx + m_w / 2;
    for (int ky = -m_h / 2; ky < m_h / 2; ky++) {
      const int my = ky + m_h / 2;

      const int x_in = x + kx;
      const int y_in = y + ky;
      if ((x_in >= 0 && x_in < i_w) && (y_in >= 0 && y_in < i_h)) {
        const auto k_v = matrix[my * m_w + mx];
        const auto *px = image + (y_in * i_w + x_in) * 3;
        r += static_cast<byte>(static_cast<float>(px[0]) * k_v);
        g += static_cast<byte>(static_cast<float>(px[1]) * k_v);
        b += static_cast<byte>(static_cast<float>(px[2]) * k_v);
      }
    }
  }

  out[3 * (x + y * i_w) + 0] = r;
  out[3 * (x + y * i_w) + 1] = g;
  out[3 * (x + y * i_w) + 2] = b;
}

int main(const int argc, const char **argv) {
  if (argc != 4 || argv[1] == std::string("-h")) {
    std::cout << "Usage: " << argv[0] << " <input image> <convolution kernel file> <output image>\n";
    return 0;
  }

  int w, h, chan;
  byte *img_ = stbi_load(argv[1], &w, &h, &chan, 3);
  if (!img_) {
    std::cerr << "Could not load image: " << argv[1] << "\n";
    return 1;
  }

  const auto conv = ld_kernel(argv[2]);
  if (!conv) {
    stbi_image_free(img_);
    return 1;
  }

  auto *img_out = new byte[3 * w * h];
  do_convolution(image{ .data = img_, .w = w, .h = h }, image{ .data = img_out, .w = w, .h = h }, *conv);

  stbi_write_png(argv[3], w, h, 3, img_out, w * 3);

  delete [] conv->buffer;
  stbi_image_free(img_); delete [] img_out;
}