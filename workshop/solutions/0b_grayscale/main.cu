//
// Created by jay on 3/18/25.
//

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <iostream>
#include "cuda_wrapper.hpp"

using byte = unsigned char;

struct pixel { byte r, g, b; };

__host__ __device__ pixel pixel_at(const byte *data, const int w, const int x, const int y) {
  const byte r = data[3 * (w * y + x) + 0];
  const byte g = data[3 * (w * y + x) + 1];
  const byte b = data[3 * (w * y + x) + 2];
  return {r, g, b};
}

__global__ void kernel(const byte *__restrict__ data, byte *__restrict__ out, int w, int h) {
  // GIMP lightness formula: 1/2 * (max(R, G, B) + min(R, G, B))
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h) return;

  const pixel px = pixel_at(data, w, x, y);
  const byte lightness = (max(max(px.r, px.g), px.b) + min(min(px.r, px.g), px.b)) / 2;
  out[w * y + x] = lightness;
}

int main(const int argc, const char **argv) {
  if (argc != 3 || argv[1] == std::string("-h")) {
    std::cout << "Usage: " << argv[0] << " <input image file> <output image file>\n";
    return 0;
  }

  // load image
  int w, h, n;
  byte *data_cpu = stbi_load(argv[1], &w, &h, &n, 3);
  if (!data_cpu) {
    std::cerr << "Could not load image: " << argv[1] << "\n";
    return 1;
  }

  byte *data_gpu, *out;
  // 1. Allocate GPU buffers
  cuCheck(cudaMallocManaged(&data_gpu, w * h * 3 * sizeof(byte))); // one per channel
  cuCheck(cudaMallocManaged(&out, w * h * sizeof(byte))); // only one channel

  // 2. Copy input data to GPU
  cuCheck(cudaMemcpy(data_gpu, data_cpu, w * h * 3 * sizeof(byte), cudaMemcpyDefault));

  // 3. Perform computation
  //      -> using 1024 threads per block, but in 2D
  //      -> using enough blocks to cover the entire image
  constexpr dim3 block_size{32, 32}; // 32 * 32 = 1024
  const dim3 grid_size{w / block_size.x + 1, h / block_size.y + 1};
  cuCheckAsync((kernel<<<grid_size, block_size>>>(data_gpu, out, w, h)));
  cuCheck(cudaDeviceSynchronize()); // wait for kernel to finish

  // 4. Copy output data to CPU
  // ! This is not needed -> unified memory allocated by cudaMallocManaged is accessible by both CPU and GPU

  stbi_write_png(argv[2], w, h, 1, out, w);

  // 5. Clean up GPU resources
  // ! We have to wait with freeing the output buffer until the output image is written!
  cuCheck(cudaFree(data_gpu));
  cuCheck(cudaFree(out));
  stbi_image_free(data_cpu);
}