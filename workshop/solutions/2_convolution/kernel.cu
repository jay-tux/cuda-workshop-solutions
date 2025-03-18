//
// Created by jay on 3/12/25.
//

#include "cuda_wrapper.hpp"
#include "kernel.hpp"

using namespace c1_convolution;

__global__ void convolution_gpu(const byte *__restrict__ image, byte *__restrict__ out, const int i_w, const int i_h, const conv_kernel kernel) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= i_w || y >= i_h) return;

  const auto m_w = kernel.w;
  const auto m_h = kernel.h;

  float r = 0, g = 0, b = 0;
  for (int kx = -m_w / 2; kx <= m_w / 2; kx++) {
    const int mx = kx + m_w / 2;
    for (int ky = -m_h / 2; ky <= m_h / 2; ky++) {
      const int my = ky + m_h / 2;

      const int x_in = x + kx;
      const int y_in = y + ky;
      if ((x_in >= 0 && x_in < i_w) && (y_in >= 0 && y_in < i_h)) {
        const auto k_v = kernel(mx, my); // matrix[my * m_w + mx];
        const auto *px = image + (y_in * i_w + x_in) * 3;
        r += static_cast<float>(px[0]) * k_v;
        g += static_cast<float>(px[1]) * k_v;
        b += static_cast<float>(px[2]) * k_v;
      }
    }
  }

  r = r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r);
  g = g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g);
  b = b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b);

  out[3 * (x + y * i_w) + 0] = static_cast<byte>(r);
  out[3 * (x + y * i_w) + 1] = static_cast<byte>(g);
  out[3 * (x + y * i_w) + 2] = static_cast<byte>(b);
}

__host__ void c1_convolution::do_convolution(const image &in_cpu, const image &out_cpu, const conv_kernel &kernel_cpu) {
  constexpr auto block = dim3(32, 32);
  const auto grid = dim3(in_cpu.w / block.x + 1, in_cpu.h / block.y + 1);

  byte *in_gpu, *out_gpu; float *kernel_gpu;
  cuCheck(cudaMalloc(&in_gpu, in_cpu.w * in_cpu.h * 3 * sizeof(byte)));
  cuCheck(cudaMalloc(&out_gpu, out_cpu.w * out_cpu.h * 3 * sizeof(byte)));
  cuCheck(cudaMalloc(&kernel_gpu, kernel_cpu.w * kernel_cpu.h * sizeof(float)));

  cuCheck(cudaMemcpy(in_gpu, in_cpu.data, in_cpu.w * in_cpu.h * 3 * sizeof(byte), cudaMemcpyDefault));
  cuCheck(cudaMemcpy(kernel_gpu, kernel_cpu.buffer, kernel_cpu.w * kernel_cpu.h * sizeof(float), cudaMemcpyDefault));

  cuCheckAsync((convolution_gpu<<<grid, block>>>(in_gpu, out_gpu, in_cpu.w, in_cpu.h, conv_kernel{ .buffer = kernel_gpu, .w = kernel_cpu.w, .h = kernel_cpu.h })));
  cuCheck(cudaDeviceSynchronize());

  cuCheck(cudaMemcpy(out_cpu.data, out_gpu, out_cpu.w * out_cpu.h * 3 * sizeof(byte), cudaMemcpyDefault));
  cuCheck(cudaFree(in_gpu));
  cuCheck(cudaFree(out_gpu));
  cuCheck(cudaFree(kernel_gpu));
}