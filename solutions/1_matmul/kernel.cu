//
// Created by jay on 3/20/25.
//

#include "kernel.hpp"
#include "cuda_wrapper.hpp"

__host__ __device__ int &matrix::at(const unsigned col, const unsigned row) {
  return data[row * n + col];
}

__host__ __device__ const int &matrix::at(const unsigned col, const unsigned row) const {
  return data[row * n + col];
}

__global__ void kernel_entry(const matrix a, const matrix b, matrix c) {
  const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= c.n || row >= c.n) return;

  int acc = 0;
  for (unsigned k = 0; k < c.n; k++) {
    acc += a.at(row, k) * b.at(k, col);
  }
  c.at(row, col) = acc;
}

__host__ matrix kernel::make_gpu(unsigned n) {
  matrix gpu{ .data = nullptr, .n = n };
  cuCheck(cudaMalloc(&gpu.data, n * n * sizeof(int)));
  return gpu;
}


__host__ matrix kernel::copy_to_gpu(const matrix& cpu) {
  matrix gpu{ .data = nullptr, .n = cpu.n };
  cuCheck(cudaMalloc(&gpu.data, cpu.n * cpu.n * sizeof(int)));
  cuCheck(cudaMemcpy(gpu.data, cpu.data, cpu.n * cpu.n * sizeof(int), cudaMemcpyDefault));
  return gpu;
}

__host__ matrix kernel::copy_to_cpu(const matrix& gpu) {
  matrix cpu{
    .data = new int[gpu.n * gpu.n],
    .n = gpu.n
  };
  cuCheck(cudaMemcpy(cpu.data, gpu.data, gpu.n * gpu.n * sizeof(int), cudaMemcpyDefault));
  return cpu;
}

__host__ void kernel::matmul(const matrix& a_gpu, const matrix& b_gpu, matrix& c_gpu) {
  constexpr dim3 block_size{32, 32};
  const dim3 grid_size{a_gpu.n / block_size.x + 1, a_gpu.n / block_size.y + 1};
  cuCheckAsync((kernel_entry<<<grid_size, block_size>>>(a_gpu, b_gpu, c_gpu)));
  cuCheck(cudaDeviceSynchronize());
}

__host__ void kernel::cleanup(matrix& gpu) {
  cuCheck(cudaFree(gpu.data));
  gpu.data = nullptr;
  gpu.n = 0;
}

__host__ matrix kernel::full_program(const matrix& a_cpu, const matrix& b_cpu) {
  // 1. Copy input to GPU
  matrix a_gpu = copy_to_gpu(a_cpu);
  matrix b_gpu = copy_to_gpu(b_cpu);
  matrix c_gpu = make_gpu(a_cpu.n);

  // 2. Run kernel & wait for completion
  matmul(a_gpu, b_gpu, c_gpu);

  // 3. Copy output to CPU
  const matrix c = copy_to_cpu(c_gpu);

  // 4. Clean up resources
  cleanup(a_gpu);
  cleanup(b_gpu);
  cleanup(c_gpu);

  // 5. Done!
  return c;
}
