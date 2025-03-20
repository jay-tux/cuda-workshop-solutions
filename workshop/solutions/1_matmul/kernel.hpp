//
// Created by jay on 3/20/25.
//

#ifndef KERNEL_HPP
#define KERNEL_HPP

struct matrix
{
  int *data;
  unsigned n;

  [[nodiscard]] __host__ __device__ int &at(unsigned col, unsigned row);
  [[nodiscard]] __host__ __device__ const int &at(unsigned col, unsigned row) const;
};

struct kernel
{
  __host__ static matrix make_gpu(unsigned n);
  __host__ static matrix copy_to_gpu(const matrix &cpu);
  __host__ static matrix copy_to_cpu(const matrix &gpu);
  __host__ static void matmul(const matrix &a_gpu, const matrix &b_gpu, matrix &c_gpu);
  __host__ static void cleanup(matrix &gpu);

  __host__ static matrix full_program(const matrix &a_cpu, const matrix &b_cpu);
};

#endif //KERNEL_HPP
