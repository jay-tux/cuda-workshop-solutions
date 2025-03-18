//
// Created by jay on 3/12/25.
//

#ifndef KERNEL_HPP
#define KERNEL_HPP

namespace c1_convolution {
  using byte = unsigned char;

  struct image
  {
    byte *data;
    int w, h;
  };

  struct conv_kernel
  {
    float *buffer;
    int w, h;

    __host__ __device__ constexpr float &operator()(const int x, const int y) const noexcept {
      return buffer[y * w + x];
    }
  };

  __host__ void do_convolution(const image &in_cpu, const image &out_cpu, const conv_kernel &kernel_cpu);
}

#endif //KERNEL_HPP
