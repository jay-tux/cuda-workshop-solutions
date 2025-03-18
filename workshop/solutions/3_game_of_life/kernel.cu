//
// Created by jay on 3/15/25.
//

#include "cuda_wrapper.hpp"
#include "kernel.hpp"


using namespace c2_game_of_life;

__global__ void do_step(const buffer in, buffer out) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= in.width || y >= in.height) return;

  int alive = 0;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      if (!(i == 0 && j == 0) && in.is_live(x + i, y + j))
        ++alive;
    }
  }

  switch (alive) {
  case 2:{
    if (in.is_live(x, y)) out.set_alive(x, y);
    else out.set_dead(x, y);
    break;
  }

  case 3:{
    out.set_alive(x, y);
    break;
  }

  default:{
    out.set_dead(x, y);
    break;
  }
  }
}

__host__ void c2_game_of_life::step(const buffer& in, const buffer& out) {
  constexpr auto block_size = dim3(32, 32);
  const auto grid_size = dim3(in.width / block_size.x + 1, in.height / block_size.y + 1);

  cuCheckAsync((do_step<<<block_size, grid_size>>>(in, out)));
  cuCheck(cudaDeviceSynchronize());
}
