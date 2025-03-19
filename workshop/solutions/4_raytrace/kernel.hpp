//
// Created by jay on 3/18/25.
//

#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cstdint>

#include "scene.hpp"

using byte = unsigned char;

__device__ constexpr static uint8_t bounces = 4;
__device__ constexpr static float epsilon = 1e-3f;

__device__ static constexpr float3 color_to_id[32] {
  {1.0, 0.0, 0.0},  // Red
  {0.0, 1.0, 0.0},  // Green
  {0.0, 0.0, 1.0},  // Blue
  {1.0, 1.0, 0.0},  // Yellow
  {1.0, 0.0, 1.0},  // Magenta
  {0.0, 1.0, 1.0},  // Cyan
  {1.0, 0.5, 0.0},  // Orange
  {0.6, 0.2, 0.8},  // Purple
  {0.0, 0.5, 0.2},  // Teal
  {0.8, 0.3, 0.0},  // Brownish-orange
  {0.5, 0.0, 0.5},  // Deep purple
  {0.3, 0.7, 0.2},  // Olive green
  {0.7, 0.0, 0.3},  // Dark pink
  {0.0, 0.7, 0.7},  // Deep cyan
  {0.7, 0.7, 0.0},  // Mustard
  {0.5, 0.2, 0.0},  // Dark brown
  {0.2, 0.4, 0.6},  // Steel blue
  {0.6, 0.1, 0.2},  // Crimson
  {0.3, 0.8, 0.6},  // Mint green
  {0.9, 0.4, 0.6},  // Salmon
  {0.2, 0.3, 0.8},  // Royal blue
  {0.7, 0.5, 0.9},  // Lavender
  {0.1, 0.8, 0.3},  // Bright green
  {0.6, 0.9, 0.2},  // Lime green
  {0.8, 0.1, 0.7},  // Vivid magenta
  {0.9, 0.7, 0.0},  // Goldenrod
  {0.3, 0.9, 0.9},  // Aqua
  {0.6, 0.3, 0.1},  // Rust
  {0.2, 0.6, 0.8},  // Sky blue
  {0.7, 0.1, 0.4},  // Deep rose
  {0.4, 0.6, 0.2},  // Moss green
  {0.8, 0.5, 0.3},  // Warm peach
};

struct image
{
  byte *data;
  int width, height;

  __device__ void set_pixel(int x, int y, float3 color);
  __host__ void cleanup();
};

struct kernel
{
  enum struct mode { ID, DIST, COLOR };
  __host__ static void render(const scene_gpu &scene, image &img, mode m);
  __device__ static bool intersect(const scene_gpu &scene, const ray &ray, float &dist, size_t &obj_id, float3 &normal, material& mat);
};

#endif //KERNEL_CUH
