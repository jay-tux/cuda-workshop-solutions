//
// Created by jay on 3/18/25.
//

#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "scene.hpp"

using byte = unsigned char;

struct image
{
  byte *data;
  int width, height;

  __device__ void set_pixel(int x, int y, float3 color);
  __host__ void cleanup();
};

struct do_log
{
  bool enabled;
  int x;
  int y;
};

struct kernel
{
  enum struct mode { ID, DIST, COLOR };
  __host__ static void render(const scene_gpu &scene, image &img, mode m);
  __device__ static bool intersect(const scene_gpu &scene, const ray &ray, float &dist, size_t &obj_id, const do_log &log);
};

#endif //KERNEL_CUH
