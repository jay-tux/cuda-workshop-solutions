//
// Created by jay on 3/18/25.
//

#ifndef SCENE_CUH
#define SCENE_CUH

#include <vector>

struct ray
{
  float3 start;
  float3 direction;
};

struct plane
{
  float3 point;
  float3 normal;
  size_t id;

  __device__ bool intersect(const ray &ray, float &t) const;
};

struct sphere
{
  float3 center;
  float radius;
  size_t id;

  __device__ bool intersect(const ray &ray, float &t) const;
};

struct triangle
{
  float3 v0;
  float3 v1;
  float3 v2;
  size_t id;

  __device__ bool intersect(const ray &ray, float &t) const;
};

struct image;

struct scene_gpu
{
  plane *planes; size_t num_planes;
  sphere *spheres; size_t num_spheres;
  triangle *triangles; size_t num_triangles;

  float3 camera;
  float3 cam_forward, cam_right, cam_up;

  __host__ void dump() const;
  __device__ void dump_gpu() const;
  __host__ void cleanup();
};

struct scene_cpu
{
  std::vector<plane> planes;
  std::vector<sphere> spheres;
  std::vector<triangle> triangles;

  float3 camera;
  float3 cam_forward, cam_right, cam_up;

  __host__ void dump() const;
  __host__ void setup_cam();
  __host__ scene_gpu to_gpu() const;
};

#endif //SCENE_CUH
