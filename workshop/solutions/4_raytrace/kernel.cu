//
// Created by jay on 3/18/25.
//

#include "kernel.hpp"
#include "float3.hpp"

#include <cuda_wrapper.hpp>

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

__device__ constexpr float clamp(const float x, const float min, const float max) {
  return x < min ? min : (x > max ? max : x);
}

__device__ void image::set_pixel(const int x, const int y, const float3 color) {
  const byte r = static_cast<byte>(clamp(color.x, 0.0f, 1.0f) * 255);
  const byte g = static_cast<byte>(clamp(color.y, 0.0f, 1.0f) * 255);
  const byte b = static_cast<byte>(clamp(color.z, 0.0f, 1.0f) * 255);

  data[3 * (width * y + x) + 0] = r;
  data[3 * (width * y + x) + 1] = g;
  data[3 * (width * y + x) + 2] = b;
}

__host__ void image::cleanup() {
  cuCheck(cudaFree(data)); data = nullptr;
  width = 0; height = 0;
}

__global__ void renderer(const scene_gpu scene, image img, kernel::mode m) {
  int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= img.width || y >= img.height) return;

  // 1. Construct ray
  const float aspect = static_cast<float>(img.width) / static_cast<float>(img.height);
  const float3 x_v = (static_cast<float>(x) / static_cast<float>(img.width) - 0.5f) * aspect * scene.cam_right;
  const float3 y_v = (0.5f - (static_cast<float>(y) / static_cast<float>(img.height))) * scene.cam_up;
  const float3 z_v = scene.cam_forward;

  const ray r{
    .start = scene.camera,
    .direction = normalize(x_v + y_v + z_v)
  };

  // 2. Find the closest intersection
  float dist;
  size_t hit;

  if (x == 860 && y == 1000) {
    scene.dump_gpu();
  }

  if (!kernel::intersect(scene, r, dist, hit, { .enabled = x == 860 && y == 1000, .x = x, .y = y })) {
    img.set_pixel(x, y, {0, 0, 0}); // black
  }
  else {
    switch (m) {
    case kernel::mode::ID:
      img.set_pixel(x, y, color_to_id[hit % 32]);
      break;
    case kernel::mode::DIST:
      img.set_pixel(x, y, {1.0f / dist, 1.0f / dist, 1.0f / dist });
      break;
    case kernel::mode::COLOR:
      break;
    }
  }
}

__host__ void kernel::render(const scene_gpu& scene, image &img, const mode m) {
  constexpr dim3 block(16, 16);
  const dim3 grid((img.width) / block.x + 1, (img.height) / block.y + 1);
  cuCheckAsync((renderer<<<grid, block>>>(scene, img, m)));
  cuCheck(cudaDeviceSynchronize());
}

__device__ bool kernel::intersect(const scene_gpu& scene, const ray& ray, float& dist, size_t& obj_id, const do_log &log) {
  dist = INFINITY;
  if (log.enabled) {
    printf("[%03d %03d] Ray is { .start = (%f, %f, %f), .direction = (%f, %f, %f) }\n",
      log.x, log.y, ray.start.x, ray.start.y, ray.start.z, ray.direction.x, ray.direction.y, ray.direction.z
    );
  }
  for (size_t i = 0; i < scene.num_planes; i++) {
    float t;
    const bool hit = scene.planes[i].intersect(ray, t);
    if (log.enabled) {
      printf("[%03d %03d] Plane %ld [%ld]: hit? %d, t = %f\n\t-> Plane is { .point = (%f, %f, %f), .normal = (%f, %f, %f) }\n",
        log.x, log.y, i, scene.planes[i].id, hit, t,
        scene.planes[i].point.x, scene.planes[i].point.y, scene.planes[i].point.z,
        scene.planes[i].normal.x, scene.planes[i].normal.y, scene.planes[i].normal.z
      );
    }
    if (scene.planes[i].intersect(ray, t) && t > 1e-6f && t < dist) {
      dist = t;
      obj_id = scene.planes[i].id;
    }
  }

  for (size_t i = 0; i < scene.num_spheres; i++) {
    float t;
    const bool hit = scene.spheres[i].intersect(ray, t);
    if (log.enabled) {
      printf("[%03d %03d] Sphere %ld [%ld]: hit? %d, t = %f\n\t-> Sphere is { .center = (%f, %f, %f), .radius = %f }\n",
        log.x, log.y, i, scene.spheres[i].id, hit, t,
        scene.spheres[i].center.x, scene.spheres[i].center.y, scene.spheres[i].center.z, scene.spheres[i].radius
      );
    }
    if (scene.spheres[i].intersect(ray, t) && t > 1e-6f && t < dist) {
      dist = t;
      obj_id = scene.spheres[i].id;
    }
  }

  for (size_t i = 0; i < scene.num_triangles; i++) {
    float t;
    const bool hit = scene.triangles[i].intersect(ray, t);
    if (log.enabled) {
      printf("[%03d %03d] Triangle %ld [%ld]: hit? %d, t = %f\n\t-> Triangle is { .v0 = (%f, %f, %f), .v1 = (%f, %f, %f), .v2 = (%f, %f, %f) }\n",
        log.x, log.y, i, scene.triangles[i].id, hit, t,
        scene.triangles[i].v0.x, scene.triangles[i].v0.y, scene.triangles[i].v0.z,
        scene.triangles[i].v1.x, scene.triangles[i].v1.y, scene.triangles[i].v1.z,
        scene.triangles[i].v2.x, scene.triangles[i].v2.y, scene.triangles[i].v2.z
      );
    }
    if (scene.triangles[i].intersect(ray, t) && t > 1e-6f && t < dist) {
      dist = t;
      obj_id = scene.triangles[i].id;
    }
  }

  return dist != INFINITY;
}

