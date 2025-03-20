//
// Created by jay on 3/18/25.
//

#include "kernel.hpp"
#include "float3.hpp"

#include <cuda_wrapper.hpp>

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

template <uint8_t bounces_left>
__device__ float3 colorizer(const scene_gpu &scene, const ray &r) {
  float dist{};
  size_t hit{};
  float3 normal{};
  material mat{};
  float3 color = {0, 0, 0}; // black

  const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  const bool do_log = x == 40 && y == 450;

  bool is_hit = kernel::intersect(scene, r, dist, hit, normal, mat);

  if (kernel::intersect(scene, r, dist, hit, normal, mat)) {
    const float3 hit_at = r.start + dist * r.direction;
    const float3 view_dir = -normalize(r.direction);
    for (size_t p_id = 0; p_id < scene.num_points; p_id++) {
      color = color + scene.points[p_id].shade(mat, hit_at, normal, view_dir);
    }

    if constexpr(bounces_left > 0) {
      if (mat.reflect_factor > epsilon) {
        const float3 reflect_dir = reflect(r.direction, normal);
        const float3 new_origin = hit_at + epsilon * reflect_dir;
        const ray new_ray{new_origin, reflect_dir};
        const float3 reflect_color = colorizer<bounces_left - 1>(scene, new_ray);

        color = lerp(color, reflect_color, mat.reflect_factor);
      }

      if (mat.transparency > epsilon) {
        const float3 new_origin = hit_at + epsilon * r.direction;
        const ray new_ray{new_origin, r.direction};
        const float3 passthrough_color = colorizer<bounces_left - 1>(scene, new_ray);

        color = lerp(color, passthrough_color, mat.transparency);
      }
    }
  }

  if (do_log) {
    printf("Ray {(%f, %f, %f)->(%f, %f, %f)}: is_hit=%d, hit=%lu, dist=%f => output color = (%f, %f, %f)\n",
      r.start.x, r.start.y, r.start.z, r.direction.x, r.direction.y, r.direction.z, is_hit, hit, dist,
      color.x, color.y, color.z
    );
  }

  return color;
}

__global__ void dump(const scene_gpu scene) { scene.dump_gpu(); }

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

  float dist = INFINITY;
  size_t hit{};
  float3 _n{};
  material _m{};

  if (m == kernel::mode::ID || m == kernel::mode::DIST) {
    if (kernel::intersect(scene, r, dist, hit, _n, _m)) {
      if (m == kernel::mode::ID) {
        img.set_pixel(x, y, color_to_id[hit % 32]);
      }
      else { // mode::DIST
        img.set_pixel(x, y, {1.0f / dist, 1.0f / dist, 1.0f / dist });
      }
    }
    else {
      img.set_pixel(x, y, {0, 0, 0}); // black
    }
  }
  else { // mode::COLOR
    img.set_pixel(x, y, colorizer<bounces>(scene, r));
  }
}

__host__ void kernel::render(const scene_gpu& scene, image &img, const mode m) {
  constexpr dim3 block(16, 16);
  const dim3 grid((img.width) / block.x + 1, (img.height) / block.y + 1);
  cuCheckAsync((dump<<<1, 1>>>(scene)));
  cuCheckAsync((renderer<<<grid, block>>>(scene, img, m)));
  cuCheck(cudaDeviceSynchronize());
}

__device__ bool kernel::intersect(const scene_gpu& scene, const ray& ray, float& dist, size_t& obj_id, float3 &normal, material& mat) {
  dist = INFINITY;

  for (size_t i = 0; i < scene.num_planes; i++) {
    float t; float3 n;
    if (scene.planes[i].intersect(ray, t, n) && t > epsilon && t < dist) {
      dist = t;
      obj_id = scene.planes[i].id;
      normal = n;
      mat = scene.materials[scene.planes[i].material_idx];
    }
  }

  for (size_t i = 0; i < scene.num_spheres; i++) {
    float t; float3 n;
    if (scene.spheres[i].intersect(ray, t, n) && t > epsilon && t < dist) {
      dist = t;
      obj_id = scene.spheres[i].id;
      normal = n;
      mat = scene.materials[scene.spheres[i].material_idx];
    }
  }

  for (size_t i = 0; i < scene.num_triangles; i++) {
    float t; float3 n;
    if (scene.triangles[i].intersect(ray, t, n) && t > epsilon && t < dist) {
      dist = t;
      obj_id = scene.triangles[i].id;
      normal = n;
      mat = scene.materials[scene.triangles[i].material_idx];
    }
  }

  return dist != INFINITY;
}

