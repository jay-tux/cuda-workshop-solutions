//
// Created by jay on 3/18/25.
//

#include <iostream>

#include "scene.hpp"
#include "float3.hpp"
#include "cuda_wrapper.hpp"
#include "matrix.hpp"

void scene_cpu::setup_cam() {
  cam_forward = normalize(cam_forward);
  cam_up = normalize(cam_up);
  cam_right = cross(cam_forward, cam_up);
  cam_up = cross(cam_right, cam_forward);
}

__device__ bool plane::intersect(const ray &ray, float &t) const {
  const float n_dot_d = dot(normal, ray.direction);
  if (abs(n_dot_d) < 1e-6f) return false; // no intersection, parallel to plane
  t = dot(normal, point - ray.start) / n_dot_d;
  return t >= 0.0f;
}

__device__ bool sphere::intersect(const ray &ray, float &t) const {
  const auto d = normalize(ray.direction), c = center, e = ray.start;
  const float R = radius;

  float dec = dot(-d, e - c);
  float sub = dec * dec - dot(d, d) * (dot(e - c, e - c) - R * R);

  float t0 = (dec - sqrt(sub)) / dot(d, d), t1 = (dec + sqrt(sub)) / dot(d, d);
  bool t0v = isfinite(t0) && 1e-6f <= t0, t1v = isfinite(t1) && 1e-6f <= t1;
  int condition = (t0v ? 2 : 0) + (t1v ? 1 : 0);

  switch(condition) {
  case 0: return false;
  case 1: t = t1; break;
  case 2: t = t0; break;
  case 3: t = min(t0, t1); break;
  default: break; // impossible
  }

  return true;
}

__device__ bool triangle::intersect(const ray &ray, float &t) const {
  const auto a = v1 - v0, b = v1 - v2, c = ray.direction, d = v1 - ray.start;
  const matrix A{a, b, d}, B{a, b, c}, A1{d, b, c}, A2{a, d, c};

  const float alpha = B.determinant();
  const float beta = A1.determinant() / alpha;
  const float gamma = A2.determinant() / alpha;
  t = A.determinant() / alpha;

  return beta >= 0 && gamma >= 0 && beta + gamma <= 1 && isfinite(t) && 1e-6f <= t;
}

__host__ void scene_gpu::cleanup() {
  cuCheck(cudaFree(planes)); planes = nullptr; num_planes = 0;
  cuCheck(cudaFree(spheres)); spheres = nullptr; num_spheres = 0;
  cuCheck(cudaFree(triangles)); triangles = nullptr; num_triangles = 0;
}

__host__ scene_gpu scene_cpu::to_gpu() const {
  scene_gpu gpu {
    .planes = nullptr, .num_planes = planes.size(),
    .spheres = nullptr, .num_spheres = spheres.size(),
    .triangles = nullptr, .num_triangles = triangles.size(),
    .camera = camera,
    .cam_forward = cam_forward, .cam_right = cam_right, .cam_up = cam_up
  };

  cuCheck(cudaMallocManaged(&gpu.planes, sizeof(plane) * gpu.num_planes));
  cuCheck(cudaMallocManaged(&gpu.spheres, sizeof(sphere) * gpu.num_spheres));
  cuCheck(cudaMallocManaged(&gpu.triangles, sizeof(triangle) * gpu.num_triangles));

  cuCheck(cudaMemcpy(gpu.planes, planes.data(), sizeof(plane) * gpu.num_planes, cudaMemcpyDefault));
  cuCheck(cudaMemcpy(gpu.spheres, spheres.data(), sizeof(sphere) * gpu.num_spheres, cudaMemcpyDefault));
  cuCheck(cudaMemcpy(gpu.triangles, triangles.data(), sizeof(triangle) * gpu.num_triangles, cudaMemcpyDefault));

  return gpu;
}

std::ostream &operator<<(std::ostream &os, const float3 &v) {
  return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}

__host__ void scene_cpu::dump() const {
  std::cout << " === Scene (CPU) ===\n"
            << "Camera data:\n"
            << "  -> Position: " << camera << "\n"
            << "  -> Forward: " << cam_forward << "\n"
            << "  -> Up: " << cam_up << "\n"
            << "  -> Right: " << cam_right << "\n"
            << "\n"
            << "  -> Planes (" << planes.size() << "):\n";
  for (size_t i = 0; i < planes.size(); i++) {
    const auto & [point, normal, id] = planes[i];
    std::cout << "    -> Plane #" << i << " [ID " << id << "]:\n"
              << "      -> Point: " << point << "\n"
              << "      -> Normal: " << normal << "\n";
  }
  std::cout << "\n"
            << "  -> Spheres (" << spheres.size() << "):\n";
  for (size_t i = 0; i < spheres.size(); i++) {
    const auto & [center, radius, id] = spheres[i];
    std::cout << "    -> Sphere #" << i << " [ID " << id << "]:\n"
              << "      -> Center: " << center << "\n"
              << "      -> Radius: " << radius << "\n";
  }
  std::cout << "\n"
              << "  -> Triangles (" << triangles.size() << "):\n";
  for (size_t i = 0; i < triangles.size(); i++) {
    const auto & [v0, v1, v2, id] = triangles[i];
    std::cout << "    -> Triangle #" << i << " [ID " << id << "]:\n"
              << "      -> v0: " << v0 << "\n"
              << "      -> v1: " << v1 << "\n"
              << "      -> v2: " << v2 << "\n";
  }
  std::cout << "=== End Scene ===\n";
}

__host__ void scene_gpu::dump() const {
  std::cout << " === Scene (GPU; host) ===\n"
            << "Camera data:\n"
            << "  -> Position: " << camera << "\n"
            << "  -> Forward: " << cam_forward << "\n"
            << "  -> Up: " << cam_up << "\n"
            << "  -> Right: " << cam_right << "\n"
            << "\n"
            << "  -> Planes (" << num_planes << "):\n";
  for (size_t i = 0; i < num_planes; i++) {
    const auto & [point, normal, id] = planes[i];
    std::cout << "    -> Plane #" << i << " [ID " << id << "]:\n"
              << "      -> Point: " << point << "\n"
              << "      -> Normal: " << normal << "\n";
  }
  std::cout << "\n"
            << "  -> Spheres (" << num_spheres << "):\n";
  for (size_t i = 0; i < num_spheres; i++) {
    const auto & [center, radius, id] = spheres[i];
    std::cout << "    -> Sphere #" << i << " [ID " << id << "]:\n"
              << "      -> Center: " << center << "\n"
              << "      -> Radius: " << radius << "\n";
  }
  std::cout << "\n"
              << "  -> Triangles (" << num_triangles << "):\n";
  for (size_t i = 0; i < num_triangles; i++) {
    const auto & [v0, v1, v2, id] = triangles[i];
    std::cout << "    -> Triangle #" << i << " [ID " << id << "]:\n"
              << "      -> v0: " << v0 << "\n"
              << "      -> v1: " << v1 << "\n"
              << "      -> v2: " << v2 << "\n";
  }
  std::cout << "=== End Scene ===\n";
}

__device__ void scene_gpu::dump_gpu() const {
  printf(" === Scene (GPU; device) ===\n"
            "Camera data:\n"
            "  -> Position: (%f, %f, %f)\n"
            "  -> Forward: (%f, %f, %f)\n"
            "  -> Up: (%f, %f, %f)\n"
            "  -> Right: (%f, %f, %f)\n"
            "\n"
            "  -> Planes (%ld):\n",
            camera.x, camera.y, camera.z,
            cam_forward.x, cam_forward.y, cam_forward.z,
            cam_up.x, cam_up.y, cam_up.z,
            cam_right.x, cam_right.y, cam_right.z,
            num_planes
  );
  for (size_t i = 0; i < num_planes; i++) {
    const auto & [point, normal, id] = planes[i];
    printf("    -> Plane #%ld [ID %ld]:\n"
              "      -> Point: (%f, %f, %f)\n"
              "      -> Normal: (%f, %f, %f)\n",
              i, id,
              point.x, point.y, point.z,
              normal.x, normal.y, normal.z
    );
  }

  printf("\n  -> Spheres (%ld):\n", num_spheres);
  for (size_t i = 0; i < num_spheres; i++) {
    const auto & [center, radius, id] = spheres[i];
    printf("    -> Sphere #%ld [ID %ld]:\n"
              "      -> Center: (%f, %f, %f)\n"
              "      -> Radius: %f\n",
              i, id,
              center.x, center.y, center.z,
              radius
    );
  }

  printf("\n  -> Triangles (%ld):\n", num_triangles);
  for (size_t i = 0; i < num_triangles; i++) {
    const auto & [v0, v1, v2, id] = triangles[i];
    printf("    -> Triangle #%ld [ID %ld]:\n"
              "      -> v0: (%f, %f, %f)\n"
              "      -> v1: (%f, %f, %f)\n"
              "      -> v2: (%f, %f, %f)\n",
              i, id,
              v0.x, v0.y, v0.z,
              v1.x, v1.y, v1.z,
              v2.x, v2.y, v2.z
    );
  }
  printf("=== End Scene ===\n");
}