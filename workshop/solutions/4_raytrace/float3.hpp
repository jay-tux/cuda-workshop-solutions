//
// Created by jay on 3/18/25.
//

#ifndef FLOAT3_HPP
#define FLOAT3_HPP

__host__ __device__ inline float3 normalize(const float3 &p) {
  float denom = sqrtf(p.x * p.x + p.y * p.y + p.z * p.z);
  return make_float3(p.x / denom, p.y / denom, p.z / denom);
}

__host__ __device__ inline float3 cross(const float3 &a, const float3 &b) {
  return make_float3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}

__host__ __device__ inline float3 operator+(const float3 a, const float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3 a, const float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float a, const float3 b) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ inline float3 operator/(const float3 a, const float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ inline float3 operator-(const float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ constexpr float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

#endif //FLOAT3_HPP
