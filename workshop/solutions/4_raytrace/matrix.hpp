//
// Created by jay on 3/19/25.
//

#ifndef MATRIX_HPP
#define MATRIX_HPP

struct matrix
{
  float3 col0, col1, col2;

  __device__ inline float determinant() const {
    const float a = col0.x, b = col1.x, c = col2.x,
                d = col0.y, e = col1.y, f = col2.y,
                g = col0.z, h = col1.z, i = col2.z;

    return a*e*i + b*f*g + c*d*h - c*e*g - a*f*h - b*d*i;
  }
};

#endif //MATRIX_HPP
