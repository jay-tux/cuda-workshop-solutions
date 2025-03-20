//
// Created by jay on 3/15/25.
//

#ifndef KERNEL_HPP
#define KERNEL_HPP

namespace c2_game_of_life {
    struct buffer
    {
        char *data;
        int width, height;

        __host__ __device__ constexpr char *at(const int x, const int y) const {
            return data + ((y + height) % height) * width + ((x + width) % width);
        }

        __host__ __device__ constexpr bool is_live(const int x, const int y) const {
            return *at(x, y) == 'X';
        }

        __host__ __device__ void set_alive(const int x, const int y) {
            *at(x, y) = 'X';
        }
        __host__ __device__ void set_dead(const int x, const int y) {
            *at(x, y) = ' ';
        }
    };

    __host__ void step(const buffer &in, const buffer &out);
}

#endif //KERNEL_HPP
