#include <immintrin.h>
#include "cuda_wrapper.hpp"
#include "framework.hpp"

template <bool row_major>
struct matrix
{
  float *data;
  size_t n;

  [[nodiscard]] __host__ __device__ constexpr float *at(const size_t row, const size_t col) const {
    if constexpr(row_major) return data + row * n + col;
    else return data + col * n + row;
  }
};

__host__ void kernel_cpu_chunked(const matrix<true> a, const matrix<false> b, matrix<true> c, const size_t n, const size_t start, const size_t end) {
  for (size_t i = start; i < end; i++) {
    for (size_t j = 0; j < n; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < n; k++) {
        sum += *a.at(i, k) * *b.at(k, j);
      }
      *c.at(i, j) = sum;
    }
  }
}
__host__ void kernel_cpu(const matrix<true> &a, const matrix<false> &b, matrix<true> &c, const size_t n) {
  kernel_cpu_chunked(a, b, c, n, 0, n);
}

__host__ void kernel_avx_chunked(const matrix<true> a, const matrix<false> b, matrix<true> c, const size_t n, const size_t start, const size_t end) {
  for (size_t i = start; i < end; i++) {
    for (size_t j = 0; j < n; j++) {
      float sum = 0.0f;
      size_t k = 0;
      for (; k + 16 <= n; k += 16) {
        const auto as = _mm512_loadu_ps(a.at(i, k));
        const auto bs = _mm512_loadu_ps(b.at(k, j));
        const auto mul = _mm512_mul_ps(as, bs);
        sum += _mm512_reduce_add_ps(mul);
      }
      for (; k < n; k++) {
        sum += *a.at(i, k) * *b.at(k, j);
      }
      *c.at(i, j) = sum;
    }
  }
}
__host__ void kernel_avx(const matrix<true> &a, const matrix<false> &b, matrix<true> &c, const size_t n) {
  kernel_avx_chunked(a, b, c, n, 0, n);
}

template <std::invocable<const matrix<true> &, const matrix<false> &, matrix<true> &, size_t, size_t, size_t> auto F>
__host__ void kernel_mt(const matrix<true> &a, const matrix<false> &b, matrix<true> &c, const size_t n) {
  std::vector<std::thread> threads;
  const size_t num_threads = std::thread::hardware_concurrency();
  const size_t chunk_size = n / num_threads;
  threads.reserve(num_threads);
  for (size_t i = 0; i < num_threads - 1; i++) {
  // for (; start + chunk_size < n; start += chunk_size) {
    threads.emplace_back(F, a, b, c, n, i * chunk_size, (i + 1) + chunk_size);
  }
  F(a, b, c, n, (num_threads - 1) * chunk_size, n);

  for (auto &t : threads) t.join();
}

__global__ void gpu_entry(const matrix<true> a, const matrix<false> b, matrix<true> c, const size_t n) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= n || y >= n) return;

  float sum = 0.0f;
  for (size_t k = 0; k < n; k++) {
    sum += *a.at(x, k) * *b.at(k, y);
  }
  *c.at(x, y) = sum;
}

__host__ void kernel_gpu(const matrix<true> &a, const matrix<false> &b, matrix<true> &c, const size_t n) {
  dim3 block(32, 32);
  const size_t blocks = n / 32 + 1;
  dim3 grid(blocks, blocks);
  cuCheckAsync((gpu_entry<<<grid, block>>>(a, b, c, n)));
  cuCheck(cudaDeviceSynchronize());
}

static std::mt19937 rng{std::random_device{}()};

int main() {
  std::vector<record> records;
  for (size_t shift = 4; shift <= 14; shift++) { // can go up to 23?
    const size_t in = 1ul << shift;
    auto a = matrix<true> { .data = new float[in * in], .n = in };
    auto b = matrix<false> { .data = new float[in * in], .n = in };
    auto c = matrix<true> { .data = new float[in * in], .n = in };
    matrix<true> a_um{ .data = nullptr, .n = in };
    matrix<false> b_um{ .data = nullptr, .n = in };
    matrix<true> c_um{ .data = nullptr, .n = in };
    std::cout << "  -> Input size: " << in << " (2^" << shift << ")\n";
    std::cout << "  -> Requiring " << 3 * static_cast<float>(in * in * sizeof(float)) / (1024.0f * 1024) << " MB of GPU VRAM\n";
    cuCheck(cudaMallocManaged(&a_um.data, sizeof(float) * in * in));
    cuCheck(cudaMallocManaged(&b_um.data, sizeof(float) * in * in));
    cuCheck(cudaMallocManaged(&c_um.data, sizeof(float) * in * in));

    for (size_t i = 0; i < in * in; i++) {
      const float _1 = rng() / 1000.0f;
      const float _2 = rng() / 1000.0f;
      a.data[i] = _1; b.data[i] = _2;
      a_um.data[i] = _1; b_um.data[i] = _2;
    }

    const auto rec = run_benchmark<const matrix<true> &, const matrix<false> &, matrix<true> &, size_t>(in,
      kernel_cpu, kernel_avx, kernel_mt<kernel_cpu_chunked>, kernel_mt<kernel_avx_chunked>, kernel_gpu,
      a, b, c, in, a_um, b_um, c_um, in
    );
    records.push_back(rec);
  }

  save_results_to(DATA_DIR "/results_0b.csv", records);
}
