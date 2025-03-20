//
// Created by jay on 3/9/25.
//

#ifndef CUDA_WRAPPER_HPP
#define CUDA_WRAPPER_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

inline void log_cuda_error(const cudaError_t err, const char *file, const int line) {
  std::cerr << "CUDA error at " << file << ":" << line << ":  " << cudaGetErrorName(err) << " (" << err << ") - " << cudaGetErrorString(err) << "\n";
  std::exit(-1);
}

#define cuCheck(e) do { const auto _ = (e); if(_ != cudaSuccess) { log_cuda_error(_, __FILE__, __LINE__); } } while(0)
#define cuCheckAsync(e) do { (e); cuCheck(cudaGetLastError()); } while(0)

#endif //CUDA_WRAPPER_HPP
