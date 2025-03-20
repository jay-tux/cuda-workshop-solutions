#include <iostream>
#include <random>
#include <cstdio>
#include "kernel.hpp"

void print_matrix(const std::string &name, const matrix &m) {
  std::printf(" -- Matrix %s -- \n", name.c_str());
  for (unsigned col = 0; col < m.n; col++) {
    for (unsigned row = 0; row < m.n; row++) {
      std::printf("%+4d  ", m.at(col, row));
    }
    std::printf("\n");
  }
}

int main(const int argc, const char **argv) {
  if (argc != 3 || argv[1] == std::string("-h")) {
    std::cout << "Usage: " << argv[0] << " <size> <seed>\n";
    return 0;
  }

  const unsigned n = std::stoul(argv[1]);
  const size_t seed = std::stoull(argv[2]);
  std::mt19937 rng(seed);
  std::uniform_int_distribution dist(-10, 10);

  matrix a {
    .data = new int[n * n],
    .n = n
  };
  matrix b {
    .data = new int[n * n],
    .n = n
  };

  for (int i = 0; i < n * n; i++) {
    a.data[i] = dist(rng);
    b.data[i] = dist(rng);
  }

  const matrix c = kernel::full_program(a, b);

  print_matrix("A", a); std::cout << "\n";
  print_matrix("B", b); std::cout << "\n";
  print_matrix("C", c);

  delete [] a.data;
  delete [] b.data;
  delete [] c.data;
}
