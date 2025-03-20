//
// Created by jay on 3/15/25.
//

#include <clocale>
#include <curses.h>
#include <string>
#include <fstream>
#include <sstream>
#include "cuda_wrapper.hpp"
#include "kernel.hpp"

void render(const char *buf, const int w, const int h) {
  for (int y = 0; y < h; y++) {
    mvaddnstr(y, 0, buf + (y * w), w);
  }
}

void read_input(const std::string &file, c2_game_of_life::buffer buf) {
  std::ifstream f(file);
  if (!f.is_open()) {
    endwin();
    std::cerr << "Could not open file " << file << "\n";
    exit(1);
  }

  for (int y = 0; y < buf.height; y++) {
    for (int x = 0; x < buf.width; x++) {
      buf.set_dead(x, y);
    }
  }

  std::string line;
  while (!f.eof()) {
    std::getline(f, line);
    std::stringstream ss(line);
    int x, y;
    ss >> x >> y;
    buf.set_alive(x, y);
  }
}

int main(int argc, const char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <input file>\n";
    return 0;
  }

  setlocale(LC_ALL, "");
  initscr(); cbreak(); noecho();

  int x, y;
  getmaxyx(stdscr, y, x);

  char *buf0, *buf1;
  cuCheck(cudaMallocManaged(&buf0, x * y * sizeof(char)));
  cuCheck(cudaMallocManaged(&buf1, x * y * sizeof(char)));
  read_input(argv[1], {.data = buf0, .width = x, .height = y});

  render(buf0, x, y);
  char last_input = getch();
  while (last_input != 'q') {
    step(
      c2_game_of_life::buffer{ .data = buf0, .width = x, .height = y },
      c2_game_of_life::buffer{ .data = buf1, .width = x, .height = y }
    );

    std::swap(buf0, buf1);
    render(buf0, x, y);

    last_input = getch();
  }

  cuCheck(cudaFree(buf0));
  cuCheck(cudaFree(buf1));
  endwin();
}
