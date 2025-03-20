cmake_minimum_required(VERSION 3.28)

add_executable(3_game_of_life main.cu kernel.cu)
target_link_libraries(3_game_of_life Curses::Curses)