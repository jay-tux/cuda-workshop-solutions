//
// Created by jay on 3/18/25.
//

#ifndef LOADER_HPP
#define LOADER_HPP

#include <string>

#include "scene.hpp"

struct loader
{
  static scene_cpu load(const std::string& filename, bool &all_okay);
};

#endif //LOADER_HPP
