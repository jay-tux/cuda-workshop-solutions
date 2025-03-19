//
// Created by jay on 3/18/25.
//

#include <iostream>
#include <inipp.h>
#include <fstream>
#include <ranges>
#include <sstream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "loader.hpp"

float3 parse_vec3(const std::string &str) {
  float x, y, z;
  std::stringstream ss(str);
  ss >> x >> y >> z;
  return make_float3(x, y, z);
}

void parse_sphere(const std::string &name, const inipp::Ini<char>::Section &sec, const size_t id, std::vector<sphere> &out) {
  const auto center = sec.find("center");
  if (center == sec.end()) {
    std::cerr << "Sphere " << name << " has no center\n";
    return;
  }

  const auto radius = sec.find("radius");
  if (radius == sec.end()) {
    std::cerr << "Sphere " << name << " has no radius\n";
    return;
  }

  out.push_back({ .center = parse_vec3(center->second), .radius = std::stof(radius->second), .id = id });
}

void parse_plane(const std::string &name, const inipp::Ini<char>::Section &sec, const size_t id, std::vector<plane> &out) {
  const auto normal = sec.find("normal");
  if (normal == sec.end()) {
    std::cerr << "Plane " << name << " has no normal\n";
    return;
  }

  const auto point = sec.find("point");
  if (point == sec.end()) {
    std::cerr << "Plane " << name << " has no point\n";
    return;
  }
  out.push_back({ .point = parse_vec3(point->second), .normal = parse_vec3(normal->second), .id = id });
}

void parse_triangle(const std::string &name, const inipp::Ini<char>::Section &sec, const size_t id, std::vector<triangle> &out) {
  const auto a = sec.find("a");
  if (a == sec.end()) {
    std::cerr << "Triangle " << name << " has no a\n";
    return;
  }

  const auto b = sec.find("b");
  if (b == sec.end()) {
    std::cerr << "Triangle " << name << " has no b\n";
    return;
  }

  const auto c = sec.find("c");
  if (c == sec.end()) {
    std::cerr << "Triangle " << name << " has no c\n";
    return;
  }

  out.push_back({ .v0 = parse_vec3(a->second), .v1 = parse_vec3(b->second), .v2 = parse_vec3(c->second), .id = id });
}

triangle from_aiFace(const aiMesh *mesh, const aiFace &face, const size_t id) {
  return {
    .v0 = make_float3(mesh->mVertices[face.mIndices[0]].x, mesh->mVertices[face.mIndices[0]].y, mesh->mVertices[face.mIndices[0]].z),
    .v1 = make_float3(mesh->mVertices[face.mIndices[1]].x, mesh->mVertices[face.mIndices[1]].y, mesh->mVertices[face.mIndices[1]].z),
    .v2 = make_float3(mesh->mVertices[face.mIndices[2]].x, mesh->mVertices[face.mIndices[2]].y, mesh->mVertices[face.mIndices[2]].z),
    .id = id
  };
}

void parse_mesh(const std::string &name, const inipp::Ini<char>::Section &sec, const size_t id, std::vector<triangle> &out) {
  const auto filename = sec.find("filename");
  if (filename == sec.end()) {
    std::cerr << "Mesh " << name << " has no filename\n";
    return;
  }

  Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(filename->second, aiProcess_Triangulate | aiProcess_GenNormals);
  if (!scene) {
    std::cerr << "Failed to load mesh " << filename->second << "\n";
    return;
  }

  for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
    const auto *mesh = scene->mMeshes[i];
    for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
      const auto f = mesh->mFaces[j];
      if (f.mNumIndices != 3) {
        std::cerr << "Face " << j << " in mesh " << i << " has " << f.mNumIndices << " vertices\n";
        continue;
      }

      out.push_back(from_aiFace(mesh, f, id));
    }
  }
}

scene_cpu loader::load(const std::string& filename) {
  std::ifstream strm(filename);
  if (!strm) {
    std::cerr << "Failed to open file " << filename << "\n";
    exit(1);
  }

  inipp::Ini<char> ini;
  ini.parse(strm);

  scene_cpu scene;
  size_t id = 0;
  for (const auto &[n, section] : ini.sections) {
    if (n == "camera") {
      auto it = section.find("position");
      if (it == section.end()) {
        std::cerr << "Camera has no position\n";
        continue;
      }
      scene.camera = parse_vec3(it->second);
      it = section.find("forward");
      if (it == section.end()) {
        std::cerr << "Camera has no forward\n";
        continue;
      }
      scene.cam_forward = parse_vec3(it->second);
      it = section.find("up");
      scene.cam_up = (it == section.end()) ? make_float3(0, 1, 0) : parse_vec3(it->second);

      scene.setup_cam();
    }
    else {
      auto it = section.find("type");
      if (it == section.end()) {
        std::cerr << "Section " << n << " has no type\n";
        continue;
      }

      size_t id_override = id;
      if (const auto id_it = section.find("id"); id_it != section.end()) {
        id_override = std::stoul(id_it->second);
      }

      std::cout << "Object " << n << " (" << it->second << ") has ID " << id_override << "\n";

      if (it->second == "sphere") parse_sphere(n, section, id_override, scene.spheres);
      else if (it->second == "plane") parse_plane(n, section, id_override, scene.planes);
      else if (it->second == "triangle") parse_triangle(n, section, id_override, scene.triangles);
      else if (it->second == "mesh") parse_mesh(n, section, id_override, scene.triangles);
      else {
        std::cerr << "Unknown type " << it->second << " (for section " << n << ")\n";
      }
    }

    ++id;
  }

  return scene;
}
