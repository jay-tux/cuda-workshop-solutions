//
// Created by jay on 3/18/25.
//

#include <iostream>
#include <inipp.h>
#include <fstream>
#include <ranges>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <filesystem>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "loader.hpp"

#include "float3.hpp"

float3 parse_vec3(const std::string &str) {
  float x, y, z;
  std::stringstream ss(str);
  ss >> x >> y >> z;
  return make_float3(x, y, z);
}

bool parse_sphere(const std::string &name, const inipp::Ini<char>::Section &sec, const size_t id, size_t &mat_id, std::vector<sphere> &out) {
  const auto center = sec.find("center");
  if (center == sec.end()) {
    std::cerr << "Sphere " << name << " has no center\n";
    return false;
  }

  const auto radius = sec.find("radius");
  if (radius == sec.end()) {
    std::cerr << "Sphere " << name << " has no radius\n";
    return false;
  }

  const auto material = sec.find("material");
  if (material == sec.end()) {
    std::cerr << "Sphere " << name << " has no material\n";
    return false;
  }

  mat_id = std::stoull(material->second);

  out.push_back({ .center = parse_vec3(center->second), .radius = std::stof(radius->second), .id = id, .material_idx = mat_id });
  return true;
}

bool parse_plane(const std::string &name, const inipp::Ini<char>::Section &sec, const size_t id, size_t &mat_id, std::vector<plane> &out) {
  const auto normal = sec.find("normal");
  if (normal == sec.end()) {
    std::cerr << "Plane " << name << " has no normal\n";
    return false;
  }

  const auto point = sec.find("point");
  if (point == sec.end()) {
    std::cerr << "Plane " << name << " has no point\n";
    return false;
  }

  const auto material = sec.find("material");
  if (material == sec.end()) {
    std::cerr << "Sphere " << name << " has no material\n";
    return false;
  }

  mat_id = std::stoull(material->second);

  out.push_back({ .point = parse_vec3(point->second), .normal = parse_vec3(normal->second), .id = id, .material_idx = mat_id });
  return true;
}

bool parse_triangle(const std::string &name, const inipp::Ini<char>::Section &sec, const size_t id, size_t &mat_id, std::vector<triangle> &out) {
  const auto a = sec.find("a");
  if (a == sec.end()) {
    std::cerr << "Triangle " << name << " has no a\n";
    return false;
  }

  const auto b = sec.find("b");
  if (b == sec.end()) {
    std::cerr << "Triangle " << name << " has no b\n";
    return false;
  }

  const auto c = sec.find("c");
  if (c == sec.end()) {
    std::cerr << "Triangle " << name << " has no c\n";
    return false;
  }

  const auto normal = sec.find("normal");
  if (normal == sec.end()) {
    std::cerr << "Triangle " << name << " has no normal\n";
    return false;
  }

  const auto material = sec.find("material");
  if (material == sec.end()) {
    std::cerr << "Sphere " << name << " has no material\n";
    return false;
  }

  mat_id = std::stoull(material->second);

  out.push_back({
    .v0 = parse_vec3(a->second), .v1 = parse_vec3(b->second), .v2 = parse_vec3(c->second),
    .normal = normalize(parse_vec3(normal->second)), .id = id, .material_idx = mat_id
  });
  return true;
}

triangle from_aiFace(const aiMesh *mesh, const aiFace &face, const size_t id, const size_t mat_id) {
  return {
    .v0 = make_float3(mesh->mVertices[face.mIndices[0]].x, mesh->mVertices[face.mIndices[0]].y, mesh->mVertices[face.mIndices[0]].z),
    .v1 = make_float3(mesh->mVertices[face.mIndices[1]].x, mesh->mVertices[face.mIndices[1]].y, mesh->mVertices[face.mIndices[1]].z),
    .v2 = make_float3(mesh->mVertices[face.mIndices[2]].x, mesh->mVertices[face.mIndices[2]].y, mesh->mVertices[face.mIndices[2]].z),
    .normal = make_float3(mesh->mNormals[face.mIndices[0]].x, mesh->mNormals[face.mIndices[0]].y, mesh->mNormals[face.mIndices[0]].z),
    .id = id,
    .material_idx = mat_id
  };
}

bool parse_mesh(const std::string &infile, const std::string &name, const inipp::Ini<char>::Section &sec, const size_t id, size_t &mat_id, std::vector<triangle> &out) {
  const auto filename = sec.find("filename");
  if (filename == sec.end()) {
    std::cerr << "Mesh " << name << " has no filename\n";
    return false;
  }

  const auto material = sec.find("material");
  if (material == sec.end()) {
    std::cerr << "Sphere " << name << " has no material\n";
    return false;
  }

  mat_id = std::stoull(material->second);

  Assimp::Importer importer;

  const auto path = (std::filesystem::absolute(infile).parent_path() / filename->second).string();

  const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenNormals);
  if (!scene) {
    std::cerr << "Failed to load mesh " << filename->second << " (attempted: " << path << ")\n";
    return false;
  }

  for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
    const auto *mesh = scene->mMeshes[i];
    for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
      const auto f = mesh->mFaces[j];
      if (f.mNumIndices != 3) {
        std::cerr << "Face " << j << " in mesh " << i << " has " << f.mNumIndices << " vertices\n";
        continue;
      }

      out.push_back(from_aiFace(mesh, f, id, mat_id));
    }
  }
  return true;
}

bool parse_mat(const std::string &name, const inipp::Ini<char>::Section &sec, size_t &id_out, std::vector<material> &out) {
  const auto id = sec.find("id");
  const auto color = sec.find("color");
  const auto phong = sec.find("phong");
  const auto reflect = sec.find("reflect");
  const auto transparency = sec.find("transparency");
  if (id == sec.end()) {
    std::cerr << "Material " << name << " has no id\n";
    return false;
  }
  if (color == sec.end()) {
    std::cerr << "Material " << name << " has no color\n";
    return false;
  }
  if (phong == sec.end()) {
    std::cerr << "Material " << name << " has no phong\n";
    return false;
  }
  if (reflect == sec.end()) {
    std::cerr << "Material " << name << " has no reflect\n";
    return false;
  }
  if (transparency == sec.end()) {
    std::cerr << "Material " << name << " has no transparency\n";
    return false;
  }

  id_out = std::stoull(id->second);
  out.push_back({
    .color = parse_vec3(color->second),
    .phong_exponent = std::stof(phong->second),
    .reflect_factor = std::stof(reflect->second),
    .transparency = std::stof(transparency->second),
    .id = id_out
  });
  return true;
}

bool parse_point(const std::string &name, const inipp::Ini<char>::Section &sec, std::vector<point_light> &out) {
  const auto point = sec.find("point");
  const auto color = sec.find("color");
  const auto attenuation = sec.find("attenuation");
  const auto intensity = sec.find("intensity");
  if (point == sec.end()) {
    std::cerr << "Point " << name << " has no point\n";
    return false;
  }
  if (color == sec.end()) {
    std::cerr << "Point " << name << " has no color\n";
    return false;
  }
  if (attenuation == sec.end()) {
    std::cerr << "Point " << name << " has no attenuation\n";
    return false;
  }
  if (intensity == sec.end()) {
    std::cerr << "Point " << name << " has no intensity\n";
    return false;
  }

  out.push_back({
    .point = parse_vec3(point->second),
    .color = clamp_01(parse_vec3(color->second)),
    .attenuation = parse_vec3(attenuation->second),
    .intensity = std::stof(intensity->second)
  });
  return true;
}

scene_cpu loader::load(const std::string &filename, bool &all_okay) {
  std::ifstream strm(filename);
  if (!strm) {
    std::cerr << "Failed to open file " << filename << "\n";
    exit(1);
  }

  inipp::Ini<char> ini;
  ini.parse(strm);

  scene_cpu scene;
  size_t id = 0;

  all_okay = true;
  std::unordered_set<size_t> missing_material_ids;
  std::unordered_map<size_t, size_t> material_id_to_idx;
  for (const auto &[n, section] : ini.sections) {
    if (n == "camera") {
      auto it = section.find("position");
      if (it == section.end()) {
        std::cerr << "Camera has no position\n";
        all_okay = false;
        continue;
      }
      scene.camera = parse_vec3(it->second);
      it = section.find("forward");
      if (it == section.end()) {
        std::cerr << "Camera has no forward\n";
        all_okay = false;
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
        all_okay = false;
        continue;
      }

      if (it->second == "point") {
        all_okay = all_okay && parse_point(n, section, scene.points);
      }
      else if (it->second == "material") {
        if (size_t mat_id = 0; parse_mat(n, section, mat_id, scene.materials)) {
          missing_material_ids.erase(mat_id);
          material_id_to_idx[mat_id] = scene.materials.size() - 1;
        }
        else {
          all_okay = false;
        }
      }
      else {
        size_t id_override = id;
        if (const auto id_it = section.find("id"); id_it != section.end()) {
          id_override = std::stoul(id_it->second);
        }

        std::cout << "Object " << n << " (" << it->second << ") has ID " << id_override << "\n";

        size_t m_id = 0;
        if (it->second == "sphere") {
          if(parse_sphere(n, section, id_override, m_id, scene.spheres)) {
            if (!missing_material_ids.contains(m_id) && !material_id_to_idx.contains(m_id)) {
              missing_material_ids.insert(m_id);
            }
          }
          else {
            all_okay = false;
          }
        }
        else if (it->second == "plane") {
          if(parse_plane(n, section, id_override, m_id, scene.planes)) {
            if (!missing_material_ids.contains(m_id) && !material_id_to_idx.contains(m_id)) {
              missing_material_ids.insert(m_id);
            }
          }
          else {
            all_okay = false;
          }
        }
        else if (it->second == "triangle") {
          if(parse_triangle(n, section, id_override, m_id, scene.triangles)) {
            if (!missing_material_ids.contains(m_id) && !material_id_to_idx.contains(m_id)) {
              missing_material_ids.insert(m_id);
            }
          }
          else {
            all_okay = false;
          }
        }
        else if (it->second == "mesh") {
          if(parse_mesh(filename, n, section, id_override, m_id, scene.triangles)) {
            if (!missing_material_ids.contains(m_id) && !material_id_to_idx.contains(m_id)) {
              missing_material_ids.insert(m_id);
            }
          }
          else {
            all_okay = false;
          }
        }
        else {
          std::cerr << "Unknown type " << it->second << " (for section " << n << ")\n";
          all_okay = false;
        }
        ++id;
      }
    }
  }

  if (!missing_material_ids.empty()) {
    std::cerr << "Missing material IDs: ";
    for (const auto i : missing_material_ids) {
      std::cerr << i << " ";
    }
    std::cerr << "\n";
    all_okay = false;
  }
  else {
    for (auto &p: scene.planes) p.material_idx = material_id_to_idx[p.material_idx];
    for (auto &s: scene.spheres) s.material_idx = material_id_to_idx[s.material_idx];
    for (auto &t: scene.triangles) t.material_idx = material_id_to_idx[t.material_idx];
  }

  return scene;
}
