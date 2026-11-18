// Scan files read by the engine itself (its static VTK has no mesh readers): what
// intraoral and lab scanners write. Binary or ASCII STL, binary little endian or
// ASCII PLY with any vertex properties, OBJ's v and f lines. Used by mesh.thumbnail
// (mesh_preview.cpp) and register (register_main.cpp).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace odent {

struct Mesh {
    std::vector<float> points;           // x, y, z
    std::vector<std::uint32_t> tris;     // three point indices each
    std::vector<float> facet_normals;    // STL only: the file's normal, x, y, z per triangle
    std::size_t vertices() const { return points.size() / 3; }
    std::size_t triangles() const { return tris.size() / 3; }
};

// The file by its extension (.stl, .ply, .obj); throws std::runtime_error with what
// is wrong (not a scan file, cut short, no faces, a face using a missing point).
Mesh read_mesh(const std::string& path);

// The points the registration starts from (x, y, z), as Open3D's io::ReadTriangleMesh
// gave them to MeshRegAuto: an STL went through Assimp, every triangle with its own
// three corners carrying the facet normal, then Assimp's JoinIdenticalVertices merged
// corners whose position and normal are both equal (in the order first seen); PLY
// (Open3D's own reader): all its vertices; OBJ (Assimp): the vertices faces use.
std::vector<double> registration_points(const std::string& path);

}  // namespace odent
