// mesh.thumbnail: the Load Data dialog's preview of a scan file (STL, PLY, OBJ)
// before it is imported, with its size (the file read by mesh_io).
//
// The mesh is seen along the axis it spreads least (an arch: the occlusal
// direction), from the side most of its surface faces, the front teeth up,
// centred on its surface: from the file's own coordinates, so no position or
// rotation of an object matters (Blender's OBJ import adds X 90 degrees). Drawn
// with a depth buffer and one light into a float tile: 0.2..1 shading, -1 empty.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "commands.h"
#include "mesh_io.h"

namespace odent {

namespace {

// --- the view ----------------------------------------------------------------

using Vec = std::array<double, 3>;
double dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec cross(const Vec& a, const Vec& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// The eigenvectors of a symmetric 3x3 matrix (Jacobi), largest eigenvalue first.
std::array<Vec, 3> principal_axes(std::array<std::array<double, 3>, 3> a) {
    std::array<std::array<double, 3>, 3> v{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    for (int sweep = 0; sweep < 50; ++sweep) {
        const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if (off < 1e-30) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                if (std::abs(a[p][q]) < 1e-300) continue;
                const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                const double t = (theta >= 0 ? 1 : -1) / (std::abs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < 3; ++k) {   // a = J^T a J
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = v[k][p], vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
    }
    std::array<int, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end(), [&](int i, int j) { return a[i][i] > a[j][j]; });
    std::array<Vec, 3> axes;
    for (int k = 0; k < 3; ++k) axes[k] = {v[0][order[k]], v[1][order[k]], v[2][order[k]]};
    return axes;
}

// mesh.thumbnail {file, size (px, 512)}: the tile (1, size, size) float32, row 0
// at the top; triangles, vertices; size_mm: the extent along the view's axes
// (largest first; the last is along the view).
Json::Value mesh_thumbnail(const Json::Value& p) {
    const std::string path = require_string(p, "file");
    const int size = std::clamp(p.get("size", 512).asInt(), 16, 2048);
    const Mesh m = read_mesh(path);
    const std::size_t nv = m.vertices();
    auto P = [&](std::size_t i) {
        return Vec{m.points[3 * i], m.points[3 * i + 1], m.points[3 * i + 2]};
    };
    // the axes of the surface, not of the points: its exact first and second
    // moments, triangle by triangle (an STL repeats each corner once per triangle,
    // scanners put more points where there is more detail, and a face's diagonal
    // breaks the symmetry of its centres: all three tilted simpler axes).
    // Over a triangle a, b, c of area A: the integral of p is A (a + b + c) / 3,
    // of p p^T  A / 12 ((a + b + c)(a + b + c)^T + a a^T + b b^T + c c^T).
    const Vec o = P(0);   // moments about a point of the mesh: no loss of precision far out
    Vec first{0, 0, 0};
    std::array<std::array<double, 3>, 3> second{};
    double total = 0;
    for (std::size_t t = 0; t < m.triangles(); ++t) {
        Vec v[3];
        for (int j = 0; j < 3; ++j) {
            const Vec q = P(m.tris[3 * t + j]);
            v[j] = {q[0] - o[0], q[1] - o[1], q[2] - o[2]};
        }
        const Vec n = cross({v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]},
                            {v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]});
        const double A = 0.5 * std::sqrt(dot(n, n));
        const Vec s{v[0][0] + v[1][0] + v[2][0], v[0][1] + v[1][1] + v[2][1], v[0][2] + v[1][2] + v[2][2]};
        for (int r = 0; r < 3; ++r) {
            first[r] += A * s[r] / 3;
            for (int c = 0; c < 3; ++c)
                second[r][c] += A / 12 * (s[r] * s[c] + v[0][r] * v[0][c] + v[1][r] * v[1][c] + v[2][r] * v[2][c]);
        }
        total += A;
    }
    if (total <= 0) throw std::runtime_error("the scan's faces have no area");
    Vec mean;
    for (int k = 0; k < 3; ++k) mean[k] = first[k] / total;
    std::array<std::array<double, 3>, 3> cov{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) cov[r][c] = second[r][c] / total - mean[r] * mean[c];
    for (int k = 0; k < 3; ++k) mean[k] += o[k];
    auto axes = principal_axes(cov);
    // look from the side most of the surface faces (an arch: the teeth, not its base)
    double facing = 0;
    for (std::size_t t = 0; t < m.triangles(); ++t) {
        const Vec a = P(m.tris[3 * t]), b = P(m.tris[3 * t + 1]), c = P(m.tris[3 * t + 2]);
        facing += dot(cross({b[0] - a[0], b[1] - a[1], b[2] - a[2]}, {c[0] - a[0], c[1] - a[1], c[2] - a[2]}),
                      axes[2]);
    }
    if (facing < 0) axes[2] = {-axes[2][0], -axes[2][1], -axes[2][2]};
    // in the picture, the same way up whatever the file's position and rotation:
    // up is the narrower half along the second axis (an arch is a U: narrow at the
    // front teeth, wide at the molars), sideways the axis that makes it not mirrored
    double half_w[2] = {0, 0};
    std::size_t half_n[2] = {0, 0};
    for (std::size_t i = 0; i < nv; ++i) {
        const Vec q = P(i), d{q[0] - mean[0], q[1] - mean[1], q[2] - mean[2]};
        const int h = dot(d, axes[1]) > 0 ? 1 : 0;
        half_w[h] += std::abs(dot(d, axes[0]));
        ++half_n[h];
    }
    if (half_n[0] && half_n[1] && half_w[1] / double(half_n[1]) > half_w[0] / double(half_n[0]))
        axes[1] = {-axes[1][0], -axes[1][1], -axes[1][2]};
    axes[0] = cross(axes[1], axes[2]);   // right-handed with the view: not mirrored

    std::vector<double> sx(nv), sy(nv), sz(nv);
    std::array<double, 3> lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
    for (std::size_t i = 0; i < nv; ++i) {
        const Vec q = P(i), d{q[0] - mean[0], q[1] - mean[1], q[2] - mean[2]};
        const double c[3] = {dot(d, axes[0]), dot(d, axes[1]), dot(d, axes[2])};
        for (int k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], c[k]);
            hi[k] = std::max(hi[k], c[k]);
        }
        sx[i] = c[0];
        sy[i] = c[1];
        sz[i] = c[2];
    }
    const double margin = 0.04 * size;
    const double span = std::max({hi[0] - lo[0], hi[1] - lo[1], 1e-9});
    const double scale = (size - 2 * margin) / span;
    const double cx = 0.5 * (lo[0] + hi[0]), cy = 0.5 * (lo[1] + hi[1]);
    for (std::size_t i = 0; i < nv; ++i) {
        sx[i] = size * 0.5 + (sx[i] - cx) * scale;
        sy[i] = size * 0.5 - (sy[i] - cy) * scale;   // rows go down
    }
    // the light: from the viewer, a little from the top left (slopes and cusps show)
    Vec lamp{axes[2][0] + 0.45 * axes[1][0] - 0.35 * axes[0][0],
             axes[2][1] + 0.45 * axes[1][1] - 0.35 * axes[0][1],
             axes[2][2] + 0.45 * axes[1][2] - 0.35 * axes[0][2]};
    const double lamp_len = std::sqrt(dot(lamp, lamp));
    for (auto& c : lamp) c /= lamp_len;
    auto tile = Volume::table(PixelType::Float32, std::size_t(size), std::size_t(size));
    auto* shade = static_cast<float*>(tile->buffer->data());
    std::fill(shade, shade + std::size_t(size) * size, -1.0f);
    std::vector<double> depth(std::size_t(size) * size, -std::numeric_limits<double>::infinity());
    for (std::size_t t = 0; t < m.triangles(); ++t) {
        const std::uint32_t i0 = m.tris[3 * t], i1 = m.tris[3 * t + 1], i2 = m.tris[3 * t + 2];
        const Vec a = P(i0), b = P(i1), c = P(i2);
        Vec n = cross({b[0] - a[0], b[1] - a[1], b[2] - a[2]}, {c[0] - a[0], c[1] - a[1], c[2] - a[2]});
        const double len = std::sqrt(dot(n, n));
        if (len <= 0) continue;
        const float light = static_cast<float>(0.2 + 0.8 * std::abs(dot(n, lamp)) / len);
        const double x0 = sx[i0], y0 = sy[i0], x1 = sx[i1], y1 = sy[i1], x2 = sx[i2], y2 = sy[i2];
        const double area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (std::abs(area) < 1e-12) continue;
        const int bx0 = std::max(0, int(std::floor(std::min({x0, x1, x2}))));
        const int bx1 = std::min(size - 1, int(std::ceil(std::max({x0, x1, x2}))));
        const int by0 = std::max(0, int(std::floor(std::min({y0, y1, y2}))));
        const int by1 = std::min(size - 1, int(std::ceil(std::max({y0, y1, y2}))));
        for (int y = by0; y <= by1; ++y)
            for (int x = bx0; x <= bx1; ++x) {
                const double px = x + 0.5, py = y + 0.5;
                const double w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area;
                const double w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area;
                const double w2 = 1 - w0 - w1;
                if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                const double z = w0 * sz[i0] + w1 * sz[i1] + w2 * sz[i2];
                const std::size_t k = std::size_t(y) * size + x;
                if (z > depth[k]) {
                    depth[k] = z;
                    shade[k] = light;
                }
            }
    }
    const Volume& ref = *tile;
    Json::Value out = volume_json(store().add(std::move(tile)), ref);
    out["triangles"] = static_cast<Json::UInt64>(m.triangles());
    out["vertices"] = static_cast<Json::UInt64>(nv);
    Json::Value extent(Json::arrayValue);
    for (int k = 0; k < 3; ++k) extent.append(hi[k] - lo[k]);
    out["size_mm"] = extent;
    return out;
}

}  // namespace

void add_mesh_preview_commands(Commands& c) {
    c["mesh.thumbnail"] = mesh_thumbnail;
}

}  // namespace odent
