// Scan files read by the engine (see mesh_io.h). Moved from mesh_preview.cpp
// (2026-10-31) for the registration; STL keeps its facet normals.

#include "mesh_io.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace odent {

namespace {

std::string read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("can not open the scan file");
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// A polygon's corners as triangles (a fan; scanners write triangles anyway).
void add_polygon(Mesh& m, const std::uint32_t* idx, std::size_t n) {
    for (std::size_t k = 1; k + 1 < n; ++k) {
        m.tris.push_back(idx[0]);
        m.tris.push_back(idx[k]);
        m.tris.push_back(idx[k + 1]);
    }
}

// --- STL ---------------------------------------------------------------------

Mesh read_stl(const std::string& data) {
    Mesh m;
    if (data.size() >= 84) {
        std::uint32_t n = 0;
        std::memcpy(&n, data.data() + 80, 4);
        if (84 + std::uint64_t(n) * 50 == data.size()) {   // binary: its size tells
            m.points.resize(std::size_t(n) * 9);
            m.tris.resize(std::size_t(n) * 3);
            m.facet_normals.resize(std::size_t(n) * 3);
            for (std::uint32_t t = 0; t < n; ++t) {
                std::memcpy(&m.facet_normals[std::size_t(t) * 3], data.data() + 84 + std::size_t(t) * 50, 12);
                std::memcpy(&m.points[std::size_t(t) * 9], data.data() + 84 + std::size_t(t) * 50 + 12, 36);
                for (int k = 0; k < 3; ++k) m.tris[std::size_t(t) * 3 + k] = t * 3 + k;
            }
            return m;
        }
    }
    // ASCII: every "facet normal x y z" and "vertex x y z"
    const char* p = data.c_str();
    const char* end = p + data.size();
    const char* next_normal = std::strstr(p, "normal");
    while ((p = std::strstr(p, "vertex")) != nullptr && p < end) {
        while (next_normal && next_normal < p) {        // the facet's normal, before its corners
            const char* s = next_normal + 6;
            char* q = nullptr;
            for (int k = 0; k < 3; ++k) {
                m.facet_normals.push_back(std::strtof(s, &q));
                s = q;
            }
            next_normal = std::strstr(s, "normal");
        }
        p += 6;
        char* q = nullptr;
        for (int k = 0; k < 3; ++k) {
            m.points.push_back(std::strtof(p, &q));
            p = q;
        }
    }
    if (m.points.size() % 9) throw std::runtime_error("the STL file is cut short");
    if (m.facet_normals.size() != m.points.size() / 3) m.facet_normals.clear();   // (no usable normals)
    for (std::uint32_t i = 0; i < m.points.size() / 3; ++i) m.tris.push_back(i);
    return m;
}

// --- PLY ---------------------------------------------------------------------

struct PlyProperty {
    std::string name, type;          // type: char uchar short ushort int uint float double (+ aliases)
    bool list = false;
    std::string count_type;
};
struct PlyElement {
    std::string name;
    std::size_t count = 0;
    std::vector<PlyProperty> props;
};

std::size_t ply_size(const std::string& t) {
    if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
    if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
    if (t == "int" || t == "uint" || t == "float" || t == "int32" || t == "uint32" || t == "float32") return 4;
    if (t == "double" || t == "float64") return 8;
    throw std::runtime_error("PLY property type '" + t + "' is not known");
}

double ply_value(const char* b, const std::string& t) {
    if (t == "char" || t == "int8") return *reinterpret_cast<const std::int8_t*>(b);
    if (t == "uchar" || t == "uint8") return *reinterpret_cast<const std::uint8_t*>(b);
    std::int16_t i16; std::uint16_t u16; std::int32_t i32; std::uint32_t u32; float f; double d;
    if (t == "short" || t == "int16") { std::memcpy(&i16, b, 2); return i16; }
    if (t == "ushort" || t == "uint16") { std::memcpy(&u16, b, 2); return u16; }
    if (t == "int" || t == "int32") { std::memcpy(&i32, b, 4); return i32; }
    if (t == "uint" || t == "uint32") { std::memcpy(&u32, b, 4); return u32; }
    if (t == "float" || t == "float32") { std::memcpy(&f, b, 4); return f; }
    std::memcpy(&d, b, 8);
    return d;
}

Mesh read_ply(const std::string& data) {
    const auto header_end = data.find("end_header");
    if (data.compare(0, 3, "ply") != 0 || header_end == std::string::npos)
        throw std::runtime_error("not a PLY file");
    std::istringstream header(data.substr(0, header_end));
    std::string line, format;
    std::vector<PlyElement> elements;
    while (std::getline(header, line)) {
        std::istringstream w(line);
        std::string key;
        w >> key;
        if (key == "format") {
            w >> format;
        } else if (key == "element") {
            PlyElement e;
            w >> e.name >> e.count;
            elements.push_back(e);
        } else if (key == "property" && !elements.empty()) {
            PlyProperty pr;
            w >> pr.type;
            if (pr.type == "list") {
                pr.list = true;
                w >> pr.count_type >> pr.type;
            }
            w >> pr.name;
            elements.back().props.push_back(pr);
        }
    }
    if (format != "ascii" && format != "binary_little_endian")
        throw std::runtime_error("PLY format '" + format + "' is not read (only ascii, little endian)");
    std::size_t pos = data.find('\n', header_end);
    if (pos == std::string::npos) throw std::runtime_error("the PLY file is cut short");
    ++pos;
    Mesh m;
    std::vector<std::uint32_t> poly;
    const bool ascii = format == "ascii";
    std::istringstream text(ascii ? data.substr(pos) : std::string());
    for (const auto& e : elements) {
        const bool is_vertex = e.name == "vertex", is_face = e.name == "face";
        int xi = -1, yi = -1, zi = -1;
        for (std::size_t k = 0; k < e.props.size(); ++k) {
            if (e.props[k].name == "x") xi = int(k);
            if (e.props[k].name == "y") yi = int(k);
            if (e.props[k].name == "z") zi = int(k);
        }
        if (is_vertex && (xi < 0 || yi < 0 || zi < 0)) throw std::runtime_error("PLY vertices without x, y, z");
        if (is_vertex) m.points.reserve(e.count * 3);
        for (std::size_t r = 0; r < e.count; ++r) {
            std::array<double, 3> xyz{};
            for (std::size_t k = 0; k < e.props.size(); ++k) {
                const auto& pr = e.props[k];
                if (pr.list) {
                    double nd = 0;
                    if (ascii) {
                        text >> nd;
                    } else {
                        const std::size_t cs = ply_size(pr.count_type);
                        if (pos + cs > data.size()) throw std::runtime_error("the PLY file is cut short");
                        nd = ply_value(data.data() + pos, pr.count_type);
                        pos += cs;
                    }
                    const auto n = static_cast<std::size_t>(nd);
                    poly.resize(n);
                    const std::size_t is = ascii ? 0 : ply_size(pr.type);
                    if (!ascii && pos + n * is > data.size()) throw std::runtime_error("the PLY file is cut short");
                    for (std::size_t j = 0; j < n; ++j) {
                        double v = 0;
                        if (ascii) {
                            text >> v;
                        } else {
                            v = ply_value(data.data() + pos, pr.type);
                            pos += is;
                        }
                        poly[j] = static_cast<std::uint32_t>(v);
                    }
                    if (is_face && (pr.name == "vertex_indices" || pr.name == "vertex_index"))
                        add_polygon(m, poly.data(), n);
                } else {
                    double v = 0;
                    if (ascii) {
                        text >> v;
                    } else {
                        const std::size_t s = ply_size(pr.type);
                        if (pos + s > data.size()) throw std::runtime_error("the PLY file is cut short");
                        v = ply_value(data.data() + pos, pr.type);
                        pos += s;
                    }
                    if (int(k) == xi) xyz[0] = v;
                    if (int(k) == yi) xyz[1] = v;
                    if (int(k) == zi) xyz[2] = v;
                }
            }
            if (is_vertex)
                for (double c : xyz) m.points.push_back(static_cast<float>(c));
        }
        if (ascii && !text) throw std::runtime_error("the PLY file is cut short");
    }
    return m;
}

// --- OBJ ---------------------------------------------------------------------

Mesh read_obj(const std::string& data) {
    Mesh m;
    std::vector<std::uint32_t> poly;
    const char* p = data.c_str();
    const char* end = p + data.size();
    while (p < end) {
        const char* eol = static_cast<const char*>(std::memchr(p, '\n', std::size_t(end - p)));
        if (!eol) eol = end;
        if (p[0] == 'v' && (p[1] == ' ' || p[1] == '\t')) {
            char* q = nullptr;
            const char* s = p + 2;
            for (int k = 0; k < 3; ++k) {
                m.points.push_back(std::strtof(s, &q));
                s = q;
            }
        } else if (p[0] == 'f' && (p[1] == ' ' || p[1] == '\t')) {
            poly.clear();
            const char* s = p + 2;
            while (s < eol) {
                while (s < eol && (*s == ' ' || *s == '\t' || *s == '\r')) ++s;
                if (s >= eol) break;
                char* q = nullptr;
                const long i = std::strtol(s, &q, 10);   // "a/b/c": the point index first
                if (q == s) break;
                const long nv = static_cast<long>(m.vertices());
                const long idx = i > 0 ? i - 1 : nv + i;   // negative: counted from the last point
                if (idx < 0 || idx >= nv) throw std::runtime_error("an OBJ face uses a point it does not have");
                poly.push_back(static_cast<std::uint32_t>(idx));
                s = q;
                while (s < eol && *s != ' ' && *s != '\t') ++s;   // skip "/b/c"
            }
            add_polygon(m, poly.data(), poly.size());
        }
        p = eol + 1;
    }
    return m;
}

std::string lower_name(const std::string& path) { return lower(path); }

bool ends_with(const std::string& name, const char* s) {
    const std::size_t n = std::strlen(s);
    return name.size() >= n && name.compare(name.size() - n, n, s) == 0;
}

}  // namespace

Mesh read_mesh(const std::string& path) {
    const std::string name = lower(path);
    auto ends = [&](const char* s) {
        const std::size_t n = std::strlen(s);
        return name.size() >= n && name.compare(name.size() - n, n, s) == 0;
    };
    const std::string data = read_all(path);
    Mesh m;
    if (ends(".stl")) m = read_stl(data);
    else if (ends(".ply")) m = read_ply(data);
    else if (ends(".obj")) m = read_obj(data);
    else throw std::runtime_error("not a scan file (STL, PLY, OBJ)");
    const std::size_t nv = m.vertices();
    for (std::uint32_t i : m.tris)
        if (i >= nv) throw std::runtime_error("a face uses a point the file does not have");
    if (m.triangles() == 0) throw std::runtime_error("the scan file has no faces");
    return m;
}

std::vector<double> registration_points(const std::string& path) {
    const Mesh m = read_mesh(path);
    std::vector<double> out;
    const std::string name = lower_name(path);
    if (ends_with(name, ".ply")) {                      // all its vertices
        out.assign(m.points.begin(), m.points.end());
        return out;
    }
    if (ends_with(name, ".stl")) {
        // Assimp: each triangle's corners with its facet normal; JoinIdenticalVertices
        // keeps the first of the corners equal in position and normal (bit for bit:
        // its hash map finds only those)
        struct Key {
            std::array<float, 6> v;
            bool operator==(const Key& o) const { return std::memcmp(v.data(), o.v.data(), sizeof v) == 0; }
        };
        struct Hash {
            std::size_t operator()(const Key& k) const {
                std::size_t h = 1469598103934665603ull;
                const auto* b = reinterpret_cast<const unsigned char*>(k.v.data());
                for (std::size_t i = 0; i < sizeof k.v; ++i) h = (h ^ b[i]) * 1099511628211ull;
                return h;
            }
        };
        const bool normals = m.facet_normals.size() == m.triangles() * 3;
        std::unordered_map<Key, int, Hash> seen;
        seen.reserve(m.vertices());
        out.reserve(m.vertices() * 3);
        for (std::size_t c = 0; c < m.vertices(); ++c) {
            Key k{};
            for (int j = 0; j < 3; ++j) k.v[j] = m.points[3 * c + j];
            for (int j = 0; j < 3; ++j) k.v[3 + j] = normals ? m.facet_normals[3 * (c / 3) + j] : 0.0f;
            if (seen.emplace(k, int(c)).second)
                for (int j = 0; j < 3; ++j) out.push_back(m.points[3 * c + j]);
        }
        return out;
    }
    // OBJ (Assimp): the vertices its faces use, in order
    std::vector<char> used(m.vertices(), 0);
    for (std::uint32_t i : m.tris) used[i] = 1;
    for (std::size_t v = 0; v < m.vertices(); ++v)
        if (used[v])
            for (int j = 0; j < 3; ++j) out.push_back(m.points[3 * v + j]);
    return out;
}

}  // namespace odent
