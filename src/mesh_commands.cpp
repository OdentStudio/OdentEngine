// Meshes from the CT: Dicom to Mesh (dicom_ops ODENT_OT_DicomToMesh, which ran
// vtkMarchingCubes on a VTK copy of the CT in Python).

#include <vtkAOSDataArrayTemplate.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkCellArray.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkFlyingEdges3D.h>
#include <vtkMarchingCubes.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkSurfaceNets3D.h>
#include <vtkUnsignedCharArray.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

#include "commands.h"

namespace odent {

namespace {

bool identity(const std::array<double, 9>& d) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (d[3 * i + j] != (i == j ? 1.0 : 0.0)) return false;
    return true;
}

// A vtkImageData over the volume's voxels (no copy). With an identity direction
// it carries the CT's origin, so VTK places the points as the addon's VTK did;
// else the points come out in the CT frame and are turned here (VTK 9.1's
// marching cubes ignores a direction matrix, the addon's 9.7 applied it).
template <class T>
vtkSmartPointer<vtkImageData> vtk_view(const Volume& ct, bool with_origin) {
    auto scalars = vtkSmartPointer<vtkAOSDataArrayTemplate<T>>::New();
    scalars->SetNumberOfComponents(1);
    scalars->SetArray(static_cast<T*>(ct.buffer->data()), static_cast<vtkIdType>(ct.count()), 1);
    auto img = vtkSmartPointer<vtkImageData>::New();
    img->SetDimensions(static_cast<int>(ct.size[0]), static_cast<int>(ct.size[1]),
                       static_cast<int>(ct.size[2]));
    img->SetSpacing(ct.spacing.data());
    if (with_origin) img->SetOrigin(ct.origin.data());
    img->GetPointData()->SetScalars(scalars);
    return img;
}

// The CT as two labels for Surface Nets (1: at or above `threshold`, 0: below), with the
// view's geometry; Surface Nets meshes the boundary between labels, not a CT value.
template <class T>
vtkSmartPointer<vtkImageData> label_image(const Volume& ct, double threshold, bool with_origin) {
    const T* v = static_cast<const T*>(ct.buffer->data());
    const std::size_t n = ct.count();
    auto labels = vtkSmartPointer<vtkUnsignedCharArray>::New();
    labels->SetNumberOfComponents(1);
    labels->SetNumberOfTuples(static_cast<vtkIdType>(n));
    unsigned char* out = labels->GetPointer(0);
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<double>(v[i]) >= threshold ? 1 : 0;
    auto img = vtkSmartPointer<vtkImageData>::New();
    img->SetDimensions(static_cast<int>(ct.size[0]), static_cast<int>(ct.size[1]),
                       static_cast<int>(ct.size[2]));
    img->SetSpacing(ct.spacing.data());
    if (with_origin) img->SetOrigin(ct.origin.data());
    img->GetPointData()->SetScalars(labels);
    return img;
}

// mesh.threshold {volume, threshold, method} -- the surface where the CT crosses
// `threshold` (a stored value: ct_values.surface_threshold), in physical space, by
// `method` (the addon preference Dicom to Mesh method):
//   "marching_cubes" (default): vtkMarchingCubes, the surface interpolated in each voxel;
//   "flying_edges": vtkFlyingEdges3D, the same kind of surface, made faster (threaded);
//   "surface_nets": vtkSurfaceNets3D on the CT split at the threshold (two labels), its
//     points smoothed: a smoother surface with fewer triangles.
// The faces keep the method's winding in the CT's index space, outward for straight axes
// (checked for each on a ball); for a mirrored direction (determinant < 0) they come out
// inside out, and the addon reverses them (dicom_ops._d2m_engine).
// -> {vertices: float32 (n, 3), faces: int32 (m, 3), method} (volume infos to map).
Json::Value mesh_threshold(const Json::Value& p) {
    const Volume& ct = store().get(require_int(p, "volume"));
    const double threshold = require_double(p, "threshold");
    const std::string method = p.get("method", "marching_cubes").asString();
    const bool plain = identity(ct.direction);
    vtkSmartPointer<vtkPolyDataAlgorithm> mesher;
    if (method == "surface_nets") {
        auto img = dispatch(ct.type, [&](auto pixel) {
            return label_image<decltype(pixel)>(ct, threshold, plain);
        });
        auto nets = vtkSmartPointer<vtkSurfaceNets3D>::New();
        nets->SetInputData(img);
        nets->SetBackgroundLabel(0);
        nets->SetValue(0, 1);                       // the label above the threshold
        nets->SetOutputMeshTypeToTriangles();
        nets->SmoothingOn();                        // VTK's defaults: 16 passes, relaxation 0.5
        mesher = nets;
    } else if (method == "flying_edges" || method == "marching_cubes") {
        auto img = dispatch(ct.type, [&](auto pixel) { return vtk_view<decltype(pixel)>(ct, plain); });
        if (method == "flying_edges") {
            auto fe = vtkSmartPointer<vtkFlyingEdges3D>::New();
            fe->SetInputData(img);
            fe->SetValue(0, threshold);
            fe->ComputeNormalsOff();
            fe->ComputeGradientsOff();
            fe->ComputeScalarsOff();
            mesher = fe;
        } else {
            auto mc = vtkSmartPointer<vtkMarchingCubes>::New();
            mc->SetInputData(img);
            mc->SetValue(0, threshold);
            mc->ComputeNormalsOff();   // the mesh is made without them
            mesher = mc;
        }
    } else {
        throw std::runtime_error("unknown method: " + method);
    }
    // its progress (VTK's ProgressEvent) to the addon's card: the meshing is most of the
    // time (case 3: ~5 of 6 s with marching cubes), the copies below the rest (90 % -> 100 %)
    vtkNew<vtkCallbackCommand> progress;
    progress->SetCallback([](vtkObject* caller, unsigned long, void*, void*) {
        report_progress(0.9 * static_cast<vtkAlgorithm*>(caller)->GetProgress());
    });
    mesher->AddObserver(vtkCommand::ProgressEvent, progress);
    mesher->Update();
    report_progress(0.9);
    vtkPolyData* out = mesher->GetOutput();
    const vtkIdType n = out->GetNumberOfPoints();
    auto verts = Volume::table(PixelType::Float32, static_cast<std::size_t>(n), 3);
    auto* vp = static_cast<float*>(verts->buffer->data());
    const auto& d = ct.direction;
    for (vtkIdType i = 0; i < n; ++i) {
        double x[3];
        out->GetPoint(i, x);
        if (!plain) {
            const double r[3] = {x[0], x[1], x[2]};
            for (int k = 0; k < 3; ++k)
                x[k] = ct.origin[k] + d[3 * k] * r[0] + d[3 * k + 1] * r[1] + d[3 * k + 2] * r[2];
        }
        for (int k = 0; k < 3; ++k) vp[3 * i + k] = static_cast<float>(x[k]);
    }
    report_progress(0.95);
    vtkCellArray* polys = out->GetPolys();
    const vtkIdType m = polys ? polys->GetNumberOfCells() : 0;
    auto faces = Volume::table(PixelType::Int32, static_cast<std::size_t>(m), 3);
    auto* fp = static_cast<std::int32_t*>(faces->buffer->data());
    if (m > 0) {
        vtkDataArray* conn = polys->GetConnectivityArray();
        if (conn->GetNumberOfTuples() != 3 * m)
            throw std::runtime_error("marching cubes gave faces that are not triangles");
        for (vtkIdType i = 0; i < 3 * m; ++i) fp[i] = static_cast<std::int32_t>(conn->GetTuple1(i));
    }
    Json::Value result;
    const Volume& vref = *verts;
    const Volume& fref = *faces;
    result["vertices"] = volume_json(store().add(std::move(verts)), vref);
    result["faces"] = volume_json(store().add(std::move(faces)), fref);
    return result;
}

// A table of the addon's (or a reply's): rows x cols of `type`, else an error naming it.
const Volume& table_param(const Json::Value& p, const char* key, std::size_t cols) {
    const Volume& v = store().get(require_int(p, key));
    if (v.size[0] != cols || v.size[2] != 1)
        throw std::runtime_error(std::string("'") + key + "' must be a table of " +
                                 std::to_string(cols) + " columns");
    return v;
}

// fn(a, b) on [0, n) cut in blocks over the hardware threads (at most 16)
template <class F>
void parallel_blocks(std::size_t n, F&& fn) {
    const unsigned threads = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
    const std::size_t block = (n + threads - 1) / threads;
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        const std::size_t a = t * block, b = std::min(n, a + block);
        if (a >= b) break;
        pool.emplace_back([&fn, a, b] { fn(a, b); });
    }
    for (auto& th : pool) th.join();
}

// The closest point of triangle (a, b, c) to p: its squared distance (Ericson, Real-Time
// Collision Detection 5.1.5, by the regions of the triangle).
double dist2_triangle(const double* p, const double* a, const double* b, const double* c) {
    double ab[3], ac[3], ap[3], q[3];
    for (int k = 0; k < 3; ++k) { ab[k] = b[k] - a[k]; ac[k] = c[k] - a[k]; ap[k] = p[k] - a[k]; }
    auto dot = [](const double* u, const double* v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; };
    auto at = [&](double v, double w) {
        for (int k = 0; k < 3; ++k) q[k] = a[k] + v * ab[k] + w * ac[k];
        double d = 0;
        for (int k = 0; k < 3; ++k) d += (p[k] - q[k]) * (p[k] - q[k]);
        return d;
    };
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return at(0, 0);
    double bp[3];
    for (int k = 0; k < 3; ++k) bp[k] = p[k] - b[k];
    const double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return at(1, 0);
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return at(d1 / (d1 - d3), 0);
    double cp[3];
    for (int k = 0; k < 3; ++k) cp[k] = p[k] - c[k];
    const double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return at(0, 1);
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return at(0, d2 / (d2 - d6));
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return at(1 - w, w);
    }
    const double den = 1.0 / (va + vb + vc);
    return at(vb * den, vc * den);
}

// mesh.distance {points, vertices, faces, max_distance} -- for each point (a table of
// rows x 3, float32 or float64), its distance to the surface (vertices: rows x 3, faces:
// rows x 3 of indices; e.g. mesh.threshold's reply, kept here: nothing goes back to the
// addon), all in the same space; past `max_distance` (mm): +inf. The Heat Map's and ALIGN
// Points' gaps (2026-11-02). Measured on 173k points against a 1.1M-face CT surface:
// Blender's BVHTree from Python 1.5 s (lists 0.41, tree 0.45, a query a call 0.65);
// vtkStaticCellLocator 0.65 s, and ~100 points' gaps wrong or missing (its bucket size 1-150
// tried: 10-25 best); this: a uniform grid of the triangles (cells 3 mean edges wide, at most
// 2^24) searched in rings out from the point's cell up to max_distance, cells farther than
// the best found skipped, exact point-triangle distances, in parallel over 8 threads:
// 0.29 s (grid 0.12, queries 0.17), every gap within 0.00001 mm of Blender's BVHTree.
// -> {distances: float32 (n, 1), faces}
Json::Value mesh_distance(const Json::Value& p) {
    const Volume& pts = table_param(p, "points", 3);
    const Volume& verts = table_param(p, "vertices", 3);
    const Volume& faces = table_param(p, "faces", 3);
    const double max_d = p.get("max_distance", 2.5).asDouble();   // not `far`: a macro in Windows' headers
    const std::size_t n = pts.size[1], nv = verts.size[1], nf = faces.size[1];
    std::vector<double> co(3 * nv), q(3 * n);
    dispatch(verts.type, [&](auto pixel) {
        const auto* src = static_cast<const decltype(pixel)*>(verts.buffer->data());
        parallel_blocks(3 * nv, [&](std::size_t a, std::size_t b) {
            for (std::size_t i = a; i < b; ++i) co[i] = static_cast<double>(src[i]);
        });
    });
    dispatch(pts.type, [&](auto pixel) {
        const auto* src = static_cast<const decltype(pixel)*>(pts.buffer->data());
        for (std::size_t i = 0; i < 3 * n; ++i) q[i] = static_cast<double>(src[i]);
    });
    std::vector<std::uint32_t> tri(3 * nf);
    std::atomic<bool> bad{false};
    dispatch(faces.type, [&](auto pixel) {
        const auto* src = static_cast<const decltype(pixel)*>(faces.buffer->data());
        parallel_blocks(3 * nf, [&](std::size_t a, std::size_t b) {
            for (std::size_t i = a; i < b; ++i) {
                const auto id = static_cast<long long>(src[i]);
                if (id < 0 || id >= static_cast<long long>(nv)) bad = true;
                tri[i] = static_cast<std::uint32_t>(id);
            }
        });
    });
    if (bad) throw std::runtime_error("a face index is out of range");
    auto out = Volume::table(PixelType::Float32, n, 1);
    auto* d = static_cast<float*>(out->buffer->data());
    std::fill(d, d + n, std::numeric_limits<float>::infinity());
    if (nf > 0 && n > 0 && max_d > 0) {
        // the grid: over the surface's box, cells three mean edges wide (2, 3, 4, 5 tried: 3 best)
        double lo[3], hi[3], edge = 0;
        for (int k = 0; k < 3; ++k) { lo[k] = std::numeric_limits<double>::max(); hi[k] = -lo[k]; }
        for (std::size_t i = 0; i < nv; ++i)
            for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], co[3 * i + k]); hi[k] = std::max(hi[k], co[3 * i + k]); }
        const std::size_t sample = std::min<std::size_t>(nf, 20000);
        for (std::size_t f = 0; f < sample; ++f) {
            const double* a = &co[3 * tri[3 * (f * nf / sample)]];
            const double* b = &co[3 * tri[3 * (f * nf / sample) + 1]];
            edge += std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
        }
        double h = std::clamp(3.0 * edge / static_cast<double>(sample), 0.05, max_d);
        long long dim[3];
        for (;;) {
            for (int k = 0; k < 3; ++k) dim[k] = std::max(1LL, static_cast<long long>((hi[k] - lo[k]) / h) + 1);
            if (dim[0] * dim[1] * dim[2] <= (1LL << 24)) break;
            h *= 1.25;
        }
        auto cell_of = [&](double x, int k) {
            return std::clamp(static_cast<long long>((x - lo[k]) / h), 0LL, dim[k] - 1);
        };
        // each triangle in the cells its box covers (counted, then placed: one array); the
        // cell ranges worked out once, in parallel
        const std::size_t ncell = static_cast<std::size_t>(dim[0] * dim[1] * dim[2]);
        std::vector<std::uint32_t> start(ncell + 1, 0);
        std::vector<std::int32_t> range(6 * nf);
        parallel_blocks(nf, [&](std::size_t fa, std::size_t fb) {
            for (std::size_t f = fa; f < fb; ++f)
                for (int k = 0; k < 3; ++k) {
                    double mn = co[3 * tri[3 * f] + k], mx = mn;
                    for (int j = 1; j < 3; ++j) { const double v = co[3 * tri[3 * f + j] + k]; mn = std::min(mn, v); mx = std::max(mx, v); }
                    range[6 * f + k] = static_cast<std::int32_t>(cell_of(mn, k));
                    range[6 * f + 3 + k] = static_cast<std::int32_t>(cell_of(mx, k));
                }
        });
        auto boxes = [&](std::size_t f, auto&& visit) {
            const std::int32_t* r = &range[6 * f];
            for (long long z = r[2]; z <= r[5]; ++z)
                for (long long y = r[1]; y <= r[4]; ++y)
                    for (long long x = r[0]; x <= r[3]; ++x)
                        visit(static_cast<std::size_t>((z * dim[1] + y) * dim[0] + x));
        };
        for (std::size_t f = 0; f < nf; ++f) boxes(f, [&](std::size_t c) { ++start[c + 1]; });
        for (std::size_t c = 0; c < ncell; ++c) start[c + 1] += start[c];
        std::vector<std::uint32_t> fill(start.begin(), start.end() - 1), items(start[ncell]);
        for (std::size_t f = 0; f < nf; ++f)
            boxes(f, [&](std::size_t c) { items[fill[c]++] = static_cast<std::uint32_t>(f); });
        const long long rings = static_cast<long long>(std::ceil(max_d / h)) + 1;
        const double max_d2 = max_d * max_d;
        auto query = [&](std::size_t i) {
            const double* x = &q[3 * i];
            long long c[3];
            for (int k = 0; k < 3; ++k) {
                // a point farther than `max_d` from the grid's box: nothing near it
                if (x[k] < lo[k] - max_d || x[k] > hi[k] + max_d) return;
                c[k] = static_cast<long long>(std::floor((x[k] - lo[k]) / h));
            }
            double best = max_d2;
            bool found = false;
            for (long long r = 0; r <= rings; ++r) {
                for (long long dz = -r; dz <= r; ++dz) {
                    const long long z = c[2] + dz;
                    if (z < 0 || z >= dim[2]) continue;
                    for (long long dy = -r; dy <= r; ++dy) {
                        const long long y = c[1] + dy;
                        if (y < 0 || y >= dim[1]) continue;
                        const bool shell = std::llabs(dz) == r || std::llabs(dy) == r;
                        for (long long dx = -r; dx <= r; dx += (shell ? 1 : 2 * std::max(r, 1LL))) {
                            const long long xx = c[0] + dx;
                            if (xx < 0 || xx >= dim[0]) continue;
                            const std::size_t cell = static_cast<std::size_t>((z * dim[1] + y) * dim[0] + xx);
                            // the cell's box farther than the best found (or than `max_d`): skipped
                            double gap = 0;
                            const long long cc[3] = {xx, y, z};
                            for (int k = 0; k < 3; ++k) {
                                const double c0 = lo[k] + static_cast<double>(cc[k]) * h, c1 = c0 + h;
                                const double g = x[k] < c0 ? c0 - x[k] : (x[k] > c1 ? x[k] - c1 : 0.0);
                                gap += g * g;
                            }
                            if (gap > best) {
                                if (r == 0) break;
                                continue;
                            }
                            for (std::uint32_t k = start[cell]; k < start[cell + 1]; ++k) {
                                const std::uint32_t f = items[k];
                                const double dd = dist2_triangle(x, &co[3 * tri[3 * f]], &co[3 * tri[3 * f + 1]],
                                                                 &co[3 * tri[3 * f + 2]]);
                                if (dd <= best) { best = dd; found = true; }
                            }
                            if (r == 0) break;
                        }
                    }
                }
                // a cell r + 1 rings out is at least r cells away: nothing nearer there
                const double reach = static_cast<double>(r) * h;
                if (found && best <= reach * reach) break;
                if (reach > max_d) break;
            }
            if (found) d[i] = static_cast<float>(std::sqrt(best));
        };
        parallel_blocks(n, [&](std::size_t a, std::size_t b) { for (std::size_t i = a; i < b; ++i) query(i); });
    }
    Json::Value result;
    const Volume& ref = *out;
    result["distances"] = volume_json(store().add(std::move(out)), ref);
    result["faces"] = static_cast<Json::UInt64>(nf);
    return result;
}

// buffer.new {dtype, rows, cols} -> a table the addon fills (numpy (1, rows, cols)). Here
// (2026-11-02) so that an engine without the cut (VTK 9.1) has it too: mesh.distance's
// reference mesh comes this way.
Json::Value buffer_new(const Json::Value& p) {
    const auto rows = static_cast<std::size_t>(require_int(p, "rows"));
    const auto cols = static_cast<std::size_t>(require_int(p, "cols"));
    auto v = Volume::table(pixel_type_named(require_string(p, "dtype")), rows, cols);
    const Volume& ref = *v;
    return volume_json(store().add(std::move(v)), ref);
}

}  // namespace

void add_mesh_commands(Commands& c) {
    c["mesh.threshold"] = mesh_threshold;
    c["mesh.distance"] = mesh_distance;
    c["buffer.new"] = buffer_new;
}

}  // namespace odent
