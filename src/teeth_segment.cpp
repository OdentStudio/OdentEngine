// segment-teeth: FastTeeth's tooth segmentation inside odent_engine (the same
// binary, run as its own process: the fine pass takes gigabytes, and running out
// of memory there must not take down the engine that holds the CT).
//
// A port of github.com/OdentStudio/fastteeth (fastteeth/pipeline.py, io.py,
// infer.py, mesh.py), step for step: its CLAUDE.md has the reasons and the
// measurements behind every number here, and its Python run is the reference the
// results are compared with (bit-identical meshes). The models and
// pipeline.json are FastTeeth's, unchanged (addon/resources/teeth).
//
//   odent_engine segment-teeth <CT file> --out <folder> [--models <folder>]
//                [--threads N] [--tiled] [--crowns] [--box x0 x1 y0 y1 z0 z1] [--window-only]
//
// ODent's own options, not FastTeeth's:
// --crowns: the fine pass on a band of the window about the bite only, the crowns the
//   scans are registered on (the AI align; see crown_band below).
// --box: only this box of the CT, as fractions of the input's image axes (ODent AI Teeth
//   Seg's red box: the whole teeth in it).
// --window-only: the coarse stage only (a few seconds): result.json's window_fractions,
//   the fine window (the teeth with their roots, the margin) as fractions of the input's
//   image axes (where ODent AI Teeth Seg's red box starts); no meshes, exit 0.
//
// Output: upper.stl, lower.stl, result.json. Exit codes as FastTeeth's: 0 done,
// 2 input not readable, 3 no teeth found, 4 internal error.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkIdentityTransform.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkResampleImageFilter.h>

#include <onnxruntime_cxx_api.h>

#include <vtkCellArray.h>
#include <vtkCellArrayIterator.h>
#include <vtkFlyingEdges3D.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkReverseSense.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTriangle.h>
#include <vtk_jsoncpp.h>

namespace fs = std::filesystem;

namespace {

using Image = itk::Image<float, 3>;
constexpr int OK = 0, ERR_INPUT = 2, ERR_NO_TEETH = 3, ERR_INTERNAL = 4;

// pipeline.py's constants (the reasons and measurements are there): a single pass
// when the window fits 0.5 GB per million voxels into 45% of the memory (0.73
// until 2026-10-25 sent a 16 GB machine to tiles: 25 s against 12 s)
constexpr double GB_PER_MVOXEL = 0.5;
constexpr double RAM_SHARE = 0.45;
constexpr std::array<int, 3> FINE_PATCH = {128, 160, 192};   // z, y, x
constexpr double BYTES_PER_VOXEL_TILED = 16.0;
constexpr double SANE_SPACING_LO = 0.05, SANE_SPACING_HI = 0.8;
constexpr int DIVISOR = 8;   // the network halves three times (infer.py)

struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

double py_round(double x, int digits) {   // round(x, n) as Python prints it (the log)
    const double f = std::pow(10.0, digits);
    return std::nearbyint(x * f) / f;
}

double total_ram_gb() {
#ifdef _WIN32
    MEMORYSTATUSEX m;
    m.dwLength = sizeof(m);
    GlobalMemoryStatusEx(&m);
    return double(m.ullTotalPhys) / double(1ull << 30);
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page <= 0) return 8.0;   // FastTeeth's cautious guess
    return double(pages) * double(page) / double(1ull << 30);
#endif
}

// --- an image's grid --------------------------------------------------------------

std::array<int, 3> size_of(const Image* img) {
    const auto s = img->GetLargestPossibleRegion().GetSize();
    return {int(s[0]), int(s[1]), int(s[2])};
}

std::array<double, 3> physical(const Image* img, double i, double j, double k) {
    itk::ContinuousIndex<double, 3> c;
    c[0] = i; c[1] = j; c[2] = k;
    itk::Point<double, 3> p;
    img->TransformContinuousIndexToPhysicalPoint(c, p);
    return {p[0], p[1], p[2]};
}

std::array<long, 3> to_index(const Image* img, const std::array<double, 3>& p) {
    itk::Point<double, 3> pt;
    for (int i = 0; i < 3; ++i) pt[i] = p[i];
    const auto idx = img->TransformPhysicalPointToIndex(pt);   // rounded, as SimpleITK
    return {long(idx[0]), long(idx[1]), long(idx[2])};
}

// io.to_orientation: sitk.DICOMOrient(img, "RPI"). Each axis of the result is the
// input axis closest to R (-x), P (+y), I (-z) in LPS, flipped when it runs the
// other way; the voxels are only reordered, their physical places stay.
Image::Pointer orient_rpi(const Image* in) {
    const auto D = in->GetDirection();
    const std::array<std::array<double, 3>, 3> target = {{{-1, 0, 0}, {0, 1, 0}, {0, 0, -1}}};
    std::array<int, 3> from{};
    std::array<bool, 3> flip{};
    std::array<bool, 3> used{};
    for (int k = 0; k < 3; ++k) {
        int best = -1;
        double best_dot = -1.0, signed_dot = 0.0;
        for (int j = 0; j < 3; ++j) {
            if (used[j]) continue;
            double dot = 0.0;
            for (int r = 0; r < 3; ++r) dot += D[r][j] * target[k][r];
            if (std::abs(dot) > best_dot) {
                best_dot = std::abs(dot);
                best = j;
                signed_dot = dot;
            }
        }
        used[best] = true;
        from[k] = best;
        flip[k] = signed_dot < 0.0;
    }
    const auto n = size_of(in);
    const auto sp = in->GetSpacing();
    Image::Pointer out = Image::New();
    Image::SizeType size;
    Image::SpacingType spacing;
    Image::DirectionType dir;
    for (int k = 0; k < 3; ++k) {
        size[k] = n[from[k]];
        spacing[k] = sp[from[k]];
        for (int r = 0; r < 3; ++r) dir[r][k] = D[r][from[k]] * (flip[k] ? -1.0 : 1.0);
    }
    // the new first voxel: the input voxel the flips start from
    std::array<double, 3> start{};
    for (int k = 0; k < 3; ++k) start[from[k]] = flip[k] ? n[from[k]] - 1 : 0;
    const auto o = physical(in, start[0], start[1], start[2]);
    Image::PointType origin;
    for (int i = 0; i < 3; ++i) origin[i] = o[i];
    Image::RegionType region;
    region.SetSize(size);
    out->SetRegions(region);
    out->SetSpacing(spacing);
    out->SetDirection(dir);
    out->SetOrigin(origin);
    out->Allocate();
    const float* src = in->GetBufferPointer();
    float* dst = out->GetBufferPointer();
    const long nx = n[0], ny = n[1];
    const std::array<long, 3> stride_in = {1, nx, nx * ny};
    long q = 0;
    for (long c = 0; c < long(size[2]); ++c)
        for (long b = 0; b < long(size[1]); ++b)
            for (long a = 0; a < long(size[0]); ++a, ++q) {
                const std::array<long, 3> out_idx = {a, b, c};
                long p = 0;
                for (int k = 0; k < 3; ++k) {
                    const long i = flip[k] ? long(size[k]) - 1 - out_idx[k] : out_idx[k];
                    p += i * stride_in[from[k]];
                }
                dst[q] = src[p];
            }
    return out;
}

// io.resample: isotropic `spacing`, linear, the same origin and direction, 0 outside
Image::Pointer resample(const Image* img, double spacing) {
    const auto n = size_of(img);
    const auto sp = img->GetSpacing();
    Image::SizeType size;
    for (int i = 0; i < 3; ++i)
        size[i] = std::max<long>(1, long(std::nearbyint(n[i] * sp[i] / spacing)));
    using Resampler = itk::ResampleImageFilter<Image, Image>;
    auto f = Resampler::New();
    f->SetInput(img);
    f->SetTransform(itk::IdentityTransform<double, 3>::New());
    f->SetInterpolator(itk::LinearInterpolateImageFunction<Image, double>::New());
    f->SetSize(size);
    f->SetOutputOrigin(img->GetOrigin());
    Image::SpacingType s;
    s.Fill(spacing);
    f->SetOutputSpacing(s);
    f->SetOutputDirection(img->GetDirection());
    f->SetDefaultPixelValue(0.0f);
    f->Update();
    return f->GetOutput();
}

// img[a0:b0+1, a1:b1+1, a2:b2+1] as SimpleITK slices it: the origin moves to the box
Image::Pointer crop(const Image* img, const std::array<long, 3>& a, const std::array<long, 3>& b) {
    Image::RegionType region;
    Image::IndexType start;
    Image::SizeType size;
    for (int i = 0; i < 3; ++i) {
        start[i] = a[i];
        size[i] = b[i] - a[i] + 1;
    }
    region.SetIndex(start);
    region.SetSize(size);
    auto f = itk::RegionOfInterestImageFilter<Image, Image>::New();
    f->SetInput(img);
    f->SetRegionOfInterest(region);
    f->Update();
    return f->GetOutput();
}

// np.rint(...).astype(int16), then pipeline._normalize: clip to -1000..3000, /4000
std::vector<float> normalized(const Image* img) {
    const auto n = size_of(img);
    const std::size_t count = std::size_t(n[0]) * n[1] * n[2];
    std::vector<float> out(count);
    const float* p = img->GetBufferPointer();
    for (std::size_t i = 0; i < count; ++i) {
        float v = float(int16_t(std::nearbyint(p[i])));   // rint: ties to even, as numpy
        v = std::min(3000.0f, std::max(-1000.0f, v));
        out[i] = (v + 1000.0f) / 4000.0f;
    }
    return out;
}

// --- the networks (infer.py) ----------------------------------------------------------

struct Net {
    Ort::Session session{nullptr};
    Net(Ort::Env& env, const fs::path& model, int threads) {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (threads > 0) opts.SetIntraOpNumThreads(threads);
        session = Ort::Session(env, model.c_str(), opts);
    }
    // logits (C, d, h, w) of a (d, h, w) volume whose sides are multiples of 8
    std::vector<float> run(const std::vector<float>& vol, const std::array<long, 3>& dhw, int& classes) {
        const std::array<int64_t, 5> shape = {1, 1, dhw[0], dhw[1], dhw[2]};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(vol.data()), vol.size(),
                                                     shape.data(), shape.size());
        const char* in_names[] = {"input"};
        const char* out_names[] = {"logits"};
        auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
        const auto oshape = out[0].GetTensorTypeAndShapeInfo().GetShape();
        classes = int(oshape[1]);
        const float* data = out[0].GetTensorData<float>();
        return std::vector<float>(data, data + std::size_t(classes) * dhw[0] * dhw[1] * dhw[2]);
    }
};

// _pad_to_divisor: edge padding at the end of each axis
std::vector<float> pad_edge(const std::vector<float>& v, const std::array<long, 3>& n,
                            const std::array<long, 3>& m) {
    if (n == m) return v;
    std::vector<float> out(std::size_t(m[0]) * m[1] * m[2]);
    for (long z = 0; z < m[0]; ++z)
        for (long y = 0; y < m[1]; ++y)
            for (long x = 0; x < m[2]; ++x) {
                const long sz = std::min(z, n[0] - 1), sy = std::min(y, n[1] - 1), sx = std::min(x, n[2] - 1);
                out[(std::size_t(z) * m[1] + y) * m[2] + x] = v[(std::size_t(sz) * n[1] + sy) * n[2] + sx];
            }
    return out;
}

long up8(long n) { return n + ((DIVISOR - n % DIVISOR) % DIVISOR); }

// run_whole + argmax(0): labels (d, h, w) of a (d, h, w) volume
std::vector<uint8_t> argmax_of(const std::vector<float>& logits, int classes,
                               const std::array<long, 3>& m, const std::array<long, 3>& n) {
    const std::size_t plane = std::size_t(m[0]) * m[1] * m[2];
    std::vector<uint8_t> out(std::size_t(n[0]) * n[1] * n[2]);
    for (long z = 0; z < n[0]; ++z)
        for (long y = 0; y < n[1]; ++y)
            for (long x = 0; x < n[2]; ++x) {
                const std::size_t p = (std::size_t(z) * m[1] + y) * m[2] + x;
                int best = 0;
                float bv = logits[p];
                for (int c = 1; c < classes; ++c)     // the first of equal ones, as numpy
                    if (logits[c * plane + p] > bv) {
                        bv = logits[c * plane + p];
                        best = c;
                    }
                out[(std::size_t(z) * n[1] + y) * n[2] + x] = uint8_t(best);
            }
    return out;
}

std::vector<uint8_t> run_whole(Net& net, const std::vector<float>& vol, const std::array<long, 3>& n) {
    const std::array<long, 3> m = {up8(n[0]), up8(n[1]), up8(n[2])};
    int classes = 0;
    const auto logits = net.run(pad_edge(vol, n, m), m, classes);
    return argmax_of(logits, classes, m, n);
}

// run_tiled: patches with a 0.25 overlap, logits summed and averaged, then argmax
std::vector<uint8_t> run_tiled(Net& net, const std::vector<float>& vol, const std::array<long, 3>& n) {
    const std::array<long, 3> m = {up8(n[0]), up8(n[1]), up8(n[2])};
    const auto padded = pad_edge(vol, n, m);
    std::array<long, 3> patch{}, step{};
    std::array<std::vector<long>, 3> starts;
    for (int i = 0; i < 3; ++i) {
        long p = std::min<long>(FINE_PATCH[i], m[i]);
        p = p - p % DIVISOR;
        if (p == 0) p = DIVISOR;
        patch[i] = p;
        step[i] = std::max<long>(DIVISOR, long(p * (1 - 0.25)));
        std::vector<long> s;
        for (long v = 0; v < std::max<long>(1, m[i] - p + 1); v += step[i]) s.push_back(v);
        s.push_back(std::max<long>(0, m[i] - p));
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
        starts[i] = s;
    }
    const std::size_t plane = std::size_t(m[0]) * m[1] * m[2];
    std::vector<float> acc;
    std::vector<float> count(plane, 0.0f);
    int classes = 0;
    std::vector<float> tile(std::size_t(patch[0]) * patch[1] * patch[2]);
    for (long z0 : starts[0])
        for (long y0 : starts[1])
            for (long x0 : starts[2]) {
                for (long z = 0; z < patch[0]; ++z)
                    for (long y = 0; y < patch[1]; ++y)
                        std::memcpy(&tile[(std::size_t(z) * patch[1] + y) * patch[2]],
                                    &padded[(std::size_t(z0 + z) * m[1] + y0 + y) * m[2] + x0],
                                    sizeof(float) * patch[2]);
                const auto out = net.run(tile, patch, classes);
                if (acc.empty()) acc.assign(std::size_t(classes) * plane, 0.0f);
                const std::size_t tplane = tile.size();
                for (long z = 0; z < patch[0]; ++z)
                    for (long y = 0; y < patch[1]; ++y)
                        for (long x = 0; x < patch[2]; ++x) {
                            const std::size_t p = (std::size_t(z0 + z) * m[1] + y0 + y) * m[2] + x0 + x;
                            const std::size_t t = (std::size_t(z) * patch[1] + y) * patch[2] + x;
                            for (int c = 0; c < classes; ++c) acc[c * plane + p] += out[c * tplane + t];
                            count[p] += 1.0f;
                        }
            }
    for (int c = 0; c < classes; ++c)
        for (std::size_t p = 0; p < plane; ++p) acc[c * plane + p] /= std::max(count[p], 1.0f);
    return argmax_of(acc, classes, m, n);
}

// --- masks ---------------------------------------------------------------------------
// Arrays are (d, h, w) = (z, y, x), x fastest: an ITK image's buffer order.

struct Components {
    std::vector<int> label;                         // 0 background, 1.. in raster order
    std::vector<long> size;                         // per label (index 0 unused)
    std::vector<std::array<double, 3>> index_sum;   // per label: sum of (x, y, z) indices
};

// sitk.ConnectedComponent (face connectivity, its default)
Components components(const std::vector<uint8_t>& mask, const std::array<long, 3>& n) {
    Components c;
    c.label.assign(mask.size(), 0);
    c.size.push_back(0);
    c.index_sum.push_back({0, 0, 0});
    std::vector<std::size_t> stack;
    const long nz = n[0], ny = n[1], nx = n[2];
    for (std::size_t s = 0; s < mask.size(); ++s) {
        if (!mask[s] || c.label[s]) continue;
        const int id = int(c.size.size());
        c.size.push_back(0);
        c.index_sum.push_back({0, 0, 0});
        c.label[s] = id;
        stack.push_back(s);
        while (!stack.empty()) {
            const std::size_t p = stack.back();
            stack.pop_back();
            const long x = long(p % nx), y = long((p / nx) % ny), z = long(p / (std::size_t(nx) * ny));
            c.size[id] += 1;
            c.index_sum[id][0] += x;
            c.index_sum[id][1] += y;
            c.index_sum[id][2] += z;
            const long nb[6][3] = {{x - 1, y, z}, {x + 1, y, z}, {x, y - 1, z},
                                   {x, y + 1, z}, {x, y, z - 1}, {x, y, z + 1}};
            for (const auto& q : nb) {
                if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= nx || q[1] >= ny || q[2] >= nz) continue;
                const std::size_t qi = (std::size_t(q[2]) * ny + q[1]) * nx + q[0];
                if (mask[qi] && !c.label[qi]) {
                    c.label[qi] = id;
                    stack.push_back(qi);
                }
            }
        }
    }
    return c;
}

double voxel_cm3(const Image* img) {
    const auto s = img->GetSpacing();
    return s[0] * s[1] * s[2] / 1000.0;
}

// pipeline._drop_far_specks: drop detections BOTH small (a share) AND far from the
// main mass; returns (count dropped, cm3 dropped)
std::pair<int, double> drop_far_specks(std::vector<uint8_t>& mask, const std::array<long, 3>& n,
                                       const Image* ref, double max_share, double min_dist) {
    const auto c = components(mask, n);
    const int labels = int(c.size.size()) - 1;
    if (labels <= 1) return {0, 0.0};
    double total = 0;
    for (int l = 1; l <= labels; ++l) total += double(c.size[l]);
    std::vector<std::array<double, 3>> cent(labels + 1);
    int big = 1;
    for (int l = 1; l <= labels; ++l) {
        const auto& s = c.index_sum[l];
        cent[l] = physical(ref, s[0] / c.size[l], s[1] / c.size[l], s[2] / c.size[l]);
        if (c.size[l] > c.size[big]) big = l;
    }
    std::vector<bool> keep(labels + 1, true);
    int dropped = 0;
    long dropped_vox = 0;
    for (int l = 1; l <= labels; ++l) {
        const double d = std::sqrt(std::pow(cent[l][0] - cent[big][0], 2) + std::pow(cent[l][1] - cent[big][1], 2) +
                                   std::pow(cent[l][2] - cent[big][2], 2));
        if (double(c.size[l]) / total < max_share && d > min_dist) {
            keep[l] = false;
            ++dropped;
            dropped_vox += c.size[l];
        }
    }
    for (std::size_t p = 0; p < mask.size(); ++p)
        if (mask[p] && !keep[c.label[p]]) mask[p] = 0;
    return {dropped, double(dropped_vox) * voxel_cm3(ref)};
}

// mesh.drop_small_components: every component of at least min_cm3 stays
void drop_small_components(std::vector<uint8_t>& mask, const std::array<long, 3>& n, const Image* ref,
                           double min_cm3) {
    if (min_cm3 <= 0) return;
    const long min_size = long(min_cm3 / voxel_cm3(ref));
    const auto c = components(mask, n);
    for (std::size_t p = 0; p < mask.size(); ++p)
        if (mask[p] && c.size[c.label[p]] < min_size) mask[p] = 0;
}

// the (z, y, x) index box of a mask
bool mask_box(const std::vector<uint8_t>& mask, const std::array<long, 3>& n, std::array<long, 3>& lo,
              std::array<long, 3>& hi) {
    lo = {n[0], n[1], n[2]};
    hi = {-1, -1, -1};
    for (long z = 0; z < n[0]; ++z)
        for (long y = 0; y < n[1]; ++y)
            for (long x = 0; x < n[2]; ++x)
                if (mask[(std::size_t(z) * n[1] + y) * n[2] + x]) {
                    const long v[3] = {z, y, x};
                    for (int i = 0; i < 3; ++i) {
                        lo[i] = std::min(lo[i], v[i]);
                        hi[i] = std::max(hi[i], v[i]);
                    }
                }
    return hi[0] >= 0;
}

// pipeline._physical_bbox: all eight corners (a direction may flip axes), then min/max
void physical_bbox(const Image* img, const std::array<long, 3>& lo, const std::array<long, 3>& hi,
                   std::array<double, 3>& pmin, std::array<double, 3>& pmax) {
    pmin = {1e300, 1e300, 1e300};
    pmax = {-1e300, -1e300, -1e300};
    for (long a : {lo[0], hi[0]})
        for (long b : {lo[1], hi[1]})
            for (long c : {lo[2], hi[2]}) {
                const auto p = physical(img, double(c), double(b), double(a));
                for (int i = 0; i < 3; ++i) {
                    pmin[i] = std::min(pmin[i], p[i]);
                    pmax[i] = std::max(pmax[i], p[i]);
                }
            }
}

// the index box of physical points p1, p2 in img, clipped to it (min/max per axis)
void index_box(const Image* img, const std::array<double, 3>& p1, const std::array<double, 3>& p2,
               std::array<long, 3>& a, std::array<long, 3>& b) {
    const auto c1 = to_index(img, p1), c2 = to_index(img, p2);
    const auto n = size_of(img);
    for (int i = 0; i < 3; ++i) {
        a[i] = std::clamp(std::min(c1[i], c2[i]), 0L, long(n[i] - 1));
        b[i] = std::clamp(std::max(c1[i], c2[i]), 0L, long(n[i] - 1));
    }
}

long count(const std::vector<uint8_t>& m) {
    return long(std::count_if(m.begin(), m.end(), [](uint8_t v) { return v != 0; }));
}

std::array<long, 3> dhw(const Image* img) {
    const auto n = size_of(img);
    return {n[2], n[1], n[0]};
}

// --- crowns only (--crowns) ------------------------------------------------------------
// ODent's AI align registers the scans, which hold the crowns only, on the arches: the
// roots are not needed, and on a large window they cost the most (a whole window of
// 17 M voxels went to tiles on a 12 GB computer: 126 s; a VOI the doctor drew about the
// crowns: 5 M, one pass, 14 s, the scan placed 0.02 mm from the full run's place,
// 2026-10-26). The coarse model does not tell the arches apart, so the bite is first
// guessed at the coarse mask's mean height (the crowns are the teeth's bulk), the fine
// pass runs on a band of CROWN_BAND_MM about it over the window's width and depth,
// and the fine labels give the bite (the upper arch's bottom, the lower's top): a guess
// more than CROWN_TOLERANCE_MM off runs the band again about the bite found.
constexpr double CROWN_BAND_MM = 30.0;       // the bite +- 15 mm: crowns ~11 mm, 4 mm of play
constexpr double CROWN_TOLERANCE_MM = 4.0;   // that play

// the mean physical height (z: superior +) of a mask's voxels
double mask_mean_z(const std::vector<uint8_t>& m, const std::array<long, 3>& n, const Image* img) {
    const std::size_t plane = std::size_t(n[1]) * n[2];
    double sum = 0.0, cnt = 0.0;
    for (long z = 0; z < n[0]; ++z) {
        const long c = long(std::count_if(m.begin() + z * plane, m.begin() + (z + 1) * plane,
                                          [](uint8_t v) { return v != 0; }));
        sum += double(c) * z;
        cnt += double(c);
    }
    return physical(img, (n[2] - 1) / 2.0, (n[1] - 1) / 2.0, cnt > 0 ? sum / cnt : (n[0] - 1) / 2.0)[2];
}

// the height (physical z) below which `share` of the voxels labelled `label` lie;
// false when the label has (almost) none
bool label_height(const std::vector<uint8_t>& pred, const std::array<long, 3>& n, const Image* img,
                  uint8_t label, double share, double& height) {
    const std::size_t plane = std::size_t(n[1]) * n[2];
    std::vector<long> rows(n[0], 0);
    long total = 0;
    for (long z = 0; z < n[0]; ++z) {
        rows[z] = long(std::count(pred.begin() + z * plane, pred.begin() + (z + 1) * plane, label));
        total += rows[z];
    }
    if (total < 1000) return false;
    const auto z_of = [&](double row) { return physical(img, (n[2] - 1) / 2.0, (n[1] - 1) / 2.0, row)[2]; };
    const bool up = z_of(1) > z_of(0);       // the rows from the lowest up, or from the top down
    long acc = 0;
    for (long k = 0; k < n[0]; ++k) {
        const long z = up ? k : n[0] - 1 - k;
        acc += rows[z];
        if (acc >= share * total) {
            height = z_of(double(z));
            return true;
        }
    }
    height = z_of(up ? n[0] - 1 : 0);
    return true;
}

// the window a..b (indices of img) cut to the band of CROWN_BAND_MM about height c;
// false (the window left as it is) when the band misses it
bool crown_band(const Image* img, double c, std::array<long, 3>& a, std::array<long, 3>& b) {
    auto p1 = physical(img, (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0);
    auto p2 = p1;
    p1[2] = c - CROWN_BAND_MM / 2.0;
    p2[2] = c + CROWN_BAND_MM / 2.0;
    const long k1 = to_index(img, p1)[2], k2 = to_index(img, p2)[2];
    const long lo = std::max(a[2], std::min(k1, k2)), hi = std::min(b[2], std::max(k1, k2));
    if (lo >= hi) return false;
    a[2] = lo;
    b[2] = hi;
    return true;
}

// --- meshes (mesh.py) ------------------------------------------------------------------

// mask_to_polydata: flying edges on a grid without origin and direction, then the
// direction and origin as a transform (the addon's own path); winding reversed when
// the direction flips space; consistent normals
vtkSmartPointer<vtkPolyData> mask_to_polydata(const std::vector<uint8_t>& mask, const Image* img) {
    const auto n = size_of(img);
    const auto sp = img->GetSpacing();
    vtkNew<vtkImageData> grid;
    grid->SetDimensions(n[0], n[1], n[2]);
    grid->SetSpacing(sp[0], sp[1], sp[2]);
    vtkNew<vtkFloatArray> scalars;
    scalars->SetNumberOfValues(vtkIdType(mask.size()));
    for (std::size_t i = 0; i < mask.size(); ++i) scalars->SetValue(vtkIdType(i), float(mask[i]));
    grid->GetPointData()->SetScalars(scalars);

    vtkNew<vtkFlyingEdges3D> surf;
    surf->SetInputData(grid);
    surf->SetValue(0, 0.5);
    surf->ComputeNormalsOff();

    const auto D = img->GetDirection();
    const auto o = img->GetOrigin();
    vtkNew<vtkMatrix4x4> mat;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) mat->SetElement(i, j, D[i][j]);
        mat->SetElement(i, 3, o[i]);
    }
    vtkNew<vtkTransform> t;
    t->SetMatrix(mat);
    vtkNew<vtkTransformPolyDataFilter> tf;
    tf->SetInputConnection(surf->GetOutputPort());
    tf->SetTransform(t);

    vtkNew<vtkPolyDataNormals> nrm;
    const double det = D[0][0] * (D[1][1] * D[2][2] - D[1][2] * D[2][1]) -
                       D[0][1] * (D[1][0] * D[2][2] - D[1][2] * D[2][0]) +
                       D[0][2] * (D[1][0] * D[2][1] - D[1][1] * D[2][0]);
    vtkNew<vtkReverseSense> rev;
    if (det < 0) {
        rev->SetInputConnection(tf->GetOutputPort());
        rev->ReverseCellsOn();
        rev->ReverseNormalsOff();
        nrm->SetInputConnection(rev->GetOutputPort());
    } else {
        nrm->SetInputConnection(tf->GetOutputPort());
    }
    nrm->ConsistencyOn();
    nrm->SplittingOff();
    nrm->Update();
    vtkSmartPointer<vtkPolyData> out = nrm->GetOutput();
    return out;
}

// a binary STL as vtkSTLWriter writes it (the static VTK has no IO module): facet
// normals from the triangles' points
void write_stl(vtkPolyData* pd, const fs::path& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path.string());
    char header[80] = {};
    std::strncpy(header, "odent_engine segment-teeth (binary STL)", sizeof(header) - 1);
    f.write(header, 80);
    vtkCellArray* polys = pd->GetPolys();
    vtkPoints* pts = pd->GetPoints();
    uint32_t n = 0;
    {
        auto it = vtk::TakeSmartPointer(polys->NewIterator());
        for (it->GoToFirstCell(); !it->IsDoneWithTraversal(); it->GoToNextCell()) {
            vtkIdType npts;
            const vtkIdType* ids;
            it->GetCurrentCell(npts, ids);
            if (npts == 3) ++n;
        }
    }
    f.write(reinterpret_cast<const char*>(&n), 4);
    auto it = vtk::TakeSmartPointer(polys->NewIterator());
    for (it->GoToFirstCell(); !it->IsDoneWithTraversal(); it->GoToNextCell()) {
        vtkIdType npts;
        const vtkIdType* ids;
        it->GetCurrentCell(npts, ids);
        if (npts != 3) continue;
        double v[3][3], nrm[3];
        for (int k = 0; k < 3; ++k) pts->GetPoint(ids[k], v[k]);
        vtkTriangle::ComputeNormal(v[0], v[1], v[2], nrm);
        float rec[12];
        for (int i = 0; i < 3; ++i) rec[i] = float(nrm[i]);
        for (int k = 0; k < 3; ++k)
            for (int i = 0; i < 3; ++i) rec[3 + 3 * k + i] = float(v[k][i]);
        f.write(reinterpret_cast<const char*>(rec), sizeof(rec));
        const uint16_t attr = 0;
        f.write(reinterpret_cast<const char*>(&attr), 2);
    }
}

// --- the pipeline (pipeline.run) ----------------------------------------------------------

struct Options {
    int threads = 0;
    bool tiled = false, crowns = false, window_only = false, has_box = false;
    std::array<double, 6> box{};   // x0 x1 y0 y1 z0 z1: fractions of the input's image axes
};

int run(const fs::path& input, const fs::path& out_dir, const fs::path& models, const Options& opt,
        Json::Value& log) {
    const int threads = opt.threads;
    const bool force_tiled = opt.tiled, crowns = opt.crowns;
    Json::Value cfg;
    {
        std::ifstream f(models / "pipeline.json");
        if (!f) throw std::runtime_error("pipeline.json not found in " + models.string());
        Json::CharReaderBuilder rb;
        std::string err;
        if (!Json::parseFromStream(rb, f, &cfg, &err)) throw std::runtime_error("pipeline.json: " + err);
    }
    if (cfg["orientation"].asString() != "RPI")
        throw std::runtime_error("only the RPI orientation is ported (pipeline.json)");
    fs::create_directories(out_dir);
    log["input"] = fs::absolute(input).string();
    log["output"] = fs::absolute(out_dir).string();

    // --- reading: the project's CT file (the VOI read, already cropped) ---
    if (!fs::exists(input)) throw InputError("path does not exist: " + input.string());
    Image::Pointer img;
    try {
        auto reader = itk::ImageFileReader<Image>::New();
        reader->SetFileName(input.string());
        reader->Update();
        img = reader->GetOutput();
    } catch (const itk::ExceptionObject& e) {
        throw InputError(std::string("cannot read the CT: ") + e.GetDescription());
    }
    // the input's grid, without its voxels: --window-only's fractions are of it
    auto grid = Image::New();
    grid->SetRegions(img->GetLargestPossibleRegion());
    grid->SetSpacing(img->GetSpacing());
    grid->SetOrigin(img->GetOrigin());
    grid->SetDirection(img->GetDirection());
    if (opt.has_box) {
        const auto n = size_of(img);
        std::array<long, 3> a, b;
        for (int i = 0; i < 3; ++i) {
            a[i] = std::clamp(long(std::floor(opt.box[2 * i] * n[i])), 0L, long(n[i] - 1));
            b[i] = std::clamp(long(std::ceil(opt.box[2 * i + 1] * n[i])) - 1, a[i], long(n[i] - 1));
            log["box_voxels"].append(Json::Int64(a[i]));
            log["box_voxels"].append(Json::Int64(b[i]));
        }
        img = crop(img, a, b);
    }
    img = orient_rpi(img);
    log["source"] = "file";
    const auto n0 = size_of(img);
    const auto sp0 = img->GetSpacing();
    std::array<double, 3> fov{};
    for (int i = 0; i < 3; ++i) {
        log["size"].append(n0[i]);
        log["spacing"].append(py_round(sp0[i], 4));
        fov[i] = py_round(n0[i] * sp0[i], 1);
        log["fov_mm"].append(fov[i]);
    }
    for (int i = 0; i < 3; ++i)
        if (!(SANE_SPACING_LO <= sp0[i] && sp0[i] <= SANE_SPACING_HI))
            throw InputError("voxel spacing is outside the plausible range for dental CBCT (0.05-0.8 mm): "
                             "the scan metadata is most likely corrupted");

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "odent_teeth");

    // --- stage 1: coarse ---
    auto t = std::chrono::steady_clock::now();
    auto seconds = [](std::chrono::steady_clock::time_point since) {
        return py_round(std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count(), 2);
    };
    const auto& cc = cfg["coarse"];
    const double sp_c = cc["spacing"].asDouble();
    Net coarse(env, models / "models" / "coarse.onnx", threads);
    auto ic = resample(img, sp_c);
    auto n_c = dhw(ic);
    auto teeth = run_whole(coarse, normalized(ic), n_c);
    for (auto& v : teeth) v = v > 0;

    const double trained_mm = cc.get("trained_fov_mm", 0.0).asDouble();
    const auto& speck = cc["speck_filter"];
    const double max_share = speck.get("max_share", 0.05).asDouble();
    const double min_dist = speck.get("min_dist_mm", 60.0).asDouble();
    if (trained_mm > 0 && count(teeth) > 0 && *std::max_element(fov.begin(), fov.end()) > trained_mm) {
        // a frame larger than the model ever saw: crop it to the trained size
        // around the teeth found, and look again
        auto big = teeth;
        if (speck.isObject()) drop_far_specks(big, n_c, ic, max_share, min_dist);
        std::array<long, 3> lo, hi;
        mask_box(count(big) ? big : teeth, n_c, lo, hi);
        std::array<double, 3> plo, phi, c1, c2;
        physical_bbox(ic, lo, hi, plo, phi);
        for (int i = 0; i < 3; ++i) {
            const double centre = (plo[i] + phi[i]) / 2;
            c1[i] = centre - trained_mm / 2.0;
            c2[i] = centre + trained_mm / 2.0;
        }
        std::array<long, 3> a2, b2;
        index_box(img, c1, c2, a2, b2);
        auto cropped = crop(img, a2, b2);
        ic = resample(cropped, sp_c);
        n_c = dhw(ic);
        teeth = run_whole(coarse, normalized(ic), n_c);
        for (auto& v : teeth) v = v > 0;
        const auto cn = size_of(cropped);
        const auto cs = cropped->GetSpacing();
        for (int i = 0; i < 3; ++i) log["coarse_second_pass"].append(py_round(cn[i] * cs[i], 1));
    }
    if (speck.isObject() && count(teeth) > 0) {
        const auto [dropped, dropped_cm3] = drop_far_specks(teeth, n_c, ic, max_share, min_dist);
        if (dropped) {
            log["coarse_dropped"] = dropped;
            log["coarse_dropped_cm3"] = py_round(dropped_cm3, 2);
        }
    }
    log["coarse_s"] = seconds(t);
    const double vol_cm3 = double(count(teeth)) * sp_c * sp_c * sp_c / 1000.0;
    log["coarse_volume_cm3"] = py_round(vol_cm3, 2);

    // --- window ---
    const double vmin = cfg["fallback_volume_cm3"][0].asDouble(), vmax = cfg["fallback_volume_cm3"][1].asDouble();
    const double ram = total_ram_gb();
    const double max_mvox = (ram * RAM_SHARE * double(1ull << 30)) / (BYTES_PER_VOXEL_TILED * 1e6);
    const double sp_f = cfg["fine"]["spacing"].asDouble();
    Image::Pointer sub;
    std::array<long, 3> win_a{}, win_b{};    // the window (indices of img), when there is one
    bool windowed = false;
    if (count(teeth) < 20 || !(vmin <= vol_cm3 && vol_cm3 <= vmax)) {
        log["window"] = "whole frame";
        char why[160];
        std::snprintf(why, sizeof(why), "coarse stage gave %.1f cm3, outside %.1f-%.1f", vol_cm3, vmin, vmax);
        std::string reason = why;
        sub = img;
        double whole_mvox = 1.0;
        for (int i = 0; i < 3; ++i) whole_mvox *= std::ceil(fov[i] / sp_f);
        whole_mvox /= 1e6;
        if (whole_mvox > max_mvox) {
            // bounded by memory: a cube of what fits, around the teeth (or the centre)
            const double side = std::cbrt(max_mvox * 1e6) * sp_f;
            const auto n = size_of(img);
            auto c = physical(img, (n[0] - 1) / 2.0, (n[1] - 1) / 2.0, (n[2] - 1) / 2.0);
            std::array<long, 3> lo, hi;
            if (mask_box(teeth, n_c, lo, hi)) {
                std::array<double, 3> pl, ph;
                physical_bbox(ic, lo, hi, pl, ph);
                for (int i = 0; i < 3; ++i) c[i] = (pl[i] + ph[i]) / 2;
            }
            std::array<double, 3> p1, p2;
            for (int i = 0; i < 3; ++i) {
                p1[i] = c[i] - side / 2;
                p2[i] = c[i] + side / 2;
            }
            std::array<long, 3> a0, b0;
            index_box(img, p1, p2, a0, b0);
            sub = crop(img, a0, b0);
            char more[200];
            std::snprintf(more, sizeof(more),
                          "; the whole frame is %.0f M voxels against a memory cap of %.0f — cropped to %.0f mm "
                          "around the centre", whole_mvox, max_mvox, side);
            reason += more;
            log["warning"] = "the frame does not fit in memory as a whole, the result may be incomplete";
        }
        log["window_reason"] = reason;
    } else {
        std::array<long, 3> lo, hi;
        mask_box(teeth, n_c, lo, hi);
        std::array<double, 3> plo, phi;
        physical_bbox(ic, lo, hi, plo, phi);
        const double m = cfg["fine"]["margin_mm"].asDouble();
        std::array<double, 3> p1, p2;
        for (int i = 0; i < 3; ++i) {
            p1[i] = plo[i] - m;
            p2[i] = phi[i] + m;
        }
        std::array<long, 3> a, b;
        index_box(img, p1, p2, a, b);
        sub = crop(img, a, b);
        for (int i = 0; i < 3; ++i) log["window"].append(py_round((b[i] - a[i] + 1) * sp0[i], 1));
        win_a = a;
        win_b = b;
        windowed = true;
    }
    if (opt.window_only) {
        // the window as fractions (x0, x1, y0, y1, z0, z1) of the input's image axes: its
        // corner voxels' outer sides, through physical space (img is reoriented, maybe boxed)
        std::array<double, 6> f = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
        if (windowed) {
            const auto n = size_of(grid);
            f = {1e300, -1e300, 1e300, -1e300, 1e300, -1e300};
            for (double x : {win_a[0] - 0.5, win_b[0] + 0.5})
                for (double y : {win_a[1] - 0.5, win_b[1] + 0.5})
                    for (double z : {win_a[2] - 0.5, win_b[2] + 0.5}) {
                        const auto p = physical(img, x, y, z);
                        itk::Point<double, 3> pt;
                        for (int i = 0; i < 3; ++i) pt[i] = p[i];
                        itk::ContinuousIndex<double, 3> ci;
                        grid->TransformPhysicalPointToContinuousIndex(pt, ci);
                        for (int i = 0; i < 3; ++i) {
                            const double v = std::clamp((ci[i] + 0.5) / n[i], 0.0, 1.0);
                            f[2 * i] = std::min(f[2 * i], v);
                            f[2 * i + 1] = std::max(f[2 * i + 1], v);
                        }
                    }
        }
        for (double v : f) log["window_fractions"].append(py_round(v, 4));
        log["status"] = "window";
        return OK;
    }
    // --crowns: the window cut to the band about the bite guessed (crown_band)
    const bool by_band = crowns && windowed;
    double band_at = 0.0;
    if (by_band) {
        band_at = mask_mean_z(teeth, n_c, ic);
        auto a = win_a, b = win_b;
        log["crowns"]["guess_mm"] = py_round(band_at, 1);
        if (crown_band(img, band_at, a, b)) {
            sub = crop(img, a, b);
            log["crowns"]["band_mm"] = py_round((b[2] - a[2] + 1) * sp0[2], 1);
        }
    }

    // --- stage 2: fine ---
    t = std::chrono::steady_clock::now();
    const double budget_mvox = force_tiled ? 0.0 : (ram * RAM_SHARE) / GB_PER_MVOXEL;
    log["ram_gb"] = py_round(ram, 1);
    Net fine(env, models / "models" / "fine.onnx", threads);
    Image::Pointer fi;
    std::array<long, 3> n_f{};
    std::vector<uint8_t> pred;
    auto fine_pass = [&](Image::Pointer s) {
        fi = resample(s, sp_f);
        // the whole CT is not needed any more: freed before the network's gigabytes
        // (a test case: a 556 MB CT held twice as float peaked 6.6 GB for a 12 M window, 4.9 freed);
        // by band it is kept for a second band (a small window: about 0.5 GB more)
        s = nullptr;
        sub = nullptr;
        if (!by_band) {
            ic = nullptr;
            img = nullptr;
        }
        n_f = dhw(fi);
        const double mvox = double(n_f[0]) * n_f[1] * n_f[2] / 1e6;
        log["fine_voxels_M"] = py_round(mvox, 2);
        const bool whole = mvox * 1e6 <= double(long(budget_mvox * 1e6));
        log["fine_mode"] = whole ? "single pass" : "tiled";
        const auto vol = normalized(fi);
        pred = whole ? run_whole(fine, vol, n_f) : run_tiled(fine, vol, n_f);
    };
    fine_pass(sub);
    if (by_band) {
        // the bite the fine labels give (1 upper, 2 lower): the upper arch's bottom, the lower's top
        double up_bottom = 0.0, low_top = 0.0;
        const bool has_up = label_height(pred, n_f, fi, 1, 0.01, up_bottom);
        const bool has_low = label_height(pred, n_f, fi, 2, 0.99, low_top);
        if (has_up || has_low) {
            const double bite = has_up && has_low ? (up_bottom + low_top) / 2.0 : (has_up ? up_bottom : low_top);
            log["crowns"]["bite_mm"] = py_round(bite, 1);
            if (std::fabs(bite - band_at) > CROWN_TOLERANCE_MM) {
                auto a = win_a, b = win_b;
                if (crown_band(img, bite, a, b)) {
                    log["crowns"]["again_about_the_bite"] = true;
                    fine_pass(crop(img, a, b));
                }
            }
        }
        ic = nullptr;
        img = nullptr;
    }
    log["fine_s"] = seconds(t);

    // --- stage 3: meshes ---
    t = std::chrono::steady_clock::now();
    const double min_comp = cfg["mesh"]["min_component_cm3"].asDouble();
    if (cfg["mesh"].get("taubin_iterations", 0).asInt() > 0)
        throw std::runtime_error("Taubin smoothing is not ported (pipeline.json: taubin_iterations)");
    int found = 0;
    log["arches"] = Json::Value(Json::objectValue);
    const std::pair<int, const char*> arches[] = {{1, "upper"}, {2, "lower"}};
    for (const auto& [val, name] : arches) {
        std::vector<uint8_t> m(pred.size());
        for (std::size_t i = 0; i < pred.size(); ++i) m[i] = pred[i] == val;
        const double cm3 = double(count(m)) * sp_f * sp_f * sp_f / 1000.0;
        if (cm3 < 0.5) {
            log["arches"][name]["found"] = false;
            log["arches"][name]["reason"] = "arch not in frame or not found";
            continue;
        }
        drop_small_components(m, n_f, fi, min_comp);
        auto pd = mask_to_polydata(m, fi);
        const fs::path p = out_dir / (std::string(name) + ".stl");
        write_stl(pd, p);
        log["arches"][name]["found"] = true;
        log["arches"][name]["volume_cm3"] = py_round(cm3, 2);
        log["arches"][name]["vertices"] = Json::Int64(pd->GetNumberOfPoints());
        log["arches"][name]["stl"] = fs::absolute(p).string();
        ++found;
    }
    log["mesh_s"] = seconds(t);
    log["status"] = found ? "ok" : "no_teeth";
    return found ? OK : ERR_NO_TEETH;
}

}  // namespace

namespace odent {

// odent_engine segment-teeth ... (main.cpp hands the arguments after the mode)
int segment_teeth_main(int argc, char** argv) {
    std::string input, out, models;
    Options opt;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--models" && i + 1 < argc) models = argv[++i];
        else if (a == "--threads" && i + 1 < argc) opt.threads = std::atoi(argv[++i]);
        else if (a == "--tiled") opt.tiled = true;
        else if (a == "--crowns") opt.crowns = true;
        else if (a == "--window-only") opt.window_only = true;
        else if (a == "--box" && i + 6 < argc) {
            for (int k = 0; k < 6; ++k) opt.box[k] = std::atof(argv[++i]);
            opt.has_box = true;
        }
        else if (a == "--progress" || a == "--quiet") continue;   // FastTeeth's flags: accepted
        else if (input.empty()) input = a;
    }
    if (input.empty() || out.empty()) {
        std::cerr << "usage: odent_engine segment-teeth <CT file> --out <folder> [--models <folder>] "
                     "[--threads N] [--tiled] [--crowns] [--box x0 x1 y0 y1 z0 z1] [--window-only]\n";
        return ERR_INPUT;
    }
    for (int i = 0; opt.has_box && i < 3; ++i)
        if (!(0.0 <= opt.box[2 * i] && opt.box[2 * i] < opt.box[2 * i + 1] && opt.box[2 * i + 1] <= 1.0)) {
            std::cerr << "--box: six fractions 0 <= x0 < x1 <= 1 (y, z alike)\n";
            return ERR_INPUT;
        }
    if (models.empty()) models = "teeth";
    const auto t0 = std::chrono::steady_clock::now();
    Json::Value log;
    int code;
    try {
        code = run(fs::u8path(input), fs::u8path(out), fs::u8path(models), opt, log);
    } catch (const InputError& e) {
        std::cerr << "cannot read input: " << e.what() << "\n";
        return ERR_INPUT;
    } catch (const std::exception& e) {
        std::cerr << "internal error: " << e.what() << "\n";
        return ERR_INTERNAL;
    }
    log["total_s"] = py_round(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 2);
    log["engine"] = "odent_engine";
    Json::StreamWriterBuilder w;
    w["indentation"] = " ";
    w["emitUTF8"] = true;
    w["precision"] = 10;   // 6.65, not 6.6500000000000004 (the values are rounded already)
    std::ofstream f(fs::u8path(out) / "result.json", std::ios::binary);
    f << Json::writeString(w, log);
    return code;
}

}  // namespace odent
