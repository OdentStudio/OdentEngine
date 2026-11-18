// Slices cut on the CPU: the addon's fallback when the GPU cannot read them
// from the volume texture (slice_views.compute_slices). The same sampling as
// its SimpleITK resample: a reference grid per plane (origin, direction,
// size, spacing; z = the thin slices across a thick one), linear, identity
// transform, `fill` outside, the result in the CT's pixel type, then the mean
// over the thin slices.

#include <itkIdentityTransform.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>

#include "commands.h"

namespace odent {

namespace {

std::array<double, 3> vec3(const Json::Value& p, const char* key) {
    const auto& v = require(p, key);
    if (!v.isArray() || v.size() != 3) throw std::runtime_error(std::string("'") + key + "' must be 3 numbers");
    return {v[0].asDouble(), v[1].asDouble(), v[2].asDouble()};
}

template <class T>
std::unique_ptr<Volume> cut(const Volume& ct, const Json::Value& plane, double fill) {
    using Img = Image3<T>;
    const auto origin = vec3(plane, "origin");
    const auto spacing = vec3(plane, "spacing");
    const auto& size_v = require(plane, "size");
    const auto& dir_v = require(plane, "direction");
    if (!size_v.isArray() || size_v.size() != 3 || !dir_v.isArray() || dir_v.size() != 9)
        throw std::runtime_error("a plane needs 'size' (3) and 'direction' (9)");
    typename Img::SizeType size;
    typename Img::PointType o;
    typename Img::SpacingType sp;
    typename Img::DirectionType dir;
    for (unsigned i = 0; i < 3; ++i) {
        size[i] = size_v[i].asUInt();
        if (size[i] == 0) throw std::runtime_error("a plane's size cannot be 0");
        o[i] = origin[i];
        sp[i] = spacing[i];
        for (unsigned j = 0; j < 3; ++j) dir(i, j) = dir_v[3 * i + j].asDouble();
    }
    auto f = itk::ResampleImageFilter<Img, Img>::New();
    f->SetInput(ct.itk_view<T>());
    f->SetTransform(itk::IdentityTransform<double, 3>::New());
    f->SetInterpolator(itk::LinearInterpolateImageFunction<Img, double>::New());
    f->SetSize(size);
    f->SetOutputOrigin(o);
    f->SetOutputSpacing(sp);
    f->SetOutputDirection(dir);
    f->SetDefaultPixelValue(static_cast<T>(fill));
    f->Update();
    // the mean over the thin slices (numpy's mean: double sums), as float32
    auto tile = std::make_unique<Volume>();
    tile->type = PixelType::Float32;
    tile->size = {size[0], size[1], 1};
    tile->spacing = {sp[0], sp[1], 1.0};
    tile->origin = origin;
    tile->buffer = SharedBuffer::create(size[0] * size[1] * sizeof(float));
    const T* in = f->GetOutput()->GetBufferPointer();
    auto* out = static_cast<float*>(tile->buffer->data());
    const std::size_t n = size[0] * size[1];
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t k = 0; k < size[2]; ++k) sum += static_cast<double>(in[k * n + i]);
        out[i] = static_cast<float>(sum / static_cast<double>(size[2]));
    }
    return tile;
}

// slices.compute {volume, fill, planes: [{origin, direction, size, spacing}]}
// -> {tiles: [volume info of a float32 (ny, nx) tile, stored values]}
Json::Value slices_compute(const Json::Value& p) {
    const Volume& ct = store().get(require_int(p, "volume"));
    const double fill = require_double(p, "fill");
    const auto& planes = require(p, "planes");
    if (!planes.isArray()) throw std::runtime_error("'planes' must be a list");
    Json::Value tiles(Json::arrayValue);
    for (const auto& plane : planes) {
        auto tile = dispatch(ct.type, [&](auto pixel) { return cut<decltype(pixel)>(ct, plane, fill); });
        const Volume& ref = *tile;
        tiles.append(volume_json(store().add(std::move(tile)), ref));
    }
    Json::Value out;
    out["tiles"] = tiles;
    return out;
}

}  // namespace

void add_slice_commands(Commands& c) {
    c["slices.compute"] = slices_compute;
}

}  // namespace odent
