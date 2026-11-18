#include "volume.h"

namespace odent {

std::size_t pixel_size(PixelType t) {
    return dispatch(t, [](auto v) { return sizeof(v); });
}

const char* numpy_dtype(PixelType t) {
    switch (t) {
        case PixelType::UInt8: return "uint8";
        case PixelType::Int16: return "int16";
        case PixelType::UInt16: return "uint16";
        case PixelType::Int32: return "int32";
        case PixelType::Float32: return "float32";
        case PixelType::Float64: return "float64";
        case PixelType::Int64: return "int64";
    }
    return "";
}

const char* sitk_name(PixelType t) {
    switch (t) {
        case PixelType::UInt8: return "8-bit unsigned integer";
        case PixelType::Int16: return "16-bit signed integer";
        case PixelType::UInt16: return "16-bit unsigned integer";
        case PixelType::Int32: return "32-bit signed integer";
        case PixelType::Float32: return "32-bit float";
        case PixelType::Float64: return "64-bit float";
        case PixelType::Int64: return "64-bit signed integer";
    }
    return "";
}

std::unique_ptr<Volume> Volume::like(const Volume& src, PixelType type) {
    auto v = std::make_unique<Volume>();
    v->type = type;
    v->size = src.size;
    v->spacing = src.spacing;
    v->origin = src.origin;
    v->direction = src.direction;
    v->metadata = src.metadata;
    v->buffer = SharedBuffer::create(v->count() * pixel_size(type));
    return v;
}

std::unique_ptr<Volume> Volume::table(PixelType type, std::size_t rows, std::size_t cols) {
    auto v = std::make_unique<Volume>();
    v->type = type;
    v->size = {cols, rows, 1};
    v->buffer = SharedBuffer::create(rows * cols * pixel_size(type));
    return v;
}

PixelType pixel_type_named(const std::string& dtype) {
    for (PixelType t : {PixelType::UInt8, PixelType::Int16, PixelType::UInt16, PixelType::Int32,
                        PixelType::Float32, PixelType::Float64, PixelType::Int64})
        if (dtype == numpy_dtype(t)) return t;
    throw std::runtime_error("unknown dtype " + dtype);
}

int Store::add(std::unique_ptr<Volume> v) {
    const int id = next_++;
    volumes_[id] = std::move(v);
    return id;
}

Volume& Store::get(int id) {
    auto it = volumes_.find(id);
    if (it == volumes_.end()) throw std::runtime_error("no volume " + std::to_string(id));
    return *it->second;
}

void Store::remove(int id) {
    if (volumes_.erase(id) == 0) throw std::runtime_error("no volume " + std::to_string(id));
}

Store& store() {
    static Store s;
    return s;
}

}  // namespace odent
