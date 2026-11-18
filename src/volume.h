// A 3D image the engine holds (a CT): its voxels in a shared memory block (the
// addon maps the same block as a numpy array, z, y, x), its geometry and
// metadata. ITK images are views on the block (no copy).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <itkCommand.h>
#include <itkImage.h>
#include <itkMetaDataObject.h>
#include <itkProcessObject.h>

#include "shm.h"

namespace odent {

// A long command's progress (commands.h; main.cpp writes the lines)
void report_progress(double fraction);

// ITK's ProgressEvent of a reader, filter or writer to report_progress (the CT read's
// card: image.load, image.resample_iso, image.write)
inline void watch_progress(itk::ProcessObject* process) {
    auto command = itk::CStyleCommand::New();
    command->SetCallback([](itk::Object* caller, const itk::EventObject&, void*) {
        if (auto* p = dynamic_cast<itk::ProcessObject*>(caller)) report_progress(p->GetProgress());
    });
    process->AddObserver(itk::ProgressEvent(), command);
}

// Float64 and Int64: mesh data from the addon (coordinates, polygon indices)
enum class PixelType { UInt8, Int16, UInt16, Int32, Float32, Float64, Int64 };

std::size_t pixel_size(PixelType t);
// numpy's dtype name ("int16") and SimpleITK's name ("16-bit signed integer"),
// which the addon checks today
const char* numpy_dtype(PixelType t);
const char* sitk_name(PixelType t);
PixelType pixel_type_named(const std::string& numpy_dtype);

template <class T> struct PixelTypeOf;
template <> struct PixelTypeOf<std::uint8_t> { static constexpr PixelType value = PixelType::UInt8; };
template <> struct PixelTypeOf<std::int16_t> { static constexpr PixelType value = PixelType::Int16; };
template <> struct PixelTypeOf<std::uint16_t> { static constexpr PixelType value = PixelType::UInt16; };
template <> struct PixelTypeOf<std::int32_t> { static constexpr PixelType value = PixelType::Int32; };
template <> struct PixelTypeOf<float> { static constexpr PixelType value = PixelType::Float32; };
template <> struct PixelTypeOf<double> { static constexpr PixelType value = PixelType::Float64; };
template <> struct PixelTypeOf<std::int64_t> { static constexpr PixelType value = PixelType::Int64; };

// Call f(T{}) with the C++ type of `t`.
template <class F>
decltype(auto) dispatch(PixelType t, F&& f) {
    switch (t) {
        case PixelType::UInt8: return f(std::uint8_t{});
        case PixelType::Int16: return f(std::int16_t{});
        case PixelType::UInt16: return f(std::uint16_t{});
        case PixelType::Int32: return f(std::int32_t{});
        case PixelType::Float32: return f(float{});
        case PixelType::Float64: return f(double{});
        case PixelType::Int64: return f(std::int64_t{});
    }
    throw std::logic_error("unknown pixel type");
}

template <class T>
using Image3 = itk::Image<T, 3>;

struct Volume {
    PixelType type = PixelType::Int16;
    std::array<std::size_t, 3> size{};       // x, y, z
    std::array<double, 3> spacing{1, 1, 1};
    std::array<double, 3> origin{};
    std::array<double, 9> direction{1, 0, 0, 0, 1, 0, 0, 0, 1};  // row major
    std::map<std::string, std::string> metadata;
    std::unique_ptr<SharedBuffer> buffer;

    std::size_t count() const { return size[0] * size[1] * size[2]; }

    // An ITK image over the shared block (it does not own the memory).
    template <class T>
    typename Image3<T>::Pointer itk_view() const {
        if (PixelTypeOf<T>::value != type) throw std::logic_error("itk_view: wrong pixel type");
        auto img = Image3<T>::New();
        typename Image3<T>::RegionType region;
        typename Image3<T>::SizeType sz;
        typename Image3<T>::DirectionType dir;
        for (unsigned i = 0; i < 3; ++i) {
            sz[i] = size[i];
            for (unsigned j = 0; j < 3; ++j) dir(i, j) = direction[3 * i + j];
        }
        region.SetSize(sz);
        img->SetRegions(region);
        img->SetSpacing(spacing.data());
        img->SetOrigin(origin.data());
        img->SetDirection(dir);
        img->GetPixelContainer()->SetImportPointer(static_cast<T*>(buffer->data()), count(), false);
        auto& dict = img->GetMetaDataDictionary();
        for (const auto& [k, v] : metadata) itk::EncapsulateMetaData<std::string>(dict, k, v);
        return img;
    }

    // A new volume holding a copy of an ITK image (its voxels go to a new block).
    template <class T>
    static std::unique_ptr<Volume> from_itk(const Image3<T>* img) {
        auto v = geometry_of<T>(img);
        v->buffer = SharedBuffer::create(v->count() * sizeof(T));
        std::copy(img->GetBufferPointer(), img->GetBufferPointer() + v->count(),
                  static_cast<T*>(v->buffer->data()));
        return v;
    }

    // A new volume made by an ITK reader (file or series) or filter (resample)
    // straight into its block: no second copy of the voxels (the copy in from_itk
    // was a whole CT more at the peak of a load). It must not have run yet.
    // `fill` (optional) may put the voxels in the block itself instead of the
    // reader's Update (the DICOM fast path); it returns false to let the reader do it.
    template <class T, class Reader>
    static std::unique_ptr<Volume> read_by(Reader* reader,
                                           const std::function<bool(T*, std::size_t)>& fill = {}) {
        reader->UpdateOutputInformation();
        auto* out = reader->GetOutput();
        const auto region = out->GetLargestPossibleRegion();
        const std::size_t n = region.GetNumberOfPixels();
        auto buffer = SharedBuffer::create(n * sizeof(T));
        // ITK keeps a buffer that is big enough (ImportImageContainer::Reserve),
        // unless the output is reset before the update (ReleaseDataBeforeUpdate)
        if (fill && fill(static_cast<T*>(buffer->data()), n)) {
            auto v = geometry_of<T>(out);   // geometry and metadata from the header read
            v->buffer = std::move(buffer);
            return v;
        }
        reader->SetReleaseDataBeforeUpdateFlag(false);
        watch_progress(reader);
        out->SetRequestedRegion(region);
        out->SetBufferedRegion(region);
        out->GetPixelContainer()->SetImportPointer(static_cast<T*>(buffer->data()), n, false);
        reader->Update();
        out = reader->GetOutput();
        if (out->GetBufferPointer() != static_cast<T*>(buffer->data()))
            return from_itk<T>(out);   // ITK allocated its own after all: copy as before
        auto v = geometry_of<T>(out);
        if (v->count() != n) throw std::runtime_error("read: the image changed size while read");
        v->buffer = std::move(buffer);
        return v;
    }

    // A volume with the geometry and metadata of an ITK image, no block yet.
    template <class T>
    static std::unique_ptr<Volume> geometry_of(const Image3<T>* img) {
        auto v = std::make_unique<Volume>();
        v->type = PixelTypeOf<T>::value;
        const auto region = img->GetLargestPossibleRegion();
        for (unsigned i = 0; i < 3; ++i) {
            v->size[i] = region.GetSize()[i];
            v->spacing[i] = img->GetSpacing()[i];
            v->origin[i] = img->GetOrigin()[i];
            for (unsigned j = 0; j < 3; ++j) v->direction[3 * i + j] = img->GetDirection()(i, j);
        }
        const auto& dict = img->GetMetaDataDictionary();
        for (const auto& key : dict.GetKeys()) {
            std::string value;
            if (key.rfind("ITK_", 0) == 0) continue;   // ITK's own notes (reader name ...), not the image's
            if (itk::ExposeMetaData<std::string>(dict, key, value)) v->metadata[key] = value;
        }
        return v;
    }

    // An empty volume with the geometry of `like` and its own block.
    static std::unique_ptr<Volume> like(const Volume& like, PixelType type);
    // A table (rows x cols) in its own block: numpy shape (1, rows, cols).
    static std::unique_ptr<Volume> table(PixelType type, std::size_t rows, std::size_t cols);
};

// The volumes the engine holds, by id.
class Store {
public:
    int add(std::unique_ptr<Volume> v);
    Volume& get(int id);
    void remove(int id);

private:
    std::map<int, std::unique_ptr<Volume>> volumes_;
    int next_ = 1;
};

Store& store();

}  // namespace odent
