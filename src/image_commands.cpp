// CT images: find the series of a DICOM folder, read a series or a 3D file,
// make it isotropic, store it as int16 HU, write the project's .mha.
// Each command does what the addon did with SimpleITK (named in its comment),
// with the same parameters, so the two can be compared.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <utility>
#include <type_traits>
#include <vector>
#include <limits>

#include <gdcmReader.h>
#include <gdcmScanner.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkIdentityTransform.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageIOFactory.h>
#include <itkImageSeriesReader.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>

#include "commands.h"

namespace odent {

namespace {

// A file's path in one spelling ("a//b/./c" and "a/b/c" are the same file:
// GDCM's series list and the addon's folder walk write them differently).
std::string same_path(const std::string& path) {
    std::error_code ec;
    const auto p = std::filesystem::weakly_canonical(std::filesystem::path(path), ec);
    return ec ? path : p.string();
}

// A file's identity for the header cache: its path, size and modification time
// ("" when they cannot be read: no caching then).
std::string file_key(const std::string& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) return "";
    const auto time = std::filesystem::last_write_time(path, ec);
    if (ec) return "";
    return same_path(path) + "|" + std::to_string(size) + ":" +
           std::to_string(time.time_since_epoch().count());
}

// ITK's GDCM reader reads a file's whole content to get its header (pixels
// included): a 668^3 multi-frame CBCT was read in full 4 times by one load
// (dicom.scan, image.header, the type check and the reader's own information
// step), 0.4-0.8 s each. This one does that read once per file and keeps the
// result while the file is unchanged; reading the voxels still reads them.
class CachedGDCMImageIO : public itk::GDCMImageIO {
public:
    using Self = CachedGDCMImageIO;
    using Superclass = itk::GDCMImageIO;
    using Pointer = itk::SmartPointer<Self>;
    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(CachedGDCMImageIO);

    void ReadImageInformation() override {
        const std::string key = file_key(GetFileName());
        if (!key.empty() && key == m_ReadKey) return;
        Superclass::ReadImageInformation();
        m_ReadKey = key;
    }

protected:
    CachedGDCMImageIO() { LoadPrivateTagsOn(); }

private:
    std::string m_ReadKey;
};

// The GDCM reader of a file, kept between commands (dicom.scan, image.header
// and image.load of the same file share its header read). A few files only.
CachedGDCMImageIO* dicom_io(const std::string& path) {
    static std::map<std::string, CachedGDCMImageIO::Pointer> cache;
    const std::string key = same_path(path);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second.GetPointer();
    if (cache.size() >= 8) cache.erase(cache.begin());
    auto io = CachedGDCMImageIO::New();
    io->SetFileName(path);
    return cache.emplace(key, io).first->second.GetPointer();
}

bool fast_routes() { return std::getenv("ODENT_ENGINE_NO_FAST_DICOM") == nullptr; }

// The IO for a 3D file: the cached GDCM reader for DICOM, ITK's choice otherwise.
// A DICOM file is recognised by GDCM's own check (its first bytes): ITK's
// (GDCMImageIO::CanReadFile) reads the whole file, 0.40 s for 596 MB.
itk::ImageIOBase::Pointer image_io(const std::string& path) {
    if (fast_routes()) {
        gdcm::Reader reader;
        reader.SetFileName(path.c_str());
        if (reader.CanRead()) return dicom_io(path);
    }
    auto io = itk::ImageIOFactory::CreateImageIO(path.c_str(),
                                                  itk::ImageIOFactory::IOFileModeEnum::ReadMode);
    if (io && dynamic_cast<itk::GDCMImageIO*>(io.GetPointer())) return dicom_io(path);
    return io;
}

}  // namespace

const Json::Value& require(const Json::Value& params, const char* key) {
    if (!params.isObject() || !params.isMember(key))
        throw std::runtime_error(std::string("missing parameter '") + key + "'");
    return params[key];
}

std::string require_string(const Json::Value& params, const char* key) {
    const auto& v = require(params, key);
    if (!v.isString()) throw std::runtime_error(std::string("'") + key + "' must be a string");
    return v.asString();
}

int require_int(const Json::Value& params, const char* key) {
    const auto& v = require(params, key);
    if (!v.isInt()) throw std::runtime_error(std::string("'") + key + "' must be an integer");
    return v.asInt();
}

double require_double(const Json::Value& params, const char* key) {
    const auto& v = require(params, key);
    if (!v.isNumeric()) throw std::runtime_error(std::string("'") + key + "' must be a number");
    return v.asDouble();
}

Json::Value volume_json(int id, const Volume& v) {
    Json::Value out;
    out["volume"] = id;
    Json::Value size(Json::arrayValue), shape(Json::arrayValue), spacing(Json::arrayValue),
        origin(Json::arrayValue), direction(Json::arrayValue);
    for (int i = 0; i < 3; ++i) {
        size.append(static_cast<Json::UInt64>(v.size[i]));
        shape.append(static_cast<Json::UInt64>(v.size[2 - i]));  // numpy: z, y, x
        spacing.append(v.spacing[i]);
        origin.append(v.origin[i]);
    }
    for (double d : v.direction) direction.append(d);
    out["size"] = size;
    out["shape"] = shape;
    out["spacing"] = spacing;
    out["origin"] = origin;
    out["direction"] = direction;
    out["pixel_type"] = sitk_name(v.type);
    out["dtype"] = numpy_dtype(v.type);
    out["shm"] = v.buffer->name();
    out["nbytes"] = static_cast<Json::UInt64>(v.count() * pixel_size(v.type));
    Json::Value meta(Json::objectValue);
    for (const auto& [k, value] : v.metadata) meta[k] = value;
    out["metadata"] = meta;
    return out;
}

namespace {

PixelType from_itk_component(itk::IOComponentEnum c) {
    using C = itk::IOComponentEnum;
    switch (c) {
        case C::UCHAR: return PixelType::UInt8;
        case C::SHORT: return PixelType::Int16;
        case C::USHORT: return PixelType::UInt16;
        case C::INT: return PixelType::Int32;
        case C::FLOAT: return PixelType::Float32;
        default:
            throw std::runtime_error("unsupported pixel type " +
                                     itk::ImageIOBase::GetComponentTypeAsString(c));
    }
}

// The series ID GDCM gives a Series Instance UID (SerieHelper::
// CreateUniqueSeriesIdentifier without series details): only letters, digits and
// dots kept.
std::string series_id(const std::string& uid) {
    std::string id;
    for (char c : uid)
        if (c == '.' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
            id += c;
    return id;
}

// A folder holding one DICOM image file (a multi-frame CBCT: every clinic case in
// the test data): {file, its series ID}, read from a few header tags. Else
// {"", ""}: ITK lists the series. ITK's listing (GDCMSeriesFileNames) reads the
// whole file to do it, 0.37 s for 596 MB; this is under 10 ms.
std::pair<std::string, std::string> single_dicom(const std::string& dir) {
    if (!fast_routes()) return {};
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file(ec)) continue;
        files.push_back(entry.path().string());
        if (files.size() > 8) return {};   // a series of files: ITK sorts them
    }
    std::string found;
    for (const auto& f : files) {
        gdcm::Reader reader;
        reader.SetFileName(f.c_str());
        if (!reader.CanRead()) continue;
        if (!found.empty()) return {};      // two DICOM files: ITK decides
        found = f;
    }
    if (found.empty()) return {};
    const gdcm::Tag uid(0x0020, 0x000e), rows(0x0028, 0x0010);
    gdcm::Scanner scanner;
    scanner.AddTag(uid);
    scanner.AddTag(rows);
    if (!scanner.Scan({found})) return {};
    const char* u = scanner.GetValue(found.c_str(), uid);
    const char* r = scanner.GetValue(found.c_str(), rows);
    if (!u || !r) return {};                // not an image with a series: ITK decides
    const std::string id = series_id(u);
    if (id.empty()) return {};
    return {found, id};
}

// The file names of one series, as SimpleITK's ImageSeriesReader gives them
// (GetGDCMSeriesFileNames: no series details, not recursive).
std::vector<std::string> series_files(const std::string& dir, const std::string& series) {
    const auto single = single_dicom(dir);
    if (!single.first.empty()) {
        if (single.second == series) return {single.first};
        return {};
    }
    auto names = itk::GDCMSeriesFileNames::New();
    names->SetUseSeriesDetails(false);
    names->SetDirectory(dir);
    return names->GetFileNames(series);
}

// Why a series cannot be read as one volume, from its headers only ("" if it
// can): slices of different sizes (the addon found it by reading the series).
std::string series_problem(const std::vector<std::string>& files) {
    const gdcm::Tag rows(0x0028, 0x0010), cols(0x0028, 0x0011);
    gdcm::Scanner scanner;
    scanner.AddTag(rows);
    scanner.AddTag(cols);
    if (!scanner.Scan(files)) return "the files of this series cannot be read";
    if (scanner.GetValues(rows).size() > 1 || scanner.GetValues(cols).size() > 1)
        return "the slices of this series are not all the same size";
    return "";
}

// A DICOM header value as text, without the padding ("" when absent).
std::string header_tag(const itk::MetaDataDictionary& dict, const char* key) {
    std::string v;
    itk::ExposeMetaData<std::string>(dict, key, v);
    const auto end = v.find_last_not_of(std::string(" \0", 2));
    return end == std::string::npos ? std::string() : v.substr(0, end + 1);
}

// dicom.scan -- odent_utils.parse_dicom_series, fast_import_ops.longest_dicom_series:
// every series of a folder with its file count and spacing. Read from the
// headers (the addon read every series whole to get its spacing, and skipped
// the ones it could not read: here they carry an "error").
Json::Value dicom_scan(const Json::Value& p) {
    const std::string dir = require_string(p, "dir");
    std::vector<std::pair<std::string, std::vector<std::string>>> found;   // id, files
    const auto single = single_dicom(dir);
    if (!single.first.empty()) {
        found.push_back({single.second, {single.first}});
    } else {
        auto names = itk::GDCMSeriesFileNames::New();
        names->SetUseSeriesDetails(false);
        names->SetDirectory(dir);
        for (const auto& id : names->GetSeriesUIDs()) found.push_back({id, names->GetFileNames(id)});
    }
    Json::Value series(Json::arrayValue);
    for (const auto& [id, files] : found) {
        Json::Value s;
        s["id"] = id;
        s["count"] = static_cast<Json::UInt64>(files.size());
        if (files.size() == 1) s["file"] = files.front();   // Load Data names a one-file CT by it
        const std::string problem = files.size() > 1 ? series_problem(files) : "";
        if (!problem.empty()) {
            s["error"] = problem;
            series.append(s);
            continue;
        }
        try {
            itk::GDCMImageIO::Pointer io = files.size() == 1
                ? itk::GDCMImageIO::Pointer(dicom_io(files.front()))
                : itk::GDCMImageIO::Pointer(CachedGDCMImageIO::New());
            auto reader = itk::ImageSeriesReader<Image3<float>>::New();
            reader->SetImageIO(io);
            reader->SetFileNames(files);
            reader->UpdateOutputInformation();
            Json::Value spacing(Json::arrayValue);
            for (int i = 0; i < 3; ++i) spacing.append(reader->GetOutput()->GetSpacing()[i]);
            s["spacing"] = spacing;
            // the volume's size: a multi-frame file is one file of many slices
            const auto sz = reader->GetOutput()->GetLargestPossibleRegion().GetSize();
            Json::Value size(Json::arrayValue);
            for (int i = 0; i < 3; ++i) size.append(static_cast<Json::UInt64>(sz[i]));
            s["size"] = size;
            // what the Load Data dialog shows of it (the first file's header)
            const auto& dict = io->GetMetaDataDictionary();
            std::string date = header_tag(dict, "0008|0021");     // series date
            if (date.empty()) date = header_tag(dict, "0008|0020");   // study date
            s["description"] = header_tag(dict, "0008|103e");
            s["date"] = date;
            s["modality"] = header_tag(dict, "0008|0060");
        } catch (const std::exception& e) {
            s["error"] = e.what();
        }
        series.append(s);
    }
    Json::Value out;
    out["series"] = series;
    return out;
}

// The voxels of a single DICOM file (a multi-frame CBCT, as every clinic case in
// the test data is) read straight from the file into `dst`, when they are stored
// plainly: one grey 16 bit sample, MONOCHROME2, no overlay, slope 1 and a whole
// intercept, and the file ending with the pixel data as one uncompressed little
// endian element of exactly n * 2 bytes (its tag and length are checked: a big
// endian, compressed or deflated file fails that check). False otherwise: ITK
// reads it as before.
//
// Why: for 12 bits stored in 16, GDCM's GetBuffer cleans the unused bits through
// string streams (0.73 s and several copies for 239 MB), after reading the whole
// file once more. This reads only the pixels and does GDCM's own cleanup
// (ImageCodec::CleanupUnusedBits) and rescale (Rescaler: slope * value +
// intercept, cast) in one pass. The result must be identical to ITK's: compared
// on every test case with ODENT_ENGINE_NO_FAST_DICOM=1 (ITK's way).
//
// Split in two since 2026-10-17: plain_pixels checks the layout, read_plain
// converts pixels; image.thumbnail reads one frame (the middle slice) the same way.

// Where and how a plain file stores its n pixels (the checks above), or nothing:
// then ITK reads it.
struct PlainPixels {
    std::uint64_t data_start = 0;   // the first pixel's byte in the file
    bool cleanup = false, is_signed = false;
    unsigned shift = 0;
    std::uint16_t pmask = 0, smask = 0, nmask = 0;
    long off = 0;                   // the whole intercept
};

std::optional<PlainPixels> plain_pixels(const std::string& path, const itk::GDCMImageIO* io,
                                        std::size_t n) {
    {
        if (std::getenv("ODENT_ENGINE_NO_FAST_DICOM")) return std::nullopt;
        const std::uint16_t one = 1;   // the file's little endian read as is: this computer too
        if (*reinterpret_cast<const unsigned char*>(&one) != 1) return std::nullopt;
        const double slope = io->GetRescaleSlope(), intercept = io->GetRescaleIntercept();
        if (slope != 1.0 || intercept != std::floor(intercept)) return std::nullopt;
        // the pixel layout, from the header ITK read (its metadata)
        const auto& dict = io->GetMetaDataDictionary();
        auto tag = [&](const char* key) {
            std::string v;
            itk::ExposeMetaData<std::string>(dict, key, v);
            const auto end = v.find_last_not_of(" \0");
            return end == std::string::npos ? std::string() : v.substr(0, end + 1);
        };
        for (const auto& key : dict.GetKeys())
            if (key.size() > 2 && key[0] == '6' && key[1] == '0') return std::nullopt;   // overlays
        if (tag("0028|0002") != "1" || tag("0028|0004") != "MONOCHROME2" || tag("0028|0100") != "16")
            return std::nullopt;
        unsigned stored = 0, high = 0, representation = 0;
        try {
            stored = static_cast<unsigned>(std::stoul(tag("0028|0101")));
            high = static_cast<unsigned>(std::stoul(tag("0028|0102")));
            representation = static_cast<unsigned>(std::stoul(tag("0028|0103")));
        } catch (...) {
            return std::nullopt;
        }
        if (stored == 0 || stored > 16 || high + 1 < stored || high > 15 || representation > 1)
            return std::nullopt;
        // the pixel data element at the end of the file
        std::error_code ec;
        const auto file_size = std::filesystem::file_size(path, ec);
        const std::uint64_t bytes = static_cast<std::uint64_t>(n) * 2;
        if (ec || file_size < bytes + 12) return std::nullopt;
        std::ifstream f(path, std::ios::binary);
        if (!f) return std::nullopt;
        unsigned char h[12];
        f.seekg(static_cast<std::streamoff>(file_size - bytes - 12));
        if (!f.read(reinterpret_cast<char*>(h), 12)) return std::nullopt;
        auto le32 = [](const unsigned char* b) {
            return static_cast<std::uint64_t>(b[0]) | static_cast<std::uint64_t>(b[1]) << 8 |
                   static_cast<std::uint64_t>(b[2]) << 16 | static_cast<std::uint64_t>(b[3]) << 24;
        };
        const bool is_tag_e = h[0] == 0xE0 && h[1] == 0x7F && h[2] == 0x10 && h[3] == 0x00;
        const bool explicit_vr = is_tag_e && (h[4] == 'O') && (h[5] == 'W' || h[5] == 'B') &&
                                 h[6] == 0 && h[7] == 0 && le32(h + 8) == bytes;
        const bool is_tag_i = h[4] == 0xE0 && h[5] == 0x7F && h[6] == 0x10 && h[7] == 0x00;
        const bool implicit_vr = is_tag_i && le32(h + 8) == bytes;
        if (!explicit_vr && !implicit_vr) return std::nullopt;

        PlainPixels L;
        L.data_start = file_size - bytes;
        L.cleanup = stored != 16;                     // Pixmap::UnusedBitsPresentInPixelData
        L.is_signed = representation != 0;
        L.shift = stored - high - 1;                  // 0 when the high bit is stored - 1
        L.pmask = static_cast<std::uint16_t>(0xffffu >> (16 - stored));
        L.smask = static_cast<std::uint16_t>(1u << (stored - 1));
        L.nmask = static_cast<std::uint16_t>(~L.pmask);
        L.off = static_cast<long>(intercept);
        return L;
    }
}

// `count` pixels of a plain file from pixel `first` on, cleaned up and rescaled
// as GDCM does, into dst.
template <class T>
bool read_plain(const std::string& path, const PlainPixels& L, std::size_t first, T* dst,
                std::size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<std::uint16_t> chunk(std::min(count, std::size_t(1) << 21));   // 4 MB at a time
    f.seekg(static_cast<std::streamoff>(L.data_start + static_cast<std::uint64_t>(first) * 2));
    for (std::size_t done = 0; done < count;) {
        const std::size_t m = std::min(chunk.size(), count - done);
        if (!f.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(m * 2)))
            return false;
        report_progress(static_cast<double>(done) / static_cast<double>(count));
        T* out = dst + done;
        for (std::size_t i = 0; i < m; ++i) {
            std::uint16_t c = chunk[i];
            if (L.cleanup) {
                c = static_cast<std::uint16_t>(c >> L.shift);
                c = (L.is_signed && (c & L.smask)) ? static_cast<std::uint16_t>(c | L.nmask)
                                                   : static_cast<std::uint16_t>(c & L.pmask);
            }
            const long value = L.is_signed ? static_cast<long>(static_cast<std::int16_t>(c))
                                           : static_cast<long>(c);
            out[i] = static_cast<T>(value + L.off);
        }
        done += m;
    }
    return true;
}

template <class T>
bool fast_dicom_pixels(const std::string& path, const itk::GDCMImageIO* io, T* dst,
                       std::size_t n) {
    if constexpr (!std::is_same_v<T, std::int16_t> && !std::is_same_v<T, std::uint16_t>) {
        return false;
    } else {
        const auto L = plain_pixels(path, io, n);
        return L && read_plain<T>(path, *L, 0, dst, n);
    }
}

template <class T>
std::unique_ptr<Volume> read_series(const std::vector<std::string>& files,
                                    itk::GDCMImageIO* io) {
    auto reader = itk::ImageSeriesReader<Image3<T>>::New();
    reader->SetImageIO(io);
    reader->SetFileNames(files);
    if (files.size() == 1) {
        // one multi-frame file: the fast path, with the header ITK has read
        return Volume::read_by<T>(reader.GetPointer(), [&](T* dst, std::size_t n) {
            return fast_dicom_pixels<T>(files.front(), io, dst, n);
        });
    }
    return Volume::read_by<T>(reader.GetPointer());
}

template <class T>
std::unique_ptr<Volume> read_file(const std::string& path, itk::ImageIOBase* io) {
    auto reader = itk::ImageFileReader<Image3<T>>::New();
    reader->SetImageIO(io);
    reader->SetFileName(path);
    if (auto* gdcm_io = dynamic_cast<itk::GDCMImageIO*>(io)) {
        return Volume::read_by<T>(reader.GetPointer(), [&](T* dst, std::size_t n) {
            return fast_dicom_pixels<T>(path, gdcm_io, dst, n);
        });
    }
    return Volume::read_by<T>(reader.GetPointer());
}

// image.load -- dicom_ops get_sitk_image_from_fileSeries (dir + series) and
// get_sitk_image_from_singlFile (file): the voxels in the pixel type the file
// gives (after the DICOM rescale), private tags in the metadata.
Json::Value image_load(const Json::Value& p) {
    std::unique_ptr<Volume> v;
    if (p.isMember("series")) {
        const auto files = series_files(require_string(p, "dir"), require_string(p, "series"));
        if (files.empty()) throw std::runtime_error("no files in this DICOM series");
        itk::GDCMImageIO::Pointer io = files.size() == 1
            ? itk::GDCMImageIO::Pointer(dicom_io(files.front()))
            : itk::GDCMImageIO::Pointer(CachedGDCMImageIO::New());
        io->SetFileName(files.front());
        io->ReadImageInformation();
        if (io->GetNumberOfComponents() != 1)
            throw std::runtime_error("not a grey-level image");
        const PixelType t = from_itk_component(io->GetComponentType());
        v = dispatch(t, [&](auto pixel) { return read_series<decltype(pixel)>(files, io); });
    } else {
        const std::string path = require_string(p, "file");
        auto io = image_io(path);
        if (!io) throw std::runtime_error("can not read image file");
        io->SetFileName(path);
        io->ReadImageInformation();
        if (io->GetNumberOfDimensions() < 3 || io->GetDimensions(2) <= 2)
            throw std::runtime_error("not a 3D image");
        if (io->GetNumberOfComponents() != 1)
            throw std::runtime_error("not a grey-level image");
        const PixelType t = from_itk_component(io->GetComponentType());
        v = dispatch(t, [&](auto pixel) { return read_file<decltype(pixel)>(path, io); });
    }
    const Volume& ref = *v;
    const int id = store().add(std::move(v));
    return volume_json(id, ref);
}

// image.resample_iso -- odent_utils.ResizeImage's size rule (size * spacing /
// target, whole voxels), linear, identity transform, the input's origin and
// direction, `fill` outside, no metadata. Already isotropic at the target
// (within kSpacingTolerance): the same volume comes back, voxels untouched.
//
// Both use a tolerance since 2026-10-11. A multi-frame DICOM gives its z spacing
// as 0.2000000000000009: compared exactly, a CT already at 0.2 mm was resampled
// (0.62 s, interpolated voxels). And the size was truncated: float noise below a
// whole number dropped the last slice (334 voxels of 0.3 mm to 0.2 mm:
// 500.99999999999994 -> 500), in 2420 of the common spacing / size pairs.
constexpr double kSpacingTolerance = 1e-6;   // mm
constexpr double kSizeTolerance = 1e-6;      // voxels
template <class T>
std::unique_ptr<Volume> resample_iso(const Volume& src, double target, double fill) {
    using Img = Image3<T>;
    auto in = src.itk_view<T>();
    typename Img::SizeType size;
    typename Img::SpacingType spacing;
    for (unsigned i = 0; i < 3; ++i) {
        size[i] = static_cast<itk::SizeValueType>(std::floor(
            static_cast<double>(src.size[i]) * (src.spacing[i] / target) + kSizeTolerance));
        spacing[i] = target;
    }
    auto f = itk::ResampleImageFilter<Img, Img>::New();
    f->SetInput(in);
    f->SetTransform(itk::IdentityTransform<double, 3>::New());
    f->SetInterpolator(itk::LinearInterpolateImageFunction<Img, double>::New());
    f->SetSize(size);
    f->SetOutputOrigin(in->GetOrigin());
    f->SetOutputSpacing(spacing);
    f->SetOutputDirection(in->GetDirection());
    f->SetDefaultPixelValue(static_cast<T>(fill));
    // straight into the new block, not ITK's buffer and a copy (668^3 from 501^3:
    // a whole 596 MB CT less at the peak). No metadata: an ITK filter does not
    // carry it (the addon's .mha had only the tags it set after SimpleITK's resample)
    return Volume::read_by<T>(f.GetPointer());
}

Json::Value image_resample_iso(const Json::Value& p) {
    const int id = require_int(p, "volume");
    const double target = require_double(p, "spacing");
    const double fill = require_double(p, "fill");
    if (!(target > 0)) throw std::runtime_error("'spacing' must be above 0");
    const Volume& src = store().get(id);
    if (std::abs(src.spacing[0] - target) <= kSpacingTolerance &&
        std::abs(src.spacing[1] - target) <= kSpacingTolerance &&
        std::abs(src.spacing[2] - target) <= kSpacingTolerance)
        return volume_json(id, src);
    auto v = dispatch(src.type, [&](auto pixel) {
        return resample_iso<decltype(pixel)>(src, target, fill);
    });
    const Volume& ref = *v;
    return volume_json(store().add(std::move(v)), ref);
}

// image.crop {volume, lo: [x, y, z], hi: [x, y, z]} -- Load Data's VOI: the voxels
// lo..hi-1 along each axis as a new volume at the same place in space (its origin
// moved to the first voxel kept), metadata kept. The CT is read whole first (a
// series or a file cannot be read in part), cropped before anything else.
template <class T>
std::unique_ptr<Volume> crop(const Volume& src, const std::array<std::size_t, 3>& lo,
                             const std::array<std::size_t, 3>& n) {
    auto v = std::make_unique<Volume>();
    v->type = src.type;
    v->size = n;
    v->spacing = src.spacing;
    v->direction = src.direction;
    v->metadata = src.metadata;
    for (int i = 0; i < 3; ++i) {   // origin + D (spacing * lo), D row major (its columns: the axes)
        v->origin[i] = src.origin[i];
        for (int j = 0; j < 3; ++j) v->origin[i] += src.direction[3 * i + j] * src.spacing[j] * double(lo[j]);
    }
    v->buffer = SharedBuffer::create(v->count() * sizeof(T));
    const T* in = static_cast<const T*>(src.buffer->data());
    T* out = static_cast<T*>(v->buffer->data());
    const std::size_t sx = src.size[0], sy = src.size[1];
    for (std::size_t z = 0; z < n[2]; ++z)
        for (std::size_t y = 0; y < n[1]; ++y)
            std::memcpy(out + (z * n[1] + y) * n[0], in + ((z + lo[2]) * sy + (y + lo[1])) * sx + lo[0],
                        n[0] * sizeof(T));
    return v;
}

Json::Value image_crop(const Json::Value& p) {
    const Volume& src = store().get(require_int(p, "volume"));
    const auto& jlo = require(p, "lo");
    const auto& jhi = require(p, "hi");
    if (!jlo.isArray() || !jhi.isArray() || jlo.size() != 3 || jhi.size() != 3)
        throw std::runtime_error("'lo' and 'hi' must be 3 voxel indices");
    std::array<std::size_t, 3> lo{}, n{};
    for (int i = 0; i < 3; ++i) {
        const auto a = std::clamp<long long>(jlo[i].asInt64(), 0, (long long)src.size[i] - 1);
        const auto b = std::clamp<long long>(jhi[i].asInt64(), a + 1, (long long)src.size[i]);
        lo[i] = std::size_t(a);
        n[i] = std::size_t(b - a);
    }
    auto v = dispatch(src.type, [&](auto pixel) { return crop<decltype(pixel)>(src, lo, n); });
    const Volume& ref = *v;
    return volume_json(store().add(std::move(v)), ref);
}

// image.clamp_cast -- ct_values.to_hu_image: clamp to lo..hi in the input's
// type (sitk.Clamp), then int16 (sitk.Cast); the result has only the
// `metadata` entries given (SimpleITK's filters keep none). With "in_place"
// and an int16 input, the input volume itself is clamped and returned (the
// reader's volume, not needed any more): no second CT at the peak of a load.
template <class T>
void clamp_to_int16(const Volume& src, Volume& dst, double lo, double hi) {
    const T* in = static_cast<const T*>(src.buffer->data());
    auto* out = static_cast<std::int16_t*>(dst.buffer->data());
    // the bounds in the input's type, as sitk.Clamp casts them
    const T tlo = static_cast<T>(lo), thi = static_cast<T>(hi);
    const std::size_t n = src.count();
    for (std::size_t i = 0; i < n; ++i)
        out[i] = static_cast<std::int16_t>(std::min(std::max(in[i], tlo), thi));
}

Json::Value image_clamp_cast(const Json::Value& p) {
    const int id = require_int(p, "volume");
    const double lo = require_double(p, "lo"), hi = require_double(p, "hi");
    if (require_string(p, "dtype") != "int16") throw std::runtime_error("only dtype int16");
    if (lo < std::numeric_limits<std::int16_t>::min() || hi > std::numeric_limits<std::int16_t>::max())
        throw std::runtime_error("lo..hi does not fit int16");
    Volume& src = store().get(id);
    auto set_metadata = [&](Volume& v) {
        v.metadata.clear();
        if (p.isMember("metadata")) {
            const auto& meta = p["metadata"];
            for (const auto& key : meta.getMemberNames()) v.metadata[key] = meta[key].asString();
        }
    };
    if (p.get("in_place", false).asBool() && src.type == PixelType::Int16) {
        auto* a = static_cast<std::int16_t*>(src.buffer->data());
        const auto tlo = static_cast<std::int16_t>(lo), thi = static_cast<std::int16_t>(hi);
        const std::size_t n = src.count();
        for (std::size_t i = 0; i < n; ++i) a[i] = std::min(std::max(a[i], tlo), thi);
        set_metadata(src);
        return volume_json(id, src);
    }
    auto v = Volume::like(src, PixelType::Int16);
    v->metadata.clear();
    dispatch(src.type, [&](auto pixel) { clamp_to_int16<decltype(pixel)>(src, *v, lo, hi); });
    set_metadata(*v);
    const Volume& ref = *v;
    return volume_json(store().add(std::move(v)), ref);
}

// image.write -- sitk.WriteImage of the project CT (.mha, no compression, the
// volume's metadata plus `metadata`).
template <class T>
void write_image(const Volume& v, const std::string& path, const Json::Value& extra) {
    auto img = v.itk_view<T>();
    auto& dict = img->GetMetaDataDictionary();
    for (const auto& key : extra.getMemberNames())
        itk::EncapsulateMetaData<std::string>(dict, key, extra[key].asString());
    auto writer = itk::ImageFileWriter<Image3<T>>::New();
    writer->SetInput(img);
    writer->SetFileName(path);
    writer->SetUseCompression(false);
    // its progress: 0 and 1 only (16 stream divisions tried: still written whole, the
    // file identical, no more events; the CT read's card holds ~1.2 s on this step)
    watch_progress(writer);
    writer->Update();
}

Json::Value image_write(const Json::Value& p) {
    const int id = require_int(p, "volume");
    const std::string path = require_string(p, "path");
    const Json::Value extra = p.isMember("metadata") ? p["metadata"] : Json::Value(Json::objectValue);
    const Volume& v = store().get(id);
    dispatch(v.type, [&](auto pixel) { write_image<decltype(pixel)>(v, path, extra); });
    Json::Value out;
    out["path"] = path;
    return out;
}

// image.header {file} -- the dialog's check of a 3D file (dicom_ops
// get_sitk_image_from_singlFile): size, spacing, origin, direction, pixel type
// and metadata, without reading the voxels.
Json::Value image_header(const Json::Value& p) {
    const std::string path = require_string(p, "file");
    auto io = image_io(path);
    if (!io) throw std::runtime_error("can not read image file");
    io->SetFileName(path);
    io->ReadImageInformation();
    Json::Value out;
    const unsigned dims = io->GetNumberOfDimensions();
    Json::Value size(Json::arrayValue), spacing(Json::arrayValue), origin(Json::arrayValue),
        direction(Json::arrayValue);
    for (unsigned i = 0; i < 3; ++i) {
        size.append(static_cast<Json::UInt64>(i < dims ? io->GetDimensions(i) : 1));
        spacing.append(i < dims ? io->GetSpacing(i) : 1.0);
        origin.append(i < dims ? io->GetOrigin(i) : 0.0);
        for (unsigned j = 0; j < 3; ++j)
            direction.append(i < dims && j < dims ? io->GetDirection(j)[i] : (i == j ? 1.0 : 0.0));
    }
    out["dimensions"] = dims;
    out["components"] = io->GetNumberOfComponents();
    out["size"] = size;
    out["spacing"] = spacing;
    out["origin"] = origin;
    out["direction"] = direction;
    try {
        out["pixel_type"] = sitk_name(from_itk_component(io->GetComponentType()));
    } catch (const std::exception&) {
        out["pixel_type"] = itk::ImageIOBase::GetComponentTypeAsString(io->GetComponentType());
    }
    Json::Value meta(Json::objectValue);
    const auto& dict = io->GetMetaDataDictionary();
    for (const auto& key : dict.GetKeys()) {
        std::string value;
        if (key.rfind("ITK_", 0) == 0) continue;   // ITK's own notes, not the image's
        if (itk::ExposeMetaData<std::string>(dict, key, value)) meta[key] = value;
    }
    out["metadata"] = meta;
    return out;
}

// The coronal and sagittal maximum intensity projections of the Load Data
// preview, built a slice at a time (nothing of the CT is kept but one slice): for
// each slice taken, the maximum over y of each x (coronal) and over x of each y
// (sagittal): teeth, bone and metal stand out (mean projections came out flat).
struct Projections {
    std::size_t nx = 0, ny = 0;
    std::vector<std::vector<float>> coronal, sagittal;   // one row per slice taken, in z order
    void add(const float* px) {
        const float low = -std::numeric_limits<float>::infinity();
        coronal.emplace_back(nx, low);
        sagittal.emplace_back(ny, low);
        auto& cor = coronal.back();
        auto& sag = sagittal.back();
        for (std::size_t y = 0; y < ny; ++y)
            for (std::size_t x = 0; x < nx; ++x) {
                const float v = px[y * nx + x];
                cor[x] = std::max(cor[x], v);
                sag[y] = std::max(sag[y], v);
            }
    }
};

// The slices taken: about `rows` of nz, evenly (the preview has no more rows).
std::vector<std::size_t> sampled(std::size_t nz, int rows) {
    const std::size_t step = std::max<std::size_t>(1, nz / std::size_t(std::max(rows, 1)));
    std::vector<std::size_t> ks;
    for (std::size_t k = step / 2; k < nz; k += step) ks.push_back(k);
    return ks;
}

// `src` (h rows of w) into the w_out x h_out rectangle of `dst` (dst_cols wide) at
// column x0, each pixel the mean of the source pixels under it; flip_x / flip_y
// mirror it.
void box_into(const std::vector<std::vector<float>>& src, std::size_t w, float* dst,
              std::size_t dst_cols, std::size_t x0, std::size_t w_out, std::size_t h_out,
              bool flip_x, bool flip_y) {
    const std::size_t h = src.size();
    for (std::size_t r = 0; r < h_out; ++r) {
        const std::size_t y0 = r * h / h_out, y1 = std::max(y0 + 1, (r + 1) * h / h_out);
        for (std::size_t c = 0; c < w_out; ++c) {
            const std::size_t a = c * w / w_out, b = std::max(a + 1, (c + 1) * w / w_out);
            double sum = 0;
            for (std::size_t y = y0; y < y1; ++y)
                for (std::size_t x = a; x < b; ++x) sum += src[y][x];
            const std::size_t rr = flip_y ? h_out - 1 - r : r, cc = flip_x ? w_out - 1 - c : c;
            dst[rr * dst_cols + x0 + cc] = static_cast<float>(sum / double((y1 - y0) * (b - a)));
        }
    }
}

// image.thumbnail {file | dir + series, size (px, 512)} -- the Load Data dialog's
// preview of a CT: its coronal and sagittal maximum intensity projections side by
// side (the head up; coronal left: the patient's right on the left; sagittal
// right: the front on the right), in its values (HU after the DICOM rescale), as
// one float32 tile (1, rows, cols), NaN between and around them, the longer side
// `size` px, the shapes in mm kept. Also what the VOI of the dialog needs: each
// panel's place ("panels": [x0, width]), which image axes run against the picture
// ("flip": x, y, z) and the CT's size in mm along x, y, z ("extent_mm"). About `size` slices are read: of a series its
// files, of a plain multi-frame DICOM its frames (read_plain); other files are
// read whole by ITK.
Json::Value image_thumbnail(const Json::Value& p) {
    const int size = std::clamp(p.get("size", 512).asInt(), 16, 2048);
    Projections pr;
    std::size_t nz = 0;
    double sx = 1, sy = 1, sz = 1;
    int sign_x = 1, sign_y = 1, sign_z = 1;   // the image axes along the patient's (LPS)
    auto sign = [](double d) { return d < 0 ? -1 : 1; };
    std::string file;
    std::vector<std::string> files;
    if (p.isMember("series")) {
        files = series_files(require_string(p, "dir"), require_string(p, "series"));
        if (files.empty()) throw std::runtime_error("no files in this DICOM series");
    } else {
        file = require_string(p, "file");
    }
    if (files.size() > 1) {
        // a file per slice (ITK's order: along the slices); every few of them
        nz = files.size();
        std::array<double, 3> first{}, last{};
        double spacing_z_of_first = 0;
        const auto ks = sampled(nz, size);
        for (std::size_t j = 0; j < ks.size(); ++j) {
            auto reader = itk::ImageFileReader<Image3<float>>::New();
            reader->SetImageIO(itk::GDCMImageIO::New());
            reader->SetFileName(files[ks[j]]);
            reader->Update();
            const auto* img = reader->GetOutput();
            const auto s = img->GetLargestPossibleRegion().GetSize();
            if (j == 0) {
                pr.nx = s[0];
                pr.ny = s[1];
                sx = img->GetSpacing()[0];
                sy = img->GetSpacing()[1];
                sign_y = sign(img->GetDirection()(1, 1));
                sign_x = sign(img->GetDirection()(0, 0));
                spacing_z_of_first = img->GetSpacing()[2];
                for (int i = 0; i < 3; ++i) first[i] = img->GetOrigin()[i];
            } else if (s[0] != pr.nx || s[1] != pr.ny) {
                throw std::runtime_error("the slices of this series are not all the same size");
            }
            for (int i = 0; i < 3; ++i) last[i] = img->GetOrigin()[i];
            pr.add(img->GetBufferPointer());
        }
        // the slice spacing from the first and last slice taken; files without a
        // position (all at the same origin) take the file's own z spacing (its
        // slice thickness), else 1 mm: a spacing of 0 made the preview 1 px high
        sz = 0;
        if (ks.size() > 1) {
            const double d = std::sqrt(std::pow(last[0] - first[0], 2) + std::pow(last[1] - first[1], 2) +
                                       std::pow(last[2] - first[2], 2));
            sz = d / double(ks.back() - ks.front());
            if (d > 1e-6) sign_z = sign(last[2] - first[2]);
        }
        if (!(sz > 1e-6)) sz = spacing_z_of_first > 1e-6 ? spacing_z_of_first : 1.0;
    } else {
        if (!files.empty()) file = files.front();
        auto io = image_io(file);
        if (!io) throw std::runtime_error("can not read image file");
        io->SetFileName(file);
        io->ReadImageInformation();
        if (io->GetNumberOfDimensions() < 3 || io->GetDimensions(2) <= 2)
            throw std::runtime_error("not a 3D image");
        if (io->GetNumberOfComponents() != 1) throw std::runtime_error("not a grey-level image");
        pr.nx = io->GetDimensions(0);
        pr.ny = io->GetDimensions(1);
        nz = io->GetDimensions(2);
        sx = io->GetSpacing(0);
        sy = io->GetSpacing(1);
        sign_y = sign(io->GetDirection(1)[1]);
        sz = io->GetSpacing(2);
        sign_x = sign(io->GetDirection(0)[0]);
        sign_z = sign(io->GetDirection(2)[2]);
        // a multi-frame DICOM can give its frame spacing negative (the frames run
        // against the slice direction): the frames then go the other way up. The
        // preview took it as is: a height of -251 px, a block of 2^64 bytes refused
        if (sz < 0) {
            sz = -sz;
            sign_z = -sign_z;
        }
        const std::size_t n2 = pr.nx * pr.ny;
        const auto ks = sampled(nz, size);
        auto* gdcm_io = dynamic_cast<itk::GDCMImageIO*>(io.GetPointer());
        const auto c = io->GetComponentType();
        const bool sixteen = c == itk::IOComponentEnum::SHORT || c == itk::IOComponentEnum::USHORT;
        bool done = false;
        if (gdcm_io && sixteen) {
            if (const auto L = plain_pixels(file, gdcm_io, n2 * nz)) {
                std::vector<float> px(n2);
                done = true;
                for (std::size_t k : ks) {
                    if (!read_plain<float>(file, *L, k * n2, px.data(), n2)) {
                        done = false;
                        pr.coronal.clear();
                        break;
                    }
                    pr.add(px.data());
                }
            }
        }
        if (!done) {
            // read whole, in its own pixel type (half the memory of float for 16 bit)
            const PixelType t = from_itk_component(c);
            auto v = dispatch(t, [&](auto pixel) { return read_file<decltype(pixel)>(file, io); });
            std::vector<float> px(n2);
            dispatch(v->type, [&](auto pixel) {
                using T = decltype(pixel);
                const T* a = static_cast<const T*>(v->buffer->data());
                for (std::size_t k : ks) {
                    for (std::size_t i = 0; i < n2; ++i) px[i] = static_cast<float>(a[k * n2 + i]);
                    pr.add(px.data());
                }
            });
        }
    }
    if (pr.coronal.empty() || pr.nx == 0 || pr.ny == 0) throw std::runtime_error("the image is empty");

    // the two side by side, a gap between, in mm then in pixels
    const double h_mm = nz * std::abs(sz), wc_mm = pr.nx * std::abs(sx), ws_mm = pr.ny * std::abs(sy),
                 gap_mm = 0.04 * std::max(wc_mm, ws_mm);
    const double scale = size / std::max(wc_mm + gap_mm + ws_mm, h_mm);
    if (!(scale > 0) || !std::isfinite(scale)) throw std::runtime_error("the image has no size");
    const auto px_of = [&](double mm) {   // never below 1 (nor a negative wrapped round)
        return std::size_t(std::max(1L, std::lround(std::abs(mm) * scale)));
    };
    const std::size_t rows = px_of(h_mm), wc = px_of(wc_mm), ws = px_of(ws_mm), gap = px_of(gap_mm);
    const std::size_t cols = wc + gap + ws;
    auto tile = Volume::table(PixelType::Float32, rows, cols);
    auto* out = static_cast<float*>(tile->buffer->data());
    std::fill(out, out + rows * cols, std::numeric_limits<float>::quiet_NaN());
    // row 0 at the top = the head: the slices in z order go up when z goes up (LPS +z
    // is the head); coronal: +x is the patient's left, shown on the right; sagittal:
    // +y is the back, shown on the left (the face to the right)
    const bool flip_x = sign_x < 0, flip_y = sign_y > 0, flip_z = sign_z > 0;
    box_into(pr.coronal, pr.nx, out, cols, 0, wc, rows, flip_x, flip_z);
    box_into(pr.sagittal, pr.ny, out, cols, wc + gap, ws, rows, flip_y, flip_z);
    tile->spacing = {1.0 / scale, 1.0 / scale, 1.0};
    const Volume& ref = *tile;
    Json::Value out_json = volume_json(store().add(std::move(tile)), ref);
    Json::Value panels(Json::arrayValue);   // [x0, width] of each, in the tile
    for (const auto& [x0, w] : {std::pair<std::size_t, std::size_t>{0, wc}, {wc + gap, ws}}) {
        Json::Value pnl(Json::arrayValue);
        pnl.append(static_cast<Json::UInt64>(x0));
        pnl.append(static_cast<Json::UInt64>(w));
        panels.append(pnl);
    }
    out_json["panels"] = panels;
    // which image axes run against the picture: coronal x (picture left to right),
    // sagittal y (picture left to right), z (picture bottom to top)
    Json::Value flip(Json::arrayValue);
    for (bool f : {flip_x, flip_y, !flip_z}) flip.append(f);
    out_json["flip"] = flip;
    Json::Value extent(Json::arrayValue);
    for (double mm : {wc_mm, ws_mm, h_mm}) extent.append(mm);
    out_json["extent_mm"] = extent;
    out_json["slices_read"] = static_cast<Json::UInt64>(pr.coronal.size());
    return out_json;
}

Json::Value image_info(const Json::Value& p) {
    const int id = require_int(p, "volume");
    return volume_json(id, store().get(id));
}

Json::Value image_free(const Json::Value& p) {
    store().remove(require_int(p, "volume"));
    return Json::Value(Json::objectValue);
}

}  // namespace

void add_image_commands(Commands& c) {
    c["dicom.scan"] = dicom_scan;
    c["image.load"] = image_load;
    c["image.resample_iso"] = image_resample_iso;
    c["image.clamp_cast"] = image_clamp_cast;
    c["image.write"] = image_write;
    c["image.info"] = image_info;
    c["image.header"] = image_header;
    c["image.thumbnail"] = image_thumbnail;
    c["image.crop"] = image_crop;
    c["image.free"] = image_free;
}

}  // namespace odent
