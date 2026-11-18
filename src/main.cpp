// odent_engine: the CT and mesh work of the ODent addon, out of Blender's
// process.
//
// `odent_engine segment-teeth <CT file> --out <folder> ...` runs the teeth
// segmentation instead, as a process of its own (teeth_segment.cpp).
//
// Protocol: one JSON object per line on stdin, {"id": n, "cmd": "...",
// "params": {...}}; one reply per line on stdout, {"id": n, "ok": {...}} or
// {"id": n, "error": "message"}. Big data (voxels, slices, meshes) goes through
// shared memory blocks named in the replies. Logs go to stderr.

#include <algorithm>
#include <cmath>
#include <mutex>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <itkConfigure.h>
#include <itkGDCMImageIOFactory.h>
#include <itkMetaImageIOFactory.h>
#include <itkNiftiImageIOFactory.h>
#include <itkNrrdImageIOFactory.h>
#include <vtkVersion.h>

#include "commands.h"

#ifdef ODENT_HAS_TEETH
namespace odent {
int segment_teeth_main(int argc, char** argv);
}
#endif
namespace odent {
int register_main(int argc, char** argv);   // register_main.cpp: the AI align's registration
}

#ifndef ODENT_ENGINE_VERSION
#define ODENT_ENGINE_VERSION "0.0.0"
#endif
#ifndef ODENT_ENGINE_API
#define ODENT_ENGINE_API 1
#endif

namespace {

std::string to_line(const Json::Value& v) {
    Json::StreamWriterBuilder w;
    w["indentation"] = "";
    return Json::writeString(w, v);
}

// the request running: its id, and whether it asked for progress lines (report_progress)
Json::Value g_request_id;
bool g_progress = false;
double g_progress_sent = -1.0;
std::mutex g_progress_lock;   // ITK may report from its worker threads

Json::Value ping(const Json::Value&) {
    Json::Value out;
    out["engine"] = "odent_engine";
    out["api"] = ODENT_ENGINE_API;
    out["version"] = ODENT_ENGINE_VERSION;   // VERSION (CMake): the release v<version>
    out["itk"] = ITK_VERSION_STRING;
    out["vtk"] = vtkVersion::GetVTKVersion();
    return out;
}

}  // namespace

namespace odent {
void report_progress(double fraction) {
    if (!g_progress) return;
    std::lock_guard<std::mutex> hold(g_progress_lock);
    fraction = std::clamp(fraction, 0.0, 1.0);
    if (fraction - g_progress_sent < 0.01 && fraction < 1.0) return;
    g_progress_sent = fraction;
    Json::Value line;
    line["id"] = g_request_id;
    line["progress"] = std::round(fraction * 1000.0) / 1000.0;
    std::cout << to_line(line) << std::endl;
}
}  // namespace odent

int main(int argc, char** argv) {
#ifdef _WIN32
    // binary pipes: no "\r\n" in the replies, no Ctrl+Z read as the end of input
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::ios::sync_with_stdio(false);
    // the image formats the addon reads (no factory auto-registration in a
    // static build)
    itk::GDCMImageIOFactory::RegisterOneFactory();
    itk::MetaImageIOFactory::RegisterOneFactory();
    itk::NrrdImageIOFactory::RegisterOneFactory();
    itk::NiftiImageIOFactory::RegisterOneFactory();
    if (argc > 1 && std::string(argv[1]) == "register") {
        return odent::register_main(argc - 2, argv + 2);
    }
    if (argc > 1 && std::string(argv[1]) == "segment-teeth") {
#ifdef ODENT_HAS_TEETH
        return odent::segment_teeth_main(argc - 2, argv + 2);
#else
        std::cerr << "this odent_engine was built without ONNX Runtime: no segment-teeth\n";
        return 4;
#endif
    }
    odent::SharedBuffer::remove_stale();

    odent::Commands commands;
    commands["ping"] = ping;
    odent::add_image_commands(commands);
    odent::add_slice_commands(commands);
    odent::add_mesh_commands(commands);
    odent::add_mesh_preview_commands(commands);
#ifdef ODENT_HAS_CUT
    odent::add_cut_commands(commands);
#endif

    Json::CharReaderBuilder rb;
    std::unique_ptr<Json::CharReader> reader(rb.newCharReader());
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        Json::Value request, reply;
        std::string parse_error;
        if (!reader->parse(line.data(), line.data() + line.size(), &request, &parse_error) ||
            !request.isObject()) {
            reply["id"] = Json::Value();
            reply["error"] = "bad request: " + parse_error;
            std::cout << to_line(reply) << std::endl;
            continue;
        }
        reply["id"] = request.get("id", Json::Value());
        const std::string cmd = request.get("cmd", "").asString();
        if (cmd == "quit") {
            reply["ok"] = Json::Value(Json::objectValue);
            std::cout << to_line(reply) << std::endl;
            break;
        }
        g_request_id = reply["id"];
        g_progress = request.get("progress", false).asBool();
        g_progress_sent = -1.0;
        auto it = commands.find(cmd);
        if (it == commands.end()) {
            reply["error"] = "unknown command: " + cmd;
        } else {
            try {
                reply["ok"] = it->second(request.get("params", Json::Value(Json::objectValue)));
            } catch (const itk::ExceptionObject& e) {
                reply["error"] = std::string(e.GetDescription());
            } catch (const std::exception& e) {
                reply["error"] = std::string(e.what());
            }
        }
        std::cout << to_line(reply) << std::endl;
    }
    return 0;
}
