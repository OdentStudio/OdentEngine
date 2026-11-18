// A command takes its JSON parameters and returns a JSON result; it throws
// std::exception (or itk::ExceptionObject) with a message on failure.

#pragma once

#include <functional>
#include <map>
#include <string>

#include <vtk_jsoncpp.h>

#include "volume.h"

namespace odent {

using Handler = std::function<Json::Value(const Json::Value& params)>;
using Commands = std::map<std::string, Handler>;

void add_image_commands(Commands& commands);
void add_slice_commands(Commands& commands);
void add_mesh_commands(Commands& commands);
void add_cut_commands(Commands& commands);
void add_mesh_preview_commands(Commands& commands);

// What the addon needs to map a volume: id, geometry, pixel type, shared block.
Json::Value volume_json(int id, const Volume& v);

// Parameter helpers: a missing or wrong parameter is an error naming it.
const Json::Value& require(const Json::Value& params, const char* key);
std::string require_string(const Json::Value& params, const char* key);
int require_int(const Json::Value& params, const char* key);
double require_double(const Json::Value& params, const char* key);

// A long command's progress (0..1): a line {"id", "progress"} before its reply, when the
// request asked for them ("progress": true; older clients do not). Lines only when the
// value moved by 1 % or more.
void report_progress(double fraction);

}  // namespace odent
