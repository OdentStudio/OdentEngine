// The Smart Cutter's cut along a closed loop (vtk_cut.cut_with_loop in the
// addon): vtkSelectPolyData turns the loop into a loop of mesh edges (Dijkstra
// shortest paths) and gives every point a signed distance to it, then
// vtkClipPolyData cuts the triangles at distance 0: the part inside the loop
// and the rest. Point and face attributes go through (VTK interpolates point
// data along the new edge). Needs VTK 9.2+ (the Dijkstra edge search).

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCleanPolyData.h>
#include <vtkClipPolyData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkIdTypeArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSelectPolyData.h>
#include <vtkTriangleFilter.h>

#include "commands.h"

namespace odent {

namespace {

const Volume& table_of(const Json::Value& p, const char* key, PixelType type, std::size_t cols) {
    const Volume& v = store().get(require_int(p, key));
    if (v.type != type || v.size[0] != cols || v.size[2] != 1)
        throw std::runtime_error(std::string("'") + key + "' must be a " + numpy_dtype(type) +
                                 " table of " + std::to_string(cols) + " columns");
    return v;
}

// A copy of a table (rows x cols) as a VTK array named `name`.
vtkSmartPointer<vtkDoubleArray> double_array(const Volume& t, const std::string& name) {
    auto a = vtkSmartPointer<vtkDoubleArray>::New();
    a->SetName(name.c_str());
    a->SetNumberOfComponents(static_cast<int>(t.size[0]));
    a->SetNumberOfTuples(static_cast<vtkIdType>(t.size[1]));
    const auto* src = static_cast<const double*>(t.buffer->data());
    std::copy(src, src + t.size[0] * t.size[1], a->GetPointer(0));
    return a;
}

vtkSmartPointer<vtkIdTypeArray> id_array(const Volume& t) {
    auto a = vtkSmartPointer<vtkIdTypeArray>::New();
    a->SetNumberOfTuples(static_cast<vtkIdType>(t.size[1]));
    const auto* src = static_cast<const std::int64_t*>(t.buffer->data());
    for (std::size_t i = 0; i < t.size[1]; ++i) a->SetValue(static_cast<vtkIdType>(i), src[i]);
    return a;
}

int add_table(std::unique_ptr<Volume> v, Json::Value& out, const std::string& key) {
    const Volume& ref = *v;
    const int id = store().add(std::move(v));
    out[key] = volume_json(id, ref);
    return id;
}

// One output of the clip as tables; the points no face uses are dropped (both
// outputs carry the whole point list), as vtk_cut.from_polydata did.
Json::Value part_json(vtkPolyData* pd, const std::vector<std::string>& point_names,
                      const std::vector<std::string>& face_names) {
    if (pd == nullptr || pd->GetNumberOfPoints() == 0 || pd->GetNumberOfPolys() == 0)
        return Json::Value();
    vtkCellArray* polys = pd->GetPolys();
    vtkDataArray* offsets = polys->GetOffsetsArray();
    vtkDataArray* conn = polys->GetConnectivityArray();
    const vtkIdType np = pd->GetNumberOfPoints();
    std::vector<std::int64_t> remap(static_cast<std::size_t>(np), -1);
    for (vtkIdType i = 0; i < conn->GetNumberOfTuples(); ++i)
        remap[static_cast<std::size_t>(conn->GetTuple1(i))] = 0;
    std::vector<vtkIdType> used;
    for (vtkIdType i = 0; i < np; ++i)
        if (remap[static_cast<std::size_t>(i)] == 0) {
            remap[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(used.size());
            used.push_back(i);
        }
    Json::Value out;
    auto co = Volume::table(PixelType::Float64, used.size(), 3);
    auto* c = static_cast<double*>(co->buffer->data());
    for (std::size_t k = 0; k < used.size(); ++k) pd->GetPoint(used[k], c + 3 * k);
    add_table(std::move(co), out, "co");
    auto off = Volume::table(PixelType::Int64, static_cast<std::size_t>(offsets->GetNumberOfTuples()), 1);
    auto* o = static_cast<std::int64_t*>(off->buffer->data());
    for (vtkIdType i = 0; i < offsets->GetNumberOfTuples(); ++i)
        o[i] = static_cast<std::int64_t>(offsets->GetTuple1(i));
    add_table(std::move(off), out, "offsets");
    auto cn = Volume::table(PixelType::Int64, static_cast<std::size_t>(conn->GetNumberOfTuples()), 1);
    auto* q = static_cast<std::int64_t*>(cn->buffer->data());
    for (vtkIdType i = 0; i < conn->GetNumberOfTuples(); ++i)
        q[i] = remap[static_cast<std::size_t>(conn->GetTuple1(i))];
    add_table(std::move(cn), out, "conn");
    Json::Value point(Json::objectValue), face(Json::objectValue);
    for (const auto& name : point_names) {
        vtkDataArray* a = pd->GetPointData()->GetArray(name.c_str());
        if (!a) continue;
        const int w = a->GetNumberOfComponents();
        auto t = Volume::table(PixelType::Float64, used.size(), static_cast<std::size_t>(w));
        auto* d = static_cast<double*>(t->buffer->data());
        for (std::size_t k = 0; k < used.size(); ++k)
            for (int j = 0; j < w; ++j) d[k * w + j] = a->GetComponent(used[k], j);
        add_table(std::move(t), point, name);
    }
    for (const auto& name : face_names) {
        vtkDataArray* a = pd->GetCellData()->GetArray(name.c_str());
        if (!a) continue;
        const int w = a->GetNumberOfComponents();
        const vtkIdType n = a->GetNumberOfTuples();
        auto t = Volume::table(PixelType::Float64, static_cast<std::size_t>(n), static_cast<std::size_t>(w));
        auto* d = static_cast<double*>(t->buffer->data());
        for (vtkIdType k = 0; k < n; ++k)
            for (int j = 0; j < w; ++j) d[k * w + j] = a->GetComponent(k, j);
        add_table(std::move(t), face, name);
    }
    out["point"] = point;
    out["face"] = face;
    return out;
}

// mesh.cut_loop {co: float64 (n, 3), offsets: int64 (f + 1, 1), conn: int64 (l, 1),
//   point: {name: float64 table}, face: {name: float64 table},
//   loop: [[x, y, z], ...], smallest: bool}
// -> {inside: part or null, outside: part or null}; a part is
//   {co, offsets, conn, point: {name: table}, face: {name: table}}
Json::Value mesh_cut_loop(const Json::Value& p) {
    const Volume& co = table_of(p, "co", PixelType::Float64, 3);
    const Volume& offsets = table_of(p, "offsets", PixelType::Int64, 1);
    const Volume& conn = table_of(p, "conn", PixelType::Int64, 1);
    const auto& loop_v = require(p, "loop");
    if (!loop_v.isArray() || loop_v.size() < 3) throw std::runtime_error("'loop' needs 3 points or more");

    vtkNew<vtkPoints> pts;
    pts->SetDataTypeToDouble();
    pts->SetData(double_array(co, "points"));
    vtkNew<vtkCellArray> cells;
    cells->SetData(id_array(offsets), id_array(conn));
    vtkNew<vtkPolyData> pd;
    pd->SetPoints(pts);
    pd->SetPolys(cells);
    std::vector<std::string> point_names, face_names;
    for (const char* domain : {"point", "face"}) {
        if (!p.isMember(domain)) continue;
        const auto& attrs = p[domain];
        for (const auto& name : attrs.getMemberNames()) {
            const Volume& t = store().get(attrs[name].asInt());
            if (t.type != PixelType::Float64) throw std::runtime_error("attribute " + name + " must be float64");
            if (std::string(domain) == "point") {
                pd->GetPointData()->AddArray(double_array(t, name));
                point_names.push_back(name);
            } else {
                pd->GetCellData()->AddArray(double_array(t, name));
                face_names.push_back(name);
            }
        }
    }
    vtkNew<vtkPoints> loop;
    for (const auto& q : loop_v) loop->InsertNextPoint(q[0].asDouble(), q[1].asDouble(), q[2].asDouble());

    vtkNew<vtkCleanPolyData> clean;   // coincident points welded, nothing else
    clean->SetInputData(pd);
    clean->PointMergingOn();
    clean->ToleranceIsAbsoluteOn();
    clean->SetAbsoluteTolerance(1e-5);
    vtkNew<vtkTriangleFilter> tri;
    tri->SetInputConnection(clean->GetOutputPort());
    vtkNew<vtkSelectPolyData> sel;
    sel->SetInputConnection(tri->GetOutputPort());
    sel->SetLoop(loop);
    sel->SetEdgeSearchModeToDijkstra();   // the greedy walk gives up ("Can't follow edge")
    sel->GenerateSelectionScalarsOn();
    if (p.get("smallest", true).asBool())
        sel->SetSelectionModeToSmallestRegion();
    else
        sel->SetSelectionModeToLargestRegion();
    vtkNew<vtkClipPolyData> clip;
    clip->SetInputConnection(sel->GetOutputPort());
    clip->SetValue(0.0);
    clip->InsideOutOn();                  // the negative side: inside the loop
    clip->GenerateClippedOutputOn();
    clip->Update();
    Json::Value out;
    out["inside"] = part_json(clip->GetOutput(), point_names, face_names);
    out["outside"] = part_json(clip->GetClippedOutput(), point_names, face_names);
    return out;
}

}  // namespace

void add_cut_commands(Commands& c) {
    c["mesh.cut_loop"] = mesh_cut_loop;
}

}  // namespace odent
