// odent_engine register SOURCE TARGET [options]: a scan registered onto its arch (the AI
// align), in place of MeshRegAuto.exe: the same command line, defaults, steps and
// matrix file (registration.h has the steps). Prints its stages and results; the matrix
// (4 lines of 4 numbers) goes to --transform_output, the fit to --result_json.
// Exit codes: 0 done, 1 a file not read or written, 2 bad arguments, 4 an internal error.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "mesh_io.h"
#include "registration.h"

namespace odent {

namespace {

struct Options {
    std::string source, target;
    std::string method = "feature_matching";   // or "correspondence"
    std::string mode = "full";                 // full, global, icp
    bool mutual_filter = false;
    double voxel_size = 0.5;                   // mm
    double distance_multiplier = 1.5;
    int max_iterations = 1000000;              // RANSAC
    double confidence = 0.999;                 // RANSAC
    double icp_threshold = 0.5;                // mm, absolute
    int max_icp_iterations = 1000;
    std::uint64_t seed = 0;                    // 0: a new one each run
    std::string export_path, transform_output, result_json;
};

void print_help() {
    std::cout <<
        "odent_engine register: a scan registered onto its arch (MeshRegAuto's command line)\n"
        "Usage: odent_engine register SOURCE TARGET [options]   (STL, PLY, OBJ)\n"
        "  --registration_mode full (default) [full | global | icp]\n"
        "  --method feature_matching (default) [feature_matching | correspondence]\n"
        "  --voxel_size 0.5 (default, mm)\n"
        "  --distance_multiplier 1.5 (default)\n"
        "  --max_iterations 1000000 (default)\n"
        "  --confidence 0.999 (default; also --confidance)\n"
        "  --mutual_filter 0 (default) [0 | 1]\n"
        "  --icp_threshold 0.5 (default, mm)\n"
        "  --max_icp_iterations 1000 (default)\n"
        "  --seed 0 (default: a new one each run)\n"
        "  --transform_output FILE   the 4x4 matrix\n"
        "  --result_json FILE        the global and ICP fitness and RMSE\n"
        "  --export_path FILE.stl    the source moved\n"
        "  --viz N                   accepted, nothing shown\n";
}

struct BadArgs : std::runtime_error {
    using std::runtime_error::runtime_error;
};

Options parse(int argc, char** argv) {
    std::vector<std::string> pos;
    std::map<std::string, std::string> kv;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_help();
            std::exit(0);
        }
        if (a.rfind("--", 0) == 0) {
            if (i + 1 >= argc) throw BadArgs(a + " needs a value");
            kv[a] = argv[++i];
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() != 2) throw BadArgs("give the source and the target files");
    Options o;
    o.source = pos[0];
    o.target = pos[1];
    auto get = [&](const char* key, auto& out) {
        const auto it = kv.find(key);
        if (it == kv.end()) return;
        using T = std::decay_t<decltype(out)>;
        try {
            if constexpr (std::is_same_v<T, std::string>) out = it->second;
            else if constexpr (std::is_same_v<T, int>) out = std::stoi(it->second);
            else if constexpr (std::is_same_v<T, std::uint64_t>) out = std::stoull(it->second);
            else out = std::stod(it->second);
        } catch (const std::exception&) {
            throw BadArgs(std::string(key) + ": not a number: " + it->second);
        }
    };
    int mutual = 0, viz = 0;
    get("--method", o.method);
    get("--registration_mode", o.mode);
    get("--mutual_filter", mutual);
    get("--voxel_size", o.voxel_size);
    get("--distance_multiplier", o.distance_multiplier);
    get("--max_iterations", o.max_iterations);
    get("--confidance", o.confidence);         // MeshRegAuto's spelling
    get("--confidence", o.confidence);
    get("--icp_threshold", o.icp_threshold);
    get("--max_icp_iterations", o.max_icp_iterations);
    get("--seed", o.seed);
    get("--viz", viz);
    get("--export_path", o.export_path);
    get("--transform_output", o.transform_output);
    get("--result_json", o.result_json);
    if (o.method != "feature_matching" && o.method != "correspondence")
        throw BadArgs("--method must be feature_matching or correspondence");
    if (o.mode != "full" && o.mode != "global" && o.mode != "icp")
        throw BadArgs("--registration_mode must be full, global or icp");
    if (mutual != 0 && mutual != 1) throw BadArgs("--mutual_filter must be 0 or 1");
    if (!(o.voxel_size > 0) || !(o.distance_multiplier > 0) || !(o.icp_threshold > 0))
        throw BadArgs("--voxel_size, --distance_multiplier and --icp_threshold must be > 0");
    if (o.max_iterations < 1 || o.max_icp_iterations < 0)
        throw BadArgs("--max_iterations must be >= 1, --max_icp_iterations >= 0");
    o.mutual_filter = mutual == 1;
    o.confidence = std::clamp(o.confidence, 0.0, 1.0);   // as MeshRegAuto
    return o;
}

reg::Points as_points(const std::vector<double>& xyz) {
    reg::Points p(xyz.size() / 3);
    for (std::size_t i = 0; i < p.size(); ++i) p[i] = {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
    return p;
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

bool write_matrix(const std::string& path, const Eigen::Matrix4d& m) {
    std::ofstream f(path);
    if (!f) return false;
    f << std::setprecision(17);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) f << m(r, c) << (c < 3 ? " " : "\n");
    return bool(f);
}

// the source mesh moved, as a binary STL (its normals from the moved triangles)
bool write_moved_stl(const std::string& path, const Mesh& mesh, const Eigen::Matrix4d& t) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    char header[80] = "odent_engine register: the source moved";
    f.write(header, 80);
    const auto n = static_cast<std::uint32_t>(mesh.triangles());
    f.write(reinterpret_cast<const char*>(&n), 4);
    for (std::uint32_t k = 0; k < n; ++k) {
        Eigen::Vector3d v[3];
        for (int j = 0; j < 3; ++j) {
            const std::uint32_t i = mesh.tris[3 * k + j];
            const Eigen::Vector4d p(mesh.points[3 * i], mesh.points[3 * i + 1], mesh.points[3 * i + 2], 1.0);
            v[j] = (t * p).head<3>();
        }
        Eigen::Vector3d nrm = (v[1] - v[0]).cross(v[2] - v[0]);
        if (nrm.norm() > 0) nrm.normalize();
        float rec[12];
        for (int j = 0; j < 3; ++j) rec[j] = float(nrm(j));
        for (int c = 0; c < 3; ++c)
            for (int j = 0; j < 3; ++j) rec[3 + 3 * c + j] = float(v[c](j));
        f.write(reinterpret_cast<const char*>(rec), sizeof rec);
        const std::uint16_t attr = 0;
        f.write(reinterpret_cast<const char*>(&attr), 2);
    }
    return bool(f);
}

}  // namespace

int register_main(int argc, char** argv) {
    Options o;
    try {
        o = parse(argc, argv);
    } catch (const BadArgs& er) {
        std::cerr << "register: " << er.what() << "\n";
        print_help();
        return 2;
    }
    try {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<double> src_xyz, tgt_xyz;
        try {
            src_xyz = registration_points(o.source);
            tgt_xyz = registration_points(o.target);
        } catch (const std::exception& er) {
            std::cerr << "Unable to read source or target file: " << er.what() << "\n";
            return 1;
        }
        if (src_xyz.empty() || tgt_xyz.empty()) {
            std::cerr << "Unable to read source or target file: no points\n";
            return 1;
        }
        std::cout << "Points: source " << src_xyz.size() / 3 << ", target " << tgt_xyz.size() / 3
                  << " (" << reg::thread_count() << " threads)\n";

        // preprocess (MeshRegAuto: VoxelDownSample, EstimateNormals(Hybrid(2v, 30), fast),
        // ComputeFPFHFeature(Hybrid(5v, 100)))
        const double v = o.voxel_size;
        auto prepare = [&](const std::vector<double>& xyz) {
            reg::Cloud down = reg::voxel_down_sample(as_points(xyz), v);
            reg::estimate_normals(down, 2.0 * v, 30);
            reg::Feature fpfh = reg::compute_fpfh(down, 5.0 * v, 100);
            return std::make_pair(std::move(down), std::move(fpfh));
        };
        auto [src, src_fpfh] = prepare(src_xyz);
        auto [tgt, tgt_fpfh] = prepare(tgt_xyz);
        std::cout << "Down-sampled: source " << src.points.size() << ", target " << tgt.points.size()
                  << " (" << std::fixed << std::setprecision(2) << seconds_since(t0) << " s)\n";
        std::cout.unsetf(std::ios::floatfield);

        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        reg::Result global, icp;
        bool have_global = false, have_icp = false;
        if (o.mode == "full" || o.mode == "global") {
            const auto t1 = std::chrono::steady_clock::now();
            const reg::CorrespondenceSet corres =
                    o.method == "feature_matching"
                            ? reg::correspondences_from_features(src_fpfh, tgt_fpfh, o.mutual_filter)
                            : reg::feature_matches(src_fpfh, tgt_fpfh, o.mutual_filter);
            reg::RansacOptions ro;
            ro.max_correspondence_distance = ro.checker_distance = v * o.distance_multiplier;
            ro.max_iteration = o.max_iterations;
            ro.confidence = o.confidence;
            ro.seed = o.seed;
            global = reg::ransac_on_correspondences(src, tgt, corres, ro);
            have_global = true;
            transform = global.transformation;
            std::cout << "Global: fitness " << global.fitness << ", RMSE " << global.inlier_rmse
                      << " (" << corres.size() << " matches, " << seconds_since(t1) << " s)\n";
        }
        if (o.mode == "full" || o.mode == "icp") {
            const auto t2 = std::chrono::steady_clock::now();
            icp = reg::icp_point_to_plane(src, tgt, o.icp_threshold, transform, 1e-6, 1e-6,
                                          o.max_icp_iterations);
            have_icp = true;
            transform = icp.transformation;
            std::cout << "ICP: fitness " << icp.fitness << ", RMSE " << icp.inlier_rmse << " ("
                      << seconds_since(t2) << " s)\n";
        }
        std::cout << "Matrix:\n" << transform << "\n" << "Total " << seconds_since(t0) << " s\n";

        if (!o.export_path.empty()) {
            Mesh mesh;
            try {
                mesh = read_mesh(o.source);
            } catch (const std::exception& er) {
                std::cerr << "Unable to read source file: " << er.what() << "\n";
                return 1;
            }
            if (!write_moved_stl(o.export_path, mesh, transform)) {
                std::cerr << "Unable to open file " << o.export_path << "\n";
                return 1;
            }
        }
        if (!o.transform_output.empty() && !write_matrix(o.transform_output, transform)) {
            std::cerr << "Unable to open file " << o.transform_output << "\n";
            return 1;
        }
        if (!o.result_json.empty()) {
            std::ofstream f(o.result_json);
            f << std::setprecision(17) << "{";
            if (have_global)
                f << "\"global\": {\"fitness\": " << global.fitness << ", \"rmse\": " << global.inlier_rmse << "}";
            if (have_global && have_icp) f << ", ";
            if (have_icp)
                f << "\"icp\": {\"fitness\": " << icp.fitness << ", \"rmse\": " << icp.inlier_rmse << "}";
            f << ", \"seconds\": " << seconds_since(t0) << "}\n";
            if (!f) {
                std::cerr << "Unable to open file " << o.result_json << "\n";
                return 1;
            }
        }
        return 0;
    } catch (const std::exception& er) {
        std::cerr << "register: " << er.what() << "\n";
        return 4;
    }
}

}  // namespace odent
