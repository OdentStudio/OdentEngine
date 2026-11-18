// The registration of a scan onto its arch (the AI align): what MeshRegAuto.exe did
// with Open3D 0.19, ported from Open3D's own code (MIT, www.open3d.org) so that the
// engine needs neither: voxel down-sampling, normals, FPFH features, RANSAC on feature
// matches, point-to-plane ICP. Same formulas, constants and order of operations as
// Open3D (the file and function each comes from is named at it); KD-trees by nanoflann
// 1.5.0 as in Open3D; threads by std::thread in the fixed blocks OpenMP's static
// schedule gives.

#pragma once

#include <cstdint>
#include <vector>

#include "itk_eigen.h"
#include ITK_EIGEN(Dense)

namespace odent::reg {

using Points = std::vector<Eigen::Vector3d>;
using Feature = Eigen::MatrixXd;                    // 33 x n (Open3D's Feature::data_)
using CorrespondenceSet = std::vector<Eigen::Vector2i>;

struct Cloud {
    Points points;
    Points normals;                                 // empty, or one per point
};

struct Result {                                     // Open3D's RegistrationResult
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    CorrespondenceSet correspondence_set;
    double fitness = 0.0;
    double inlier_rmse = 0.0;
    bool better_ransac_than(const Result& o) const {
        return fitness > o.fitness || (fitness == o.fitness && inlier_rmse < o.inlier_rmse);
    }
};

// geometry/PointCloud.cpp VoxelDownSample: the mean of the points in each voxel
Cloud voxel_down_sample(const Points& points, double voxel);
// geometry/EstimateNormals.cpp (fast normal computation, no orientation): hybrid search
void estimate_normals(Cloud& cloud, double radius, int max_nn);
// pipelines/registration/Feature.cpp ComputeFPFHFeature: hybrid search
Feature compute_fpfh(const Cloud& cloud, double radius, int max_nn);
// Feature.cpp CorrespondencesFromFeatures: nearest feature source -> target; with the
// mutual filter, the pairs both ways, unless fewer than `ratio` of the source's (then all)
CorrespondenceSet correspondences_from_features(const Feature& source, const Feature& target,
                                                bool mutual_filter, double ratio = 0.1);
// MeshRegAuto's own "correspondence" method: the same nearest features, the mutual
// pairs kept without that fall back
CorrespondenceSet feature_matches(const Feature& source, const Feature& target, bool mutual);

struct RansacOptions {
    double max_correspondence_distance = 0.75;      // voxel x distance multiplier
    int ransac_n = 3;
    double edge_length_similarity = 0.9;            // CorrespondenceCheckerBasedOnEdgeLength
    double checker_distance = 0.75;                 // CorrespondenceCheckerBasedOnDistance
    int max_iteration = 1000000;
    double confidence = 0.999;
    std::uint64_t seed = 0;                         // 0: a new one each run (Open3D's random_device)
};
// Registration.cpp RegistrationRANSACBasedOnCorrespondence, point-to-point (Umeyama,
// no scaling)
Result ransac_on_correspondences(const Cloud& source, const Cloud& target,
                                 const CorrespondenceSet& corres, const RansacOptions& o);

// Registration.cpp RegistrationICP with TransformationEstimationPointToPlane (L2):
// the target needs normals
Result icp_point_to_plane(const Cloud& source, const Cloud& target, double max_distance,
                          const Eigen::Matrix4d& init, double relative_fitness,
                          double relative_rmse, int max_iteration);

// the threads the loops use (Open3D: OpenMP's maximum, the logical cores)
int thread_count();

}  // namespace odent::reg
