// The registration (see registration.h): ported from Open3D 0.19 (MIT licence,
// Copyright (c) 2018-2024 www.open3d.org), the source file of each part named at it.

#include "registration.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include ITK_EIGEN(Geometry)
#include <nanoflann.hpp>

namespace odent::reg {

namespace {

constexpr double kPi = 3.14159265358979323846;

// --- threads: OpenMP's static schedule, one block of the range per thread -------------

void parallel_blocks(int n, const std::function<void(int thread, int begin, int end)>& f) {
    const int threads = std::max(1, std::min(thread_count(), n));
    if (threads == 1) {
        f(0, 0, n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        // OpenMP static: the first n % threads blocks one longer
        const int base = n / threads, extra = n % threads;
        const int begin = t * base + std::min(t, extra);
        const int end = begin + base + (t < extra ? 1 : 0);
        pool.emplace_back(f, t, begin, end);
    }
    for (auto& th : pool) th.join();
}

// --- geometry/KDTreeFlann.cpp: nanoflann over the columns of a matrix ------------------

class KDTree {
public:
    explicit KDTree(Eigen::MatrixXd data) : data_(std::move(data)) {
        if (data_.size() == 0) throw std::runtime_error("no points to search");
        index_ = std::make_unique<Index>(data_.rows(), std::cref(data_), 15);   // (builds it)
    }
    // the index refers to data_: never copied or moved (returned by guaranteed elision)
    KDTree(const KDTree&) = delete;
    KDTree(KDTree&&) = delete;
    KDTree& operator=(const KDTree&) = delete;
    KDTree& operator=(KDTree&&) = delete;
    static KDTree of(const Points& p) {
        Eigen::MatrixXd m(3, static_cast<Eigen::Index>(p.size()));
        for (std::size_t i = 0; i < p.size(); ++i) m.col(static_cast<Eigen::Index>(i)) = p[i];
        return KDTree(std::move(m));
    }
    int knn(const double* q, int k, std::vector<int>& indices, std::vector<double>& d2) const {
        indices.resize(k);
        d2.resize(k);
        std::vector<Eigen::Index> idx(k);
        const int n = static_cast<int>(index_->index_->knnSearch(q, k, idx.data(), d2.data()));
        indices.resize(n);
        d2.resize(n);
        std::copy_n(idx.begin(), n, indices.begin());
        return n;
    }
    // SearchHybrid: the max_nn nearest, cut at the radius
    int hybrid(const double* q, double radius, int max_nn, std::vector<int>& indices,
               std::vector<double>& d2) const {
        d2.resize(max_nn);
        std::vector<Eigen::Index> idx(max_nn);
        int k = static_cast<int>(index_->index_->knnSearch(q, max_nn, idx.data(), d2.data()));
        k = static_cast<int>(std::distance(
                d2.begin(), std::lower_bound(d2.begin(), d2.begin() + k, radius * radius)));
        indices.resize(k);
        d2.resize(k);
        std::copy_n(idx.begin(), k, indices.begin());
        return k;
    }

private:
    using Index = nanoflann::KDTreeEigenMatrixAdaptor<const Eigen::MatrixXd, -1,
                                                       nanoflann::metric_L2, false>;
    Eigen::MatrixXd data_;
    std::unique_ptr<Index> index_;
};

Eigen::Vector3d transform_point(const Eigen::Matrix4d& t, const Eigen::Vector3d& p) {
    return (t * Eigen::Vector4d(p(0), p(1), p(2), 1.0)).head<3>();
}

Points transformed(const Points& p, const Eigen::Matrix4d& t) {
    Points out(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) out[i] = transform_point(t, p[i]);
    return out;
}

// --- geometry/EstimateNormals.cpp -----------------------------------------------------

Eigen::Vector3d ComputeEigenvector0(const Eigen::Matrix3d& A, double eval0) {
    Eigen::Vector3d row0(A(0, 0) - eval0, A(0, 1), A(0, 2));
    Eigen::Vector3d row1(A(0, 1), A(1, 1) - eval0, A(1, 2));
    Eigen::Vector3d row2(A(0, 2), A(1, 2), A(2, 2) - eval0);
    Eigen::Vector3d r0xr1 = row0.cross(row1);
    Eigen::Vector3d r0xr2 = row0.cross(row2);
    Eigen::Vector3d r1xr2 = row1.cross(row2);
    double d0 = r0xr1.dot(r0xr1);
    double d1 = r0xr2.dot(r0xr2);
    double d2 = r1xr2.dot(r1xr2);
    double dmax = d0;
    int imax = 0;
    if (d1 > dmax) {
        dmax = d1;
        imax = 1;
    }
    if (d2 > dmax) imax = 2;
    if (imax == 0) return r0xr1 / std::sqrt(d0);
    if (imax == 1) return r0xr2 / std::sqrt(d1);
    return r1xr2 / std::sqrt(d2);
}

Eigen::Vector3d ComputeEigenvector1(const Eigen::Matrix3d& A, const Eigen::Vector3d& evec0,
                                    double eval1) {
    Eigen::Vector3d U, V;
    if (std::abs(evec0(0)) > std::abs(evec0(1))) {
        double inv_length = 1 / std::sqrt(evec0(0) * evec0(0) + evec0(2) * evec0(2));
        U << -evec0(2) * inv_length, 0, evec0(0) * inv_length;
    } else {
        double inv_length = 1 / std::sqrt(evec0(1) * evec0(1) + evec0(2) * evec0(2));
        U << 0, evec0(2) * inv_length, -evec0(1) * inv_length;
    }
    V = evec0.cross(U);
    Eigen::Vector3d AU(A(0, 0) * U(0) + A(0, 1) * U(1) + A(0, 2) * U(2),
                       A(0, 1) * U(0) + A(1, 1) * U(1) + A(1, 2) * U(2),
                       A(0, 2) * U(0) + A(1, 2) * U(1) + A(2, 2) * U(2));
    Eigen::Vector3d AV = {A(0, 0) * V(0) + A(0, 1) * V(1) + A(0, 2) * V(2),
                          A(0, 1) * V(0) + A(1, 1) * V(1) + A(1, 2) * V(2),
                          A(0, 2) * V(0) + A(1, 2) * V(1) + A(2, 2) * V(2)};
    double m00 = U(0) * AU(0) + U(1) * AU(1) + U(2) * AU(2) - eval1;
    double m01 = U(0) * AV(0) + U(1) * AV(1) + U(2) * AV(2);
    double m11 = V(0) * AV(0) + V(1) * AV(1) + V(2) * AV(2) - eval1;
    double absM00 = std::abs(m00);
    double absM01 = std::abs(m01);
    double absM11 = std::abs(m11);
    double max_abs_comp;
    if (absM00 >= absM11) {
        max_abs_comp = std::max(absM00, absM01);
        if (max_abs_comp > 0) {
            if (absM00 >= absM01) {
                m01 /= m00;
                m00 = 1 / std::sqrt(1 + m01 * m01);
                m01 *= m00;
            } else {
                m00 /= m01;
                m01 = 1 / std::sqrt(1 + m00 * m00);
                m00 *= m01;
            }
            return m01 * U - m00 * V;
        }
        return U;
    }
    max_abs_comp = std::max(absM11, absM01);
    if (max_abs_comp > 0) {
        if (absM11 >= absM01) {
            m01 /= m11;
            m11 = 1 / std::sqrt(1 + m01 * m01);
            m01 *= m11;
        } else {
            m11 /= m01;
            m01 = 1 / std::sqrt(1 + m11 * m11);
            m11 *= m01;
        }
        return m11 * U - m01 * V;
    }
    return U;
}

// the eigenvector of the smallest eigenvalue (geometrictools' robust 3x3 method)
Eigen::Vector3d FastEigen3x3(const Eigen::Matrix3d& covariance) {
    Eigen::Matrix3d A = covariance;
    double max_coeff = A.maxCoeff();
    if (max_coeff == 0) return Eigen::Vector3d::Zero();
    A /= max_coeff;
    double norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    if (norm > 0) {
        Eigen::Vector3d eval;
        Eigen::Vector3d evec0, evec1, evec2;
        double q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;
        double b00 = A(0, 0) - q;
        double b11 = A(1, 1) - q;
        double b22 = A(2, 2) - q;
        double p = std::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);
        double c00 = b11 * b22 - A(1, 2) * A(1, 2);
        double c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
        double c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
        double det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);
        double half_det = det * 0.5;
        half_det = std::min(std::max(half_det, -1.0), 1.0);
        double angle = std::acos(half_det) / (double)3;
        double const two_thirds_pi = 2.09439510239319549;
        double beta2 = std::cos(angle) * 2;
        double beta0 = std::cos(angle + two_thirds_pi) * 2;
        double beta1 = -(beta0 + beta2);
        eval(0) = q + p * beta0;
        eval(1) = q + p * beta1;
        eval(2) = q + p * beta2;
        if (half_det >= 0) {
            evec2 = ComputeEigenvector0(A, eval(2));
            if (eval(2) < eval(0) && eval(2) < eval(1)) return evec2;
            evec1 = ComputeEigenvector1(A, evec2, eval(1));
            if (eval(1) < eval(0) && eval(1) < eval(2)) return evec1;
            evec0 = evec1.cross(evec2);
            return evec0;
        }
        evec0 = ComputeEigenvector0(A, eval(0));
        if (eval(0) < eval(1) && eval(0) < eval(2)) return evec0;
        evec1 = ComputeEigenvector1(A, evec0, eval(1));
        if (eval(1) < eval(0) && eval(1) < eval(2)) return evec1;
        evec2 = evec0.cross(evec1);
        return evec2;
    }
    A *= max_coeff;
    if (A(0, 0) < A(1, 1) && A(0, 0) < A(2, 2)) return Eigen::Vector3d(1, 0, 0);
    if (A(1, 1) < A(0, 0) && A(1, 1) < A(2, 2)) return Eigen::Vector3d(0, 1, 0);
    return Eigen::Vector3d(0, 0, 1);
}

// utility/Eigen.cpp ComputeCovariance
Eigen::Matrix3d ComputeCovariance(const Points& points, const std::vector<int>& indices) {
    if (indices.empty()) return Eigen::Matrix3d::Identity();
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 9, 1> cumulants;
    cumulants.setZero();
    for (const auto& idx : indices) {
        const Eigen::Vector3d& point = points[idx];
        cumulants(0) += point(0);
        cumulants(1) += point(1);
        cumulants(2) += point(2);
        cumulants(3) += point(0) * point(0);
        cumulants(4) += point(0) * point(1);
        cumulants(5) += point(0) * point(2);
        cumulants(6) += point(1) * point(1);
        cumulants(7) += point(1) * point(2);
        cumulants(8) += point(2) * point(2);
    }
    cumulants /= (double)indices.size();
    covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
    covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
    covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
    covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
    covariance(1, 0) = covariance(0, 1);
    covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
    covariance(2, 0) = covariance(0, 2);
    covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
    covariance(2, 1) = covariance(1, 2);
    return covariance;
}

// --- pipelines/registration/Feature.cpp -----------------------------------------------

Eigen::Vector4d ComputePairFeatures(const Eigen::Vector3d& p1, const Eigen::Vector3d& n1,
                                    const Eigen::Vector3d& p2, const Eigen::Vector3d& n2) {
    Eigen::Vector4d result;
    Eigen::Vector3d dp2p1 = p2 - p1;
    result(3) = dp2p1.norm();
    if (result(3) == 0.0) return Eigen::Vector4d::Zero();
    auto n1_copy = n1;
    auto n2_copy = n2;
    double angle1 = n1_copy.dot(dp2p1) / result(3);
    double angle2 = n2_copy.dot(dp2p1) / result(3);
    if (std::acos(std::fabs(angle1)) > std::acos(std::fabs(angle2))) {
        n1_copy = n2;
        n2_copy = n1;
        dp2p1 *= -1.0;
        result(2) = -angle2;
    } else {
        result(2) = angle1;
    }
    auto v = dp2p1.cross(n1_copy);
    double v_norm = v.norm();
    if (v_norm == 0.0) return Eigen::Vector4d::Zero();
    v /= v_norm;
    auto w = n1_copy.cross(v);
    result(1) = v.dot(n2_copy);
    result(0) = std::atan2(w.dot(n2_copy), n1_copy.dot(n2_copy));
    return result;
}

// the nearest of `to`'s features for each of `from`'s (a KD-tree on `to`)
std::vector<int> nearest_features(const Feature& from, const Feature& to) {
    const KDTree tree{Eigen::MatrixXd(to)};
    const int n = static_cast<int>(from.cols());
    std::vector<int> best(n, -1);
    parallel_blocks(n, [&](int, int begin, int end) {
        std::vector<int> idx(1);
        std::vector<double> d2(1);
        Eigen::VectorXd q;
        for (int i = begin; i < end; ++i) {
            q = from.col(i);
            if (tree.knn(q.data(), 1, idx, d2) > 0) best[i] = idx[0];
        }
    });
    return best;
}

// --- pipelines/registration/Registration.cpp ------------------------------------------

// GetRegistrationResultAndCorrespondences (source already moved); `parallel` false
// inside RANSAC's threads (OpenMP's nested regions run on one thread there)
Result evaluate(const Points& source, const KDTree& target_tree,
                double max_distance, const Eigen::Matrix4d& transformation, bool parallel) {
    Result result;
    result.transformation = transformation;
    if (max_distance <= 0.0) return result;
    const int n = static_cast<int>(source.size());
    double error2 = 0.0;
    std::mutex lock;
    auto block = [&](int, int begin, int end) {
        double error2_private = 0.0;
        CorrespondenceSet corres_private;
        std::vector<int> indices(1);
        std::vector<double> dists(1);
        for (int i = begin; i < end; ++i) {
            if (target_tree.hybrid(source[i].data(), max_distance, 1, indices, dists) > 0) {
                error2_private += dists[0];
                corres_private.emplace_back(i, indices[0]);
            }
        }
        std::lock_guard<std::mutex> hold(lock);
        result.correspondence_set.insert(result.correspondence_set.end(), corres_private.begin(),
                                         corres_private.end());
        error2 += error2_private;
    };
    if (parallel) parallel_blocks(n, block);
    else block(0, 0, n);
    if (result.correspondence_set.empty()) {
        result.fitness = 0.0;
        result.inlier_rmse = 0.0;
    } else {
        const auto count = static_cast<double>(result.correspondence_set.size());
        result.fitness = count / static_cast<double>(source.size());
        result.inlier_rmse = std::sqrt(error2 / count);
    }
    return result;
}

double inlier_correspondence_ratio(const Points& source, const Points& target,
                                   const CorrespondenceSet& corres, double max_distance) {
    int inlier = 0;
    const double max_dis2 = max_distance * max_distance;
    for (const auto& c : corres)
        if ((source[c[0]] - target[c[1]]).squaredNorm() < max_dis2) ++inlier;
    return double(inlier) / double(corres.size());
}

// TransformationEstimationPointToPoint (no scaling): Umeyama
Eigen::Matrix4d point_to_point(const Points& source, const Points& target,
                               const CorrespondenceSet& corres) {
    if (corres.empty()) return Eigen::Matrix4d::Identity();
    Eigen::MatrixXd s(3, corres.size()), t(3, corres.size());
    for (std::size_t i = 0; i < corres.size(); ++i) {
        s.col(static_cast<Eigen::Index>(i)) = source[corres[i][0]];
        t.col(static_cast<Eigen::Index>(i)) = target[corres[i][1]];
    }
    return Eigen::umeyama(s, t, false);
}

// CorrespondenceCheckerBasedOnEdgeLength and ...BasedOnDistance
bool checks_pass(const Points& source, const Points& target, const CorrespondenceSet& corres,
                 const Eigen::Matrix4d& transformation, const RansacOptions& o) {
    for (std::size_t i = 0; i < corres.size(); ++i) {
        for (std::size_t j = i + 1; j < corres.size(); ++j) {
            const double dis_source = (source[corres[i](0)] - source[corres[j](0)]).norm();
            const double dis_target = (target[corres[i](1)] - target[corres[j](1)]).norm();
            if (dis_source < dis_target * o.edge_length_similarity ||
                dis_target < dis_source * o.edge_length_similarity)
                return false;
        }
    }
    for (const auto& c : corres) {
        const Eigen::Vector3d pt = transform_point(transformation, source[c(0)]);
        if ((target[c(1)] - pt).norm() > o.checker_distance) return false;
    }
    return true;
}

// utility/Eigen.cpp TransformVector6dToMatrix4d
Eigen::Matrix4d vector6_to_matrix(const Eigen::Matrix<double, 6, 1>& v) {
    Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
    out.block<3, 3>(0, 0) = (Eigen::AngleAxisd(v(2), Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(v(1), Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(v(0), Eigen::Vector3d::UnitX()))
                                    .matrix();
    out.block<3, 1>(0, 3) = v.block<3, 1>(3, 0);
    return out;
}

// TransformationEstimationPointToPlane::ComputeTransformation (L2 kernel: weight 1):
// ComputeJTJandJTr, then SolveJacobianSystemAndObtainExtrinsicMatrix (LDLT)
Eigen::Matrix4d point_to_plane(const Points& source, const Cloud& target,
                               const CorrespondenceSet& corres) {
    if (corres.empty() || target.normals.empty()) return Eigen::Matrix4d::Identity();
    using Mat6 = Eigen::Matrix<double, 6, 6>;
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    Mat6 JTJ = Mat6::Zero();
    Vec6 JTr = Vec6::Zero();
    std::mutex lock;
    parallel_blocks(static_cast<int>(corres.size()), [&](int, int begin, int end) {
        Mat6 JTJ_private = Mat6::Zero();
        Vec6 JTr_private = Vec6::Zero();
        Vec6 J_r = Vec6::Zero();
        for (int i = begin; i < end; ++i) {
            const Eigen::Vector3d& vs = source[corres[i][0]];
            const Eigen::Vector3d& vt = target.points[corres[i][1]];
            const Eigen::Vector3d& nt = target.normals[corres[i][1]];
            const double r = (vs - vt).dot(nt);
            const double w = 1.0;
            J_r.block<3, 1>(0, 0) = vs.cross(nt);
            J_r.block<3, 1>(3, 0) = nt;
            JTJ_private.noalias() += J_r * w * J_r.transpose();
            JTr_private.noalias() += J_r * w * r;
        }
        std::lock_guard<std::mutex> hold(lock);
        JTJ += JTJ_private;
        JTr += JTr_private;
    });
    const Eigen::MatrixXd A = JTJ;
    const Eigen::VectorXd b = -JTr;
    const Eigen::VectorXd x = A.ldlt().solve(b);
    return vector6_to_matrix(x);
}

}  // namespace

int thread_count() {
    static const int n = std::max(1u, std::thread::hardware_concurrency());
    return n;
}

// geometry/PointCloud.cpp VoxelDownSample (the voxels in the order first met)
Cloud voxel_down_sample(const Points& points, double voxel) {
    if (voxel <= 0.0) throw std::runtime_error("voxel_size <= 0");
    if (points.empty()) return {};
    Eigen::Vector3d min_bound = points[0], max_bound = points[0];
    for (const auto& p : points) {
        min_bound = min_bound.cwiseMin(p);
        max_bound = max_bound.cwiseMax(p);
    }
    const Eigen::Vector3d voxel3(voxel, voxel, voxel);
    const Eigen::Vector3d voxel_min_bound = min_bound - voxel3 * 0.5;
    const Eigen::Vector3d voxel_max_bound = max_bound + voxel3 * 0.5;
    if (voxel * std::numeric_limits<int>::max() < (voxel_max_bound - voxel_min_bound).maxCoeff())
        throw std::runtime_error("voxel_size is too small");
    struct Hash {
        std::size_t operator()(const Eigen::Vector3i& v) const {
            std::size_t seed = 0;                     // utility::hash_eigen (boost's combine)
            for (int i = 0; i < 3; ++i)
                seed ^= std::hash<int>()(v(i)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
    struct Acc {
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        int n = 0;
    };
    std::unordered_map<Eigen::Vector3i, int, Hash> slot;
    std::vector<Acc> acc;
    for (const auto& p : points) {
        const Eigen::Vector3d ref = (p - voxel_min_bound) / voxel;
        const Eigen::Vector3i index(int(std::floor(ref(0))), int(std::floor(ref(1))),
                                    int(std::floor(ref(2))));
        auto it = slot.find(index);
        if (it == slot.end()) {
            it = slot.emplace(index, static_cast<int>(acc.size())).first;
            acc.emplace_back();
        }
        acc[it->second].sum += p;
        acc[it->second].n++;
    }
    Cloud out;
    out.points.reserve(acc.size());
    for (const auto& a : acc) out.points.push_back(a.sum / double(a.n));
    return out;
}

// geometry/EstimateNormals.cpp EstimateNormals (+ PointCloud::EstimatePerPointCovariances)
void estimate_normals(Cloud& cloud, double radius, int max_nn) {
    const KDTree tree = KDTree::of(cloud.points);
    const int n = static_cast<int>(cloud.points.size());
    cloud.normals.assign(n, Eigen::Vector3d::Zero());
    parallel_blocks(n, [&](int, int begin, int end) {
        std::vector<int> indices;
        std::vector<double> distance2;
        for (int i = begin; i < end; ++i) {
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
            if (tree.hybrid(cloud.points[i].data(), radius, max_nn, indices, distance2) >= 3)
                covariance = ComputeCovariance(cloud.points, indices);
            Eigen::Vector3d normal = FastEigen3x3(covariance);
            if (normal.norm() == 0.0) normal = Eigen::Vector3d(0.0, 0.0, 1.0);
            cloud.normals[i] = normal;
        }
    });
}

// pipelines/registration/Feature.cpp ComputeSPFHFeature + ComputeFPFHFeature
Feature compute_fpfh(const Cloud& cloud, double radius, int max_nn) {
    if (cloud.normals.size() != cloud.points.size())
        throw std::runtime_error("FPFH needs the cloud's normals");
    const KDTree tree = KDTree::of(cloud.points);
    const int n = static_cast<int>(cloud.points.size());
    Feature spfh = Feature::Zero(33, n);
    parallel_blocks(n, [&](int, int begin, int end) {
        std::vector<int> indices;
        std::vector<double> distance2;
        for (int i = begin; i < end; ++i) {
            const auto& point = cloud.points[i];
            const auto& normal = cloud.normals[i];
            if (tree.hybrid(point.data(), radius, max_nn, indices, distance2) > 1) {
                const double hist_incr = 100.0 / (double)(indices.size() - 1);
                for (std::size_t k = 1; k < indices.size(); k++) {   // (k 0: the point itself)
                    const auto pf = ComputePairFeatures(point, normal, cloud.points[indices[k]],
                                                        cloud.normals[indices[k]]);
                    int h_index = (int)(std::floor(11 * (pf(0) + kPi) / (2.0 * kPi)));
                    if (h_index < 0) h_index = 0;
                    if (h_index >= 11) h_index = 10;
                    spfh(h_index, i) += hist_incr;
                    h_index = (int)(std::floor(11 * (pf(1) + 1.0) * 0.5));
                    if (h_index < 0) h_index = 0;
                    if (h_index >= 11) h_index = 10;
                    spfh(h_index + 11, i) += hist_incr;
                    h_index = (int)(std::floor(11 * (pf(2) + 1.0) * 0.5));
                    if (h_index < 0) h_index = 0;
                    if (h_index >= 11) h_index = 10;
                    spfh(h_index + 22, i) += hist_incr;
                }
            }
        }
    });
    Feature feature = Feature::Zero(33, n);
    parallel_blocks(n, [&](int, int begin, int end) {
        std::vector<int> indices;
        std::vector<double> distance2;
        for (int i = begin; i < end; ++i) {
            if (tree.hybrid(cloud.points[i].data(), radius, max_nn, indices, distance2) > 1) {
                double sum[3] = {0.0, 0.0, 0.0};
                for (std::size_t k = 1; k < indices.size(); k++) {
                    const double dist = distance2[k];
                    if (dist == 0.0) continue;
                    for (int j = 0; j < 33; j++) {
                        const double val = spfh(j, indices[k]) / dist;
                        sum[j / 11] += val;
                        feature(j, i) += val;
                    }
                }
                for (int j = 0; j < 3; j++)
                    if (sum[j] != 0.0) sum[j] = 100.0 / sum[j];
                for (int j = 0; j < 33; j++) {
                    feature(j, i) *= sum[j / 11];
                    feature(j, i) += spfh(j, i);
                }
            }
        }
    });
    return feature;
}

CorrespondenceSet correspondences_from_features(const Feature& source, const Feature& target,
                                                bool mutual_filter, double ratio) {
    const std::vector<int> fwd = nearest_features(source, target);
    CorrespondenceSet ij;
    ij.reserve(fwd.size());
    for (int i = 0; i < static_cast<int>(fwd.size()); ++i) ij.emplace_back(i, fwd[i]);
    if (!mutual_filter) return ij;
    const std::vector<int> back = nearest_features(target, source);
    CorrespondenceSet mutual;
    for (int i = 0; i < static_cast<int>(fwd.size()); ++i)
        if (back[fwd[i]] == i) mutual.emplace_back(i, fwd[i]);
    if (int(mutual.size()) >= int(ratio * static_cast<double>(fwd.size()))) return mutual;
    return ij;                                     // too few after the filter: all of them
}

CorrespondenceSet feature_matches(const Feature& source, const Feature& target, bool mutual) {
    const std::vector<int> fwd = nearest_features(source, target);
    std::vector<int> back;
    if (mutual) back = nearest_features(target, source);
    CorrespondenceSet out;
    for (int i = 0; i < static_cast<int>(fwd.size()); ++i) {
        const int j = fwd[i];
        if (j < 0) continue;
        if (mutual && back[j] != i) continue;
        out.emplace_back(i, j);
    }
    return out;
}

// Registration.cpp RegistrationRANSACBasedOnCorrespondence. As there: the iterations
// split in fixed blocks over the threads, an iteration run only while its number is
// below the estimate of how many are needed (est_k), lowered by any thread
Result ransac_on_correspondences(const Cloud& source, const Cloud& target,
                                 const CorrespondenceSet& corres, const RansacOptions& o) {
    if (o.ransac_n < 3 || (int)corres.size() < o.ransac_n || o.max_correspondence_distance <= 0.0)
        return Result();
    const double confidence = std::clamp(o.confidence, 0.0, 1.0);
    const KDTree tree = KDTree::of(target.points);
    std::atomic<int> est_k_global{o.max_iteration};
    std::mutex lock;
    Result best;
    std::random_device rd;
    const std::uint64_t seed = o.seed ? o.seed : (std::uint64_t(rd()) << 32 | rd());
    parallel_blocks(o.max_iteration, [&](int thread, int begin, int end) {
        CorrespondenceSet sample(o.ransac_n);
        Result best_local;
        int est_k_local = o.max_iteration;
        std::mt19937 engine(static_cast<std::uint32_t>(seed + 0x9e3779b97f4a7c15ull * (thread + 1)));
        std::uniform_int_distribution<int> pick(0, static_cast<int>(corres.size()) - 1);
        for (int itr = begin; itr < end; itr++) {
            if (itr >= est_k_global.load(std::memory_order_relaxed)) break;   // (no later one runs)
            for (int j = 0; j < o.ransac_n; j++) sample[j] = corres[pick(engine)];
            const Eigen::Matrix4d transformation = point_to_point(source.points, target.points, sample);
            if (!checks_pass(source.points, target.points, sample, transformation, o)) continue;
            const Points moved = transformed(source.points, transformation);
            Result result = evaluate(moved, tree, o.max_correspondence_distance,
                                     transformation, false);
            if (result.better_ransac_than(best_local)) {
                best_local = std::move(result);
                const double ratio = inlier_correspondence_ratio(moved, target.points, corres,
                                                                 o.max_correspondence_distance);
                double est_k_local_d = std::log(1.0 - confidence) /
                                       std::log(1.0 - std::pow(ratio, o.ransac_n));
                est_k_local_d = est_k_local_d < 0 ? est_k_local : est_k_local_d;
                est_k_local = est_k_local_d < est_k_global.load()
                                      ? static_cast<int>(std::ceil(est_k_local_d))
                                      : est_k_local;
            }
            int cur = est_k_global.load();
            while (est_k_local < cur && !est_k_global.compare_exchange_weak(cur, est_k_local)) {}
        }
        std::lock_guard<std::mutex> hold(lock);
        if (best_local.better_ransac_than(best)) best = std::move(best_local);
    });
    return best;
}

// Registration.cpp RegistrationICP (TransformationEstimationPointToPlane)
Result icp_point_to_plane(const Cloud& source, const Cloud& target, double max_distance,
                          const Eigen::Matrix4d& init, double relative_fitness,
                          double relative_rmse, int max_iteration) {
    if (max_distance <= 0.0) throw std::runtime_error("ICP: the distance must be > 0");
    if (target.normals.size() != target.points.size())
        throw std::runtime_error("point-to-plane ICP needs the target's normals");
    const KDTree tree = KDTree::of(target.points);
    Eigen::Matrix4d transformation = init;
    Points pcd = init.isIdentity() ? source.points : transformed(source.points, init);
    Result result = evaluate(pcd, tree, max_distance, transformation, true);
    for (int i = 0; i < max_iteration; i++) {
        const Eigen::Matrix4d update = point_to_plane(pcd, target, result.correspondence_set);
        transformation = update * transformation;
        pcd = transformed(pcd, update);
        const Result backup = result;
        result = evaluate(pcd, tree, max_distance, transformation, true);
        if (std::abs(backup.fitness - result.fitness) < relative_fitness &&
            std::abs(backup.inlier_rmse - result.inlier_rmse) < relative_rmse)
            break;
    }
    return result;
}

}  // namespace odent::reg
