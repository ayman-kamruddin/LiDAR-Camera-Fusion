#include "Optimization.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <numeric>

namespace {

Eigen::Vector3d solveLeastSquares(const std::vector<Eigen::Vector3d>& normals,
                                  const std::vector<double>& rhs,
                                  const std::vector<int>& indices) {
    Eigen::MatrixXd A(indices.size(), 3);
    Eigen::VectorXd b(indices.size());

    for (size_t row = 0; row < indices.size(); ++row) {
        int idx = indices[row];
        A.row(row) = normals[idx].transpose();
        b(row) = rhs[idx];
    }

    return A.colPivHouseholderQr().solve(b);
}

} // namespace

Eigen::Matrix4d Optimization::run(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                  const std::vector<Eigen::Vector4d>& cameraPlanes) {
    if (!validateNumberOfPlanes(lidarPlanes, cameraPlanes)) {
        throw std::runtime_error("Mismatch between the number of LiDAR and camera planes.");
    }

    if (lidarPlanes.size() < 3 || cameraPlanes.size() < 3) {
        throw std::runtime_error("Not enough valid planes for optimization. At least 3 planes are required.");
    }

    // Step 1: Optimize rotation matrix
    Eigen::Matrix3d rotation = optimizeRotation(lidarPlanes, cameraPlanes);

    // Step 2: Optimize translation vector
    Eigen::Vector3d translation = optimizeTranslation(lidarPlanes, cameraPlanes, rotation);

    // Step 3: Construct the full transformation matrix
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = rotation;
    transformation.block<3, 1>(0, 3) = translation;

    return transformation;
}

Eigen::Matrix3d Optimization::optimizeRotation(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                               const std::vector<Eigen::Vector4d>& cameraPlanes) {
    size_t numPlanes = lidarPlanes.size();
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();

    // Compute the cross-covariance matrix
    for (size_t i = 0; i < numPlanes; ++i) {
        Eigen::Vector3d lidarNormal = lidarPlanes[i].head<3>();
        Eigen::Vector3d cameraNormal = cameraPlanes[i].head<3>();
        H += lidarNormal * cameraNormal.transpose();
    }

    // Perform Singular Value Decomposition (SVD) on H
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Ensure the determinant is positive for a proper rotation
    Eigen::Matrix3d rotation = V * U.transpose();
    if (rotation.determinant() < 0) {
        V.col(2) *= -1; // Flip the last column
        rotation = V * U.transpose();
    }

    return rotation;
}

Eigen::Vector3d Optimization::optimizeTranslation(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                                  const std::vector<Eigen::Vector4d>& cameraPlanes,
                                                  const Eigen::Matrix3d& rotation) {
    size_t numPlanes = lidarPlanes.size();

    std::vector<Eigen::Vector3d> normals;
    std::vector<double> rhs;
    normals.reserve(numPlanes);
    rhs.reserve(numPlanes);

    for (size_t i = 0; i < numPlanes; ++i) {
        Eigen::Vector3d lidarNormal = lidarPlanes[i].head<3>();
        Eigen::Vector3d cameraNormal = cameraPlanes[i].head<3>();
        double lidarD = lidarPlanes[i][3];
        double cameraD = cameraPlanes[i][3];

        double lidarNorm = lidarNormal.norm();
        double cameraNorm = cameraNormal.norm();

        if (lidarNorm < 1e-9 || cameraNorm < 1e-9) {
            continue;
        }

        lidarNormal /= lidarNorm;
        lidarD /= lidarNorm;

        cameraNormal /= cameraNorm;
        cameraD /= cameraNorm;

        Eigen::Vector3d predictedCameraNormal = rotation * lidarNormal;
        double alignment = predictedCameraNormal.dot(cameraNormal);
        if (alignment < 0.0) {
            cameraNormal = -cameraNormal;
            cameraD = -cameraD;
            alignment = -alignment;
        }

        // If the normals disagree by more than ~45 degrees, skip this pair—it likely came from a bad fit.
        const double alignmentThreshold = std::sqrt(0.5);
        if (alignment < alignmentThreshold) {
            continue;
        }

        normals.push_back(cameraNormal);
        rhs.push_back(lidarD - cameraD);
    }

    if (normals.size() < 3) {
        throw std::runtime_error("Insufficient valid planes to estimate translation.");
    }

    std::vector<int> allIndices(normals.size());
    std::iota(allIndices.begin(), allIndices.end(), 0);

    Eigen::Vector3d translation = solveLeastSquares(normals, rhs, allIndices);

    const double residualThreshold = 0.02; // 2 cm tolerance
    std::vector<int> inliers;
    inliers.reserve(normals.size());

    for (size_t idx = 0; idx < normals.size(); ++idx) {
        double residual = std::abs(normals[idx].dot(translation) - rhs[idx]);
        if (residual <= residualThreshold) {
            inliers.push_back(static_cast<int>(idx));
        }
    }

    if (inliers.size() >= 3 && inliers.size() < normals.size()) {
        translation = solveLeastSquares(normals, rhs, inliers);
    }

    return translation;
}

bool Optimization::validateNumberOfPlanes(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                          const std::vector<Eigen::Vector4d>& cameraPlanes) {
    return lidarPlanes.size() == cameraPlanes.size();
}
