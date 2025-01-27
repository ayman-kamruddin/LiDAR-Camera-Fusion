#include "Optimization.h"
#include <Eigen/Dense>
#include <iostream>

Eigen::Matrix4d Optimization::run(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                  const std::vector<Eigen::Vector4d>& cameraPlanes) {
    if (lidarPlanes.size() != cameraPlanes.size()) {
        throw std::runtime_error("Mismatch between the number of LiDAR and camera planes.");
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

    Eigen::MatrixXd A(3 * numPlanes, 3);
    Eigen::VectorXd b(3 * numPlanes);

    for (size_t i = 0; i < numPlanes; ++i) {
        Eigen::Vector3d lidarNormal = lidarPlanes[i].head<3>();
        Eigen::Vector3d cameraNormal = cameraPlanes[i].head<3>();
        double lidarD = lidarPlanes[i][3];
        double cameraD = cameraPlanes[i][3];

        A.block<3, 3>(3 * i, 0) = lidarNormal.asDiagonal();
        b.segment<3>(3 * i) = cameraNormal * cameraD - rotation * lidarNormal * lidarD;
    }

    Eigen::Vector3d translation = A.colPivHouseholderQr().solve(b);
    return translation;
}
