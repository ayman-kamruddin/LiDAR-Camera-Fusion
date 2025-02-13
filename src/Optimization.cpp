/******************************************************************************
 *  Optimization.cpp
 *
 *  This file implements the methods declared in Optimization.h.
 *  It fixes the main pitfalls from the original code by:
 *    1) Using exactly one scalar constraint per plane in optimizeTranslation()
 *    2) Optionally flipping plane normals if they are reversed
 *    3) Optionally normalizing plane normals for consistent offset interpretation
 *
 *  Make sure to #include "Optimization.h" in your build system along with Eigen.
 ******************************************************************************/

#include "Optimization.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <iostream>
#include <stdexcept>

Eigen::Matrix4d Optimization::run(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                  const std::vector<Eigen::Vector4d>& cameraPlanes)
{
    // 1. Validate plane counts
    if (!validateNumberOfPlanes(lidarPlanes, cameraPlanes)) {
        throw std::runtime_error("[Optimization::run] Mismatch between number of LiDAR and camera planes.");
    }

    // 2. Check sufficient planes
    if (lidarPlanes.size() < 3 || cameraPlanes.size() < 3) {
        throw std::runtime_error("[Optimization::run] Not enough planes for optimization (need >= 3).");
    }

    // 3. Optimize rotation
    Eigen::Matrix3d rotation = optimizeRotation(lidarPlanes, cameraPlanes);

    // 4. Optimize translation
    Eigen::Vector3d translation = optimizeTranslation(lidarPlanes, cameraPlanes, rotation);

    // 5. Build final 4x4 transformation
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = rotation;
    transformation.block<3, 1>(0, 3) = translation;

    return transformation;
}

bool Optimization::validateNumberOfPlanes(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                          const std::vector<Eigen::Vector4d>& cameraPlanes)
{
    return (lidarPlanes.size() == cameraPlanes.size());
}


Eigen::Matrix3d Optimization::optimizeRotation(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                               const std::vector<Eigen::Vector4d>& cameraPlanes)
{
    // We accumulate a cross-covariance matrix H from corresponding plane normals
    size_t numPlanes = lidarPlanes.size();
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();

    for (size_t i = 0; i < numPlanes; ++i) {
        // Extract the 3D normal
        Eigen::Vector3d lidarNormal  = lidarPlanes[i].head<3>();
        Eigen::Vector3d cameraNormal = cameraPlanes[i].head<3>();

        // OPTIONALLY: Normalize each normal if your data isn't guaranteed normalized.
        double nL = lidarNormal.norm();
        double nC = cameraNormal.norm();
        if (nL > 1e-9) {
            lidarNormal /= nL;
         }
         if (nC > 1e-9) {
             cameraNormal /= nC;
         }

        // Accumulate
        H += lidarNormal * cameraNormal.transpose();
    }

    // SVD to get rotation
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Proposed rotation
    Eigen::Matrix3d rotation = V * U.transpose();

    // Enforce a right-handed coordinate system (det=+1)
    if (rotation.determinant() < 0.0) {
        V.col(2) *= -1.0;
        rotation = V * U.transpose();
    }

    return rotation;
}


Eigen::Vector3d Optimization::optimizeTranslation(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                                  const std::vector<Eigen::Vector4d>& cameraPlanes,
                                                  const Eigen::Matrix3d& rotation)
{
    // We'll create an N x 3 matrix A and N x 1 vector b (one plane -> one constraint).
    //
    // For each plane i, if LiDAR plane is: n_l^T x + d_l = 0
    // then camera plane is: n_c^T x + d_c = 0
    // where n_c = R * n_l,  d_c = d_l - n_c^T t  (transform x_c = R x_l + t)
    //
    // => n_c^T t = d_l - d_c
    // => (R n_l)^T t = d_l - d_c
    //
    // We'll store each scalar equation as a row in A, with b the RHS.

    size_t numPlanes = lidarPlanes.size();
    Eigen::MatrixXd A(numPlanes, 3);
    Eigen::VectorXd b(numPlanes);

    for (size_t i = 0; i < numPlanes; ++i) {
        Eigen::Vector3d lidarNormal  = lidarPlanes[i].head<3>();
        double           lidarD      = lidarPlanes[i](3);

        Eigen::Vector3d cameraNormal = cameraPlanes[i].head<3>();
        double           cameraD     = cameraPlanes[i](3);

        // OPTIONAL: Normalize if not guaranteed
        double nL = lidarNormal.norm();
        double nC = cameraNormal.norm();
        if (nL > 1e-9) {
             lidarNormal /= nL;
             lidarD /= nL;  // scale offset by same factor
         }
         if (nC > 1e-9) {
             cameraNormal /= nC;
             cameraD /= nC;
         }

        // Compute the predicted camera-frame normal via the rotation
        Eigen::Vector3d predictedN_c = rotation * lidarNormal;

        // OPTIONAL: Check direction consistency
        // If predictedN_c is "the opposite" of cameraNormal, you may flip one set:
        if (predictedN_c.dot(cameraNormal) < 0.0) {
            // Flip LiDAR side
             lidarNormal  = -lidarNormal;
             lidarD       = -lidarD;
             // Recompute predictedN_c:
             predictedN_c = rotation * lidarNormal;
         }

        // Single scalar equation: predictedN_c^T * t = lidarD - cameraD
        A.row(i) = predictedN_c.transpose();
        b(i) = lidarD - cameraD;
    }

    // Solve the system A * t = b in least-squares sense
    Eigen::Vector3d translation = A.colPivHouseholderQr().solve(b);

    return translation;
}
