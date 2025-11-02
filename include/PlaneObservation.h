#ifndef PLANE_OBSERVATION_H
#define PLANE_OBSERVATION_H

#include <Eigen/Dense>

struct PlaneObservation {
    Eigen::Vector4d plane = Eigen::Vector4d::Zero();
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();

    bool isValid(double tolerance = 1e-8) const {
        return !plane.isZero(tolerance);
    }
};

#endif // PLANE_OBSERVATION_H
