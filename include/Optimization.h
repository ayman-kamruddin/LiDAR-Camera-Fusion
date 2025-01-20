#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <Eigen/Dense>
#include <vector>

class Optimization {
public:
    static Eigen::Matrix4d run(const std::vector<Eigen::Vector4d>& lidarPlanes,
                               const std::vector<Eigen::Vector4d>& cameraPlanes);

private:
    static Eigen::Matrix3d optimizeRotation(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                            const std::vector<Eigen::Vector4d>& cameraPlanes);

    static Eigen::Vector3d optimizeTranslation(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                               const std::vector<Eigen::Vector4d>& cameraPlanes,
                                               const Eigen::Matrix3d& rotation);

    static bool validateNumberOfPlanes(const std::vector<Eigen::Vector4d>& lidarPlanes,
                                       const std::vector<Eigen::Vector4d>& cameraPlanes);
};

#endif // OPTIMIZATION_H
