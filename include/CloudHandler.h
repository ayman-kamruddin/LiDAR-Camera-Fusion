#ifndef CLOUD_HANDLER_H
#define CLOUD_HANDLER_H

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class CloudHandler {
public:
    CloudHandler(double min_bound_x, double min_bound_y, double min_bound_z,
                 double max_bound_x, double max_bound_y, double max_bound_z,
                 double plane_ransac_thresh, int plane_min_points, bool debug);

    Eigen::Vector4d extractPlane(const std::string& cloudFile);

private:
    Eigen::Vector3d min_bound_;
    Eigen::Vector3d max_bound_;
    double plane_ransac_thresh_;
    int plane_min_points_;
    bool debug_;
};

#endif // CLOUD_HANDLER_H
