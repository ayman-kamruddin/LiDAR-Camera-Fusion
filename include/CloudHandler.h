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
                 double plane_ransac_thresh, int plane_min_points, bool debug,
                 double cluster_tolerance, 
                 double bounding_box_size,
                 double bounding_box_tolerance);

    Eigen::Vector4d extractPlane(const std::string& cloudFile);

private:
    Eigen::Vector3d min_bound_;
    Eigen::Vector3d max_bound_;
    double plane_ransac_thresh_;
    int plane_min_points_;
    bool debug_;
    double cluster_tolerance_;
    double bounding_box_size_;
    double bounding_box_tolerance_;
};

#endif // CLOUD_HANDLER_H
