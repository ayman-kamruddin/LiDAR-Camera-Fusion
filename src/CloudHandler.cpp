#include "CloudHandler.h"
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iostream>

CloudHandler::CloudHandler(double min_bound_x, double min_bound_y, double min_bound_z,
                           double max_bound_x, double max_bound_y, double max_bound_z,
                           double plane_ransac_thresh, int plane_min_points, bool debug,
                           double cluster_tolerance, double bounding_box_tolerance)
    : min_bound_(min_bound_x, min_bound_y, min_bound_z),
      max_bound_(max_bound_x, max_bound_y, max_bound_z),
      plane_ransac_thresh_(plane_ransac_thresh),
      plane_min_points_(plane_min_points),
      debug_(debug),
      cluster_tolerance_(cluster_tolerance),
      bounding_box_tolerance_(bounding_box_tolerance) {}

Eigen::Vector4d CloudHandler::extractPlane(const std::string& cloudFile) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>());

    // Load the point cloud
    if (pcl::io::loadPCDFile(cloudFile, *cloud) == -1) {
        std::cerr << "Failed to load point cloud: " << cloudFile << std::endl;
        return Eigen::Vector4d::Zero();
    }

    // Pass-through filter
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(min_bound_.x(), max_bound_.x());
    pass.filter(*filteredCloud);

    pass.setInputCloud(filteredCloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(min_bound_.y(), max_bound_.y());
    pass.filter(*filteredCloud);

    pass.setInputCloud(filteredCloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_bound_.z(), max_bound_.z());
    pass.filter(*filteredCloud);

    if (filteredCloud->points.size() < plane_min_points_) {
        std::cerr << "Not enough points after filtering for plane extraction." << std::endl;
        return Eigen::Vector4d::Zero();
    }

    // Clustering
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(filteredCloud);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_); // Adjust as needed
    ec.setMinClusterSize(100);   // Minimum number of points per cluster
    ec.setMaxClusterSize(100000); // Maximum number of points per cluster
    ec.setSearchMethod(tree);
    ec.setInputCloud(filteredCloud);
    ec.extract(clusterIndices);

    if (clusterIndices.empty()) {
        std::cerr << "No clusters found in the filtered cloud." << std::endl;
        return Eigen::Vector4d::Zero();
    }

    // Select the largest valid cluster
    pcl::PointCloud<pcl::PointXYZ>::Ptr selectedCluster(new pcl::PointCloud<pcl::PointXYZ>());
    size_t maxClusterSize = 0;

    for (const auto& indices : clusterIndices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& idx : indices.indices) {
            cluster->points.push_back(filteredCloud->points[idx]);
        }

        // Calculate bounding box
        pcl::PointXYZ minPt, maxPt;
        pcl::getMinMax3D(*cluster, minPt, maxPt);
        double extentX = maxPt.x - minPt.x;
        double extentY = maxPt.y - minPt.y;
        double extentZ = maxPt.z - minPt.z;

        // Check each dimension is <= 1.2 + bounding_box_tolerance_
        if (extentX <= (1.2 + bounding_box_tolerance_) && extentY <= (1.2 + bounding_box_tolerance_) && extentZ <= (1.2 + bounding_box_tolerance_)) {
            if (cluster->points.size() > maxClusterSize) {
                selectedCluster = cluster;
                maxClusterSize = cluster->points.size();
            }
        }
    }

    if (selectedCluster->points.empty()) {
        std::cerr << "No valid clusters found." << std::endl;
        return Eigen::Vector4d::Zero();
    }

    // Plane fitting using RANSAC
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(plane_ransac_thresh_);
    seg.setInputCloud(selectedCluster);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() < plane_min_points_) {
        std::cerr << "Failed to extract plane from the selected cluster." << std::endl;
        return Eigen::Vector4d::Zero();
    }

    // Debug output
    if (debug_) {
        pcl::io::savePCDFile("/tmp/selected_cluster.pcd", *selectedCluster);
        std::cout << "Selected cluster saved to /tmp/selected_cluster.pcd" << std::endl;
    }

    // Return plane coefficients
    if (coefficients->values.size() == 4) {
        return Eigen::Vector4d(coefficients->values[0], coefficients->values[1],
                               coefficients->values[2], coefficients->values[3]);
    } else {
        std::cerr << "Invalid plane coefficients." << std::endl;
        return Eigen::Vector4d::Zero();
    }
}
