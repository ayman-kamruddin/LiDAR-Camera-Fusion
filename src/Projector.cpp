#include "Projector.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <iostream>

void Projector::projectLidarToImage(const std::string& cloudFile,
                                    const std::string& imageFile,
                                    const Eigen::Matrix4d& transformation,
                                    const cv::Mat& cameraMatrix,
                                    const cv::Mat& distCoeffs) {
    // Load the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile(cloudFile, *cloud) == -1) {
        std::cerr << "Failed to load point cloud: " << cloudFile << std::endl;
        return;
    }

    // Load the image
    cv::Mat image = cv::imread(imageFile);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imageFile << std::endl;
        return;
    }

    // Extract rotation (R) and translation (T) from the transformation matrix
    cv::Mat rotationMatrix = (cv::Mat_<double>(3, 3) << 
        transformation(0, 0), transformation(0, 1), transformation(0, 2),
        transformation(1, 0), transformation(1, 1), transformation(1, 2),
        transformation(2, 0), transformation(2, 1), transformation(2, 2));

    cv::Mat rvec;
    cv::Rodrigues(rotationMatrix, rvec);

    cv::Mat T = (cv::Mat_<double>(3, 1) << 
        transformation(0, 3),
        transformation(1, 3),
        transformation(2, 3));

    // Convert the point cloud to a vector of cv::Point3f
    std::vector<cv::Point3f> lidarPoints;
    for (const auto& point : cloud->points) {
        lidarPoints.emplace_back(point.x, point.y, point.z);
    }

    // Project the LiDAR points onto the image plane
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(lidarPoints, rvec, T, cameraMatrix, distCoeffs, imagePoints);

    // Draw the projected points on the image
    cv::Mat outputImage = image.clone();
    for (const auto& point : imagePoints) {
        if (point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows) {
            cv::circle(outputImage, point, .3, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Save the result
    if (!cv::imwrite("projected_points.jpg", outputImage)) {
        std::cerr << "Failed to save projected image." << std::endl;
    } else {
        std::cout << "Projected points image saved as projected_points.jpg" << std::endl;
    }
}

void Projector::projectLidarToImageAndColorize(const std::string& cloudFile,
                                               const std::string& imageFile,
                                               const Eigen::Matrix4d& transformation,
                                               const cv::Mat& cameraMatrix,
                                               const cv::Mat& distCoeffs) {
    // Load the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile(cloudFile, *cloud) == -1) {
        std::cerr << "Failed to load point cloud: " << cloudFile << std::endl;
        return;
    }

    // Load the image
    cv::Mat image = cv::imread(imageFile);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imageFile << std::endl;
        return;
    }

    // Extract rotation (R) and translation (T) from the transformation matrix
    cv::Mat rotationMatrix = (cv::Mat_<double>(3, 3) << 
        transformation(0, 0), transformation(0, 1), transformation(0, 2),
        transformation(1, 0), transformation(1, 1), transformation(1, 2),
        transformation(2, 0), transformation(2, 1), transformation(2, 2));

    cv::Mat rvec;
    cv::Rodrigues(rotationMatrix, rvec);

    cv::Mat T = (cv::Mat_<double>(3, 1) << 
        transformation(0, 3),
        transformation(1, 3),
        transformation(2, 3));

    // Convert the point cloud to a vector of cv::Point3f
    std::vector<cv::Point3f> lidarPoints;
    for (const auto& point : cloud->points) {
        lidarPoints.emplace_back(point.x, point.y, point.z);
    }

    // Project the LiDAR points onto the image plane
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(lidarPoints, rvec, T, cameraMatrix, distCoeffs, imagePoints);

    // Create a new colored point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    // Map the projected points to the image and assign colors
    for (size_t i = 0; i < lidarPoints.size(); ++i) {
        const auto& point = imagePoints[i];
        if (point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows) {
            // Get the color from the image
            cv::Vec3b color = image.at<cv::Vec3b>(static_cast<int>(point.y), static_cast<int>(point.x));

            // Create a colored point
            pcl::PointXYZRGB coloredPoint;
            coloredPoint.x = lidarPoints[i].x;
            coloredPoint.y = lidarPoints[i].y;
            coloredPoint.z = lidarPoints[i].z;
            coloredPoint.r = color[2]; // OpenCV uses BGR format
            coloredPoint.g = color[1];
            coloredPoint.b = color[0];

            coloredCloud->points.push_back(coloredPoint);
        }
    }

    // Save the colored point cloud
    coloredCloud->width = coloredCloud->points.size();
    coloredCloud->height = 1;
    coloredCloud->is_dense = false;

    if (pcl::io::savePCDFileBinary("./colored_pointcloud.pcd", *coloredCloud) == -1) {
        std::cerr << "Failed to save colored point cloud." << std::endl;
    } else {
        std::cout << "Colored point cloud saved as colored_pointcloud.pcd" << std::endl;
    }
}
