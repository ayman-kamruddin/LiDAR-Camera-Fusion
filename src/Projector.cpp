#include "Projector.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <limits>
#include <algorithm>
#include <cmath>

namespace {
cv::Scalar mapDistanceToColor(float normalizedDistance) {
    normalizedDistance = std::clamp(normalizedDistance, 0.0f, 1.0f);

    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;

    if (normalizedDistance < 0.25f) {
        float t = normalizedDistance / 0.25f; // red -> yellow
        red = 255.0f;
        green = 255.0f * t;
        blue = 0.0f;
    } else if (normalizedDistance < 0.5f) {
        float t = (normalizedDistance - 0.25f) / 0.25f; // yellow -> green
        red = 255.0f * (1.0f - t);
        green = 255.0f;
        blue = 0.0f;
    } else if (normalizedDistance < 0.75f) {
        float t = (normalizedDistance - 0.5f) / 0.25f; // green -> cyan
        red = 0.0f;
        green = 255.0f;
        blue = 255.0f * t;
    } else {
        float t = (normalizedDistance - 0.75f) / 0.25f; // cyan -> blue
        red = 0.0f;
        green = 255.0f * (1.0f - t);
        blue = 255.0f;
    }

    return cv::Scalar(static_cast<int>(blue),
                      static_cast<int>(green),
                      static_cast<int>(red));
}
} // namespace

void Projector::projectLidarToImage(const std::string& cloudFile,
                                    const std::string& imageFile,
                                    const Eigen::Matrix4d& transformation,
                                    const cv::Mat& cameraMatrix,
                                    const cv::Mat& distCoeffs,
                                    const std::string& outputPath) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile(cloudFile, *cloud) == -1) {
        std::cerr << "Failed to load point cloud: " << cloudFile << std::endl;
        return;
    }

    cv::Mat image = cv::imread(imageFile, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imageFile << std::endl;
        return;
    }

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

    std::vector<cv::Point3f> lidarPoints;
    lidarPoints.reserve(cloud->points.size());
    std::vector<float> pointDistances;
    pointDistances.reserve(cloud->points.size());
    for (const auto& point : cloud->points) {
        lidarPoints.emplace_back(point.x, point.y, point.z);
        pointDistances.push_back(std::sqrt(point.x * point.x +
                                           point.y * point.y +
                                           point.z * point.z));
    }

    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(lidarPoints, rvec, T, cameraMatrix, distCoeffs, imagePoints);

    validateProjectedPoints(imagePoints, image.cols, image.rows);

    float minDistance = *std::min_element(pointDistances.begin(), pointDistances.end());
    float maxDistance = *std::max_element(pointDistances.begin(), pointDistances.end());
    float distanceRange = maxDistance - minDistance;
    if (distanceRange <= std::numeric_limits<float>::epsilon()) {
        distanceRange = 1.0f;
    }

    cv::Mat outputImage = image.clone();
    for (std::size_t i = 0; i < imagePoints.size(); ++i) {
        const auto& point = imagePoints[i];
        if (point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows) {
            float normalizedDistance = (pointDistances[i] - minDistance) / distanceRange;
            cv::Scalar color = mapDistanceToColor(normalizedDistance);
            cv::circle(outputImage, point, 2, color, -1);
        }
    }

    std::filesystem::path outputPathObj(outputPath);
    if (!outputPathObj.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(outputPathObj.parent_path(), ec);
        if (ec) {
            std::cerr << "Failed to create output directory: " << ec.message() << std::endl;
            return;
        }
    }

    if (!cv::imwrite(outputPathObj.string(), outputImage)) {
        std::cerr << "Failed to save projected image." << std::endl;
    } else {
        std::cout << "Projected points image saved to " << outputPathObj << std::endl;
    }
}

void Projector::projectLidarToImageAndColorize(const std::string& cloudFile,
                                               const std::string& imageFile,
                                               const Eigen::Matrix4d& transformation,
                                               const cv::Mat& cameraMatrix,
                                               const cv::Mat& distCoeffs,
                                               const std::string& outputCloudPath) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile(cloudFile, *cloud) == -1) {
        std::cerr << "Failed to load point cloud: " << cloudFile << std::endl;
        return;
    }

    cv::Mat image = cv::imread(imageFile, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imageFile << std::endl;
        return;
    }

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

    std::vector<cv::Point3f> lidarPoints;
    lidarPoints.reserve(cloud->points.size());
    for (const auto& point : cloud->points) {
        lidarPoints.emplace_back(point.x, point.y, point.z);
    }

    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(lidarPoints, rvec, T, cameraMatrix, distCoeffs, imagePoints);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    colorCloud->points.reserve(cloud->points.size());

    for (std::size_t i = 0; i < cloud->points.size(); ++i) {
        pcl::PointXYZRGB coloredPoint;
        coloredPoint.x = cloud->points[i].x;
        coloredPoint.y = cloud->points[i].y;
        coloredPoint.z = cloud->points[i].z;

        const auto& projectedPoint = imagePoints[i];
        int px = static_cast<int>(std::round(projectedPoint.x));
        int py = static_cast<int>(std::round(projectedPoint.y));

        if (px >= 0 && px < image.cols && py >= 0 && py < image.rows) {
            const cv::Vec3b& color = image.at<cv::Vec3b>(py, px);
            coloredPoint.b = color[0];
            coloredPoint.g = color[1];
            coloredPoint.r = color[2];
        } else {
            coloredPoint.r = 0;
            coloredPoint.g = 0;
            coloredPoint.b = 0;
        }

        colorCloud->points.push_back(coloredPoint);
    }

    colorCloud->width = static_cast<uint32_t>(colorCloud->points.size());
    colorCloud->height = 1;
    colorCloud->is_dense = false;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Colorized Point Cloud"));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);
    viewer->addPointCloud<pcl::PointXYZRGB>(colorCloud, "color_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "color_cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->spin();

    std::filesystem::path outputPath(outputCloudPath);
    if (!outputPath.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(outputPath.parent_path(), ec);
        if (ec) {
            std::cerr << "Failed to create output directory for point cloud: " << ec.message() << std::endl;
            return;
        }
    }

    if (pcl::io::savePCDFileBinary(outputPath.string(), *colorCloud) == -1) {
        std::cerr << "Failed to save colorized point cloud to " << outputPath << std::endl;
    } else {
        std::cout << "Colorized point cloud saved to " << outputPath << std::endl;
    }
}

void Projector::validateProjectedPoints(std::vector<cv::Point2f>& points, int imageWidth, int imageHeight) {
    for (auto& point : points) {
        if (point.x < 0) point.x = 0;
        if (point.x >= imageWidth) point.x = imageWidth - 1;
        if (point.y < 0) point.y = 0;
        if (point.y >= imageHeight) point.y = imageHeight - 1;
    }
}
