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
        -transformation(0, 3), // a bit of a hack to get the correct translation. will need to be generalized.
        transformation(1, 3),
        transformation(2, 3));

    // Convert the point cloud to a vector of cv::Point3f
    std::vector<cv::Point3f> lidarPoints;
    std::vector<float> pointDistances;
    for (const auto& point : cloud->points) {
        lidarPoints.emplace_back(point.x, point.y, point.z);
        pointDistances.push_back(sqrt(point.x * point.x + point.y * point.y + point.z * point.z));
    }

    // Project the LiDAR points onto the image plane
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(lidarPoints, rvec, T, cameraMatrix, distCoeffs, imagePoints);

    // Validate the projected points
    validateProjectedPoints(imagePoints, image.cols, image.rows);

    // Normalize distances for color mapping
    float minDistance = *std::min_element(pointDistances.begin(), pointDistances.end());
    float maxDistance = *std::max_element(pointDistances.begin(), pointDistances.end());

    // Draw the projected points on the image
    cv::Mat outputImage = image.clone();
    for (size_t i = 0; i < imagePoints.size(); ++i) {
        const auto& point = imagePoints[i];
        if (point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows) {
            // Map the distance to a color (near = red, far = blue)
            float normalizedDistance = (pointDistances[i] - minDistance) / (maxDistance - minDistance);
            int blue = static_cast<int>(255 * normalizedDistance);
            int red = static_cast<int>(255 * (1.0 - normalizedDistance));
            cv::Scalar color(blue, 0, red);

            // Draw the point on the image
            cv::circle(outputImage, point, .3, color, -1);
        }
    }

    // Save the result
    if (!cv::imwrite("projected_points.jpg", outputImage)) {
        std::cerr << "Failed to save projected image." << std::endl;
    } else {
        std::cout << "Projected points image saved as projected_points.jpg" << std::endl;
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
