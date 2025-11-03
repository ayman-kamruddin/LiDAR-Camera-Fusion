#ifndef PROJECTOR_H
#define PROJECTOR_H

#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class Projector {
public:
    // Projects LiDAR points onto the image plane and saves the projected image
    static void projectLidarToImage(const std::string& cloudFile,
                                    const std::string& imageFile,
                                    const Eigen::Matrix4d& transformation,
                                    const cv::Mat& cameraMatrix,
                                    const cv::Mat& distCoeffs,
                                    const std::string& outputPath);

    // Projects LiDAR points onto the image plane and saves a colorized point cloud
    static void projectLidarToImageAndColorize(const std::string& cloudFile,
                                               const std::string& imageFile,
                                               const Eigen::Matrix4d& transformation,
                                               const cv::Mat& cameraMatrix,
                                               const cv::Mat& distCoeffs,
                                               const std::string& outputCloudPath);

    // Validates the projected points to ensure they fall within the image boundaries
    static void validateProjectedPoints(std::vector<cv::Point2f>& points, int imageWidth, int imageHeight);
};

#endif // PROJECTOR_H
