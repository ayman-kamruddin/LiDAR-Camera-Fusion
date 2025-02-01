#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Dense>

class ImageHandler {
public:
    ImageHandler(bool debugMode, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);

    Eigen::Vector4d extractPlane(const std::string& imagePath);
    bool validateExtractedPlane(const Eigen::Vector4d& plane); // Function to validate the extracted plane
    bool calibrateCamera(const std::vector<std::string>& imageFiles, int checkerboardRows, int checkerboardCols, float checkerboardSize);

private:
    bool debugMode_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
};

#endif // IMAGE_HANDLER_H
