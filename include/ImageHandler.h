#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Dense>

class ImageHandler {
public:
    ImageHandler(int checkerboardRows, int checkerboardCols, float checkerboardSize,
                 bool debugMode, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);

    Eigen::Vector4d extractPlane(const std::string& imagePath);
    bool validateExtractedPlane(const Eigen::Vector4d& plane); // Function to validate the extracted plane

private:
    int checkerboardRows_;
    int checkerboardCols_;
    float checkerboardSize_;
    bool debugMode_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
};

#endif // IMAGE_HANDLER_H
