#ifndef CALIBRATION_HANDLER_H
#define CALIBRATION_HANDLER_H

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "ImageHandler.h"
#include "CloudHandler.h"

class CalibrationHandler {
public:
    CalibrationHandler(const std::vector<std::string>& imageFiles,
                       const std::vector<std::string>& cloudFiles,
                       const cv::Mat& cameraMatrix,
                       const cv::Mat& distCoeffs,
                       int checkerboardRows, int checkerboardCols, float checkerboardSize);

    void run();
    Eigen::Matrix4d getTransformationMatrix() const; // Getter for transformation matrix
    void saveTransformationMatrix(const std::string& filePath) const; // Save transformation matrix
    std::string extractFilenameWithoutDirectoryAndExtension(const std::string& filepath); // Extract filename without directory and extension

private:
    std::vector<std::string> imageFiles_;
    std::vector<std::string> cloudFiles_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;

    ImageHandler imageHandler_;
    CloudHandler cloudHandler_;
    Eigen::Matrix4d transformation_; // Transformation matrix (LiDAR to Camera)
};

#endif // CALIBRATION_HANDLER_H
