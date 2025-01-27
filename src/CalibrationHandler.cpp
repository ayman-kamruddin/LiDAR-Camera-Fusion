#include "CalibrationHandler.h"
#include "Optimization.h"
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

CalibrationHandler::CalibrationHandler(const std::vector<std::string>& imageFiles,
                                       const std::vector<std::string>& cloudFiles,
                                       const cv::Mat& cameraMatrix,
                                       const cv::Mat& distCoeffs)
    : imageFiles_(imageFiles),
      cloudFiles_(cloudFiles),
      cameraMatrix_(cameraMatrix),
      distCoeffs_(distCoeffs),
      imageHandler_(6, 9, 0.079, false, cameraMatrix, distCoeffs),
      cloudHandler_(4.0, -4.0, -1.0, 9.0, 4.0, 1.5, 0.02, 500, false) {}

void CalibrationHandler::run() {
    std::cout << "Running calibration with " << imageFiles_.size() << " image-cloud pairs." << std::endl;

    std::vector<Eigen::Vector4d> cameraPlanes;
    std::vector<Eigen::Vector4d> lidarPlanes;


    // create a progress bar
    for (size_t i = 0; i < imageFiles_.size(); ++i) {
        // print processing pair _ of _
        std::cout << "Processing pair " << i+1 << " of " << imageFiles_.size() << std::endl;

        // make sure that filenames match. igore directory prefix and file extension
        std::string imageFile = imageFiles_[i].substr(imageFiles_[i].find_last_of("/\\") + 1);
        std::string cloudFile = cloudFiles_[i].substr(cloudFiles_[i].find_last_of("/\\") + 1);
        imageFile = imageFile.substr(0, imageFile.find_last_of("."));
        cloudFile = cloudFile.substr(0, cloudFile.find_last_of("."));
        if (imageFile != cloudFile) {
            std::cerr << "Image and cloud file names do not match for pair: " << imageFiles_[i]
                      << " and " << cloudFiles_[i] << std::endl;
            continue;
        }

        // Extract planes
        Eigen::Vector4d cameraPlane = imageHandler_.extractPlane(imageFiles_[i]);
        Eigen::Vector4d lidarPlane = cloudHandler_.extractPlane(cloudFiles_[i]);

        // Validate extracted planes
        if (cameraPlane.isZero() || lidarPlane.isZero()) {
            std::cerr << "Failed to extract valid planes for pair: " << imageFiles_[i]
                      << " and " << cloudFiles_[i] << std::endl;
            continue;
        }

        cameraPlanes.push_back(cameraPlane);
        lidarPlanes.push_back(lidarPlane);
    }

    // Check if there are enough planes for optimization
    if (cameraPlanes.size() < 3 || lidarPlanes.size() < 3) {
        std::cerr << "Not enough valid planes for optimization. Calibration aborted." << std::endl;
        return;
    }

    // Perform optimization to compute the transformation matrix
    try {
        transformation_ = Optimization::run(lidarPlanes, cameraPlanes);

        // Output the computed transformation matrix
        std::cout << "Calibration completed successfully!" << std::endl;
        std::cout << "Computed transformation matrix (LiDAR to Camera):\n" << transformation_ << std::endl;

        // Save the transformation matrix to a file
        saveTransformationMatrix("transformation.yaml");
    } catch (const std::exception& e) {
        std::cerr << "Error during optimization: " << e.what() << std::endl;
    }
}

Eigen::Matrix4d CalibrationHandler::getTransformationMatrix() const {
    return transformation_;
}

void CalibrationHandler::saveTransformationMatrix(const std::string& filePath) const {
    std::ofstream outFile(filePath);
    if (outFile.is_open()) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                outFile << transformation_(i, j);
                if (j < 3) {
                    outFile << " ";
                }
            }
            outFile << "\n";
        }
        outFile.close();
        std::cout << "Transformation matrix saved to " << filePath << std::endl;
    } else {
        std::cerr << "Failed to save transformation matrix to " << filePath << std::endl;
    }
}
