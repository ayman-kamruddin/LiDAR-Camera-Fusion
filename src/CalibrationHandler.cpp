#include "CalibrationHandler.h"
#include "Optimization.h"
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

CalibrationHandler::CalibrationHandler(const std::vector<std::string>& imageFiles,
                                       const std::vector<std::string>& cloudFiles,
                                       const cv::Mat& cameraMatrix,
                                       const cv::Mat& distCoeffs,
                                       int checkerboardRows,
                                       int checkerboardCols,
                                       float checkerboardSize)
    : imageFiles_(imageFiles),
      cloudFiles_(cloudFiles),
      cameraMatrix_(cameraMatrix),
      distCoeffs_(distCoeffs),
      imageHandler_(false, cameraMatrix, distCoeffs),
      cloudHandler_(4.0, -4.0, -1.0, 9.0, 4.0, 1.5, 0.02, 500, false, 0.05, 1.2, 0.1) {}

void CalibrationHandler::run() {
    std::cout << "Running calibration with " << imageFiles_.size() << " image-cloud pairs." << std::endl;

    std::vector<Eigen::Vector4d> cameraPlanes;
    std::vector<Eigen::Vector4d> lidarPlanes;

    for (size_t i = 0; i < imageFiles_.size(); ++i) {
        std::cout << "Processing pair " << i + 1 << " of " << imageFiles_.size() << std::endl;

        std::string imageFile = extractFilenameWithoutDirectoryAndExtension(imageFiles_[i]);
        std::string cloudFile = extractFilenameWithoutDirectoryAndExtension(cloudFiles_[i]);

        if (imageFile != cloudFile) {
            std::cerr << "Image and cloud file names do not match for pair: " << imageFiles_[i]
                      << " and " << cloudFiles_[i] << std::endl;
            continue;
        }

        Eigen::Vector4d cameraPlane = imageHandler_.extractPlane(imageFiles_[i]);
        Eigen::Vector4d lidarPlane = cloudHandler_.extractPlane(cloudFiles_[i]);

        if (cameraPlane.isZero() || lidarPlane.isZero()) {
            std::cerr << "Failed to extract valid planes for pair: " << imageFiles_[i]
                      << " and " << cloudFiles_[i] << std::endl;
            continue;
        }

        // Add validation for extracted plane
        if (!imageHandler_.validateExtractedPlane(cameraPlane)) {
            std::cerr << "Invalid camera plane extracted for image: " << imageFiles_[i] << std::endl;
            continue;
        }

        // Add debug output for extracted plane
        std::cout << "Extracted camera plane: " << cameraPlane.transpose() << std::endl;

        cameraPlanes.push_back(cameraPlane);
        lidarPlanes.push_back(lidarPlane);
    }

    if (cameraPlanes.size() < 3 || lidarPlanes.size() < 3) {
        std::cerr << "Not enough valid planes for optimization. Calibration aborted." << std::endl;
        return;
    }

    try {
        transformation_ = Optimization::run(lidarPlanes, cameraPlanes);

        std::cout << "Calibration completed successfully!" << std::endl;
        std::cout << "Computed transformation matrix (LiDAR to Camera):\n" << transformation_ << std::endl;

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

std::string CalibrationHandler::extractFilenameWithoutDirectoryAndExtension(const std::string& filepath) {
    std::string filename = filepath.substr(filepath.find_last_of("/\\") + 1);
    return filename.substr(0, filename.find_last_of("."));
}
