#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp> // Include the JSON library (https://github.com/nlohmann/json)
#include "DataLoader.h"
#include "CalibrationHandler.h"
#include "Projector.h"

// Function to save calibration data in JSON format
void saveCalibrationData(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                         const cv::Mat& rotationMatrix, const cv::Mat& translationVector, const std::string& fileName) {
    nlohmann::json jsonData;

    // Save the camera matrix (K)
    jsonData["K"] = {
        cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(0, 1), cameraMatrix.at<double>(0, 2),
        cameraMatrix.at<double>(1, 0), cameraMatrix.at<double>(1, 1), cameraMatrix.at<double>(1, 2),
        cameraMatrix.at<double>(2, 0), cameraMatrix.at<double>(2, 1), cameraMatrix.at<double>(2, 2)
    };

    // Save the rotation matrix
    jsonData["rotation_mat"] = {
        {rotationMatrix.at<double>(0, 0), rotationMatrix.at<double>(0, 1), rotationMatrix.at<double>(0, 2)},
        {rotationMatrix.at<double>(1, 0), rotationMatrix.at<double>(1, 1), rotationMatrix.at<double>(1, 2)},
        {rotationMatrix.at<double>(2, 0), rotationMatrix.at<double>(2, 1), rotationMatrix.at<double>(2, 2)}
    };

    // Save the translation vector
    jsonData["translation_vec"] = {
        translationVector.at<double>(0),
        translationVector.at<double>(1),
        translationVector.at<double>(2)
    };

    // Save the distortion coefficients
    jsonData["dist_coeffs"] = {
        distCoeffs.at<double>(0, 0), distCoeffs.at<double>(0, 1),
        distCoeffs.at<double>(0, 2), distCoeffs.at<double>(0, 3), distCoeffs.at<double>(0, 4)
    };

    // Write to a JSON file
    std::ofstream file(fileName);
    if (file.is_open()) {
        file << jsonData.dump(4); // Pretty print with an indent of 4 spaces
        file.close();
        std::cout << "Calibration data saved to " << fileName << std::endl;
    } else {
        std::cerr << "Failed to open file: " << fileName << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <calibration_data_path>" << std::endl;
        return 1;
    }

    std::string dataPath = argv[1];

    try {
        // Load data (image-point cloud pairs)
        DataLoader dataLoader(dataPath);
        const auto& imageFiles = dataLoader.getImageFiles();
        const auto& cloudFiles = dataLoader.getCloudFiles();

        
        if (imageFiles.empty() || cloudFiles.empty()) {
            std::cerr << "No valid image or point cloud files found in the specified directory." << std::endl;
            return 1;
        }

        int checkerboardRows = 6; // Update based on your checkerboard
        int checkerboardCols = 9;
        float checkerboardSize = 0.079; // Size in meters

        cv::Mat cameraMatrix, distCoeffs;
        ImageHandler imageHandler(false, cameraMatrix, distCoeffs);
        if (!imageHandler.calibrateCamera(imageFiles, checkerboardRows, checkerboardCols, checkerboardSize)) {
            return 1;
        }

        CalibrationHandler calibrationHandler(imageFiles, cloudFiles, cameraMatrix, distCoeffs, checkerboardRows, checkerboardCols, checkerboardSize);
        calibrationHandler.run();

        // Load the transformation matrix
        Eigen::Matrix4d transformation = calibrationHandler.getTransformationMatrix();

        // Save the calibration data
        cv::Mat rotationMatrix(3, 3, CV_64F);
        cv::Mat translationVector(3, 1, CV_64F);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                rotationMatrix.at<double>(i, j) = transformation(i, j);
            }
            translationVector.at<double>(i, 0) = transformation(i, 3);
        }
        saveCalibrationData(cameraMatrix, distCoeffs, rotationMatrix, translationVector, "calibration_parameters.json");

        // Project LiDAR points to the image for the first pair
        if (!imageFiles.empty() && !cloudFiles.empty()) {
            Projector::projectLidarToImage(cloudFiles[0], imageFiles[0], transformation, cameraMatrix, distCoeffs);
            std::cout << "Projected LiDAR points onto the image successfully." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
