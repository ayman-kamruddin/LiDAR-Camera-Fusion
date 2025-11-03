#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp> // Include the JSON library (https://github.com/nlohmann/json)
#include "DataLoader.h"
#include "CalibrationHandler.h"
#include "Projector.h"

namespace fs = std::filesystem;

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

// Function to calibrate the camera based on checkerboard images
bool calibrateCamera(const std::vector<std::string>& imageFiles, int checkerboardRows, int checkerboardCols,
                     float checkerboardSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::vector<cv::Point3f>> objectPoints;

    std::vector<cv::Point3f> objectPoint;
    for (int i = 0; i < checkerboardRows; ++i) {
        for (int j = 0; j < checkerboardCols; ++j) {
            objectPoint.emplace_back(j * checkerboardSize, i * checkerboardSize, 0.0f);
        }
    }

    for (const auto& file : imageFiles) {
        cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << file << std::endl;
            continue;
        }

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(image, cv::Size(checkerboardCols, checkerboardRows), corners,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            imagePoints.push_back(corners);
            objectPoints.push_back(objectPoint);

            // Debug: Visualize corners
            // Uncomment the following lines for debugging:
            // cv::drawChessboardCorners(image, cv::Size(checkerboardCols, checkerboardRows), corners, found);
            // cv::imshow("Checkerboard", image);
            // cv::waitKey(100);
        } else {
            std::cerr << "Checkerboard not found in image: " << file << std::endl;
        }
    }

    if (imagePoints.empty()) {
        std::cerr << "No valid checkerboard images found for calibration." << std::endl;
        return false;
    }

    cv::Size imageSize = cv::imread(imageFiles[0], cv::IMREAD_GRAYSCALE).size();
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

    std::cout << "Camera calibration completed. RMS error: " << rms << std::endl;
    std::cout << "Camera Matrix:\n" << cameraMatrix << std::endl;
    std::cout << "Distortion Coefficients:\n" << distCoeffs.t() << std::endl;

    // Verify the accuracy of the calibration parameters
    double totalError = 0.0;
    int totalPoints = 0;
    for (size_t i = 0; i < objectPoints.size(); ++i) {
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPoints);
        double error = cv::norm(imagePoints[i], projectedPoints, cv::NORM_L2);
        totalError += error * error;
        totalPoints += objectPoints[i].size();
    }
    double meanError = std::sqrt(totalError / totalPoints);
    std::cout << "Mean reprojection error: " << meanError << std::endl;

    return true;
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
        if (!calibrateCamera(imageFiles, checkerboardRows, checkerboardCols, checkerboardSize, cameraMatrix, distCoeffs)) {
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

        // Project LiDAR points to the images for all available pairs and colorize the point clouds
        if (!imageFiles.empty() && !cloudFiles.empty()) {
            fs::path projectedImageDir = fs::path(dataPath) / "projected_points";
            fs::path colorizedCloudDir = fs::path(dataPath) / "colorized_pointclouds";

            std::error_code ec;
            fs::create_directories(projectedImageDir, ec);
            if (ec) {
                std::cerr << "Failed to create projected points directory: " << ec.message() << std::endl;
                return 1;
            }

            ec.clear();
            fs::create_directories(colorizedCloudDir, ec);
            if (ec) {
                std::cerr << "Failed to create colorized point cloud directory: " << ec.message() << std::endl;
                return 1;
            }

            const std::size_t pairCount = std::min(imageFiles.size(), cloudFiles.size());
            for (std::size_t i = 0; i < pairCount; ++i) {
                fs::path imagePath = imageFiles[i];
                fs::path cloudPath = cloudFiles[i];

                std::string baseName = imagePath.stem().string();
                if (baseName.empty()) {
                    baseName = "pair_" + std::to_string(i);
                }

                fs::path projectedImagePath = projectedImageDir / (baseName + "_projected.jpg");
                Projector::projectLidarToImage(cloudPath.string(),
                                               imagePath.string(),
                                               transformation,
                                               cameraMatrix,
                                               distCoeffs,
                                               projectedImagePath.string());

                fs::path colorizedCloudPath = colorizedCloudDir / (baseName + "_colorized.pcd");
                Projector::projectLidarToImageAndColorize(cloudPath.string(),
                                                          imagePath.string(),
                                                          transformation,
                                                          cameraMatrix,
                                                          distCoeffs,
                                                          colorizedCloudPath.string());
            }

            std::cout << "Projected LiDAR points saved for " << pairCount
                      << " pairs in " << projectedImageDir << std::endl;
            std::cout << "Colorized point clouds saved for " << pairCount
                      << " pairs in " << colorizedCloudDir << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
