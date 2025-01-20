#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "DataLoader.h"
#include "CalibrationHandler.h"
#include "Projector.h"

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

    cv::destroyAllWindows();

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


        CalibrationHandler calibrationHandler(imageFiles, cloudFiles, cameraMatrix, distCoeffs);
        calibrationHandler.run();

        // Load the transformation matrix
        Eigen::Matrix4d transformation = calibrationHandler.getTransformationMatrix();

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
