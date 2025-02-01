#include "ImageHandler.h"
#include <opencv2/opencv.hpp>
#include <iostream>

ImageHandler::ImageHandler(bool debugMode, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
    : debugMode_(debugMode),
      cameraMatrix_(cameraMatrix),
      distCoeffs_(distCoeffs) {}

Eigen::Vector4d ImageHandler::extractPlane(const std::string& imagePath) {
    // Load the image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return Eigen::Vector4d::Zero();
    }

    // Find checkerboard corners
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(image, cv::Size(9, 6), corners,
                                           cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    if (!found) {
        std::cerr << "Checkerboard pattern not found in image: " << imagePath << std::endl;
        
        return Eigen::Vector4d::Zero();

    }

    // Refine corner locations
    cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

    if (debugMode_) {
        // Draw corners for debugging
        cv::Mat debugImage;
        cv::cvtColor(image, debugImage, cv::COLOR_GRAY2BGR);
        cv::drawChessboardCorners(debugImage, cv::Size(9, 6), corners, found);
        cv::imshow("Checkerboard Detection", debugImage);
        cv::waitKey(100); // Pause to view the image
    }

    // Generate 3D points for the checkerboard
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 9; ++j) {
            objectPoints.emplace_back(j * 0.079, i * 0.079, 0.0f);
        }
    }

    // Solve PnP to find the camera pose
    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(objectPoints, corners, cameraMatrix_, distCoeffs_, rvec, tvec);

    if (!success) {
        std::cerr << "Failed to solve PnP for image: " << imagePath << std::endl;
        return Eigen::Vector4d::Zero();
    }

    // Convert rotation vector to rotation matrix
    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);

    // Extract the plane normal in the camera frame
    cv::Mat normalCamera = rotationMatrix * (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
    Eigen::Vector3d planeNormal(normalCamera.at<double>(0, 0), normalCamera.at<double>(1, 0), normalCamera.at<double>(2, 0));
    double planeD = -planeNormal.dot(Eigen::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));

    // Construct the plane equation
    Eigen::Vector4d plane(planeNormal(0), planeNormal(1), planeNormal(2), planeD);

    if (debugMode_) {
        std::cout << "Extracted plane: " << plane.transpose() << std::endl;
    }

    return plane;
}

bool ImageHandler::validateExtractedPlane(const Eigen::Vector4d& plane) {
    // Check if the plane is valid (non-zero)
    return !plane.isZero();
}

bool ImageHandler::calibrateCamera(const std::vector<std::string>& imageFiles, int checkerboardRows, int checkerboardCols, float checkerboardSize) {
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
    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix_, distCoeffs_, rvecs, tvecs);

    std::cout << "Camera calibration completed. RMS error: " << rms << std::endl;
    std::cout << "Camera Matrix:\n" << cameraMatrix_ << std::endl;
    std::cout << "Distortion Coefficients:\n" << distCoeffs_.t() << std::endl;

    // Verify the accuracy of the calibration parameters
    double totalError = 0.0;
    int totalPoints = 0;
    for (size_t i = 0; i < objectPoints.size(); ++i) {
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix_, distCoeffs_, projectedPoints);
        double error = cv::norm(imagePoints[i], projectedPoints, cv::NORM_L2);
        totalError += error * error;
        totalPoints += objectPoints[i].size();
    }
    double meanError = std::sqrt(totalError / totalPoints);
    std::cout << "Mean reprojection error: " << meanError << std::endl;

    return true;
}
