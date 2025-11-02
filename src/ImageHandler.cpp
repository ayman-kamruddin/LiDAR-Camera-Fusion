#include "ImageHandler.h"
#include <opencv2/opencv.hpp>
#include <iostream>

ImageHandler::ImageHandler(int checkerboardRows, int checkerboardCols, float checkerboardSize,
                           bool debugMode, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
    : checkerboardRows_(checkerboardRows),
      checkerboardCols_(checkerboardCols),
      checkerboardSize_(checkerboardSize),
      debugMode_(debugMode),
      cameraMatrix_(cameraMatrix),
      distCoeffs_(distCoeffs) {}

PlaneObservation ImageHandler::extractPlane(const std::string& imagePath) {
    PlaneObservation observation;

    // Load the image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return observation;
    }

    // Find checkerboard corners
    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(image, cv::Size(checkerboardCols_, checkerboardRows_), corners,
                                           cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    if (!found) {
        std::cerr << "Checkerboard pattern not found in image: " << imagePath << std::endl;
        return observation;
    }

    // Refine corner locations
    cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

    if (debugMode_) {
        // Draw corners for debugging
        cv::Mat debugImage;
        cv::cvtColor(image, debugImage, cv::COLOR_GRAY2BGR);
        cv::drawChessboardCorners(debugImage, cv::Size(checkerboardCols_, checkerboardRows_), corners, found);
        cv::imshow("Checkerboard Detection", debugImage);
        cv::waitKey(100); // Pause to view the image
    }

    // Generate 3D points for the checkerboard
    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < checkerboardRows_; ++i) {
        for (int j = 0; j < checkerboardCols_; ++j) {
            objectPoints.emplace_back(j * checkerboardSize_, i * checkerboardSize_, 0.0f);
        }
    }

    // Solve PnP to find the camera pose
    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(objectPoints, corners, cameraMatrix_, distCoeffs_, rvec, tvec);

    if (!success) {
        std::cerr << "Failed to solve PnP for image: " << imagePath << std::endl;
        return observation;
    }

    // Convert rotation vector to rotation matrix
    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);

    // Extract the plane normal in the camera frame
    cv::Mat normalCamera = rotationMatrix * (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
    Eigen::Vector3d planeNormal(normalCamera.at<double>(0, 0),
                                normalCamera.at<double>(1, 0),
                                normalCamera.at<double>(2, 0));

    Eigen::Vector3d tEigen(tvec.at<double>(0, 0),
                           tvec.at<double>(1, 0),
                           tvec.at<double>(2, 0));

    double planeD = -planeNormal.dot(tEigen);

    double norm = planeNormal.norm();
    if (norm > 1e-9) {
        planeNormal /= norm;
        planeD /= norm;
    }

    // Construct the plane equation
    observation.plane = Eigen::Vector4d(planeNormal(0), planeNormal(1), planeNormal(2), planeD);

    // Compute the checkerboard center in camera coordinates for translation alignment
    double centerX = 0.5 * (checkerboardCols_ - 1) * checkerboardSize_;
    double centerY = 0.5 * (checkerboardRows_ - 1) * checkerboardSize_;
    Eigen::Vector3d boardCenterObject(centerX, centerY, 0.0);

    Eigen::Matrix3d rotationEigen;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rotationEigen(r, c) = rotationMatrix.at<double>(r, c);
        }
    }

    observation.centroid = rotationEigen * boardCenterObject + tEigen;

    if (debugMode_) {
        std::cout << "Extracted plane: " << observation.plane.transpose()
                  << " | centroid: " << observation.centroid.transpose() << std::endl;
    }

    return observation;
}
