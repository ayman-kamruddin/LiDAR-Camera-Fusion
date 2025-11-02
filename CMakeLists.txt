project(LidarCameraCalibration)

set(OpenCV_DIR "/home/ayman/Programs/opencv_build")

cmake_minimum_required(VERSION 3.10)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(include)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add the executable and source files
add_executable(lidar_camera_calibration src/main.cpp src/CalibrationHandler.cpp src/CloudHandler.cpp src/DataLoader.cpp src/ImageHandler.cpp src/Optimization.cpp src/Projector.cpp)

# Link libraries
TARGET_LINK_LIBRARIES(lidar_camera_calibration ${PCL_LIBRARIES} ${OpenCV_LIBS})
