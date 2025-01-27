#include "DataLoader.h"
#include <iostream>
#include <filesystem>
#include <unordered_map>

namespace fs = std::filesystem;

DataLoader::DataLoader(const std::string& dataPath) {
    std::string imageDir = fs::path(dataPath) / "images";
    std::string cloudDir = fs::path(dataPath) / "pointclouds";

    if (!fs::exists(imageDir) || !fs::exists(cloudDir)) {
        throw std::runtime_error("Image or point cloud directory does not exist: " + dataPath);
    }

    loadPairs(imageDir, cloudDir);

    if (imageFiles_.empty() || cloudFiles_.empty()) {
        throw std::runtime_error("No valid image-point cloud pairs found in " + dataPath);
    }
}

const std::vector<std::string>& DataLoader::getImageFiles() const {
    return imageFiles_;
}

const std::vector<std::string>& DataLoader::getCloudFiles() const {
    return cloudFiles_;
}

void DataLoader::loadPairs(const std::string& imageDir, const std::string& cloudDir) {
    std::unordered_map<std::string, std::string> imageMap;

    // Load all image files into a map (key: filename without extension)
    for (const auto& entry : fs::directory_iterator(imageDir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().stem().string(); // Get filename without extension
            imageMap[filename] = entry.path().string();
        }
    }

    // Match point clouds with corresponding images
    for (const auto& entry : fs::directory_iterator(cloudDir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().stem().string(); // Get filename without extension
            if (imageMap.find(filename) != imageMap.end()) {
                imageFiles_.push_back(imageMap[filename]);
                cloudFiles_.push_back(entry.path().string());
            }
        }
    }

    // Print summary
    std::cout << "Found " << imageFiles_.size() << " valid image-point cloud pairs." << std::endl;
}
