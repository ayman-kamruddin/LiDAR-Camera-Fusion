#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>

class DataLoader {
public:
    DataLoader(const std::string& dataPath);

    const std::vector<std::string>& getImageFiles() const;
    const std::vector<std::string>& getCloudFiles() const;

private:
    std::vector<std::string> imageFiles_;
    std::vector<std::string> cloudFiles_;

    void loadPairs(const std::string& imageDir, const std::string& cloudDir);
};

#endif // DATA_LOADER_H
