#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


/*
* Get all the file names in the specified path.
* Since it's a part of our final project,
* there's no file type checking.
* We assume all the files in the path are images.
* 
* @param path: the path to get the file names
* @return a vector of file names
*/
vector<filesystem::path> getFileNames(const string& path) {
	vector<filesystem::path> files;

	// Check if the path exists
	filesystem::path p(path);
	if (!filesystem::exists(p)) {
		fprintf(stderr, "The specified path does not exist.\n");
		return files;
	}

	// TEST USE
	int count = 0;
	// If the path exist, get all the files in the path
	for (const auto& entry : filesystem::directory_iterator(path)) {
		if (count == 10000) break;
		files.push_back(entry.path());
		++count;
	}

	return files;
}


/*
* Load all images found by getFileNames(). There's no file type checking.
* 
* @param files: a vector of file names to be loaded
* @param images: a vector of `Mat` object to store the loaded images
* @return true if successfully loaded at least one image, false otherwise
*/
bool loadImages(const vector<filesystem::path>& files, vector<Mat>& images) {
	if (!files.size()) {
		fprintf(stderr, "No files found in the specified path.\n");
		return false;
	}

	int success = 0;
	int failed = 0;

	for (int i = 0; i < files.size(); i++) {
		Mat image = imread(files[i].string(), IMREAD_GRAYSCALE);
		if (image.empty()) {
			++failed;
			continue;
		}

		++success;
		images.push_back(image);
	}

	printf("Seccussfully loaded %d images, could not load %d images.\n", success, failed);

	return true;
}