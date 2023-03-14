#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

vector<filesystem::path> getFileNames(const string& path) {
	vector<filesystem::path> files;

	// TEST USE
	int count = 0;
	for (const auto& entry : filesystem::directory_iterator(path)) {
		if(count == 100) break;
		files.push_back(entry.path());
		++count;
	}

	return files;
}

bool loadImages(const vector<filesystem::path> &files, vector<Mat> &images) {
	if (!files.size()) {
		fprintf(stderr, "No files found in the specified path.\n");
		return false;
	}

	int success = 0;
	int failed = 0;

	for (int i = 0; i < files.size(); i++) {
		Mat image = imread(files[i].string(), IMREAD_GRAYSCALE);
		if (image.empty()) {
			//fprintf(stderr, "Could not read the image: %s\n", files[i].string());
			++failed;
			continue;
		}

		++success;
		images.push_back(image);
	}

	printf("Seccussfully loaded %d images, failed loaded %d images.", success, failed);

	return true; 
}