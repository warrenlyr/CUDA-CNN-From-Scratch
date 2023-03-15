#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
		if (count == 100) break;
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


/*
* Convert OpenCV Mat to int array.
*
*/
bool convertMatToIntArr(const vector<Mat> images, int*** intImages, const int count, const int row, const int col) {
	if (!images.size()) {
		fprintf(stderr, "Error: images is empty!");
		return false;
	}

	int cnt = 0;
	for (Mat image : images) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				intImages[cnt][i][j] = image.at<uchar>(i, j);
			}
		}
		++cnt;
	}

	return true;
}


int* flatten3Dto1D(int*** arr3D, int x, int y, int z) {
	int* arr1D = new int[x * y * z];

	for (int i = 0; i < x; i++) {
		for(int j = 0; j < y; j++) {
			for (int k = 0; k < z; k++) {
				arr1D[i * z * y + j * z + k] = arr3D[i][j][k];
			} 
		}
	}
	
	return arr1D;
}


/*
* Helper function for using CUDA Convolutional Layer.
*/
cudaError_t startCudaCov2D(const int* images, const int* images_output, int count, int row, int col) {
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto ExitPoint;
	}
	else {
		cout << "cudaSetDevice success!" << endl;
	}

	// Allocate GPU buffers for 3D int array of images
	cudaExtent extent = make_cudaExtent(row * sizeof(int), count, col);
	cudaPitchedPtr dev_ptr;
	cudaStatus = cudaMalloc3D(&dev_ptr, extent);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc3D failed!");
		goto Error;
	}
	else {
		cout << "cudaMalloc3D success!" << endl;{}
	}

	// Copy images to device memory
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)images, row * sizeof(int), row, count);
	copyParams.dstPtr = dev_ptr;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaStatus = cudaMemcpy3D(&copyParams);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3D failed!");
		goto Error;
	}
	else 
		cout << "cudaMemcpy3D success!" << endl; {} {
	}

Error:
	cudaFree(dev_ptr.ptr);

ExitPoint:
	return cudaStatus;
}
