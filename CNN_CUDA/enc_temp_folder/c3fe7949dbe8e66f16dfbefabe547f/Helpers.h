#pragma once
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"

using namespace std;
using namespace cv;
using namespace chrono;


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
		//cout << entry.path() << endl;
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

	cout << "==================================================" << endl;
	cout << "=                   LOAD IMAGE                   =" << endl;
	cout << "==================================================" << endl;
	printf("Seccussfully loaded %d images, could not load %d images.\n", success, failed);

	return true;
}


/*
* Print devide properties.
*/
void printDeviceProperties() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cout << "==================================================" << endl;
	cout << "=                  DEVICE INFO                   =" << endl;
	cout << "==================================================" << endl;
	printf("Device Name: %s\n", prop.name);
	printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
	printf("Clock Rate: %d\n", prop.clockRate);
	printf("Total Global Memory: %d\n", prop.totalGlobalMem);
	printf("Total Constant Memory: %d\n", prop.totalConstMem);
	printf("Shared Memory Per Block: %d\n", prop.sharedMemPerBlock);
	printf("Registers Per Block: %d\n", prop.regsPerBlock);
	printf("Warp Size: %d\n", prop.warpSize);
	printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
	printf("Max Threads Dim: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("Max Grid Size: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("Total Constant Memory: %d\n", prop.totalConstMem);
	printf("Major: %d\n", prop.major);
	printf("Minor: %d\n", prop.minor);
	printf("Texture Alignment: %d\n", prop.textureAlignment);
	printf("Device Overlap: %d\n", prop.deviceOverlap);
	printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
	printf("Kernel Exec Timeout Enabled: %d\n", prop.kernelExecTimeoutEnabled);
}


/*
* Convert OpenCV Mat to 3D int array.
* 
* @param images: a vector of `Mat` object to be converted
* @param intImages3D: a 3D int array to store the converted images
* @param count: the number of images
* @param row: the number of rows of each image
* @param col: the number of columns of each image
* @return none
*
*/
bool convertMatToIntArr3D(const vector<Mat> images, int*** intImages3D, const int count, const int row, const int col) {
	if (!images.size()) {
		fprintf(stderr, "Error: images is empty!");
		return false;
	}

	int cnt = 0;
	for (Mat image : images) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				intImages3D[cnt][i][j] = image.at<uchar>(i, j);
			}
		}
		++cnt;
	}

	return true;
}


/*
* Convert 3D int array to OpenCV Mat.
* 
* @param intImages3D: a 3D int array to be converted
* @param images: a vector of `Mat` object to store the converted images
* @param count: the number of images
* @param row: the number of rows of each image
* @param col: the number of columns of each image
* @return true if successfully converted, false otherwise
*/
bool convertIntArr3DToMat(int*** intImages3D, vector<Mat>& images, const int count, const int row, const int col) {
	int cnt = 0;
	for (int k = 0; k < count; ++k) {
		Mat image(row, col, CV_8UC1);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				image.at<uchar>(i, j) = intImages3D[cnt][i][j];
			}
		}
		images.push_back(image);
		++cnt;
	}

	return true;
}


/*
* Convert OpenCV Mat to 1D int array.
* 
* @param images: a vector of `Mat` object to be converted
* @param intImages1D: a 1D int array to store the converted images
* @param count: the number of images
* @param row: the number of rows of each image
* @param col: the number of columns of each image
* @return true if successfully converted, false otherwise
*
*/
bool convertMatToIntArr1D(const vector<Mat> images, int* intImages1D, const int count, const int row, const int col) {
	if (!images.size()) {
		fprintf(stderr, "Error: images is empty!");
		return false;
	}

	int cnt = 0;
	int image_length = row * col;
	for (Mat image : images) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				intImages1D[cnt * image_length + i * col + j] = image.at<uchar>(i, j);
			}
		}
	}

	return true;
}


/*
* Flatten 3D int array to 1D int array.
* 
* @param arr3D: a 3D int array to be flattened
* @param x: the x axis
* @param y: the y axis
* @param z: the z axis
* @return a 1D int array
*/
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
* Build a 3D int array from a 1D int array.
* 
* @param arr1D: a 1D int array to be converted
* @param x: the x axis of the 3D array
* @param y: the y axis of the 3D array
* @param z: the z axis of the 3D array
* @return a 3D int array
*/
int*** build3Dfrom1D(int* arr1D, int x, int y, int z) {
	int*** arr3D = new int** [x];

	for (int i = 0; i < x; i++) {
		arr3D[i] = new int* [y];
		for (int j = 0; j < y; j++) {
			arr3D[i][j] = new int[z];
		}
	}

	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			for (int k = 0; k < z; k++) {
				arr3D[i][j][k] = arr1D[i * z * y + j * z + k];
			}
		}
	}

	return arr3D;
}


/*
* Check if two Mat object images are equal.
* 
* @param images_1: the first image
* @param images_2: the second image
* @param row: the number of rows of each image
* @param col: the number of columns of each image
* @return true if two images are equal, false otherwise
*/
bool checkImagesEqual(const Mat &images_1, const Mat &images_2, const int row, const int col) {

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			//printf("1: %d, 2: %d ", int(images_1.at<uchar>(i, j)), int(images_2.at<uchar>(i, j)));
			if (images_1.at<uchar>(i, j) != images_2.at<uchar>(i, j)) {
				return false;
			}
		}
	}

	return true;
}



/*
* Helper function for using CUDA Convolutional Layer with 1D-represented 3D array input.
* 
* @param images: a 1D-represented 3D int array to be convolved
* @param images_output: a 1D-represented 3D int array to store the convolved images
* @param filter: a 2D int array to be used as the filter
* @param time_memcopy: a float variable to store the time of memory copy
* @param time_kernel_run: a float variable to store the time of kernel running
* @param count: the number of images
* @param row: the number of rows of each image
* @param col: the number of columns of each image
* @param row_output: the number of rows of each convolved image
* @param col_output: the number of columns of each convolved image
* @return cudaSuccess if successfully convolved, cudaError_t otherwise
*/
cudaError_t conv2DwithCuda(
	const int* images, int* images_output, const int* filter, float &time_memcopy, float &time_kernel_run,
	const int count, const int row, const int col, const int row_output, const int col_output) 
{

	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time_memcopy_images;
	float time_memcopy_filter;
	float time_memcopy_result_in;
	float time_memcopy_result_out;
	float time_kernel;

	cudaPitchedPtr dev_ptr;
	cudaPitchedPtr dev_ptr_output;
	int* dev_filter;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for 3D int array of images
	cudaExtent extent = make_cudaExtent(row * sizeof(int), count, col);
	cudaStatus = cudaMalloc3D(&dev_ptr, extent);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc3D dev_ptr failed!");
		goto Error;
	}
	// Copy images to device memory
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)images, row * sizeof(int), row, count);
	copyParams.dstPtr = dev_ptr;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaEventRecord(start);
	cudaStatus = cudaMemcpy3D(&copyParams);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_memcopy_images, start, stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3D dev_ptr failed!");
		goto Error;
	}

	// Allocated GPU buffers for 3D int array of images output
	extent = make_cudaExtent(row_output * sizeof(int), count, col_output);
	cudaStatus = cudaMalloc3D(&dev_ptr_output, extent);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc3D dev_ptr_output failed!");
		goto Error;
	}

	// Copy images output to device memory
	copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)images_output, row_output * sizeof(int), row_output, count);
	copyParams.dstPtr = dev_ptr_output;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaEventRecord(start);
	cudaStatus = cudaMemcpy3D(&copyParams);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_memcopy_result_in, start, stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3D dev_ptr_output failed!");
		goto Error;
	}

	// Copy filter to device memory
	cudaStatus = cudaMalloc((void**)&dev_filter, 9 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc dev_ptr_output failed!");
		goto Error;
	}
	cudaEventRecord(start);
	cudaStatus = cudaMemcpy(dev_filter, filter, 9 * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_memcopy_filter, start, stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_ptr_output failed!");
		goto Error;
	}
	
	// Calculate config
	int grid_y = ceil(float(count) / 1024);
	int block_y;
	if (count < 1024)
		block_y = count;
	else
		block_y = 1024;

	// Launch kernel function
	dim3 gridDim(1, grid_y, 1);
	dim3 blockDim(1, block_y, 1);
	cudaEventRecord(start);
	conv2D_cuda3D <<<gridDim, blockDim >>> (dev_ptr, dev_ptr_output, dev_filter, count, row, col, row_output, col_output);
	//conv2D_cuda3D_opt <<<gridDim, blockDim>>> (dev_ptr, dev_ptr_output, dev_filter, count, row, col, row_output, col_output);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_kernel, start, stop);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "conv2D_cuda3D launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Copy output from device memory to host memory
	copyParams = { 0 };
	copyParams.srcPtr = dev_ptr_output;
	copyParams.dstPtr = make_cudaPitchedPtr((void*)images_output, row_output * sizeof(int), row_output, count);
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyDeviceToHost;
	cudaEventRecord(start);
	cudaStatus = cudaMemcpy3D(&copyParams);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_memcopy_result_out, start, stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3D images_output failed: %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	
	// Calculate time
	time_memcopy = time_memcopy_images + time_memcopy_filter + time_memcopy_result_in + time_memcopy_result_out;
	time_kernel_run = time_kernel;

Error:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_ptr.ptr);
	cudaFree(dev_ptr_output.ptr);
	cudaFree(dev_filter);

	return cudaStatus;
}


/*
* Helper function for using CUDA Pooling Layer with 3D array input.
* 
* @param image_array: a 1D-represented 3D int array to be convolved
* @param new_image_array: a 1D-represented 3D int array to store the result
* @param time_memcopy: time for memory copy
* @param time_kernel_run: time for kernel run
* @param count: number of images
* @param row: number of rows of each image
* @param col: number of columns of each image
* @return cudaError_t: cuda status
*/
cudaError_t poolingWithCuda(
	const int* image_array, int* new_image_array, 
	float& time_memcopy, float& time_kernel_run, 
	int count, int row, int col
)
{
	cudaError_t cudaStatus;
	cudaPitchedPtr image_arrptr;
	cudaPitchedPtr new_image_arrptr;

	// For time measurement
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time_memcopy_images;
	float time_memcopy_result_in;
	float time_memcopy_result_out;
	float time_kernel;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate 3D GPU buffer for image array & output array
	cudaExtent extent = make_cudaExtent(row * sizeof(int), count, col);
	cudaStatus = cudaMalloc3D(&image_arrptr, extent);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc3D failed!");
		goto Error;
	}

	extent = make_cudaExtent(row / POOLING_SIZE * sizeof(int), count, col / POOLING_SIZE);
	cudaStatus = cudaMalloc3D(&new_image_arrptr, extent);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc3D failed!");
		goto Error;
	}

	// Copy arrays from host memory to GPU buffers.
	cudaMemcpy3DParms cpy = { 0 };
	cpy.srcPtr = make_cudaPitchedPtr((void*)image_array, row * sizeof(int), row, count);
	cpy.dstPtr = image_arrptr;
	cpy.extent = make_cudaExtent(row * sizeof(int), count, col);
	cpy.kind = cudaMemcpyHostToDevice;
	cudaEventRecord(start);
	cudaStatus = cudaMemcpy3D(&cpy);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_memcopy_images, start, stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3D failed!");
		goto Error;
	}

	cudaMemcpy3DParms cpy2 = { 0 };
	cpy2.srcPtr = make_cudaPitchedPtr((void*)new_image_array, row / POOLING_SIZE * sizeof(int), row / POOLING_SIZE, count);
	cpy2.dstPtr = new_image_arrptr;
	cpy2.extent = make_cudaExtent(row / POOLING_SIZE * sizeof(int), count, col / POOLING_SIZE);
	cpy2.kind = cudaMemcpyHostToDevice;
	cudaEventRecord(start);
	cudaStatus = cudaMemcpy3D(&cpy2);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_memcopy_result_in, start, stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3D failed!");
		goto Error;
	}

	// Kernel parameters
	int threads_per_block = 1024;
	int num_blocks = count / threads_per_block + 1;

	// Launch the kernel
	cudaEventRecord(start);
	poolingKernel<<<num_blocks, threads_per_block>>>(image_arrptr, new_image_arrptr, count, row, col);
	//optimized_poolingKernel <<<num_blocks, threads_per_block>>> (image_arrptr, new_image_arrptr, count, row, col);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_kernel, start, stop);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "poolingKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching poolingKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output image array from GPU buffer to host memory.
	cudaMemcpy3DParms cpy3 = { 0 };
	cpy3.srcPtr = new_image_arrptr;
	cpy3.dstPtr = make_cudaPitchedPtr((void*)new_image_array, row / POOLING_SIZE * sizeof(int), row / POOLING_SIZE, count);
	cpy3.extent = make_cudaExtent(row / POOLING_SIZE * sizeof(int), count, col / POOLING_SIZE);
	cpy3.kind = cudaMemcpyDeviceToHost;
	cudaEventRecord(start);
	cudaStatus = cudaMemcpy3D(&cpy3);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_memcopy_result_out, start, stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Calculate time
	time_memcopy = time_memcopy_images + time_memcopy_result_in + time_memcopy_result_out;
	time_kernel_run = time_kernel;

Error:
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(image_arrptr.ptr);
	cudaFree(new_image_arrptr.ptr);

	return cudaStatus;
}

