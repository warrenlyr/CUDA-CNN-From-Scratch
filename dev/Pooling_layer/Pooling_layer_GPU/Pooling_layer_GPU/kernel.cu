/*
* Pooling layer GPU implementation
* Author: Yuan Ma
* Date: 3/14/2023
* For: CSS 535: High Performance Computing, Final Project
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <Windows.h>
#include <direct.h>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <chrono>
#include<cstdio>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace chrono;

#define BASE_PATH ".\\data\\"
#define CATS_PATH ".\\data\\cats\\"
#define CATS_PATH_OUTPUT ".\\data\\cats_output\\"
#define CATS_PATH_FINAL ".\\data\\cats_final\\"
#define DOGS_PATH ".\\data\\dogs\\"
#define DOGS_PATH_OUTPUT ".\\data\\dogs_output\\"
#define POOLING_SIZE 3

void getCurrDir();

vector<filesystem::path> getFileNames(const string& path);

cudaError_t poolingWithCuda(const int* image_array, int* new_image_array, int image_cnt, int row, int col);

int* flatten3Dto1D(int*** arr3D, int x, int y, int z);

bool convertMatToIntArr(const vector<Mat> images, int*** intImages, const int count, const int row, const int col);

bool loadImages(const vector<filesystem::path>& files, vector<Mat>& images);

__global__ void poolingKernel(cudaPitchedPtr image, cudaPitchedPtr new_image, int image_cnt, int row, int col)
{
    int* devPtr = (int*)image.ptr;
    size_t pitch = image.pitch;
    size_t slicePitch = pitch * row;

    int* devPtr2 = (int*)new_image.ptr;
    size_t pitch2 = new_image.pitch;
    size_t slicePitch2 = pitch2 * row;
    
    int* slice = devPtr;
    int* roww = slice;

    int* slice2 = devPtr2;
    int* roww2 = slice2;
    printf("kernel\n");
    for (int x = 0; x < 5; x++) {
        roww2[x] = roww[x];
        printf("%d ", roww2[x]);
    }
}

int main()
{
    // Get all images
    vector<filesystem::path> cats_files = getFileNames(CATS_PATH);
    vector<Mat> cats_images;
    bool load_image_status = loadImages(cats_files, cats_images);
    if (!load_image_status) {
        fprintf(stderr, "Could not load images. Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    // Transfer images to a 3d array
    const int col = cats_images[0].cols;
    const int row = cats_images[0].rows;
    const int count = cats_images.size();

    int*** image_array = new int** [count];
    int*** new_image_array = new int** [count];

    for (int cnt = 0; cnt < count; cnt++) {
        image_array[cnt] = new int* [row];
        for (int i = 0; i < row; i++) {
            image_array[cnt][i] = new int[col];
            for (int j = 0; j < col; j++) {
                image_array[cnt][i][j] = 0;
            }
        }
    }

    for (int cnt = 0; cnt < count; cnt++) {
        new_image_array[cnt] = new int* [row / POOLING_SIZE];
        for (int i = 0; i < row / POOLING_SIZE; i++) {
            new_image_array[cnt][i] = new int[col / POOLING_SIZE];
            for (int j = 0; j < col / POOLING_SIZE; j++) {
                new_image_array[cnt][i][j] = 0;
            }
        }
    }

    if (!convertMatToIntArr(cats_images, image_array, count, row, col)) {
        fprintf(stderr, "Could not convert Mat to int array. Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    int* intImages1D = flatten3Dto1D(image_array, count, row, col);
    int* intImages_output1D = flatten3Dto1D(new_image_array, count, row / POOLING_SIZE, col / POOLING_SIZE);

    // Finish pooling layer calculation in GPU
    cudaError_t cudaStatus = poolingWithCuda(intImages1D, intImages_output1D, count, row, col);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cout << endl;
    cout << "image:" << image_array[0][0][0] << image_array[0][0][1] << image_array[0][0][2] << image_array[0][0][3] << image_array[0][0][4] << endl;
    cout << "new image:" << intImages_output1D[0] << intImages_output1D[1] << intImages_output1D[2] << intImages_output1D[3] << intImages_output1D[4] << endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    
    return 0;
}


void getCurrDir() {
    char cwd[1024];
    if (_getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    }
    else {
        std::cerr << "Failed to get current working directory." << std::endl;
    }
}


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

// Helper function for using CUDA to computer pooling layer in parallel
cudaError_t poolingWithCuda(const int* image_array, int* new_image_array, int count, int row, int col)
{
    cudaError_t cudaStatus;
    cudaPitchedPtr image_arrptr;
    cudaPitchedPtr new_image_arrptr;

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
    cudaStatus = cudaMemcpy3D(&cpy);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy3D failed!");
        goto Error;
    }

    cudaMemcpy3DParms cpy2 = { 0 };
    cpy2.srcPtr = make_cudaPitchedPtr((void*)new_image_array, row / POOLING_SIZE * sizeof(int), row / POOLING_SIZE, count);
    cpy2.dstPtr = new_image_arrptr;
    cpy2.extent = make_cudaExtent(row / POOLING_SIZE * sizeof(int), count, col / POOLING_SIZE);
    cpy2.kind = cudaMemcpyHostToDevice;
    cudaStatus = cudaMemcpy3D(&cpy2);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy3D failed!");
        goto Error;
    }

    // Launch the kernel
    poolingKernel<<<1, 1 >>>(image_arrptr, new_image_arrptr, count, row, col);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output image array from GPU buffer to host memory.
    cudaMemcpy3DParms cpy3 = { 0 };
    cpy3.srcPtr = new_image_arrptr;
    cpy3.dstPtr = make_cudaPitchedPtr((void*)new_image_array, row / POOLING_SIZE * sizeof(int), row / POOLING_SIZE, count);
    cpy3.extent = make_cudaExtent(row / POOLING_SIZE * sizeof(int), count, col / POOLING_SIZE);
    cpy3.kind = cudaMemcpyDeviceToHost;
    cudaStatus = cudaMemcpy3D(&cpy3);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(image_arrptr.ptr);
    cudaFree(new_image_arrptr.ptr);

    return cudaStatus;
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
        for (int j = 0; j < y; j++) {
            for (int k = 0; k < z; k++) {
                arr1D[i * z * y + j * z + k] = arr3D[i][j][k];
            }
        }
    }

    return arr1D;
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
