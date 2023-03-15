/*-----------------------------------------------
* Author: Warren Liu, Chris Ma
* Final Project: CUDA CNN Implementation
* CSS535 - High Performance Computing
* School of STEM, Department of Computer Science & Software Engineering
* Winter 2023, University of Washington Bothell
* -----------------------------------------------
* Compile Prerequisites
* 1. Visual Studio 17 2022
* 2. CUDA Toolkit 12.0
* 3. CMake
* 4. OpenCV 4.7.0 with CUDA support (need to be compiled from source)
*/
#include "Filters.h"
#include "Helpers.h"
#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"

#include <chrono>




using namespace chrono;


#define CATS_PATH ".\\data\\Animal Images\\cats\\"
#define BASE_PATH ".\\data\\Animal Images\\"
#define CATS_PATH ".\\data\\Animal Images\\cats\\"
#define CATS_PATH_OUTPUT ".\\data\\\\Animal Imagescats_output\\"
#define DOGS_PATH ".\\data\\Animal Images\\dogs\\"
#define DOGS_PATH_OUTPUT ".\\data\\Animal Images\\dogs_output\\"


void cnn_conv_pool_cpu(vector<Mat> images);


int main()
{
    
    //cuda::printCudaDeviceInfo(0);


    // Load Images
    vector<filesystem::path> cats_files = getFileNames(CATS_PATH);

    vector<Mat> cats_images;
    bool load_image_status = loadImages(cats_files, cats_images);
    if (!load_image_status) {
        fprintf(stderr, "Could not load images. Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 10; i++) {
        cout << int(cats_images[0].at<uchar>(0, i)) << " ";
    }
    cout << endl;


    // [CPU] Convolutional and Pooling Layer
    //cnn_conv_pool_cpu(cats_images);


    // Convert Mat to int array
    const int col = cats_images[0].cols;
    const int col_output = col - 3;
    const int row = cats_images[0].rows;
    const int row_output = row - 3;
    const int count = cats_images.size();

    int*** intImages = new int** [count];
    int*** intImages_output = new int** [count];
    for (int k = 0; k < count; k++) {
        intImages[k] = new int* [row];
        intImages_output[k] = new int* [row];
        for (int i = 0; i < row; i++) {
			intImages[k][i] = new int[col];
            for (int j = 0; j < col; j++) {
                intImages[k][i][j] = 0;
            }
		}
        
        for (int i = 0; i < row_output; i++) {
			intImages_output[k][i] = new int[col_output];
            for (int j = 0; j < col_output; j++) {
				intImages_output[k][i][j] = 0;
			}
        }
    }

    if (!convertMatToIntArr3D(cats_images, intImages, count, row, col)) {
        fprintf(stderr, "Could not convert Mat to int array. Program aborted.\n");
		exit(EXIT_FAILURE);
    }

    int* intImages1D = flatten3Dto1D(intImages, count, row, col);
    int* intImages_output1D = flatten3Dto1D(intImages_output, count, row_output, col_output);
    startCudaCov2Dwith3Darr(intImages1D, intImages_output1D, count, row, col, row_output, col_output);

    /*int* intImages1D = new int[count * row * col];
    int* intImages_output1D = new int[count * row * col];
    if (convertMatToIntArr1D(cats_images, intImages1D, count, row, col)) {
		fprintf(stderr, "Could not convert Mat to int array. Program aborted.\n");
		exit(EXIT_FAILURE);
	}*/


    
    
    return 0;
}


/*
* The CPU version of the Convolutional Layer and Pooling Layer in CNN.
* 
* @param images: The images to be processed.
* @return none
*/
void cnn_conv_pool_cpu(vector<Mat> images) {
    // Convolutional Layer
    vector<Mat> conv_images;
    auto start_conv = high_resolution_clock::now();
    Filters filters;
    for (auto image : images) {
        vector<Mat> new_images = conv2D_static(image, filters);
        for (auto new_image : new_images) {
            conv_images.push_back(new_image);
        }
    }
    auto end_conv = high_resolution_clock::now();

    // Pooling Layer
    vector<Mat> pool_images;
    auto start_pool = high_resolution_clock::now();
    for (auto image : conv_images) {
        Mat new_image = pool2D_max(image);
        pool_images.push_back(new_image);
    }
    auto end_pool = high_resolution_clock::now();

    // Durations
    auto duration_conv = duration_cast<milliseconds>(end_conv - start_conv).count();
    auto duration_pool = duration_cast<milliseconds>(end_pool - start_pool).count();
    printf("Convolutional Layer took %d milliseconds to run.\n", duration_conv);
    printf("Pooling Layer took %d milliseconds to run.\n", duration_pool);
}


/*
* Catch CUDA errors and print the error message.
* 
* @param err: The CUDA function return value.
*/
void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


