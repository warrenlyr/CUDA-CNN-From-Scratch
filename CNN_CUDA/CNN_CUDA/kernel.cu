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


void cnn_conv_pool_cpu(vector<Mat> images, vector<Mat>& conv_images, vector<Mat>& pool_images);
void cnn_conv_pool_gpu(vector<Mat>, vector<Mat>& conv_images, vector<Mat>& pool_images);


int main()
{
    // TEST: OpenCV with CUDA support installation status
    //cuda::printCudaDeviceInfo(0);

    // Print CUDA device information
    printDeviceProperties();


    // Load Images
    vector<filesystem::path> cats_files = getFileNames(CATS_PATH);

    vector<Mat> cats_images;
    bool load_image_status = loadImages(cats_files, cats_images);
    if (!load_image_status) {
        fprintf(stderr, "Could not load images. Program aborted.\n");
        exit(EXIT_FAILURE);
    }

    // TEST: Print some original images
    /*for (int k = 0; k < 12; k++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                cout << int(cats_images[k].at<uchar>(i, j)) << " ";
            }
            cout << endl;
        }
        cout << endl;
        cout << endl;
    }*/
    
    vector<Mat> conv_images;
    vector<Mat> pool_images;

    // [CPU] Convolutional and Pooling Layer
    cnn_conv_pool_cpu(cats_images, conv_images, pool_images);

    // [GPU] Convolutional and Pooling Layer
    cnn_conv_pool_gpu(cats_images, conv_images, pool_images);
    
    
    
    return 0;
}


/*
* The CPU version of the Convolutional Layer and Pooling Layer in CNN.
* 
* @param images: The images to be processed.
* @return none
*/
void cnn_conv_pool_cpu(vector<Mat> images, vector<Mat>& conv_images, vector<Mat>& pool_images) {
    // Convolutional Layer
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
* The GPU version of the Convolutional Layer and Pooling Layer in CNN.
* 
* @param images: The images to be processed.
* @return none
*/
void cnn_conv_pool_gpu(vector<Mat> images, vector<Mat> &conv_images, vector<Mat> &pool_images) {
    // Convert Mat to int array for kernel function use
    const int col = images[0].cols;
    const int col_output = col - 3;
    const int row = images[0].rows;
    const int row_output = row - 3;
    const int count = images.size();

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

    if (!convertMatToIntArr3D(images, intImages, count, row, col)) {
        fprintf(stderr, "Could not convert Mat to int array. Program aborted.\n");
        exit(EXIT_FAILURE);
    }


    // Convert the 3D arr to 1D arr to pass to kernel
    int* intImages1D = flatten3Dto1D(intImages, count, row, col); // Input images
    int* intImages_output1D = flatten3Dto1D(intImages_output, count, row_output, col_output); // Store results


    // Get filters
    Filters filters;
    // Record time
    float time_total_memcopy = 0.0, time_total_kernel = 0.0;

    // Perform task on each filter
    // There is a much better way that can run multiple kernel functions at once
    // so we don't have to copy same data to device multiple times.
    // But we don't have time to do that
    for (int i = 0; i < filters.num; i++) {
        float time_memcopy = 0.0, time_kernel = 0.0;
        // Perform Convolutional Layer
        startCudaCov2Dwith3Darr(
            intImages1D, intImages_output1D, filters.filterArr[i], 
            time_memcopy, time_kernel,
            count, row, col, row_output, col_output
        );
        time_total_memcopy += time_memcopy;
        time_total_kernel += time_kernel;


        // Convert the result from 1D arr back to 3D arr
        intImages_output = build3Dfrom1D(intImages_output1D, count, row_output, col_output);


        // Rebuild the Mat image from 3D Array to visualize the result
        vector<Mat> images_output;
        if (!convertIntArr3DToMat(intImages_output, images_output, count, row_output, col_output)) {
            fprintf(stderr, "Could not convert result int array back to Mat. Program aborted.\n");
            exit(EXIT_FAILURE);
        }

        cout << "Equal: " << checkImagesEqual(conv_images[i], images_output[0], row_output, col_output);

        /*for (auto i : images_output) {
            imshow("image", i);
            waitKey(0);
        }*/
    }

    printf("Total time: %f, memcopy: %f, kernel: %f.\n",
        time_total_memcopy + time_total_kernel, time_total_memcopy, time_total_kernel
    );
    


    // Cleanup
    for (int k = 0; k < count; k++) {
        for (int i = 0; i < row; i++) {
            delete[] intImages[k][i];
        }
        delete[] intImages[k];

        for (int i = 0; i < row_output; i++) {
            delete[] intImages_output[k][i];
        }
        delete intImages_output[k];
    }
    delete[] intImages;
    delete[] intImages_output;

    delete[] intImages1D;
    delete[] intImages_output1D;
}

