#include "Filters.h"
#include "Helpers.h"
#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"

#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/cuda.hpp>

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
    // Load Images
    /*vector<filesystem::path> cats_files = getFileNames(CATS_PATH);
    vector<Mat> cats_images;
    bool load_image_status = loadImages(cats_files, cats_images);
    if (!load_image_status) {
        fprintf(stderr, "Could not load images. Program aborted.\n");
        exit(EXIT_FAILURE);
    }*/

    // [CPU] Convolutional and Pooling Layer
    //cnn_conv_pool_cpu(cats_images);

    cuda::printCudaDeviceInfo(0);

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

