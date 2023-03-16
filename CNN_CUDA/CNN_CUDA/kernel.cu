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
#define DEMO_MODE true


void cnn_conv_pool_cpu(vector<Mat> images, vector<Mat>& conv_images, vector<Mat>& pool_images);
void cnn_conv_pool_gpu(vector<Mat>, vector<Mat>& conv_images, vector<Mat>& pool_images);


int main()
{
    // TEST: OpenCV with CUDA support installation status
    //cuda::printCudaDeviceInfo(0);

    // Print CUDA device information
    printDeviceProperties();

    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);


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
    namedWindow("Original Image", WINDOW_NORMAL);
    resizeWindow("Original Image", 450, 450);
    imshow("Original Image", pool_images[0]);
    waitKey(0);
    return 1;

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

    cout << "==================================================" << endl;
    cout << "=                   CPU RESULT                   =" << endl;
    cout << "==================================================" << endl;
    printf("[CPU] Convolutional Layer took %d ms to run.\n", duration_conv);
    printf("[CPU] Pooling Layer took %d ms to run.\n", duration_pool);
}


/*
* The GPU version of the Convolutional Layer and Pooling Layer in CNN.
* 
* @param images: The images to be processed.
* @return none
*/
void cnn_conv_pool_gpu(vector<Mat> images, vector<Mat> &conv_images, vector<Mat> &pool_images) {
    if (DEMO_MODE) {
        cout << "==================================================" << endl;
        cout << "=              DEMO MODE DISPLAY                 =" << endl;
        cout << "==================================================" << endl;
    }
    // Convert Mat to int array for kernel function use
    const int col = images[0].cols;
    const int col_output = col - 3;
    const int col_output_pool = col_output / POOLING_SIZE;
    const int row = images[0].rows;
    const int row_output = row - 3;
    const int row_output_pool = row_output / POOLING_SIZE;
    const int count = images.size();

    // Allocate arr for transform between 1D array and 3Darray
    int*** intImages = new int** [count];
    int*** intImages_output_conv = new int** [count];
    int*** intImages_output_pool = new int** [count];
    for (int k = 0; k < count; k++) {
        intImages[k] = new int* [row];
        intImages_output_conv[k] = new int* [row];
        intImages_output_pool[k] = new int* [row];

        // Original images
        for (int i = 0; i < row; i++) {
            intImages[k][i] = new int[col];
            for (int j = 0; j < col; j++) {
                intImages[k][i][j] = 0;
            }
        }

        // Conv2D output images
        for (int i = 0; i < row_output; i++) {
            intImages_output_conv[k][i] = new int[col_output];
            for (int j = 0; j < col_output; j++) {
                intImages_output_conv[k][i][j] = 0;
            }
        }

        // Pool output images
        for (int i = 0; i < row_output_pool; i++) {
            intImages_output_pool[k][i] = new int[col_output_pool];
            for (int j = 0; j < col_output_pool; j++) {
                intImages_output_pool[k][i][j] = 0;
            }
        }
    }

    if (!convertMatToIntArr3D(images, intImages, count, row, col)) {
        fprintf(stderr, "Could not convert Mat to int array. Program aborted.\n");
        exit(EXIT_FAILURE);
    }


    // Convert the 3D arr to 1D arr to pass to kernel
    int* intImages1D = flatten3Dto1D(intImages, count, row, col); // Input images
    int* intImages_output_conv1D = flatten3Dto1D(intImages_output_conv, count, row_output, col_output);
    int* intImages_output_pool1D = flatten3Dto1D(intImages_output_pool, count, row_output_pool, col_output_pool);


    // Get filters
    Filters filters;
    // Record time
    float conv_time_total_memcopy = 0.0, conv_time_total_kernel = 0.0;
    float pooling_time_total_memcopy = 0.0, pooling_time_total_kernel = 0.0;

    // Perform task on each filter
    // There is a much better way that can run multiple kernel functions at once
    // so we don't have to copy same data to device multiple times.
    // But we don't have time to do that
    for (int i = 0; i < filters.num; i++) {
        float time_memcopy = 0.0, time_kernel = 0.0;
        // Perform Convolutional Layer
        conv2DwithCuda(
            intImages1D, intImages_output_conv1D, filters.filterArr[i],
            time_memcopy, time_kernel,
            count, row, col, row_output, col_output
        );
        conv_time_total_memcopy += time_memcopy;
        conv_time_total_kernel += time_kernel;

        poolingWithCuda(
            intImages_output_conv1D, intImages_output_pool1D,
            pooling_time_total_memcopy, pooling_time_total_kernel,
            count, row_output, col_output
        );


        if (DEMO_MODE) {
            // Reconstruc images from int array
            
            // Conv2D result
            // Convert the result from 1D arr back to 3D arr
            intImages_output_conv = build3Dfrom1D(intImages_output_conv1D, count, row_output, col_output);

            // Rebuild the Mat image from 3D Array to visualize the result
            vector<Mat> images_output;
            if (!convertIntArr3DToMat(intImages_output_conv, images_output, count, row_output, col_output)) {
                fprintf(stderr, "Could not convert result int array back to Mat. Program aborted.\n");
                exit(EXIT_FAILURE);
            }

            // Check if cpu and gpu results are equal
            cout << "Check if GPU result equal to CPU result: " <<
                checkImagesEqual(conv_images[i], images_output[0], row_output, col_output) << endl;

            for (auto image : images_output) {
                string name = "Image-conv2d-" + to_string(i);
                namedWindow(name, WINDOW_NORMAL);
                resizeWindow(name, 450, 450);
                imshow(name, image);
            }

            // Pooling result
            // Convert the result from 1D arr back to 3D arr
            intImages_output_pool = build3Dfrom1D(intImages_output_pool1D, count, row_output_pool, col_output_pool);

            // Rebuild the Mat image from 3D Array to visualize the result
            vector<Mat> images_output_pool;
            if (!convertIntArr3DToMat(intImages_output_pool, images_output_pool, count, row_output_pool, col_output_pool)) {
				fprintf(stderr, "Could not convert result int array back to Mat. Program aborted.\n");
				exit(EXIT_FAILURE);
			}

            // Check if cpu and gpu results are equal
            cout << "Check if GPU result equal to CPU result: " <<
                checkImagesEqual(pool_images[i], images_output_pool[0], row_output_pool, col_output_pool) << endl;

            for (auto image : images_output_pool) {
                string name = "Image-pool-" + to_string(i);
                namedWindow(name, WINDOW_NORMAL);
                resizeWindow(name, 450, 450);
                imshow(name, image);
            }
        }
    }

    if (DEMO_MODE) {
        waitKey(0);
    }

    cout << "==================================================" << endl;
    cout << "=                   GPU RESULT                   =" << endl;
    cout << "==================================================" << endl;
    printf("[GPU] Convolutional Layer total time: %f ms, memcopy: %f ms, kernel: %f ms.\n",
        conv_time_total_memcopy + conv_time_total_kernel, conv_time_total_memcopy, conv_time_total_kernel
    );
    printf("[GPU] Pooling Layer total time: %f ms, memcopy: %f ms, kernel: %f ms.\n",
        pooling_time_total_memcopy + pooling_time_total_kernel, pooling_time_total_memcopy, pooling_time_total_kernel
    );
    


    // Cleanup
    for (int k = 0; k < count; k++) {
        for (int i = 0; i < row; i++) {
            delete[] intImages[k][i];
        }
        delete[] intImages[k];

        for (int i = 0; i < row_output; i++) {
            delete[] intImages_output_conv[k][i];
        }
        delete intImages_output_conv[k];

        for (int i = 0; i < row_output_pool; i++) {
            delete[] intImages_output_pool[k][i];
        }
        delete intImages_output_pool[k];
    }
    delete[] intImages;
    delete[] intImages_output_conv;
    delete[] intImages_output_pool;

    delete[] intImages1D;
    delete[] intImages_output_conv1D;
    delete[] intImages_output_pool1D;
}

