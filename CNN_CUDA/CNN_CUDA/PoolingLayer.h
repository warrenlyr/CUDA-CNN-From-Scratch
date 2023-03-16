#pragma once
#include <string>
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#define POOLING_SIZE 3


/*
* Pooling Layer: Max Pooling 2D.
* 
* @param image: the image to be processed
*/
Mat pool2D_max(const Mat image) {

	// Original image size
	int image_width = image.cols;
	int image_height = image.rows;

	// New image size
	int new_image_width = image_width / POOLING_SIZE;
	int new_image_height = image_height / POOLING_SIZE;

	// Init the new image
	Mat new_image = Mat::zeros(new_image_height, new_image_width, CV_8UC1);

	// Loop for each pixel of new image
	for (int i = 0; i < new_image_height; i++) {
		for (int j = 0; j < new_image_width; j++) {
			// Find the left upper point in original image
			int corner_i = i * POOLING_SIZE;
			int corner_j = j * POOLING_SIZE;

			// Initialize the maximum to int_min
			int maximum = INT_MIN;

			// Loop and find the maximum
			for (int pool_i = corner_i; pool_i < corner_i + POOLING_SIZE; pool_i++) {
				for (int pool_j = corner_j; pool_j < corner_j + POOLING_SIZE; pool_j++) {
					// The value of the pixel of original image
					int image_value = image.at<uchar>(pool_i, pool_j);

					// Find maximum
					if (image_value > maximum) {
						maximum = image_value;
					}
				}
			}

			// Save the calculated new pixel to new image
			new_image.at<uchar>(i, j) = maximum;
		}
	}

	return new_image;
}

/*
* CUDA: Naive
*/
__global__ void poolingKernel(cudaPitchedPtr image, cudaPitchedPtr new_image, int count, int row, int col)
{
    // Compute the image index [k] of this thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid overflow
    if (index < count) {
        // Get the start pointer of this image
        char* imagePtrSlice = (char*)image.ptr + index * image.pitch * col;

        // Get the start pointer of this new image
        char* newimagePtrSlice = (char*)new_image.ptr + index * new_image.pitch * col / POOLING_SIZE;

        // Loop for each pixel of the new image
        for (int i = 0; i < row / POOLING_SIZE; i++) {
            // Get the start pointer of this row of new image
            int* newrowData = (int*)(newimagePtrSlice + i * new_image.pitch);
            for (int j = 0; j < col / POOLING_SIZE; j++) {
                // Find the left upper point in the original image
                int corner_i = i * POOLING_SIZE;
                int corner_j = j * POOLING_SIZE;

                // Initialize the maximum
                int maximum = newrowData[j];

                // Loop and find the maximum
                for (int pool_i = corner_i; pool_i < corner_i + POOLING_SIZE; pool_i++) {
                    // Get the start pointer of this row of image
                    int* rowData = (int*)(imagePtrSlice + pool_i * image.pitch);

                    for (int pool_j = corner_j; pool_j < corner_j + POOLING_SIZE; pool_j++) {
                        // The value of the pixel of original image
                        int pixel = rowData[pool_j];

                        // Find maximum
                        maximum = pixel > maximum ? pixel : maximum;
                    }
                }

                // Assign pooling result to the new image
                newrowData[j] = maximum;
            }
        }
    }
}

/*
* CUDA: unrolling
*/
__global__ void optimized_poolingKernel(cudaPitchedPtr image, cudaPitchedPtr new_image, int count, int row, int col)
{
    // Compute the image index [k] of this thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid overflow
    if (index < count) {
        // Get the start pointer of this image
        char* imagePtrSlice = (char*)image.ptr + index * image.pitch * col;

        // Get the start pointer of this new image
        char* newimagePtrSlice = (char*)new_image.ptr + index * new_image.pitch * col / POOLING_SIZE;

        // Loop for each pixel of the new image
        for (int i = 0; i < row / POOLING_SIZE; i++) {
            // Get the start pointer of this row of new image
            int* newrowData = (int*)(newimagePtrSlice + i * new_image.pitch);
            for (int j = 0; j < col / POOLING_SIZE; j++) {
                // Find the left upper point in the original image
                int corner_i = i * POOLING_SIZE;
                int corner_j = j * POOLING_SIZE;

                // Initialize the maximum
                int maximum = newrowData[j];

                // Find the maximum, used loop unrolling, ASSUMED POOLING_SIZE = 3!!!
                int i1 = corner_i + 1;
                int i2 = corner_i + 2;
                int j1 = corner_j + 1;
                int j2 = corner_j + 2;

                int* rowData0 = (int*)(imagePtrSlice + corner_i * image.pitch);
                int* rowData1 = (int*)(imagePtrSlice + i1 * image.pitch);
                int* rowData2 = (int*)(imagePtrSlice + i2 * image.pitch);

                int pixel0 = rowData0[corner_j];
                maximum = pixel0 > maximum ? pixel0 : maximum;

                int pixel1 = rowData0[j1];
                maximum = pixel1 > maximum ? pixel1 : maximum;

                int pixel2 = rowData0[j2];
                maximum = pixel2 > maximum ? pixel2 : maximum;

                int pixel3 = rowData1[corner_j];
                maximum = pixel3 > maximum ? pixel3 : maximum;

                int pixel4 = rowData1[j1];
                maximum = pixel4 > maximum ? pixel4 : maximum;

                int pixel5 = rowData1[j2];
                maximum = pixel5 > maximum ? pixel5 : maximum;

                int pixel6 = rowData2[corner_j];
                maximum = pixel6 > maximum ? pixel6 : maximum;

                int pixel7 = rowData2[j1];
                maximum = pixel7 > maximum ? pixel7 : maximum;

                int pixel8 = rowData2[j2];
                maximum = pixel8 > maximum ? pixel8 : maximum;

                // Assign pooling result to the new image
                newrowData[j] = maximum;
            }
        }
    }
}