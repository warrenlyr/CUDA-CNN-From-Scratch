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