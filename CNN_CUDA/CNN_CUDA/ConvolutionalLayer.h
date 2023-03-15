#include "Filters.h"
#include "KernelFunctions.cu"

#include <string>
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

#define NUM_FILTERS 6
#define FILTER_SIZE 3


/*
* Convolutional Layer 2D.
* Static version means it takes static created filters.
* Another version is dynamic version, which takes dynamic created filters.
* But that one has too much loops and is not efficient.
* 
* @param image: the image to be processed
* @param filters: the filters to be used
*/
vector<Mat> conv2D_static(
	const Mat image,
	Filters &filters
) {
	// Get the image size
	int image_width = image.cols;
	int image_height = image.rows;
	// Calculate the new image size
	int new_image_width = image_width - filters.size;
	int new_image_height = image_height - filters.size;

	// Init the new image
	Mat new_image_extract_vertical = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
	Mat new_image_extract_horizontal = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
	Mat new_image_extract_leftDiagonal = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
	Mat new_image_extract_rightDiagonal = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
	Mat new_image_extract_cross = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
	Mat new_image_extract_plus = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
	Mat new_image_extract_x = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
	Mat new_image_extract_square = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
	Mat new_image_extract_diamond = Mat::zeros(new_image_height, new_image_width, CV_8UC1);

	// Loop for each pixel of new image
	for (int i = 0; i < new_image_height; i++) {
		for (int j = 0; j < new_image_width; j++) {
			// Reset the sums
			filters.cleanSum();

			// Loop for each pixel of filter
			for (int filter_i = i; filter_i < i + filters.size; filter_i++) {
				for (int filter_j = j; filter_j < j + filters.size; filter_j++) {
					// Get the pixel value of the original image
					int image_value = image.at<uchar>(filter_i, filter_j);
					// Store the product sum of the image value and the filter value
					int filter_value;

					// Filter 1: Vertical Line
					filter_value = filters.verticalLine[filter_i - i][filter_j - j];
					filters.verticalSum += image_value * filter_value;

					// Filter 2: Horizontal Line
					filter_value = filters.horizontalLine[filter_i - i][filter_j - j];
					filters.horizontalSum += image_value * filter_value;

					// Filter 3: Left Diagonal Line
					filter_value = filters.leftDiagonalLine[filter_i - i][filter_j - j];
					filters.leftDiagonalSum += image_value * filter_value;

					// Filter 4: Right Diagonal Line
					filter_value = filters.rightDiagonalLine[filter_i - i][filter_j - j];
					filters.rightDiagonalSum += image_value * filter_value;

					// Filter 5: Cross
					filter_value = filters.cross[filter_i - i][filter_j - j];
					filters.crossSum += image_value * filter_value;

					// Filter 6: Plus
					filter_value = filters.plus[filter_i - i][filter_j - j];
					filters.plusSum += image_value * filter_value;

					// Filter 7: X
					filter_value = filters.x[filter_i - i][filter_j - j];
					filters.xSum += image_value * filter_value;

					// Filter 8: Square
					filter_value = filters.square[filter_i - i][filter_j - j];
					filters.squareSum += image_value * filter_value;

					// Filter 9: Diamond
					filter_value = filters.diamond[filter_i - i][filter_j - j];
					filters.diamondSum += image_value * filter_value;
				}
			}

			// Store the sum to the new image
			new_image_extract_vertical.at<uchar>(i, j) = filters.verticalSum;
			new_image_extract_horizontal.at<uchar>(i, j) = filters.horizontalSum;
			new_image_extract_leftDiagonal.at<uchar>(i, j) = filters.leftDiagonalSum;
			new_image_extract_rightDiagonal.at<uchar>(i, j) = filters.rightDiagonalSum;
			new_image_extract_cross.at<uchar>(i, j) = filters.crossSum;
			new_image_extract_plus.at<uchar>(i, j) = filters.plusSum;
			new_image_extract_x.at<uchar>(i, j) = filters.xSum;
			new_image_extract_square.at<uchar>(i, j) = filters.squareSum;
			new_image_extract_diamond.at<uchar>(i, j) = filters.diamondSum;
		}
	}

	// Return the new image
	vector<Mat> new_images{
		new_image_extract_vertical,
		new_image_extract_horizontal,
		new_image_extract_leftDiagonal,
		new_image_extract_rightDiagonal,
		new_image_extract_cross,
		new_image_extract_plus,
		new_image_extract_x,
		new_image_extract_square,
		new_image_extract_diamond
	};


	// Output for test
	/*
	cout << "Original Image Size: " << image.size() << endl;
	cout << "New Image (Vertical) Size: " << new_image_extract_vertical.size() << endl;
	cout << "New Image (Horizontal) Size: " << new_image_extract_horizontal.size() << endl;
	cout << "New Image (Left Diagonal) Size: " << new_image_extract_leftDiagonal.size() << endl;
	cout << "New Image (Right Diagonal) Size: " << new_image_extract_rightDiagonal.size() << endl;
	cout << "New Image (Cross) Size: " << new_image_extract_cross.size() << endl;
	cout << "New Image (Plus) Size: " << new_image_extract_plus.size() << endl;
	cout << "New Image (X) Size: " << new_image_extract_x.size() << endl;
	cout << "New Image (Square) Size: " << new_image_extract_square.size() << endl;
	cout << "New Image (Diamond) Size: " << new_image_extract_diamond.size() << endl;

	// Show images
	namedWindow("Original Image", WINDOW_NORMAL);
	resizeWindow("Original Image", 450, 450);
	namedWindow("Vertical Line", WINDOW_NORMAL);
	resizeWindow("Vertical Line", 450, 450);
	namedWindow("Horizontal Line", WINDOW_NORMAL);
	resizeWindow("Horizontal Line", 450, 450);
	namedWindow("Left Diagonal Line", WINDOW_NORMAL);
	resizeWindow("Left Diagonal Line", 450, 450);
	namedWindow("Right Diagonal Line", WINDOW_NORMAL);
	resizeWindow("Right Diagonal Line", 450, 450);
	namedWindow("Cross Line", WINDOW_NORMAL);
	resizeWindow("Cross Line", 450, 450);
	namedWindow("Plus Line", WINDOW_NORMAL);
	resizeWindow("Plus Line", 450, 450);
	namedWindow("X Line", WINDOW_NORMAL);
	resizeWindow("X Line", 450, 450);
	namedWindow("Square Line", WINDOW_NORMAL);
	resizeWindow("Square Line", 450, 450);
	namedWindow("Diamond Line", WINDOW_NORMAL);
	resizeWindow("Diamond Line", 450, 450);

	imshow("Original Image", image);
	imshow("Vertical Line", new_image_extract_vertical);
	imshow("Horizontal Line", new_image_extract_horizontal);
	imshow("Left Diagonal Line", new_image_extract_leftDiagonal);
	imshow("Right Diagonal Line", new_image_extract_rightDiagonal);
	imshow("Cross Line", new_image_extract_cross);
	imshow("Plus Line", new_image_extract_plus);
	imshow("X Line", new_image_extract_x);
	imshow("Square Line", new_image_extract_square);
	imshow("Diamond Line", new_image_extract_diamond);

	waitKey(0); // Wait for any keystroke in the window
	*/

	return new_images;
}


/*
* [DEPRECATED]
* Convolutional Layer 2D.
* Dynamic version, which takes dynamic created filters.
* But this one much is slower than the static version.
*
* @param image: the image to be processed
* @param filters: the filters to be used
*/
vector<Mat> conv2D(const string& image_path, const vector<vector<vector<int>>>& filters) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);

	if (!image.empty()) {
		// Original image size
		int image_width = image.cols;
		int image_height = image.rows;

		// New image size
		int new_image_width = image_width - FILTER_SIZE;
		int new_image_height = image_height - FILTER_SIZE;

		// Init the vector to store the new images
		vector<Mat> new_images;
		for (int i = 0; i < NUM_FILTERS; i++) {
			new_images.push_back(Mat::zeros(new_image_height, new_image_width, CV_8UC1));
		}

		// Loop for each pixel of new image
		for (int i = 0; i < new_image_height; i++) {
			for (int j = 0; j < new_image_width; j++) {
				// Init vector to store the value of this pixel of each filter
				vector<int> pixel_sum;
				for (int pixel = 0; pixel < NUM_FILTERS; pixel++) {
					pixel_sum.push_back(0);
				}

				for (int filter_i = i; filter_i < i + FILTER_SIZE; filter_i++) {
					for (int filter_j = j; filter_j < j + FILTER_SIZE; filter_j++) {
						// The value of the pixel of original image
						int image_value = image.at<uchar>(filter_i, filter_j);

						// Loop each filter
						for (int filter = 0; filter < filters.size(); filter++) {
							int filter_value = filters[filter][filter_i - i][filter_j - j];
							int filter_sum = image_value * filter_value;

							pixel_sum[filter] += filter_sum;
						}
					}
				}

				// Save the calculated new pixel to new images
				for (int image = 0; image < new_images.size(); image++) {
					new_images[image].at<uchar>(i, j) = pixel_sum[image];
				}
			}
		}

		return new_images;
	}

	vector<Mat> new_images;
	return new_images;
}



