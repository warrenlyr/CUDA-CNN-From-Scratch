// CPP CNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <Windows.h>
#include <direct.h>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <chrono>
#include<cstdio>

#include <omp.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace chrono;


#define BASE_PATH ".\\data\\"
#define CATS_PATH ".\\data\\cats\\"
#define CATS_PATH_OUTPUT ".\\data\\cats_output\\"
#define DOGS_PATH ".\\data\\dogs\\"
#define DOGS_PATH_OUTPUT ".\\data\\dogs_output\\"
#define NUM_FILTERS 6
#define FILTER_SIZE 3


void getCurrDir();
vector<filesystem::path> getFiles(const string& path);
vector<vector<vector<int>>> createFilters();

vector<Mat> conv2D_static(const string&, // This function takes static filters as parameters
	const vector<vector<int>>,
	const vector<vector<int>>,
	const vector<vector<int>>,
	const vector<vector<int>>,
	const vector<vector<int>>,
	const vector<vector<int>>);

vector<Mat> conv2D( // The function takes a vector of filters as a parameter
	const string&,
	const vector<vector<vector<int>>>&
);


int main()
{
	// Get all images
	vector<filesystem::path> cat_images = getFiles(CATS_PATH);
	vector<filesystem::path> dog_images = getFiles(DOGS_PATH);

	// Create filters (test use, for conv2D_static)
	
	vector<vector<int>> filter_vertical_line{
		{0, 1, 0},
		{0, 1, 0},
		{0, 1, 0},
	};

	vector<vector<int>> filter_horiz_line{
		{0, 0, 0},
		{1, 1, 1},
		{0, 0, 0},
	};

	vector<vector<int>> filter_diagonal_lbru_line{
		{ 0, 0, 1 },
		{ 0, 1, 0 },
		{ 1, 0, 0 },
	};

	vector<vector<int>> filter_diagonal_lurb_line{
		{ 1, 0, 0 },
		{ 0, 1, 0 },
		{ 0, 0, 1 },
	};

	vector<vector<int>> filter_diagonal_x_line{
		{ 1, 0, 1 },
		{ 0, 1, 0 },
		{ 1, 0, 1 },
	};

	vector<vector<int>> filter_round_line{
		{0, 1, 0},
		{1, 0, 1},
		{0, 1, 0},
	};
	

	// Create filters (actual use, for conv2D)
	vector<vector<vector<int>>> filters = createFilters();

	// Convolution
	auto start = high_resolution_clock::now();
	//#pragma omp parallel for
	for (int i = 0; i < cat_images.size(); i++) {
		
		vector<Mat> new_images = conv2D_static(cat_images[i].string(),
												filter_vertical_line,
												filter_horiz_line,
												filter_diagonal_lbru_line,
												filter_diagonal_lurb_line,
												filter_diagonal_x_line,
												filter_round_line
												);
		
		// Returned convoloed images
		//vector<Mat> new_images = conv2D(cat_images[i].string(), filters);

		// Test: write convolved images to output folder
		int index = 0;
		for (auto image : new_images) {
			bool success = imwrite(string(CATS_PATH_OUTPUT) + "filter_" + to_string(index++) + "_" + cat_images[i].filename().string(), image);
			cout << "Success: " << success << endl;
		}
		break;
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout << "Time taken by function: " << duration.count() << " microseconds" << endl;


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


vector<filesystem::path> getFiles(const string& path) {
	vector<filesystem::path> files;
	for (const auto& entry : filesystem::directory_iterator(path)) {
		files.push_back(entry.path());
	}

	return files;
}


vector<vector<vector<int>>> createFilters() {
	vector<vector<int>> filter_vertical_line{
		{0, 1, 0},
		{0, 1, 0},
		{0, 1, 0},
	};

	vector<vector<int>> filter_horiz_line{
		{0, 0, 0},
		{1, 1, 1},
		{0, 0, 0},
	};

	vector<vector<int>> filter_diagonal_lbru_line{
		{ 0, 0, 1 },
		{ 0, 1, 0 },
		{ 1, 0, 0 },
	};

	vector<vector<int>> filter_diagonal_lurb_line{
		{ 1, 0, 0 },
		{ 0, 1, 0 },
		{ 0, 0, 1 },
	};

	vector<vector<int>> filter_diagonal_x_line{
		{ 1, 0, 1 },
		{ 0, 1, 0 },
		{ 1, 0, 1 },
	};

	vector<vector<int>> filter_round_line{
		{0, 1, 0},
		{1, 0, 1},
		{0, 1, 0},
	};

	vector<vector<vector<int>>> filters{
		filter_vertical_line,
		filter_horiz_line,
		filter_diagonal_lbru_line,
		filter_diagonal_lurb_line,
		filter_diagonal_x_line,
		filter_round_line
	};

	return filters;
}


vector<Mat> conv2D_static(
	const string& image_path,
	const vector<vector<int>> filter_vertical_line,
	const vector<vector<int>> filter_horizontal_line,
	const vector<vector<int>> filter_diagonal_lbru_line,
	const vector<vector<int>> filter_diagonal_lurb_line,
	const vector<vector<int>> filter_diagonal_x_line,
	const vector<vector<int>> filter_round_line
) {

	Mat image = imread(image_path, IMREAD_GRAYSCALE);

	if (!image.empty()) {
		int image_width = image.cols;
		int image_height = image.rows;

		/*cout << "Image width: " << image_width << endl;
		cout << "Image height: " << image_height << endl;*/

		int new_image_width = image_width - FILTER_SIZE;
		int new_image_height = image_height - FILTER_SIZE;

		Mat new_image_extract_vertical = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
		Mat new_image_extract_horiz = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
		Mat new_image_extract_diagonal_lbru = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
		Mat new_image_extract_diagonal_lurb = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
		Mat new_image_extract_diagonal_x = Mat::zeros(new_image_height, new_image_width, CV_8UC1);
		Mat new_image_extract_round = Mat::zeros(new_image_height, new_image_width, CV_8UC1);

		for (int i = 0; i < new_image_height; i++) {
			for (int j = 0; j < new_image_width; j++) {
				int vertical_sum = 0;
				int horiz_sum = 0;
				int diagonal_lbru_sum = 0;
				int diagonal_lurb_sum = 0;
				int diagonal_x_sum = 0;
				int round_sum = 0;

				for (int filter_i = i; filter_i < i + FILTER_SIZE; filter_i++) {
					for (int filter_j = j; filter_j < j + FILTER_SIZE; filter_j++) {
						int image_value = image.at<uchar>(filter_i, filter_j);

						int filter_value = filter_vertical_line[filter_i - i][filter_j - j];
						vertical_sum += image_value * filter_value;

						filter_value = filter_horizontal_line[filter_i - i][filter_j - j];
						horiz_sum += image_value * filter_value;

						filter_value = filter_diagonal_lbru_line[filter_i - i][filter_j - j];
						diagonal_lbru_sum += image_value * filter_value;

						filter_value = filter_diagonal_lurb_line[filter_i - i][filter_j - j];
						diagonal_lurb_sum += image_value * filter_value;

						filter_value = filter_diagonal_x_line[filter_i - i][filter_j - j];
						diagonal_x_sum += image_value * filter_value;

						filter_value = filter_round_line[filter_i - i][filter_j - j];
						round_sum += image_value * filter_value;
					}
				}
				
				new_image_extract_vertical.at<uchar>(i, j) = vertical_sum;
				new_image_extract_horiz.at<uchar>(i, j) = horiz_sum;
				new_image_extract_diagonal_lbru.at<uchar>(i, j) = diagonal_lbru_sum;
				new_image_extract_diagonal_lurb.at<uchar>(i, j) = diagonal_lurb_sum;
				new_image_extract_diagonal_x.at<uchar>(i, j) = diagonal_x_sum;
				new_image_extract_round.at<uchar>(i, j) = round_sum;
			}
		}

		vector<Mat> new_images{
			new_image_extract_vertical,
			new_image_extract_horiz,
			new_image_extract_diagonal_lbru,
			new_image_extract_diagonal_lurb,
			new_image_extract_diagonal_x,
			new_image_extract_round
		};
		return new_images;

		// Output for test
		/*
		cout << "Original Image Size: " << image.size() << endl;
		cout << "New Image (Vertical) Size: " << new_image_extract_vertical.size() << endl;
		cout << "New Image (Horizontal) Size: " << new_image_extract_horiz.size() << endl;
		cout << "New Image (Diagonal lbru) Size: " << new_image_extract_diagonal_lbru.size() << endl;
		cout << "New Image (Diagonal lurb) Size: " << new_image_extract_diagonal_lurb.size() << endl;
		cout << "New Image (Diagonal x) Size: " << new_image_extract_diagonal_x.size() << endl;
		cout << "New Image (Round) Size: " << new_image_extract_round.size() << endl;

		// Show images
		namedWindow("Original Image", WINDOW_NORMAL);
		resizeWindow("Original Image", 450, 450);
		namedWindow("Vertical Line", WINDOW_NORMAL);
		resizeWindow("Vertical Line", 450, 450);
		namedWindow("Horizontal Line", WINDOW_NORMAL);
		resizeWindow("Horizontal Line", 450, 450);
		namedWindow("Diagonal lbru Line", WINDOW_NORMAL);
		resizeWindow("Diagonal lbru Line", 450, 450);
		namedWindow("Diagonal lurb Line", WINDOW_NORMAL);
		resizeWindow("Diagonal lurb Line", 450, 450);
		namedWindow("Diagonal x Line", WINDOW_NORMAL);
		resizeWindow("Diagonal x Line", 450, 450);
		namedWindow("Round Line", WINDOW_NORMAL);
		resizeWindow("Round Line", 450, 450);

		imshow("Original Image", image);
		imshow("Vertical Line", new_image_extract_vertical);
		imshow("Horizontal Line", new_image_extract_horiz);
		imshow("Diagonal lbru Line", new_image_extract_diagonal_lbru);
		imshow("Diagonal lurb Line", new_image_extract_diagonal_lurb);
		imshow("Diagonal x Line", new_image_extract_diagonal_x);
		imshow("Round Line", new_image_extract_round);

		waitKey(0); // Wait for any keystroke in the window
		*/
	}

	vector<Mat> new_images;
	return new_images;
}


vector<Mat> conv2D(const string& image_path, const vector<vector<vector<int>>> &filters) {
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