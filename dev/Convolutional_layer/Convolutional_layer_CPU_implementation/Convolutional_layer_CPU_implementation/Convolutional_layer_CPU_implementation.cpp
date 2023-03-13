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

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace chrono;
using namespace filesystem;


#define BASE_PATH ".\\data\\"
#define CATS_PATH ".\\data\\cats\\"
#define DOGS_PATH ".\\data\\dogs\\"

#define NUM_FILTERS 5


void getCurrDir();
vector<string> getFiles(const string& path);
vector<vector<vector<int>>> createFilters();

void conv2D(const string&,
	const vector<vector<int>>,
	const vector<vector<int>>,
	const vector<vector<int>>,
	const vector<vector<int>>,
	const vector<vector<int>>,
	const vector<vector<int>>);


int main()
{
	// Get all images
	vector<string> cat_images = getFiles(CATS_PATH);
	vector<string> dog_images = getFiles(DOGS_PATH);

	// Create filters
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

	// Convolution
	auto start = high_resolution_clock::now();
	for (int i = 0; i < cat_images.size(); i++) {
		conv2D(cat_images[i],
			filter_vertical_line,
			filter_horiz_line,
			filter_diagonal_lbru_line,
			filter_diagonal_lurb_line,
			filter_diagonal_x_line,
			filter_round_line);
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


vector<string> getFiles(const string& path) {
	vector<string> files;
	for (const auto& entry : filesystem::directory_iterator(path)) {
		files.push_back(entry.path().string());
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


void conv2D(
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

		int new_image_width = image_width - 3;
		int new_image_height = image_height - 3;

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

				for (int filter_i = i; filter_i < i + 3; filter_i++) {
					for (int filter_j = j; filter_j < j + 3; filter_j++) {
						int image_value = image.at<uchar>(filter_i, filter_j);

						int filter_value = filter_vertical_line[filter_i - i][filter_j - j];
						//int filter_value = *(filter_vertical_line + filter_i * 3 + filter_j);
						vertical_sum += image_value * filter_value;

						filter_value = filter_horizontal_line[filter_i - i][filter_j - j];
						//filter_value = *(filter_horizontal_line + filter_i * 3 + filter_j);
						horiz_sum += image_value * filter_value;

						filter_value = filter_diagonal_lbru_line[filter_i - i][filter_j - j];
						//filter_value = *(filter_diagonal_line + filter_i * 3 + filter_j);
						diagonal_lbru_sum += image_value * filter_value;

						filter_value = filter_diagonal_lurb_line[filter_i - i][filter_j - j];
						diagonal_lurb_sum += image_value * filter_value;

						filter_value = filter_diagonal_x_line[filter_i - i][filter_j - j];
						diagonal_x_sum += image_value * filter_value;

						filter_value = filter_round_line[filter_i - i][filter_j - j];
						//filter_value = *(filter_round_line + filter_i * 3 + filter_j);
						round_sum += image_value * filter_value;
					}
				}
				//new_image.at<uchar>(i, j) = 200;
				new_image_extract_vertical.at<uchar>(i, j) = vertical_sum;
				new_image_extract_horiz.at<uchar>(i, j) = horiz_sum;
				new_image_extract_diagonal_lbru.at<uchar>(i, j) = diagonal_lbru_sum;
				new_image_extract_diagonal_lurb.at<uchar>(i, j) = diagonal_lurb_sum;
				new_image_extract_diagonal_x.at<uchar>(i, j) = diagonal_x_sum;
				new_image_extract_round.at<uchar>(i, j) = round_sum;
			}
		}

		// Output for test

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

	}
}