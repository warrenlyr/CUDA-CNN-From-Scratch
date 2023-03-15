#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

using namespace cv;

//__global__ void testKernel(Mat& src, Mat& dst) {
//	int indexX = threadIdx.x;
//	int indexY = threadIdx.y;
//
//	src.at<uchar>(indexX, indexY) = 0;
//}