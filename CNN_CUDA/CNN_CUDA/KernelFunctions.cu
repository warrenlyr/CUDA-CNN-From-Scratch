#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

using namespace cv;

//__global__ void conv2D_cuda(
//	cudaPitchedPtr devPtr, cudaPitchedPtr devPtr_output, 
//	int count, int row, int col, int row_output, int col_output) 
//{
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//	int z = blockIdx.z * blockDim.z + threadIdx.z;
//
//	printf("%d", ((int*)devPtr.ptr)[0]);
//}
