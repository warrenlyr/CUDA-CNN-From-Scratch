# CUDA_CNN_From_Scratch
Build Convolutional Neural Network from scratch and accelerate with CUDA. 
Although there are many sophisticated frameworks out there such as Tensorflow, Sklearn, cuDNN, OpenCV, and Caffe,
this is a part of our final project for High Performance Computing class, for applying what we have learned during the semester.

The "Convolutional Layer" and the "Pooling Layer" are the most computationally intensive part of the CNN algorithm,
and it involves matrix operations that we can accelerate using CUDA. Therefore, it's a perfect choice for our final project.

Keywords: Matrix Multiplication, CUDA, Threads, Blocks, Grids, Memory, Cache, Latency, Performance.

## Implementation
- Convolutional Layer
  - CPU Naive implementation
  - CUDA Naive implementation
  - CUDA optimized implementation
- Pooling Layer
  - CPU Naive implementation
  - CUDA Naive implementation
  - CUDA optimized implementation

## Prerequisites

- `OpenCV v4.7.0+`

  - Add the `OpenCV` lib to `Visual Studio 2022` project properties following this 

    [tutorial]: https://www.geeksforgeeks.org/opencv-c-windows-setup-using-visual-studio-2019/	"OpenCV C++ Windows Setup using Visual Studio 2019"

- `Visual Studio 2022+`

  - C++ Language Standard: ISO C++ 17 Standard (17 and above)

- `CUDA Toolkit 12.0+`

  - Link to `Visual Studio 2022`
