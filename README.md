# CUDA_CNN_From_Scratch
This is our final project of CSS566: High Performance Computing class. In this project, we undertake the ambitious task of constructing a Convolutional Neural Network (CNN) from the ground up and optimizing its performance with CUDA. Though there exists a plethora of highly advanced frameworks like Tensorflow, Sklearn, cuDNN, OpenCV, and Caffe, the goal here is to apply and deepen our understanding of High Performance Computing concepts learned throughout our coursework.

The heart of any CNN algorithm lies in its Convolutional Layer and Pooling Layer, both of which are computationally demanding. These layers primarily engage in matrix operations, which we aim to significantly expedite using CUDA. This makes it an ideal candidate for our final project as we strive to push the boundaries of computing performance.

Key Concepts: Matrix Multiplication, CUDA, Threads, Blocks, Grids, Memory Management, Cache Optimization, Latency Reduction, and Performance Enhancement.

## Author

- Warren Liu
- Chris Ma

## Implementation

Given the time limitations, our focus was primarily on implementing both the Convolutional Layer and Pooling Layer, with versions optimized for CPU and GPU respectively. 

- Convolutional Layer
  - CPU Naive implementation
  - CUDA Naive implementation
  - CUDA optimized implementation
- Pooling Layer
  - CPU Naive implementation
  - CUDA Naive implementation
  - CUDA optimized implementation

## Code

- `CNN_CUDA` folder contains the final version
- `dev` folder contains some testing code during development

## Report

Our comprehensive project report is located in the `Report` folder. This document provides a detailed account of our development journey, the optimization strategies we employed, our key discoveries, and additional reflections.

## Prerequisites

- `Visual Studio 2022+`

  - C++ Language Standard: ISO C++ 17 Standard (17 and above)


- `OpenCV v4.7.0+`

  - Set up `OpenCV` and add to `Visual Studio 2022` project properties following this [link](https://www.geeksforgeeks.org/opencv-c-windows-setup-using-visual-studio-2019/)


- `CUDA Toolkit 12.0+`

  - Link to `Visual Studio 2022`
