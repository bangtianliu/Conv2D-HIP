# Simple implemenetation of 2D convolution using HIP on AMD GPU
This code repository pertains to Bangtian's introductory task, focusing on the implementation of 2D convolution using HIP on AMD GPU.

## Setup
1. Build: `make`
2. Run: `./conv2d_gpu`

parameters for convolution can be changed in `conv_config.hpp`.

## Bangtian's Convolution Implementations
<!-- Description about code -->
| ID  |            Kernel Method             |                                           Comment                                            | 
|:---:|:------------------------------------:|:--------------------------------------------------------------------------------------------:|  
| 1  | direct 2D convolution  |                        shared memory is used, FP32   (kernels/1_*.hpp)                         |
<!--  Description about code -->

(Additional implementations will be included, with the im2col implementation currently in progress.)