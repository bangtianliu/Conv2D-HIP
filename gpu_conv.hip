#include "kernels.hpp"
#include "gpu_conv.hpp"
#include "conv_config.hpp"

void run_direct_conv(float * __restrict__ in, 
               float * __restrict__ out, 
               float * __restrict__ filter, 
               int batch_size,
               int in_channels,
                int in_height,
                int in_width,
                int out_channels,
                int out_height,
                int out_width,
                int filter_height,
                int filter_width)
{

    const int BLOCK_HEIGHT = 4;
    const int BLOCK_WIDTH = 16;
    
    const int FILTER_HEIGHT = CONV_F_H;
    const int FILTER_WIDTH = CONV_F_W;
    const int INC = CONV_IN_C; 
    const int OUTC = CONV_OUT_C;
    constexpr int MALLOC_FILTER_HEIGHT = FILTER_HEIGHT % 2 == 0 ? FILTER_HEIGHT : FILTER_HEIGHT + 1;
    constexpr int MALLOC_FILTER_WIDTH = FILTER_WIDTH % 2 == 0 ? FILTER_WIDTH : FILTER_WIDTH + 1;
    constexpr int MALLOC_BLOCK_HEIGHT = (BLOCK_HEIGHT + FILTER_HEIGHT) * 2; 
    constexpr int MALLOC_BLOCK_WIDTH = (BLOCK_WIDTH + FILTER_WIDTH) * 2;
    constexpr int MALLOC_OUT_C = OUTC *4; 


    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(out_width / BLOCK_WIDTH, out_height / BLOCK_HEIGHT);

    gpu_direct_conv2d<BLOCK_HEIGHT, BLOCK_WIDTH, FILTER_HEIGHT, FILTER_WIDTH,
                      MALLOC_FILTER_HEIGHT, MALLOC_FILTER_WIDTH,
                      MALLOC_BLOCK_HEIGHT, MALLOC_BLOCK_WIDTH, MALLOC_OUT_C>
        <<<grid, block>>>(in, out, filter, batch_size, in_channels, in_height,
                          in_width, out_channels, out_height, out_width,
                          filter_height, filter_width);
}

void run_kernel(int kernel_num, 
               float * __restrict__ in, 
               float * __restrict__ out, 
               float * __restrict__ filter, 
               int batch_size,
               int in_channels,
                int in_height,
                int in_width,
                int out_channels,
                int out_height,
                int out_width,
                int filter_height,
                int filter_width)
{
    switch(kernel_num) {
        case 1: run_direct_conv(in, out, filter, batch_size, in_channels, in_height, in_width, out_channels, out_height, out_width, filter_height, filter_width);
                break;
        default: 
                throw std::invalid_argument("Invalid kernel number");
    }
}

