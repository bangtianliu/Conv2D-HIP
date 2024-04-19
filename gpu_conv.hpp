#ifndef GPU_CONV_HPP
#define GPU_CONV_HPP

using data_t = float;


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
                int filter_width);

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
                int filter_width);
                
#endif
