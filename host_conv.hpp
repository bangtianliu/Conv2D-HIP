#ifndef HOST_CONV_HPP
#define HOST_CONV_HPP

void host_conv(const float * __restrict__ in, 
               float *out, 
               const float * __restrict__ filter, 
               int batch_size,
               int in_channels,
                int in_height,
                int in_width,
                int out_channels,
                int out_height,
                int out_width,
                int filter_height,
                int filter_width);

bool verify_data(float *out_ref, float *out, int size);
                
#endif  // HOST_CONV_HPP
