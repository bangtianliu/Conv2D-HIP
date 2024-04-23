#include "host_conv.hpp"
#include <cassert>
#include <iostream>

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
                int filter_width)
{
    assert(batch_size == 1);  // only support batch_size = 1

    // 7 for loops 
    
    // 1. batch_size
    // 2. out_channels
    // 3. out_height
    // 4. out_width
    // 5. in_channels
    // 6. filter_height
    // 7. filter_width
    for (int b = 0; b < batch_size; b++)
    {
        for (int oc = 0; oc < out_channels; oc++)
        {
            for (int oh = 0; oh < out_height; oh++)
            {
                for (int ow = 0; ow < out_width; ow++)
                {
                    float sum = 0;
                    int out_pos = b * out_channels * out_height * out_width + oc * out_height * out_width + oh * out_width + ow;
                    for (int ic = 0; ic < in_channels; ic++)
                    {
                        for (int fh = 0; fh < filter_height; fh++)
                        {
                            for (int fw = 0; fw < filter_width; fw++)
                            {
                                int ih = oh + fh;
                                int iw = ow + fw;
                                int in_pos = b * in_channels * in_height * in_width + ic * in_height * in_width + ih * in_width + iw;
                                int kernel_pos = oc * in_channels * filter_height * filter_width + ic * filter_height * filter_width + fh * filter_width + fw;

                                sum += in[in_pos] * filter[kernel_pos];
                            }
                        }
                    }

                    // if(out_pos ==16)
                    // {
                    //     int count = 0;
                    //     for (int ic = 0; ic < in_channels; ic++)
                    //     {
                    //         for (int fh = 0; fh < filter_height; fh++)
                    //         {
                    //             for (int fw = 0; fw < filter_width; fw++)
                    //             {
                    //                 int ih = oh + fh;
                    //                 int iw = ow + fw;
                    //                 int in_pos = b * in_channels * in_height * in_width + ic * in_height * in_width + ih * in_width + iw;
                    //                 int kernel_pos = oc * in_channels * filter_height * filter_width + ic * filter_height * filter_width + fh * filter_width + fw;
    
                    //                 // sum += in[in_pos] * filter[kernel_pos];
                    //                 std::cout << "count = " << count << " ";
                    //                 std::cout <<"in[" << in_pos << "] = " << in[in_pos] << " filter[" << kernel_pos << "] = " << filter[kernel_pos] << std::endl;
                    //                 count++;
                    //             }
                    //         }
                    //     }
                    // }
                    out[out_pos] = sum;
                }
            }
        }
    }
}

bool verify_data(float *out_ref, float *out, int size)
{
    float diff = 0.0; 

    for(int i=0; i < size; i++)
    {
        diff = std::fabs(out_ref[i] - out[i]);
        if(diff > 1e-2)
        {
            std::cout << "Error at position " << i << " expected " << out_ref[i] << " got " << out[i] << std::endl;
            return false;
        }
    }

    return true;
}
