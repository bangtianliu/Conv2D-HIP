#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <numeric>

//AMD GPU Runtime
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>


#include "host_conv.hpp"
#include "gpu_conv.hpp"
#include "conv_config.hpp"

#define DEVICE_CHECK 

#define WARMUP_RUN 0
#define TIMING_RUN 10

// #define  USE_FP16

constexpr int error_exit_code = -1;
/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(error_exit_code);                                                     \
        }                                                                                   \
    }



// convolution implementation on AMD GPU by Bangtian Liu (bangtian.liu@amd.com)
int main()
{
    const int N = CONV_N; 
    const int inC = CONV_IN_C; 
    const int inH = CONV_IN_H; 
    const int inW = CONV_IN_W;

    const int outC = CONV_OUT_C;
    const int filterH = CONV_F_H;
    const int filterW = CONV_F_W;

    //assume stride is 1
    const int outH = CONV_OUT_H;
    const int outW = CONV_OUT_W;

    //print configuration
    printf("N = %d, inC = %d, inH = %d, inW = %d\n", N, inC, inH, inW);
    printf("filterH = %d, filterW = %d\n", filterH, filterW);
    printf("outC = %d, outH = %d, outW = %d\n", outC, outH, outW);

    float *h_input,*h_filter,*h_output;

    //NCHW format is assumed for this implementation
    int i_size = N * inC * inH * inW;
    int f_size = outC * inC * filterH * filterW;
    int o_size = N * outC * outH * outW;

    h_input = (float *)malloc(i_size * sizeof(float));
    h_filter = (float *)malloc(f_size * sizeof(float));
    h_output = (float *)malloc(o_size * sizeof(float));

    printf("BEFORE i_size =%d\n", i_size);

    for(int i = 0; i < i_size; i++)
    {
        h_input[i] = rand() % 10;
    }


    for(int i = 0; i < f_size; i+=3)
    {
        h_filter[i] = -1;
        h_filter[i+1] = 0;
        h_filter[i+2] = 1;
    }

    memset(h_output, 0, o_size * sizeof(float));

    host_conv(h_input, h_output, h_filter, N, inC, inH, inW, outC, outH, outW, filterH, filterW);

#ifdef DEVICE_CHECK

    int deviceCount;
    (void)hipGetDeviceCount(&deviceCount);
    std::cout << "deviceCount = " << deviceCount << std::endl;
    (void)hipSetDevice(0);
    hipDeviceProp_t props;
    (void)hipGetDeviceProperties(&props, 0);
    std::cout << "Device name: " << props.name << std::endl;
    std::cout << "GCN Arch name: " << props.gcnArchName << std::endl;
    std::cout << "Warpsize = " << props.warpSize << std::endl;
    std::cout << "isMultiGpuBoard = " << props.isMultiGpuBoard << std::endl;
    std::cout << "clockRate = " << props.clockRate << std::endl;
    std::cout << "maxThreadsPerBlock = " << props.maxThreadsPerBlock
              << std::endl;
    std::cout << "multiProcessorCount = " << props.multiProcessorCount
              << std::endl;
    std::cout << "maxThreadsPerMultiProcessor = "
              << props.maxThreadsPerMultiProcessor << std::endl;
#endif

#ifdef USE_FP16
    //TODO: will add support for FP16
#else
    (void) hipSetDevice(0);
    float *d_input;
    float *d_filter; 
    float *d_output;
    float *h_output_gpu = (float *)malloc(o_size * sizeof(float));

    HIP_CHECK(hipMalloc(&d_input, i_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_filter, f_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, o_size * sizeof(float)));
    HIP_CHECK(hipGetLastError());

    printf("i_size =%d\n", i_size);
    int i_size_bytes = i_size * sizeof(float);

    // hipMemcpy(d_input, h_input, i_size * sizeof(float), hipMemcpyHostToDevice);
    HIP_CHECK(hipMemcpy(d_input, h_input, i_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_filter, h_filter, f_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_output, 0, o_size * sizeof(float)));
    // HIP_CHECK(hipMemcpy(d_output, h_output, o_size * sizeof(float), hipMemcpyHostToDevice));


    // for(int row =0; row < 9; row++)
    // {
    //     for(int col = 0; col < 21; col++)
    //     {
    //         printf("%f ", h_input[row*inW + col]);
    //     }
    //     printf("\n");
    // }

    // getchar();

    int kernel_num = 1;

    for (int i = 0; i < WARMUP_RUN; i++) {
        run_kernel(kernel_num, d_input, d_output, d_filter, N, inC, inH, inW,
                   outC, outH, outW, filterH, filterW);
    }

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    std::vector<float> elapsed_times(TIMING_RUN);
    for(int i = 0; i < TIMING_RUN; i++)
    {
        HIP_CHECK(hipMemset(d_output, 0, o_size * sizeof(float)));
        HIP_CHECK(hipEventRecord(start, NULL));
        run_kernel(kernel_num, d_input, d_output, d_filter, N, inC, inH, inW,
                   outC, outH, outW, filterH, filterW);
        HIP_CHECK(hipEventRecord(stop, NULL));
        HIP_CHECK(hipEventSynchronize(stop));
        float time;
        HIP_CHECK(hipEventElapsedTime(&time, start, stop));
        //convert ms to s
        time /= 1000;
        elapsed_times[i] = time;
    }

    float mean_time = std::accumulate(elapsed_times.begin(), elapsed_times.end(), 0.0) / TIMING_RUN;

    float min_time = *std::min_element(elapsed_times.begin(), elapsed_times.end());

    float max_time = *std::max_element(elapsed_times.begin(), elapsed_times.end());


    printf("Mean time: %f s\n", mean_time);
    printf("Min time: %f s\n", min_time);
    printf("Max time: %f s\n", max_time);

    HIP_CHECK(hipMemcpy(h_output_gpu, d_output, o_size * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipGetLastError());


    // for(int i=0; i < 21*9; i++)
    // {
    //     printf("h_output[%d] = %f, h_output_gpu[%d] = %f\n", i, h_output[i], i, h_output_gpu[i]);
    // }

    if(!verify_data(h_output, h_output_gpu, o_size))
    {
        std::cout << "Verification failed" << std::endl;
    }
    else
    {
        std::cout << "Verification passed" << std::endl;
    }

    free(h_output_gpu);
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_filter));
    HIP_CHECK(hipFree(d_output));

#endif


    free(h_input);
    free(h_filter);
    free(h_output);
}
