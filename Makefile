SHELL = /bin/bash

# HIP variables
ROCM_INSTALL_DIR := /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Common variables and flags
CXX_STD   := c++17
ICXXFLAGS := -std=$(CXX_STD)

all: conv2d_gpu 

conv2d_gpu: main.o host_conv.o gpu_conv.o
	$(HIPCXX) -o $@ $^ -v

main.o: main.hip kernels.hpp 
	$(HIPCXX) -c -o $@ $< $(ICXXFLAGS) -I$(HIP_INCLUDE_DIR)

host_conv.o: host_conv.hip
	$(HIPCXX) -c -o $@ $< $(ICXXFLAGS) -I$(HIP_INCLUDE_DIR)

gpu_conv.o: gpu_conv.hip
	$(HIPCXX) -c -o $@ $< $(ICXXFLAGS) -I$(HIP_INCLUDE_DIR)

clean:
	rm -f conv2d_gpu *.o