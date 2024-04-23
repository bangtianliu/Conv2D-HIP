#ifndef CONV_CONFIG_HPP
#define CONV_CONFIG_HPP

#define CONV_N 2
#define CONV_IN_C 1280 
#define CONV_IN_H 32
#define CONV_IN_W 32

#define CONV_F_H 3
#define CONV_F_W 3
#define CONV_OUT_C 1280

#define CONV_OUT_H (CONV_IN_H - CONV_F_H + 1)
#define CONV_OUT_W (CONV_IN_W - CONV_F_W + 1)


//#define USE_FP16

#endif