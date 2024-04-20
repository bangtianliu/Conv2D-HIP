#ifndef CONV_CONFIG_HPP
#define CONV_CONFIG_HPP

#define CONV_N 1
#define CONV_IN_C 3 
#define CONV_IN_H 1920 
#define CONV_IN_W 2400

#define CONV_F_H 6 
#define CONV_F_W 6
#define CONV_OUT_C 3

#define CONV_OUT_H (CONV_IN_H - CONV_F_H + 1)
#define CONV_OUT_W (CONV_IN_W - CONV_F_W + 1)


//#define USE_FP16

#endif