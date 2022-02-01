/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
#include "xa_nnlib_common.h"
/* Uncomment the line below to enable row_unroll 16*/
//#define UNROLL_16

#define ROW_UNROLL_SINGLE 4

#define SETUP_MAT1_SINGLE_ROW_PTR \
        ae_int8x16 * _ae_int8x16_p_mat1_0  = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];

#define SETUP_MAT2_SINGLE_ROW_PTR \
        ae_int8x16 * _ae_int8x16_p_mat2_0  = (ae_int8x16 *) &p_mat2[(m_itr+0)*row_stride2];

#define kernel_mat1_vec1_8x16_single_row \
          AE_L8X8X2_IP(_ae_int8x8_mat1_0, _ae_int8x8_mat1_1_0,_ae_int8x16_p_mat1_0, 16); \
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat1_0, zero_temp, zero_temp,zero_temp,\
                         _ae_int16x4_vec1,_ae_int16x4_vec1_1);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat1_1_0, zero_temp, zero_temp,zero_temp,\
                         _ae_int16x4_vec1_2,_ae_int16x4_vec1_3);

#define kernel_mat2_vec2_8x16_single_row \
          AE_L8X8X2_IP(_ae_int8x8_mat2_0, _ae_int8x8_mat2_1_0,_ae_int8x16_p_mat2_0, 16); \
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat2_0, zero_temp, zero_temp,zero_temp,\
                         _ae_int16x4_vec2,_ae_int16x4_vec2_1);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat2_1_0, zero_temp, zero_temp,zero_temp,\
                         _ae_int16x4_vec2_2,_ae_int16x4_vec2_3);

#ifdef UNROLL_16

#ifdef ROW_UNROLL
    #undef ROW_UNROLL
    #define ROW_UNROLL 16
#else
    #define ROW_UNROLL 16
#endif
#include "xa_nnlib_common_macros_hifi5.h"

#define SETUP_MAT1_8x8_PTR_8x16 \
        ae_int8x16 * _ae_int8x16_p_mat1_0  = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_4  = (ae_int8x16 *) &p_mat1[(m_itr+4)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_8  = (ae_int8x16 *) &p_mat1[(m_itr+8)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_12 = (ae_int8x16 *) &p_mat1[(m_itr+12)*row_stride1];

#define SETUP_MAT2_8x8_PTR_8x16 \
        ae_int8x16 * _ae_int8x16_p_mat2_0  = (ae_int8x16 *) &p_mat2[(m_itr+0)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_4  = (ae_int8x16 *) &p_mat2[(m_itr+4)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_8  = (ae_int8x16 *) &p_mat2[(m_itr+8)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_12 = (ae_int8x16 *) &p_mat2[(m_itr+12)*row_stride2];

#define kernel_mat1_vec1_8x16 \
          AE_L8X8X2_X( _ae_int8x8_mat1_1, _ae_int8x8_mat1_1_1,_ae_int8x16_p_mat1_0, 1*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2, _ae_int8x8_mat1_1_2,_ae_int8x16_p_mat1_0, 2*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_3, _ae_int8x8_mat1_1_3,_ae_int8x16_p_mat1_0, 3*row_stride1);\
          AE_L8X8X2_IP(_ae_int8x8_mat1_0, _ae_int8x8_mat1_1_0,_ae_int8x16_p_mat1_0, 16);\
          AE_L8X8X2_X( _ae_int8x8_mat1_5, _ae_int8x8_mat1_1_5,_ae_int8x16_p_mat1_4, 1*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_6, _ae_int8x8_mat1_1_6,_ae_int8x16_p_mat1_4, 2*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_7, _ae_int8x8_mat1_1_7,_ae_int8x16_p_mat1_4, 3*row_stride1);\
          AE_L8X8X2_IP(_ae_int8x8_mat1_4, _ae_int8x8_mat1_1_4,_ae_int8x16_p_mat1_4, 16);\
          AE_L8X8X2_X( _ae_int8x8_mat1_9, _ae_int8x8_mat1_1_9,_ae_int8x16_p_mat1_8, 1*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_10, _ae_int8x8_mat1_1_10,_ae_int8x16_p_mat1_8, 2*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_11, _ae_int8x8_mat1_1_11,_ae_int8x16_p_mat1_8, 3*row_stride1);\
          AE_L8X8X2_IP(_ae_int8x8_mat1_8, _ae_int8x8_mat1_1_8,_ae_int8x16_p_mat1_8, 16);\
          AE_L8X8X2_X( _ae_int8x8_mat1_13, _ae_int8x8_mat1_1_13,_ae_int8x16_p_mat1_12, 1*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_14, _ae_int8x8_mat1_1_14,_ae_int8x16_p_mat1_12, 2*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_15, _ae_int8x8_mat1_1_15,_ae_int8x16_p_mat1_12, 3*row_stride1);\
          AE_L8X8X2_IP(_ae_int8x8_mat1_12, _ae_int8x8_mat1_1_12,_ae_int8x16_p_mat1_12, 16);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat1_0, _ae_int8x8_mat1_1, _ae_int8x8_mat1_2,_ae_int8x8_mat1_3,\
                         _ae_int16x4_vec1,_ae_int16x4_vec1_1);\
          AE_MULA8QW8X16(_ae_int64_acc_4  , _ae_int64_acc_5  , _ae_int64_acc_6  ,_ae_int64_acc_7,\
                         _ae_int8x8_mat1_4, _ae_int8x8_mat1_5, _ae_int8x8_mat1_6,_ae_int8x8_mat1_7,\
                         _ae_int16x4_vec1,_ae_int16x4_vec1_1);\
          AE_MULA8QW8X16(_ae_int64_acc_8  , _ae_int64_acc_9  , _ae_int64_acc_10  ,_ae_int64_acc_11,\
                         _ae_int8x8_mat1_8, _ae_int8x8_mat1_9, _ae_int8x8_mat1_10,_ae_int8x8_mat1_11,\
                         _ae_int16x4_vec1,_ae_int16x4_vec1_1);\
          AE_MULA8QW8X16(_ae_int64_acc_12  , _ae_int64_acc_13  , _ae_int64_acc_14  ,_ae_int64_acc_15,\
                         _ae_int8x8_mat1_12, _ae_int8x8_mat1_13, _ae_int8x8_mat1_14,_ae_int8x8_mat1_15,\
                         _ae_int16x4_vec1,_ae_int16x4_vec1_1);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat1_1_0, _ae_int8x8_mat1_1_1, _ae_int8x8_mat1_1_2,_ae_int8x8_mat1_1_3,\
                         _ae_int16x4_vec1_2,_ae_int16x4_vec1_3);\
          AE_MULA8QW8X16(_ae_int64_acc_4  , _ae_int64_acc_5  , _ae_int64_acc_6  ,_ae_int64_acc_7,\
                         _ae_int8x8_mat1_1_4, _ae_int8x8_mat1_1_5, _ae_int8x8_mat1_1_6,_ae_int8x8_mat1_1_7,\
                         _ae_int16x4_vec1_2,_ae_int16x4_vec1_3);\
          AE_MULA8QW8X16(_ae_int64_acc_8  , _ae_int64_acc_9  , _ae_int64_acc_10  ,_ae_int64_acc_11,\
                         _ae_int8x8_mat1_1_8, _ae_int8x8_mat1_1_9, _ae_int8x8_mat1_1_10,_ae_int8x8_mat1_1_11,\
                         _ae_int16x4_vec1_2,_ae_int16x4_vec1_3);\
          AE_MULA8QW8X16(_ae_int64_acc_12  , _ae_int64_acc_13  , _ae_int64_acc_14  ,_ae_int64_acc_15,\
                         _ae_int8x8_mat1_1_12, _ae_int8x8_mat1_1_13, _ae_int8x8_mat1_1_14,_ae_int8x8_mat1_1_15,\
                         _ae_int16x4_vec1_2,_ae_int16x4_vec1_3);

#define kernel_mat2_vec2_8x16 \
          AE_L8X8X2_X( _ae_int8x8_mat2_1, _ae_int8x8_mat2_1_1,_ae_int8x16_p_mat2_0, 1*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2, _ae_int8x8_mat2_1_2,_ae_int8x16_p_mat2_0, 2*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_3, _ae_int8x8_mat2_1_3,_ae_int8x16_p_mat2_0, 3*row_stride2);\
          AE_L8X8X2_IP(_ae_int8x8_mat2_0, _ae_int8x8_mat2_1_0,_ae_int8x16_p_mat2_0, 16); \
          AE_L8X8X2_X( _ae_int8x8_mat2_5, _ae_int8x8_mat2_1_5,_ae_int8x16_p_mat2_4, 1*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_6, _ae_int8x8_mat2_1_6,_ae_int8x16_p_mat2_4, 2*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_7, _ae_int8x8_mat2_1_7,_ae_int8x16_p_mat2_4, 3*row_stride2);\
          AE_L8X8X2_IP(_ae_int8x8_mat2_4, _ae_int8x8_mat2_1_4,_ae_int8x16_p_mat2_4, 16);\
          AE_L8X8X2_X( _ae_int8x8_mat2_9, _ae_int8x8_mat2_1_9,_ae_int8x16_p_mat2_8, 1*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_10, _ae_int8x8_mat2_1_10,_ae_int8x16_p_mat2_8, 2*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_11, _ae_int8x8_mat2_1_11,_ae_int8x16_p_mat2_8, 3*row_stride2);\
          AE_L8X8X2_IP(_ae_int8x8_mat2_8, _ae_int8x8_mat2_1_8,_ae_int8x16_p_mat2_8, 16);\
          AE_L8X8X2_X( _ae_int8x8_mat2_13, _ae_int8x8_mat2_1_13,_ae_int8x16_p_mat2_12, 1*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_14, _ae_int8x8_mat2_1_14,_ae_int8x16_p_mat2_12, 2*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_15, _ae_int8x8_mat2_1_15,_ae_int8x16_p_mat2_12, 3*row_stride2);\
          AE_L8X8X2_IP(_ae_int8x8_mat2_12, _ae_int8x8_mat2_1_12,_ae_int8x16_p_mat2_12, 16);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat2_0, _ae_int8x8_mat2_1, _ae_int8x8_mat2_2,_ae_int8x8_mat2_3,\
                         _ae_int16x4_vec2,_ae_int16x4_vec2_1);\
          AE_MULA8QW8X16(_ae_int64_acc_4  , _ae_int64_acc_5  , _ae_int64_acc_6  ,_ae_int64_acc_7,\
                         _ae_int8x8_mat2_4, _ae_int8x8_mat2_5, _ae_int8x8_mat2_6,_ae_int8x8_mat2_7,\
                         _ae_int16x4_vec2,_ae_int16x4_vec2_1);\
          AE_MULA8QW8X16(_ae_int64_acc_8  , _ae_int64_acc_9  , _ae_int64_acc_10  ,_ae_int64_acc_11,\
                         _ae_int8x8_mat2_8, _ae_int8x8_mat2_9, _ae_int8x8_mat2_10,_ae_int8x8_mat2_11,\
                         _ae_int16x4_vec2,_ae_int16x4_vec2_1);\
          AE_MULA8QW8X16(_ae_int64_acc_12  , _ae_int64_acc_13  , _ae_int64_acc_14  ,_ae_int64_acc_15,\
                         _ae_int8x8_mat2_12, _ae_int8x8_mat2_13, _ae_int8x8_mat2_14,_ae_int8x8_mat2_15,\
                         _ae_int16x4_vec2,_ae_int16x4_vec2_1);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat2_1_0, _ae_int8x8_mat2_1_1, _ae_int8x8_mat2_1_2,_ae_int8x8_mat2_1_3,\
                         _ae_int16x4_vec2_2,_ae_int16x4_vec2_3);\
          AE_MULA8QW8X16(_ae_int64_acc_4  , _ae_int64_acc_5  , _ae_int64_acc_6  ,_ae_int64_acc_7,\
                         _ae_int8x8_mat2_1_4, _ae_int8x8_mat2_1_5, _ae_int8x8_mat2_1_6,_ae_int8x8_mat2_1_7,\
                         _ae_int16x4_vec2_2,_ae_int16x4_vec2_3);\
          AE_MULA8QW8X16(_ae_int64_acc_8  , _ae_int64_acc_9  , _ae_int64_acc_10  ,_ae_int64_acc_11,\
                         _ae_int8x8_mat2_1_8, _ae_int8x8_mat2_1_9, _ae_int8x8_mat2_1_10,_ae_int8x8_mat2_1_11,\
                         _ae_int16x4_vec2_2,_ae_int16x4_vec2_3);\
          AE_MULA8QW8X16(_ae_int64_acc_12  , _ae_int64_acc_13  , _ae_int64_acc_14  ,_ae_int64_acc_15,\
                         _ae_int8x8_mat2_1_12, _ae_int8x8_mat2_1_13, _ae_int8x8_mat2_1_14,_ae_int8x8_mat2_1_15,\
                         _ae_int16x4_vec2_2,_ae_int16x4_vec2_3);\

#define TEMPLATE_mat1Xvec1_mat2Xvec2_8x16 \
        SETUP_ACC; \
        SETUP_VEC1_16b_16; \
        SETUP_MAT1;\
        SETUP_MAT1_8x8_PTR_8x16;\
        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++) \
        { \
          LOAD_VEC1_16b_16_1; \
          LOAD_VEC1_16b_16; \
          kernel_mat1_vec1_8x16; \
        } \
        SETUP_VEC2_16b_16; \
        SETUP_MAT2; \
        SETUP_MAT2_8x8_PTR_8x16; \
        for(c_itr = 0; c_itr < (cols2 >> 4); c_itr++) \
        { \
          LOAD_VEC2_16b_16_1;\
          LOAD_VEC2_16b_16;\
          kernel_mat2_vec2_8x16;\
        }

#else

#ifdef ROW_UNROLL
    #undef ROW_UNROLL
    #define ROW_UNROLL 4
#else
    #define ROW_UNROLL 4
#endif

#include "xa_nnlib_common_macros_hifi5.h"

#if 0
#define SETUP_MAT1_8x8_PTR_8x16 \
        ae_int8x16 * _ae_int8x16_p_mat1_0  = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_1  = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_2  = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_3  = (ae_int8x16 *) &p_mat1[(m_itr+3)*row_stride1];\
        ae_int8x8 _ae_int8x8_mat1_2_0, _ae_int8x8_mat1_3_0,_ae_int8x8_mat1_2_1,_ae_int8x8_mat1_2_2,_ae_int8x8_mat1_2_3;\
        ae_int8x8 _ae_int8x8_mat1_3_1, _ae_int8x8_mat1_3_2, _ae_int8x8_mat1_3_3;

#define SETUP_MAT2_8x8_PTR_8x16 \
        ae_int8x16 * _ae_int8x16_p_mat2_0  = (ae_int8x16 *) &p_mat2[(m_itr+0)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_1  = (ae_int8x16 *) &p_mat2[(m_itr+1)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_2  = (ae_int8x16 *) &p_mat2[(m_itr+2)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_3  = (ae_int8x16 *) &p_mat2[(m_itr+3)*row_stride2];\
        ae_int8x8 _ae_int8x8_mat2_2_0, _ae_int8x8_mat2_3_0,_ae_int8x8_mat2_2_1,_ae_int8x8_mat2_2_2,_ae_int8x8_mat2_2_3;\
        ae_int8x8 _ae_int8x8_mat2_3_1, _ae_int8x8_mat2_3_2, _ae_int8x8_mat2_3_3;

#define kernel_mat1_vec1_8x16 \
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_0, _ae_int8x8_mat1_3_0,_ae_int8x16_p_mat1_0, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_1, _ae_int8x8_mat1_3_1,_ae_int8x16_p_mat1_1, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_2, _ae_int8x8_mat1_3_2,_ae_int8x16_p_mat1_2, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_3, _ae_int8x8_mat1_3_3,_ae_int8x16_p_mat1_3, 16);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_0, _ae_int8x8_mat1_1_0,_ae_int8x16_p_mat1_0, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_1, _ae_int8x8_mat1_1_1,_ae_int8x16_p_mat1_1, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_2, _ae_int8x8_mat1_1_2,_ae_int8x16_p_mat1_2, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_3, _ae_int8x8_mat1_1_3,_ae_int8x16_p_mat1_3, 32);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat1_0, _ae_int8x8_mat1_1, _ae_int8x8_mat1_2,_ae_int8x8_mat1_3,\
                         _ae_int16x4_vec1,_ae_int16x4_vec1_1);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat1_1_0, _ae_int8x8_mat1_1_1, _ae_int8x8_mat1_1_2,_ae_int8x8_mat1_1_3,\
                         _ae_int16x4_vec1_2,_ae_int16x4_vec1_3);\
          LOAD_VEC1_16b_16_1; \
          LOAD_VEC1_16b_16; \
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat1_2_0, _ae_int8x8_mat1_2_1, _ae_int8x8_mat1_2_2,_ae_int8x8_mat1_2_3,\
                         _ae_int16x4_vec1,_ae_int16x4_vec1_1);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat1_3_0, _ae_int8x8_mat1_3_1, _ae_int8x8_mat1_3_2,_ae_int8x8_mat1_3_3,\
                         _ae_int16x4_vec1_2,_ae_int16x4_vec1_3);\

#define kernel_mat2_vec2_8x16 \
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_0, _ae_int8x8_mat2_3_0,_ae_int8x16_p_mat2_0, 16); \
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_1, _ae_int8x8_mat2_3_1,_ae_int8x16_p_mat2_1, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_2, _ae_int8x8_mat2_3_2,_ae_int8x16_p_mat2_2, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_3, _ae_int8x8_mat2_3_3,_ae_int8x16_p_mat2_3, 16);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_0, _ae_int8x8_mat2_1_0,_ae_int8x16_p_mat2_0, 32); \
          AE_L8X8X2_IP( _ae_int8x8_mat2_1, _ae_int8x8_mat2_1_1,_ae_int8x16_p_mat2_1, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_2, _ae_int8x8_mat2_1_2,_ae_int8x16_p_mat2_2, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_3, _ae_int8x8_mat2_1_3,_ae_int8x16_p_mat2_3, 32);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat2_0, _ae_int8x8_mat2_1, _ae_int8x8_mat2_2,_ae_int8x8_mat2_3,\
                         _ae_int16x4_vec2,_ae_int16x4_vec2_1);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat2_1_0, _ae_int8x8_mat2_1_1, _ae_int8x8_mat2_1_2,_ae_int8x8_mat2_1_3,\
                         _ae_int16x4_vec2_2,_ae_int16x4_vec2_3);\
          LOAD_VEC2_16b_16_1;\
          LOAD_VEC2_16b_16;\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat2_2_0, _ae_int8x8_mat2_2_1, _ae_int8x8_mat2_2_2,_ae_int8x8_mat2_2_3,\
                         _ae_int16x4_vec2,_ae_int16x4_vec2_1);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat2_3_0, _ae_int8x8_mat2_3_1, _ae_int8x8_mat2_3_2,_ae_int8x8_mat2_3_3,\
                         _ae_int16x4_vec2_2,_ae_int16x4_vec2_3);\


#define TEMPLATE_mat1Xvec1_mat2Xvec2_8x16 \
        SETUP_ACC; \
        SETUP_VEC1_16b_16; \
        SETUP_MAT1;\
        SETUP_MAT1_8x8_PTR_8x16;\
        for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++) \
        { \
          LOAD_VEC1_16b_16_1; \
          LOAD_VEC1_16b_16; \
          kernel_mat1_vec1_8x16; \
        } \
        SETUP_VEC2_16b_16; \
        SETUP_MAT2; \
        SETUP_MAT2_8x8_PTR_8x16; \
        for(c_itr = 0; c_itr < (cols2 >> 5); c_itr++) \
        { \
          LOAD_VEC2_16b_16_1;\
          LOAD_VEC2_16b_16;\
          kernel_mat2_vec2_8x16;\
        }

#else

#define SETUP_MAT1_8x8_PTR_8x16 \
        ae_int8x16 * _ae_int8x16_p_mat1_0  = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];\

#define SETUP_MAT2_8x8_PTR_8x16 \
        ae_int8x16 * _ae_int8x16_p_mat2_0  = (ae_int8x16 *) &p_mat2[(m_itr+0)*row_stride2];\

#define kernel_mat1_vec1_8x16 \
          AE_L8X8X2_X( _ae_int8x8_mat1_1, _ae_int8x8_mat1_1_1,_ae_int8x16_p_mat1_0, 1*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2, _ae_int8x8_mat1_1_2,_ae_int8x16_p_mat1_0, 2*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_3, _ae_int8x8_mat1_1_3,_ae_int8x16_p_mat1_0, 3*row_stride1);\
          AE_L8X8X2_IP(_ae_int8x8_mat1_0, _ae_int8x8_mat1_1_0,_ae_int8x16_p_mat1_0, 16);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat1_0, _ae_int8x8_mat1_1, _ae_int8x8_mat1_2,_ae_int8x8_mat1_3,\
                         _ae_int16x4_vec1,_ae_int16x4_vec1_1);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat1_1_0, _ae_int8x8_mat1_1_1, _ae_int8x8_mat1_1_2,_ae_int8x8_mat1_1_3,\
                         _ae_int16x4_vec1_2,_ae_int16x4_vec1_3);\

#define kernel_mat2_vec2_8x16 \
          AE_L8X8X2_X( _ae_int8x8_mat2_1, _ae_int8x8_mat2_1_1,_ae_int8x16_p_mat2_0, 1*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2, _ae_int8x8_mat2_1_2,_ae_int8x16_p_mat2_0, 2*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_3, _ae_int8x8_mat2_1_3,_ae_int8x16_p_mat2_0, 3*row_stride2);\
          AE_L8X8X2_IP(_ae_int8x8_mat2_0, _ae_int8x8_mat2_1_0,_ae_int8x16_p_mat2_0, 16); \
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat2_0, _ae_int8x8_mat2_1, _ae_int8x8_mat2_2,_ae_int8x8_mat2_3,\
                         _ae_int16x4_vec2,_ae_int16x4_vec2_1);\
          AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,\
                         _ae_int8x8_mat2_1_0, _ae_int8x8_mat2_1_1, _ae_int8x8_mat2_1_2,_ae_int8x8_mat2_1_3,\
                         _ae_int16x4_vec2_2,_ae_int16x4_vec2_3);\


#define TEMPLATE_mat1Xvec1_mat2Xvec2_8x16 \
        SETUP_ACC; \
        SETUP_VEC1_16b_16; \
        SETUP_MAT1;\
        SETUP_MAT1_8x8_PTR_8x16;\
        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++) \
        { \
          LOAD_VEC1_16b_16_1; \
          LOAD_VEC1_16b_16; \
          kernel_mat1_vec1_8x16; \
        } \
        SETUP_VEC2_16b_16; \
        SETUP_MAT2; \
        SETUP_MAT2_8x8_PTR_8x16; \
        for(c_itr = 0; c_itr < (cols2 >> 4); c_itr++) \
        { \
          LOAD_VEC2_16b_16_1;\
          LOAD_VEC2_16b_16;\
          kernel_mat2_vec2_8x16;\
        }


#endif

#endif


#define TEMPLATE_mat1Xvec1_mat2Xvec2_8x16_single_row \
        UNROLL_SETUP_ACC(0);\
        UNROLL_SETUP_ACC(1);\
        UNROLL_SETUP_ACC(2);\
        UNROLL_SETUP_ACC(3); \
        SETUP_VEC1_16b_16; \
        UNROLL_SETUP_MAT1(0);\
        ZER0_8x8_Temp_Variable;\
        SETUP_MAT1_SINGLE_ROW_PTR;\
        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)\
        { \
          LOAD_VEC1_16b_16_1;\
          LOAD_VEC1_16b_16; \
          kernel_mat1_vec1_8x16_single_row;\
        }\
        SETUP_VEC2_16b_16; \
        UNROLL_SETUP_MAT2(0);\
        SETUP_MAT2_SINGLE_ROW_PTR;\
        for(c_itr = 0; c_itr < (cols2 >> 4); c_itr++)\
        { \
          LOAD_VEC2_16b_16_1;\
          LOAD_VEC2_16b_16;\
          kernel_mat2_vec2_8x16_single_row;\
        }\

#define TEMPLATE_mat1Xvec1_8x16 \
        SETUP_ACC; \
        SETUP_VEC1_16b_16; \
        SETUP_MAT1;\
        SETUP_MAT1_8x8_PTR_8x16;\
        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++) \
        { \
          LOAD_VEC1_16b_16_1; \
          LOAD_VEC1_16b_16; \
          kernel_mat1_vec1_8x16; \
        } \

#define TEMPLATE_mat1Xvec1_8x16_single_row \
        UNROLL_SETUP_ACC(0);\
        UNROLL_SETUP_ACC(1);\
        UNROLL_SETUP_ACC(2);\
        UNROLL_SETUP_ACC(3); \
        SETUP_VEC1_16b_16; \
        UNROLL_SETUP_MAT1(0);\
        ZER0_8x8_Temp_Variable;\
        SETUP_MAT1_SINGLE_ROW_PTR;\
        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)\
        { \
          LOAD_VEC1_16b_16_1;\
          LOAD_VEC1_16b_16; \
          kernel_mat1_vec1_8x16_single_row;\
        }\

#define TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_two_rows \
              SETUP_ACC_FOR_8bx16b(0); \
              SETUP_ACC_FOR_8bx16b(1); \
              SETUP_ACC_FOR_8bx16b(2); \
              SETUP_ACC_FOR_8bx16b(3); \
              SETUP_MAT1_8b_UNALIGNED(0);\
              SETUP_MAT1_8b_UNALIGNED(1);\
              SETUP_MAT1_8b_UNALIGNED(2);\
              SETUP_VEC1_8X16b_UNALIGNED;\
              ZER0_8x8_Temp_Variable;\
              int cols_count=cols1-cols1%8;\
              for(c_itr = 0; c_itr < cols_count>>3; c_itr++)\
              {\
                  LOAD_ROW_MAT1_8b_UNALIGNED(0);\
                  LOAD_ROW_MAT1_8b_UNALIGNED(1);\
                  LOAD_ROW_MAT1_8b_UNALIGNED(2);\
                  LOAD_VEC1_8X16b_UNALIGNED;\
                  KERNEL_MAT1_VEC1_8b_16b_UNALIGNED; \
              }\
              for(c_itr = cols_count; c_itr < cols1; c_itr++)\
              {\
                  LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);\
                  LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(1);\
                  LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(2);\
                  LOAD_VEC1_16b_UNALIGNED_SINGLE_ELEMENT;\
                  KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT;\
              }\
              SETUP_MAT2_8b_UNALIGNED(0);\
              SETUP_MAT2_8b_UNALIGNED(1);\
              SETUP_MAT2_8b_UNALIGNED(2);\
              SETUP_VEC2_8X16b_UNALIGNED;\
              cols_count=cols2-cols2%8;\
              for(c_itr = 0; c_itr < cols_count>>3; c_itr++)\
              {\
                  LOAD_ROW_MAT2_8b_UNALIGNED(0);\
                  LOAD_ROW_MAT2_8b_UNALIGNED(1);\
                  LOAD_ROW_MAT2_8b_UNALIGNED(2);\
                  LOAD_VEC2_8X16b_UNALIGNED;\
                  KERNEL_MAT2_VEC2_8b_16b_UNALIGNED;\
              }\
              for(c_itr = cols_count; c_itr < cols2; c_itr++)\
              {\
                  LOAD_ROW_MAT2_8b_UNALIGNED_SINGLE_ELEMENT(0);\
                  LOAD_ROW_MAT2_8b_UNALIGNED_SINGLE_ELEMENT(1);\
                  LOAD_ROW_MAT2_8b_UNALIGNED_SINGLE_ELEMENT(2);\
                  LOAD_VEC2_16b_UNALIGNED_SINGLE_ELEMENT;\
                  KERNEL_MAT2_VEC2_8b_16b_UNALIGNED_SINGLE_ELEMENT;\
              }


#define TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_one_row \
              SETUP_ACC_FOR_8bx16b(0);\
              SETUP_ACC_FOR_8bx16b(1);\
              SETUP_ACC_FOR_8bx16b(2);\
              SETUP_ACC_FOR_8bx16b(3);\
              SETUP_MAT1_8b_UNALIGNED(0);\
              SETUP_VEC1_8X16b_UNALIGNED;\
              ZER0_8x8_Temp_Variable;\
              int cols_count=cols1-cols1%8;\
              for(c_itr = 0; c_itr < cols_count>>3; c_itr++)\
              {\
                  LOAD_ROW_MAT1_8b_UNALIGNED(0);\
                  LOAD_VEC1_8X16b_UNALIGNED;\
                  KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ROW;\
              }\
              for(c_itr = cols_count; c_itr < cols1; c_itr++)\
              {\
                  LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);\
                  LOAD_VEC1_16b_UNALIGNED_SINGLE_ELEMENT;\
                  KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW;\
              }\
              SETUP_MAT2_8b_UNALIGNED(0);\
              SETUP_VEC2_8X16b_UNALIGNED;\
              cols_count=cols2-cols2%8;\
              for(c_itr = 0; c_itr < cols_count>>3; c_itr++)\
              {\
                  LOAD_ROW_MAT2_8b_UNALIGNED(0);\
                  LOAD_VEC2_8X16b_UNALIGNED;\
                  KERNEL_MAT2_VEC2_8b_16b_UNALIGNED_SINGLE_ROW;\
              }\
              for(c_itr = cols_count; c_itr < cols2; c_itr++)\
              {\
                  LOAD_ROW_MAT2_8b_UNALIGNED_SINGLE_ELEMENT(0);\
                  LOAD_VEC2_16b_UNALIGNED_SINGLE_ELEMENT;\
                  KERNEL_MAT2_VEC2_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW;\
              }\

#define TEMPLATE_UNALIGNED_8x16_matxvec_two_rows \
              SETUP_ACC_FOR_8bx16b(0); \
              SETUP_ACC_FOR_8bx16b(1); \
              SETUP_ACC_FOR_8bx16b(2); \
              SETUP_ACC_FOR_8bx16b(3); \
              SETUP_MAT1_8b_UNALIGNED(0);\
              SETUP_MAT1_8b_UNALIGNED(1);\
              SETUP_MAT1_8b_UNALIGNED(2);\
              SETUP_VEC1_8X16b_UNALIGNED;\
              ZER0_8x8_Temp_Variable;\
              int cols_count=cols1-cols1%8;\
              for(c_itr = 0; c_itr < cols_count>>3; c_itr++)\
              {\
                  LOAD_ROW_MAT1_8b_UNALIGNED(0);\
                  LOAD_ROW_MAT1_8b_UNALIGNED(1);\
                  LOAD_ROW_MAT1_8b_UNALIGNED(2);\
                  LOAD_VEC1_8X16b_UNALIGNED;\
                  KERNEL_MAT1_VEC1_8b_16b_UNALIGNED; \
              }\
              for(c_itr = cols_count; c_itr < cols1; c_itr++)\
              {\
                  LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);\
                  LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(1);\
                  LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(2);\
                  LOAD_VEC1_16b_UNALIGNED_SINGLE_ELEMENT;\
                  KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT;\
              }\


#define TEMPLATE_UNALIGNED_8x16_matxvec_one_row \
              SETUP_ACC_FOR_8bx16b(0);\
              SETUP_ACC_FOR_8bx16b(1);\
              SETUP_ACC_FOR_8bx16b(2);\
              SETUP_ACC_FOR_8bx16b(3);\
              SETUP_MAT1_8b_UNALIGNED(0);\
              SETUP_VEC1_8X16b_UNALIGNED;\
              ZER0_8x8_Temp_Variable;\
              int cols_count=cols1-cols1%8;\
              for(c_itr = 0; c_itr < cols_count>>3; c_itr++)\
              {\
                  LOAD_ROW_MAT1_8b_UNALIGNED(0);\
                  LOAD_VEC1_8X16b_UNALIGNED;\
                  KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ROW;\
              }\
              for(c_itr = cols_count; c_itr < cols1; c_itr++)\
              {\
                  LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);\
                  LOAD_VEC1_16b_UNALIGNED_SINGLE_ELEMENT;\
                  KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW;\
              }\


WORD32 xa_nn_matXvec_8x16_16(
         WORD16 * __restrict__ p_out,           /* output */
         WORD8 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD8 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         WORD16 * __restrict__ p_bias,          /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 row_stride2,                    /* row stride for matrix2 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift)                       /* bias shift amount */
{
  /* Iterators used in for loops */
  int m_itr, c_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;

  if (!p_bias)
  {
    return -1;
  }

#define UNROLL_SETUP_ACC        SETUP_ACC_FOR_8bx16b
#define UNROLL_SETUP_MAT1       SETUP_MAT1_8b_8x16
#define UNROLL_SETUP_MAT2       SETUP_MAT2_8b_8x16
#define UNROLL_STORE_ACC        STORE_ACC_8bx16b_AT_OUT_16b
#define SETUP_BIAS              SETUP_BIAS_16b
#define UNROLL_ADD_BIAS_ACC     ADD_BIAS_16b_ACC_FOR_8bx16b


  acc_shift=32+acc_shift;
  LIMIT_ACC_LSH
  if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && cols1%16==0 && cols2%16==0 && row_stride1%16==0 && row_stride2%16==0)
  {
    /* All four pointers are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {

      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
      {
        TEMPLATE_mat1Xvec1_mat2Xvec2_8x16;
        ADD_BIAS_ACC;
#ifdef UNROLL_16
        ae_int16 * output_ptr=(ae_int16*)(p_out+m_itr);
        STORE_ACC_8bx16b_AT_OUT_16x4x4;
#else
        UNROLL_STORE_ACC(0);
        UNROLL_STORE_ACC(1);
        UNROLL_STORE_ACC(2);
        UNROLL_STORE_ACC(3);
#endif
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        TEMPLATE_mat1Xvec1_mat2Xvec2_8x16_single_row;
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_STORE_ACC(0);
      }
    }
  }
  else if(p_mat1 && p_vec1 && p_mat2 && p_vec2)
  {

      SETUP_BIAS;
      if(rows > 3)
      {
          int row_count = rows- rows%3;
          for(m_itr = 0; m_itr < row_count ; m_itr+=3)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_two_rows;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              ADD_BIAS_16b_ACC_FOR_8bx16b(1);
              ADD_BIAS_16b_ACC_FOR_8bx16b(2);
              STORE_ACC_8bx16b_AT_OUT_16b(0);
              STORE_ACC_8bx16b_AT_OUT_16b(1);
              STORE_ACC_8bx16b_AT_OUT_16b(2);
          }
      }
      {
          for(; m_itr < rows ; m_itr++)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_one_row;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              STORE_ACC_8bx16b_AT_OUT_16b(0);
          }
      }

  }
  else if (p_mat1 && p_vec1 && cols1%32==0)
  {
    /* Only mat1, vec1 are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL_SINGLE)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL_SINGLE-1)) ; m_itr += ROW_UNROLL_SINGLE)
      {
#if 1
        UNROLL_SETUP_ACC(0);UNROLL_SETUP_ACC(1)UNROLL_SETUP_ACC(2);UNROLL_SETUP_ACC(3);
        SETUP_VEC1_16b_16;
        SETUP_MAT1_8b_8x16(0);SETUP_MAT1_8b_8x16(1);SETUP_MAT1_8b_8x16(2);SETUP_MAT1_8b_8x16(3);

        ae_int8x16 * _ae_int8x16_p_mat1_0  = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];
        ae_int8x16 * _ae_int8x16_p_mat1_1  = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];
        ae_int8x16 * _ae_int8x16_p_mat1_2  = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];
        ae_int8x16 * _ae_int8x16_p_mat1_3 = (ae_int8x16 *) &p_mat1[(m_itr+3)*row_stride1];

        for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++)
        {
          AE_L16X4X2_I(_ae_int16x4_vec1_2,_ae_int16x4_vec1_3, _ae_int16x8_p_vec1, 48);
          AE_L16X4X2_I(_ae_int16x4_vec1,_ae_int16x4_vec1_1, _ae_int16x8_p_vec1, 32);

          AE_L8X8X2_I(_ae_int8x8_mat1_0, _ae_int8x8_mat1_1_0,_ae_int8x16_p_mat1_0, 16);
          AE_L8X8X2_I( _ae_int8x8_mat1_1, _ae_int8x8_mat1_1_1,_ae_int8x16_p_mat1_1, 16);
          AE_L8X8X2_I( _ae_int8x8_mat1_2, _ae_int8x8_mat1_1_2,_ae_int8x16_p_mat1_2, 16);
          AE_L8X8X2_I( _ae_int8x8_mat1_3, _ae_int8x8_mat1_1_3,_ae_int8x16_p_mat1_3, 16);
          AE_MULA8QW8X16(_ae_int64_acc_0, _ae_int64_acc_1, _ae_int64_acc_2, _ae_int64_acc_3, _ae_int8x8_mat1_0, _ae_int8x8_mat1_1, _ae_int8x8_mat1_2, _ae_int8x8_mat1_3, _ae_int16x4_vec1,_ae_int16x4_vec1_1);
          AE_MULA8QW8X16(_ae_int64_acc_0, _ae_int64_acc_1, _ae_int64_acc_2, _ae_int64_acc_3, _ae_int8x8_mat1_1_0, _ae_int8x8_mat1_1_1, _ae_int8x8_mat1_1_2, _ae_int8x8_mat1_1_3, _ae_int16x4_vec1_2, _ae_int16x4_vec1_3);

          AE_L16X4X2_I(_ae_int16x4_vec1_2,_ae_int16x4_vec1_3, _ae_int16x8_p_vec1, 16);
          AE_L16X4X2_IP(_ae_int16x4_vec1,_ae_int16x4_vec1_1, _ae_int16x8_p_vec1, 8*INCREMENT_IN_BYTES_FOR_INT16X4);

          AE_L8X8X2_IP(_ae_int8x8_mat1_0, _ae_int8x8_mat1_1_0,_ae_int8x16_p_mat1_0, 4*INCREMENT_IN_BYTES_FOR_WORD8X8);
          AE_L8X8X2_IP( _ae_int8x8_mat1_1, _ae_int8x8_mat1_1_1,_ae_int8x16_p_mat1_1, 4*INCREMENT_IN_BYTES_FOR_WORD8X8);
          AE_L8X8X2_IP( _ae_int8x8_mat1_2, _ae_int8x8_mat1_1_2,_ae_int8x16_p_mat1_2, 4*INCREMENT_IN_BYTES_FOR_WORD8X8);
          AE_L8X8X2_IP( _ae_int8x8_mat1_3, _ae_int8x8_mat1_1_3,_ae_int8x16_p_mat1_3, 4*INCREMENT_IN_BYTES_FOR_WORD8X8);
          AE_MULA8QW8X16(_ae_int64_acc_0, _ae_int64_acc_1, _ae_int64_acc_2, _ae_int64_acc_3, _ae_int8x8_mat1_0, _ae_int8x8_mat1_1, _ae_int8x8_mat1_2, _ae_int8x8_mat1_3, _ae_int16x4_vec1,_ae_int16x4_vec1_1);
          AE_MULA8QW8X16(_ae_int64_acc_0, _ae_int64_acc_1, _ae_int64_acc_2, _ae_int64_acc_3, _ae_int8x8_mat1_1_0, _ae_int8x8_mat1_1_1, _ae_int8x8_mat1_1_2, _ae_int8x8_mat1_1_3, _ae_int16x4_vec1_2, _ae_int16x4_vec1_3);
#else
          ae_int16 * output_ptr=(ae_int16*)(p_out+m_itr);
          TEMPLATE_mat1Xvec1_8x16;
#endif
        }
#if 1
        UNROLL_ADD_BIAS_ACC(0);UNROLL_ADD_BIAS_ACC(1);UNROLL_ADD_BIAS_ACC(2);UNROLL_ADD_BIAS_ACC(3);
        STORE_ACC_8bx16b_AT_OUT_16b(0);STORE_ACC_8bx16b_AT_OUT_16b(1);STORE_ACC_8bx16b_AT_OUT_16b(2);STORE_ACC_8bx16b_AT_OUT_16b(3);
#else
        ADD_BIAS_ACC;
        STORE_ACC_8bx16b_AT_OUT_16x4x4;
#endif
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        TEMPLATE_mat1Xvec1_8x16_single_row;
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_STORE_ACC(0);
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
      SETUP_BIAS;
      if(rows > 3)
      {
          int row_count = rows - rows%3;
          for(m_itr = 0; m_itr < row_count; m_itr+=3)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_two_rows;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              ADD_BIAS_16b_ACC_FOR_8bx16b(1);
              ADD_BIAS_16b_ACC_FOR_8bx16b(2);
              STORE_ACC_8bx16b_AT_OUT_16b(0);
              STORE_ACC_8bx16b_AT_OUT_16b(1);
              STORE_ACC_8bx16b_AT_OUT_16b(2);
          }
      }
      {
          for(; m_itr < rows ; m_itr++)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_one_row;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              STORE_ACC_8bx16b_AT_OUT_16b(0);
          }
      }

  }
  else
  {
    return -1;
  }

  /* Undefining the defined macro to make them available for reuse */
#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef UNROLL_STORE_ACC
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC

  return 0;
}

WORD32 xa_nn_matXvec_8x16_32(
         WORD32 * __restrict__ p_out,           /* output */
         WORD8 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD8 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         WORD16 * __restrict__ p_bias,          /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 row_stride2,                    /* row stride for matrix2 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift)                       /* bias shift amount */

{
  /* Iterators used in for loops */
  int m_itr, c_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  if (!p_bias)
  {
    return -1;
  }

#define UNROLL_SETUP_ACC        SETUP_ACC_FOR_8bx16b
#define UNROLL_SETUP_MAT1       SETUP_MAT1_8b_8x16
#define UNROLL_SETUP_MAT2       SETUP_MAT2_8b_8x16
#define UNROLL_STORE_ACC        STORE_ACC_8bx16b_AT_OUT_32b
#define SETUP_BIAS              SETUP_BIAS_16b
#define UNROLL_ADD_BIAS_ACC     ADD_BIAS_16b_ACC_FOR_8bx16b

  //ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(WORD8, WORD16, WORD32);
  acc_shift=32+acc_shift;
  LIMIT_ACC_LSH
  if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && cols1%16==0 && cols2%16==0 && row_stride1%16==0 && row_stride2%16==0)
  {
    /* All four pointers are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
      {
        TEMPLATE_mat1Xvec1_mat2Xvec2_8x16;
        ADD_BIAS_ACC;
#ifdef UNROLL_16
        ae_int32x4 * output_ptr=(ae_int32x4*)(p_out+m_itr);
        STORE_ACC_8bx16b_AT_OUT_32b_32x4x4;
#else
        UNROLL_STORE_ACC(0);
        UNROLL_STORE_ACC(1);
        UNROLL_STORE_ACC(2);
        UNROLL_STORE_ACC(3);
#endif
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        TEMPLATE_mat1Xvec1_mat2Xvec2_8x16_single_row;
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_STORE_ACC(0);
      }
    }
  }
  else if(p_mat1 && p_vec1 && p_mat2 && p_vec2)
  {
      SETUP_BIAS;
      if(rows > 3)
      {
          int row_count = rows- rows%3;
          for(m_itr = 0; m_itr < row_count ; m_itr+=3)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_two_rows;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              ADD_BIAS_16b_ACC_FOR_8bx16b(1);
              ADD_BIAS_16b_ACC_FOR_8bx16b(2);
              STORE_ACC_8bx16b_AT_OUT_32b(0);
              STORE_ACC_8bx16b_AT_OUT_32b(1);
              STORE_ACC_8bx16b_AT_OUT_32b(2);
          }
      }
      {
          for(; m_itr < rows ; m_itr++)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_one_row;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              STORE_ACC_8bx16b_AT_OUT_32b(0);
          }
      }

  }
  else if (p_mat1 && p_vec1 && cols1%16==0)
  {
    /* Only mat1, vec1 are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
      {
        TEMPLATE_mat1Xvec1_8x16;
        ADD_BIAS_ACC;
#ifdef UNROLL_16
        ae_int32x4 * output_ptr=(ae_int32x4*)(p_out+m_itr);
        STORE_ACC_8bx16b_AT_OUT_32b_32x4x4;
#else
        UNROLL_STORE_ACC(0);
        UNROLL_STORE_ACC(1);
        UNROLL_STORE_ACC(2);
        UNROLL_STORE_ACC(3);
#endif
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        TEMPLATE_mat1Xvec1_8x16_single_row;
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_STORE_ACC(0);
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
      SETUP_BIAS;
      if(rows > 3)
      {
          int row_count = rows - rows%3;
          for(m_itr = 0; m_itr < row_count; m_itr+=3)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_two_rows;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              ADD_BIAS_16b_ACC_FOR_8bx16b(1);
              ADD_BIAS_16b_ACC_FOR_8bx16b(2);
              STORE_ACC_8bx16b_AT_OUT_32b(0);
              STORE_ACC_8bx16b_AT_OUT_32b(1);
              STORE_ACC_8bx16b_AT_OUT_32b(2);
          }
      }
      {
          for(; m_itr < rows ; m_itr++)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_one_row;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              STORE_ACC_8bx16b_AT_OUT_32b(0);
          }
      }

  }
  else
  {
    return -1;
  }

  /* Undefining the defined macro to make them available for reuse */
#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef UNROLL_STORE_ACC
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
  return 0;
}


WORD32 xa_nn_matXvec_8x16_64(
         WORD64 * __restrict__ p_out,           /* output */
         WORD8 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD8 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         WORD16 * __restrict__ p_bias,          /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 row_stride2,                    /* row stride for matrix2 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift)                       /* bias shift amount */
{
  /* Iterators used in for loops */
  int m_itr, c_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  if (!p_bias)
  {
    return -1;
  }

#define UNROLL_SETUP_ACC        SETUP_ACC_FOR_8bx16b
#define UNROLL_SETUP_MAT1       SETUP_MAT1_8b_8x16
#define UNROLL_SETUP_MAT2       SETUP_MAT2_8b_8x16
#define UNROLL_STORE_ACC        STORE_ACC_8bx16b_AT_OUT_64b
#define SETUP_BIAS              SETUP_BIAS_16b
#define UNROLL_ADD_BIAS_ACC     ADD_BIAS_16b_ACC_FOR_8bx16b
  LIMIT_ACC_LSH

  if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && cols1%16==0 && cols2%16==0 && row_stride1%16==0 && row_stride2%16==0)
  {
    /* All four pointers are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
      {
          TEMPLATE_mat1Xvec1_mat2Xvec2_8x16;
          ADD_BIAS_ACC;
#ifdef UNROLL_16
          ae_int64x2 * output_ptr=(ae_int64x2*)(p_out+m_itr);
          STORE_ACC_8bx16b_AT_OUT_64b_64x2x8;
#else
        UNROLL_STORE_ACC(0);
        UNROLL_STORE_ACC(1);
        UNROLL_STORE_ACC(2);
        UNROLL_STORE_ACC(3);
#endif
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        TEMPLATE_mat1Xvec1_mat2Xvec2_8x16_single_row;
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_STORE_ACC(0);
      }
    }
  }
  else if(p_mat1 && p_vec1 && p_mat2 && p_vec2)
  {
      SETUP_BIAS;
      if(rows > 3)
      {
          int row_count = rows- rows%3;
          for(m_itr = 0; m_itr < row_count ; m_itr+=3)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_two_rows;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              ADD_BIAS_16b_ACC_FOR_8bx16b(1);
              ADD_BIAS_16b_ACC_FOR_8bx16b(2);
              STORE_ACC_8bx16b_AT_OUT_64b(0);
              STORE_ACC_8bx16b_AT_OUT_64b(1);
              STORE_ACC_8bx16b_AT_OUT_64b(2);
          }
      }
      {
          for(; m_itr < rows ; m_itr++)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_one_row;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              STORE_ACC_8bx16b_AT_OUT_64b(0);
          }
      }

  }
  else if (p_mat1 && p_vec1 && cols1%16==0)
  {
    /* Only mat1, vec1 are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
      {
          TEMPLATE_mat1Xvec1_8x16;
          ADD_BIAS_ACC;
#ifdef UNROLL_16
          ae_int64x2 * output_ptr=(ae_int64x2*)(p_out+m_itr);
          STORE_ACC_8bx16b_AT_OUT_64b_64x2x8;
#else
        UNROLL_STORE_ACC(0);
        UNROLL_STORE_ACC(1);
        UNROLL_STORE_ACC(2);
        UNROLL_STORE_ACC(3);
#endif
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        TEMPLATE_mat1Xvec1_8x16_single_row;
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_STORE_ACC(0);
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
      SETUP_BIAS;
      if(rows > 3)
      {
          int row_count = rows - rows%3;
          for(m_itr = 0; m_itr < row_count; m_itr+=3)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_two_rows;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              ADD_BIAS_16b_ACC_FOR_8bx16b(1);
              ADD_BIAS_16b_ACC_FOR_8bx16b(2);
              STORE_ACC_8bx16b_AT_OUT_64b(0);
              STORE_ACC_8bx16b_AT_OUT_64b(1);
              STORE_ACC_8bx16b_AT_OUT_64b(2);
          }
      }
      {
          for(; m_itr < rows ; m_itr++)
          {
              TEMPLATE_UNALIGNED_8x16_matxvec_one_row;
              ADD_BIAS_16b_ACC_FOR_8bx16b(0);
              STORE_ACC_8bx16b_AT_OUT_64b(0);
          }
      }

  }
  else
  {
    return -1;
  }

  /* Undefining the defined macro to make them available for reuse */
#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef UNROLL_STORE_ACC
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
  return 0;
}

WORD32 xa_nn_matXvec_8x16_16_tanh(
         WORD16 * __restrict__ p_out,     /* output */
         WORD8 * __restrict__ p_mat1,     /* matrix1: rows x cols1 */
         WORD8 * __restrict__ p_mat2,     /* matrix2: rows x cols2 */
         WORD16 * __restrict__ p_vec1,    /* vec1: cols1 x 1 */
         WORD16 * __restrict__ p_vec2,    /* vec2: cols2 x 1 */
         VOID   * __restrict__ p_bias,    /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,              /* row stride for matrix1 */
         WORD32 row_stride2,              /* row stride for matrix2 */
         WORD32 acc_shift,                  /* out accumulator shift amount */
         WORD32 bias_shift,                 /* bias shift amount */
         WORD32 bias_precision,           /* 16 or 64 */
         VOID   * __restrict__ p_scratch) /* Scratch pointer arg, only if required */
{
  /* Iterators used in for loops */
  int m_itr, c_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  if (!p_bias)
  {
    return -1;
  }

  int err=0;

  switch(bias_precision)
  {
    default:
    case 16:
      {
        err = xa_nn_matXvec_8x16_32(
        ((WORD32 *)p_scratch),       /* output stored in scratch*/
        p_mat1,          /* matrix1: rows x cols1 */
        p_mat2,          /* matrix2: rows x cols2 */
        p_vec1,          /* vec1: cols1 x 1 */
        p_vec2,          /* vec2: cols2 x 1 */
        ((WORD16 *)p_bias),          /* bias */
        rows,
        cols1,
        cols2,
        row_stride1,                  /* row stride for matrix1 */
        row_stride2,                  /* row stride for matrix2 */
        acc_shift,                    /* out accumulator shift amount */
        bias_shift);                   /* bias shift amount */

        if(err){
            return -1;
        }
        break;
      }
    case 64:
      {
#define UNROLL_SETUP_ACC        SETUP_ACC_FOR_8bx16b
#define UNROLL_SETUP_MAT1       SETUP_MAT1_8b_8x16
#define UNROLL_SETUP_MAT2       SETUP_MAT2_8b_8x16
#define UNROLL_STORE_ACC        STORE_ACC_8bx16b_AT_SCRATCH_32b
#define SETUP_BIAS              SETUP_BIAS_64b
#define UNROLL_ADD_BIAS_ACC     ADD_BIAS_64b_ACC_FOR_8bx16b
        acc_shift=32+acc_shift;
        LIMIT_ACC_LSH
        if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && cols1%16==0 && cols2%16==0 && row_stride1%16==0 && row_stride2%16==0)
        {
          /* All four pointers are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
            {

              TEMPLATE_mat1Xvec1_mat2Xvec2_8x16;
              ADD_BIAS_ACC;
#ifdef UNROLL_16
              ae_int32x4 * output_ptr=(ae_int32x4*)((int *)p_scratch+m_itr);
              STORE_ACC_8bx16b_AT_OUT_32b_32x4x4;
#else
              UNROLL_STORE_ACC(0);
              UNROLL_STORE_ACC(1);
              UNROLL_STORE_ACC(2);
              UNROLL_STORE_ACC(3);
#endif
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {

                TEMPLATE_mat1Xvec1_mat2Xvec2_8x16_single_row;
                UNROLL_ADD_BIAS_ACC(0);
                UNROLL_STORE_ACC(0);
            }
          }
        }
          else if(p_mat1 && p_vec1 && p_mat2 && p_vec2)
          {
              SETUP_BIAS;
              if(rows > 3)
              {
                  int row_count = rows- rows%3;
                  for(m_itr = 0; m_itr < row_count ; m_itr+=3)
                  {
                      TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_two_rows;
                      ADD_BIAS_64b_ACC_FOR_8bx16b(0);
                      ADD_BIAS_64b_ACC_FOR_8bx16b(1);
                      ADD_BIAS_64b_ACC_FOR_8bx16b(2);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(0);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(1);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(2);
                  }
              }
              {
                  for(; m_itr < rows ; m_itr++)
                  {
                      TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_one_row;
                      ADD_BIAS_64b_ACC_FOR_8bx16b(0);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(0);
                  }
              }

          }
        else if (p_mat1 && p_vec1 && cols1%16==0)
        {
          /* Only mat1, vec1 are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
            {
              TEMPLATE_mat1Xvec1_8x16;
              ADD_BIAS_ACC;
#ifdef UNROLL_16
              ae_int32x4 * output_ptr=(ae_int32x4*)((int *)p_scratch+m_itr);
              STORE_ACC_8bx16b_AT_OUT_32b_32x4x4;
#else
              UNROLL_STORE_ACC(0);
              UNROLL_STORE_ACC(1);
              UNROLL_STORE_ACC(2);
              UNROLL_STORE_ACC(3);
#endif
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {
                TEMPLATE_mat1Xvec1_8x16_single_row;
                UNROLL_ADD_BIAS_ACC(0);
                UNROLL_STORE_ACC(0);
            }
          }
        }
          else if(p_mat1 && p_vec1)
          {
              SETUP_BIAS;
              if(rows > 3)
              {
                  int row_count = rows - rows%3;
                  for(m_itr = 0; m_itr < row_count; m_itr+=3)
                  {
                      TEMPLATE_UNALIGNED_8x16_matxvec_two_rows;
                      ADD_BIAS_64b_ACC_FOR_8bx16b(0);
                      ADD_BIAS_64b_ACC_FOR_8bx16b(1);
                      ADD_BIAS_64b_ACC_FOR_8bx16b(2);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(0);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(1);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(2);
                  }
              }
              {
                  for(; m_itr < rows ; m_itr++)
                  {
                      TEMPLATE_UNALIGNED_8x16_matxvec_one_row;
                      ADD_BIAS_64b_ACC_FOR_8bx16b(0);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(0);
                  }
              }

          }
        else
        {
          return -1;
        }

        break;
        /* Undefining the defined macro to make them available for reuse */
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
      }
#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef UNROLL_STORE_ACC
  }

  xa_nn_vec_tanh_32_16((pWORD16) p_out, (pWORD32) p_scratch, rows);
  return 0;
}

WORD32 xa_nn_matXvec_8x16_16_sigmoid(
         WORD16 * __restrict__ p_out,     /* output */
         WORD8 * __restrict__ p_mat1,     /* matrix1: rows x cols1 */
         WORD8 * __restrict__ p_mat2,     /* matrix2: rows x cols2 */
         WORD16 * __restrict__ p_vec1,    /* vec1: cols1 x 1 */
         WORD16 * __restrict__ p_vec2,    /* vec2: cols2 x 1 */
         VOID   * __restrict__ p_bias,    /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,              /* row stride for matrix1 */
         WORD32 row_stride2,              /* row stride for matrix2 */
         WORD32 acc_shift,                  /* out accumulator shift amount */
         WORD32 bias_shift,                 /* bias shift amount */
         WORD32 bias_precision,           /* 16 or 64 */
         VOID   * __restrict__ p_scratch) /* Scratch pointer arg, only if required */
{
  /* Iterators used in for loops */
  int m_itr, c_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;

  if (!p_bias)
  {
    return -1;
  }

  int err = 0;

  switch(bias_precision)
  {
    default:
    case 16:
      {
        err = xa_nn_matXvec_8x16_32(
        ((WORD32 *)p_scratch),       /* output stored in scratch*/
        p_mat1,          /* matrix1: rows x cols1 */
        p_mat2,          /* matrix2: rows x cols2 */
        p_vec1,          /* vec1: cols1 x 1 */
        p_vec2,          /* vec2: cols2 x 1 */
        ((WORD16 *)p_bias),          /* bias */
        rows,
        cols1,
        cols2,
        row_stride1,                  /* row stride for matrix1 */
        row_stride2,                  /* row stride for matrix2 */
        acc_shift,                    /* out accumulator shift amount */
        bias_shift);                   /* bias shift amount */

        if(err){
            return -1;
        }
        break;
      }
    case 64:
      {
#define UNROLL_SETUP_ACC        SETUP_ACC_FOR_8bx16b
#define UNROLL_SETUP_MAT1       SETUP_MAT1_8b_8x16
#define UNROLL_SETUP_MAT2       SETUP_MAT2_8b_8x16
#define UNROLL_STORE_ACC        STORE_ACC_8bx16b_AT_SCRATCH_32b
#define SETUP_BIAS              SETUP_BIAS_64b
#define UNROLL_ADD_BIAS_ACC     ADD_BIAS_64b_ACC_FOR_8bx16b
        acc_shift=32+acc_shift;
        LIMIT_ACC_LSH
        if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && cols1%16==0 && cols2%16==0 && row_stride1%16==0 && row_stride2%16==0)
        {
          /* All four pointers are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
            {
                TEMPLATE_mat1Xvec1_mat2Xvec2_8x16;
                ADD_BIAS_ACC;
#ifdef UNROLL_16
                ae_int32x4 * output_ptr=(ae_int32x4*)((int*)p_scratch+m_itr);
              STORE_ACC_8bx16b_AT_OUT_32b_32x4x4;
#else
              UNROLL_STORE_ACC(0);
              UNROLL_STORE_ACC(1);
              UNROLL_STORE_ACC(2);
              UNROLL_STORE_ACC(3);
#endif
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {
              TEMPLATE_mat1Xvec1_mat2Xvec2_8x16_single_row;
              UNROLL_ADD_BIAS_ACC(0); UNROLL_STORE_ACC(0);
            }
          }
        }
        else if(p_mat1 && p_vec1 && p_mat2 && p_vec2)
          {
              SETUP_BIAS;
              if(rows > 3)
              {
                  int row_count = rows- rows%3;
                  for(m_itr = 0; m_itr < row_count ; m_itr+=3)
                  {
                      TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_two_rows;
                      ADD_BIAS_64b_ACC_FOR_8bx16b(0);
                      ADD_BIAS_64b_ACC_FOR_8bx16b(1);
                      ADD_BIAS_64b_ACC_FOR_8bx16b(2);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(0);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(1);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(2);
                  }
              }
              {
                  for(; m_itr < rows ; m_itr++)
                  {
                      TEMPLATE_UNALIGNED_8x16_matxvec_mat1xvec1_one_row;
                      ADD_BIAS_64b_ACC_FOR_8bx16b(0);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(0);
                  }
              }

          }
        else if (p_mat1 && p_vec1 &&  cols1%16==0)
        {
          /* Only mat1, vec1 are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
            {
                TEMPLATE_mat1Xvec1_8x16;
                ADD_BIAS_ACC;
#ifdef UNROLL_16
              ae_int32x4 * output_ptr=(ae_int32x4*)((int *)p_scratch+m_itr);
              STORE_ACC_8bx16b_AT_OUT_32b_32x4x4;
#else
              UNROLL_STORE_ACC(0);
              UNROLL_STORE_ACC(1);
              UNROLL_STORE_ACC(2);
              UNROLL_STORE_ACC(3);
#endif
            }
          }
          {
            for(; m_itr < rows; m_itr++)
            {
                TEMPLATE_mat1Xvec1_8x16_single_row;
                UNROLL_ADD_BIAS_ACC(0);
                UNROLL_STORE_ACC(0);
            }
          }
        }
          else if(p_mat1 && p_vec1)
          {
              SETUP_BIAS;
              if(rows > 3)
              {
                  int row_count = rows - rows%3;
                  for(m_itr = 0; m_itr < row_count; m_itr+=3)
                  {
                      TEMPLATE_UNALIGNED_8x16_matxvec_two_rows;
                      ADD_BIAS_64b_ACC_FOR_8bx16b(0);
                      ADD_BIAS_64b_ACC_FOR_8bx16b(1);
                      ADD_BIAS_64b_ACC_FOR_8bx16b(2);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(0);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(1);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(2);
                  }
              }
              {
                  for(; m_itr < rows ; m_itr++)
                  {
                      TEMPLATE_UNALIGNED_8x16_matxvec_one_row;
                      ADD_BIAS_64b_ACC_FOR_8bx16b(0);
                      STORE_ACC_8bx16b_AT_SCRATCH_32b(0);
                  }
              }

          }
        else
        {
          return -1;
        }

        break;
        /* Undefining the defined macro to make them available for reuse */
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC
      }
#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef UNROLL_STORE_ACC
  }

  xa_nn_vec_sigmoid_32_16((pWORD16) p_out, (pWORD32) p_scratch, rows);
  return 0;
}
