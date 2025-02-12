/*******************************************************************************
* Copyright (c) 2018-2025 Cadence Design Systems, Inc.
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
#include "xa_nnlib_common_macros_hifi5.h"
#define ROW_UNROLL_SINGLE 4
/* Uncomment the line below to enable unroll row 16 code */
//#define UNROLL_16

#ifdef UNROLL_16

#define ROW_UNROLL_OPT 16
#define SETUP_MAT1_8x8_PTR \
        ae_int8x16 * _ae_int8x16_p_mat1_0  = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_4  = (ae_int8x16 *) &p_mat1[(m_itr+4)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_8  = (ae_int8x16 *) &p_mat1[(m_itr+8)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_12 = (ae_int8x16 *) &p_mat1[(m_itr+12)*row_stride1];\

#define SETUP_MAT2_8x8_PTR \
        ae_int8x16 * _ae_int8x16_p_mat2_0  = (ae_int8x16 *) &p_mat2[(m_itr+0)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_4  = (ae_int8x16 *) &p_mat2[(m_itr+4)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_8  = (ae_int8x16 *) &p_mat2[(m_itr+8)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_12 = (ae_int8x16 *) &p_mat2[(m_itr+12)*row_stride2];\

#define SETUP_MAT1_8b_16(idx) \
  ae_int8x8 _ae_int8x8_mat1_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat1_1_ ## idx; \
  ae_int8x8 _ae_int8x8_mat1_2_ ## idx; \
  ae_int8x8 _ae_int8x8_mat1_3_ ## idx; \

#define SET_UP_MAT1_16_VARIABLES \
        SETUP_MAT1_8b_16(0) ;SETUP_MAT1_8b_16(1) ;SETUP_MAT1_8b_16(2) ;SETUP_MAT1_8b_16(3);\
        SETUP_MAT1_8b_16(4) ;SETUP_MAT1_8b_16(5) ;SETUP_MAT1_8b_16(6) ;SETUP_MAT1_8b_16(7);\
        SETUP_MAT1_8b_16(8) ;SETUP_MAT1_8b_16(9) ;SETUP_MAT1_8b_16(10);SETUP_MAT1_8b_16(11);\
        SETUP_MAT1_8b_16(12);SETUP_MAT1_8b_16(13);SETUP_MAT1_8b_16(14);SETUP_MAT1_8b_16(15);

#define SETUP_MAT2_8b_16(idx) \
  ae_int8x8 _ae_int8x8_mat2_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat2_1_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat2_2_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat2_3_ ## idx ; \

#define SET_UP_MAT2_16_VARIABLES \
        SETUP_MAT2_8b_16(0) ;SETUP_MAT2_8b_16(1) ;SETUP_MAT2_8b_16(2) ;SETUP_MAT2_8b_16(3);\
        SETUP_MAT2_8b_16(4) ;SETUP_MAT2_8b_16(5) ;SETUP_MAT2_8b_16(6) ;SETUP_MAT2_8b_16(7);\
        SETUP_MAT2_8b_16(8) ;SETUP_MAT2_8b_16(9) ;SETUP_MAT2_8b_16(10);SETUP_MAT2_8b_16(11);\
        SETUP_MAT2_8b_16(12);SETUP_MAT2_8b_16(13);SETUP_MAT2_8b_16(14);SETUP_MAT2_8b_16(15);

#define kernel_8x8_mat1_vec1 \
          AE_L8X8X2_X( _ae_int8x8_mat1_1, _ae_int8x8_mat1_1_1,_ae_int8x16_p_mat1_0, 1*row_stride1);\
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
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_0 , _ae_int8x8_mat1_1 , _ae_int8x8_mat1_2 , _ae_int8x8_mat1_3 , _ae_int8x8_vec1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 , _ae_int8x8_mat1_4 , _ae_int8x8_mat1_5 , _ae_int8x8_mat1_6 , _ae_int8x8_mat1_7 , _ae_int8x8_vec1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4 ,  _ae_int32x2_acc_5 , _ae_int8x8_mat1_8 , _ae_int8x8_mat1_9 , _ae_int8x8_mat1_10, _ae_int8x8_mat1_11, _ae_int8x8_vec1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6 ,  _ae_int32x2_acc_7 , _ae_int8x8_mat1_12, _ae_int8x8_mat1_13, _ae_int8x8_mat1_14, _ae_int8x8_mat1_15, _ae_int8x8_vec1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_1_0 , _ae_int8x8_mat1_1_1 , _ae_int8x8_mat1_1_2 , _ae_int8x8_mat1_1_3 , _ae_int8x8_vec1_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2,  _ae_int32x2_acc_3, _ae_int8x8_mat1_1_4 , _ae_int8x8_mat1_1_5 , _ae_int8x8_mat1_1_6 , _ae_int8x8_mat1_1_7 , _ae_int8x8_vec1_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4,  _ae_int32x2_acc_5, _ae_int8x8_mat1_1_8 , _ae_int8x8_mat1_1_9 , _ae_int8x8_mat1_1_10, _ae_int8x8_mat1_1_11, _ae_int8x8_vec1_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6,  _ae_int32x2_acc_7, _ae_int8x8_mat1_1_12, _ae_int8x8_mat1_1_13, _ae_int8x8_mat1_1_14, _ae_int8x8_mat1_1_15, _ae_int8x8_vec1_1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_1, _ae_int8x8_mat1_3_1,_ae_int8x16_p_mat1_0, 1*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_2, _ae_int8x8_mat1_3_2,_ae_int8x16_p_mat1_0, 2*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_3, _ae_int8x8_mat1_3_3,_ae_int8x16_p_mat1_0, 3*row_stride1);\
          AE_L8X8X2_IP(_ae_int8x8_mat1_2_0, _ae_int8x8_mat1_3_0,_ae_int8x16_p_mat1_0, 16);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_5, _ae_int8x8_mat1_3_5,_ae_int8x16_p_mat1_4, 1*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_6, _ae_int8x8_mat1_3_6,_ae_int8x16_p_mat1_4, 2*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_7, _ae_int8x8_mat1_3_7,_ae_int8x16_p_mat1_4, 3*row_stride1);\
          AE_L8X8X2_IP(_ae_int8x8_mat1_2_4, _ae_int8x8_mat1_3_4,_ae_int8x16_p_mat1_4, 16);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_9 , _ae_int8x8_mat1_3_9,_ae_int8x16_p_mat1_8, 1*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_10, _ae_int8x8_mat1_3_10,_ae_int8x16_p_mat1_8, 2*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_11, _ae_int8x8_mat1_3_11,_ae_int8x16_p_mat1_8, 3*row_stride1);\
          AE_L8X8X2_IP(_ae_int8x8_mat1_2_8 , _ae_int8x8_mat1_3_8,_ae_int8x16_p_mat1_8, 16);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_13, _ae_int8x8_mat1_3_13,_ae_int8x16_p_mat1_12, 1*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_14, _ae_int8x8_mat1_3_14,_ae_int8x16_p_mat1_12, 2*row_stride1);\
          AE_L8X8X2_X( _ae_int8x8_mat1_2_15, _ae_int8x8_mat1_3_15,_ae_int8x16_p_mat1_12, 3*row_stride1);\
          AE_L8X8X2_IP(_ae_int8x8_mat1_2_12, _ae_int8x8_mat1_3_12,_ae_int8x16_p_mat1_12, 16);\
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_2_0 , _ae_int8x8_mat1_2_1 , _ae_int8x8_mat1_2_2 , _ae_int8x8_mat1_2_3 , _ae_int8x8_vec1_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 , _ae_int8x8_mat1_2_4 , _ae_int8x8_mat1_2_5 , _ae_int8x8_mat1_2_6 , _ae_int8x8_mat1_2_7 , _ae_int8x8_vec1_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4 ,  _ae_int32x2_acc_5 , _ae_int8x8_mat1_2_8 , _ae_int8x8_mat1_2_9 , _ae_int8x8_mat1_2_10, _ae_int8x8_mat1_2_11, _ae_int8x8_vec1_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6 ,  _ae_int32x2_acc_7 , _ae_int8x8_mat1_2_12, _ae_int8x8_mat1_2_13, _ae_int8x8_mat1_2_14, _ae_int8x8_mat1_2_15, _ae_int8x8_vec1_2); \
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_3_0 , _ae_int8x8_mat1_3_1 , _ae_int8x8_mat1_3_2 , _ae_int8x8_mat1_3_3 , _ae_int8x8_vec1_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2,  _ae_int32x2_acc_3, _ae_int8x8_mat1_3_4 , _ae_int8x8_mat1_3_5 , _ae_int8x8_mat1_3_6 , _ae_int8x8_mat1_3_7 , _ae_int8x8_vec1_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4,  _ae_int32x2_acc_5, _ae_int8x8_mat1_3_8 , _ae_int8x8_mat1_3_9 , _ae_int8x8_mat1_3_10, _ae_int8x8_mat1_3_11, _ae_int8x8_vec1_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6,  _ae_int32x2_acc_7, _ae_int8x8_mat1_3_12, _ae_int8x8_mat1_3_13, _ae_int8x8_mat1_3_14, _ae_int8x8_mat1_3_15, _ae_int8x8_vec1_3);

#define kernel_8x8_mat2_vec2 \
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
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat2_0 , _ae_int8x8_mat2_1 , _ae_int8x8_mat2_2 , _ae_int8x8_mat2_3 , _ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 , _ae_int8x8_mat2_4 , _ae_int8x8_mat2_5 , _ae_int8x8_mat2_6 , _ae_int8x8_mat2_7 , _ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4 ,  _ae_int32x2_acc_5 , _ae_int8x8_mat2_8 , _ae_int8x8_mat2_9 , _ae_int8x8_mat2_10, _ae_int8x8_mat2_11, _ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6 ,  _ae_int32x2_acc_7 , _ae_int8x8_mat2_12, _ae_int8x8_mat2_13, _ae_int8x8_mat2_14, _ae_int8x8_mat2_15, _ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat2_1_0 , _ae_int8x8_mat2_1_1 , _ae_int8x8_mat2_1_2 , _ae_int8x8_mat2_1_3 , _ae_int8x8_vec2_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2,  _ae_int32x2_acc_3, _ae_int8x8_mat2_1_4 , _ae_int8x8_mat2_1_5 , _ae_int8x8_mat2_1_6 , _ae_int8x8_mat2_1_7 , _ae_int8x8_vec2_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4,  _ae_int32x2_acc_5, _ae_int8x8_mat2_1_8 , _ae_int8x8_mat2_1_9 , _ae_int8x8_mat2_1_10, _ae_int8x8_mat2_1_11, _ae_int8x8_vec2_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6,  _ae_int32x2_acc_7, _ae_int8x8_mat2_1_12, _ae_int8x8_mat2_1_13, _ae_int8x8_mat2_1_14, _ae_int8x8_mat2_1_15, _ae_int8x8_vec2_1);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_1, _ae_int8x8_mat2_3_1,_ae_int8x16_p_mat2_0, 1*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_2, _ae_int8x8_mat2_3_2,_ae_int8x16_p_mat2_0, 2*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_3, _ae_int8x8_mat2_3_3,_ae_int8x16_p_mat2_0, 3*row_stride2);\
          AE_L8X8X2_IP(_ae_int8x8_mat2_2_0, _ae_int8x8_mat2_3_0,_ae_int8x16_p_mat2_0, 16);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_5, _ae_int8x8_mat2_3_5,_ae_int8x16_p_mat2_4, 1*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_6, _ae_int8x8_mat2_3_6,_ae_int8x16_p_mat2_4, 2*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_7, _ae_int8x8_mat2_3_7,_ae_int8x16_p_mat2_4, 3*row_stride2);\
          AE_L8X8X2_IP(_ae_int8x8_mat2_2_4, _ae_int8x8_mat2_3_4,_ae_int8x16_p_mat2_4, 16);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_9 , _ae_int8x8_mat2_3_9,_ae_int8x16_p_mat2_8, 1*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_10, _ae_int8x8_mat2_3_10,_ae_int8x16_p_mat2_8, 2*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_11, _ae_int8x8_mat2_3_11,_ae_int8x16_p_mat2_8, 3*row_stride2);\
          AE_L8X8X2_IP(_ae_int8x8_mat2_2_8 , _ae_int8x8_mat2_3_8,_ae_int8x16_p_mat2_8, 16);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_13, _ae_int8x8_mat2_3_13,_ae_int8x16_p_mat2_12, 1*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_14, _ae_int8x8_mat2_3_14,_ae_int8x16_p_mat2_12, 2*row_stride2);\
          AE_L8X8X2_X( _ae_int8x8_mat2_2_15, _ae_int8x8_mat2_3_15,_ae_int8x16_p_mat2_12, 3*row_stride2);\
          AE_L8X8X2_IP(_ae_int8x8_mat2_2_12, _ae_int8x8_mat2_3_12,_ae_int8x16_p_mat2_12, 16);\
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat2_2_0 , _ae_int8x8_mat2_2_1 , _ae_int8x8_mat2_2_2 , _ae_int8x8_mat2_2_3 , _ae_int8x8_vec2_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 , _ae_int8x8_mat2_2_4 , _ae_int8x8_mat2_2_5 , _ae_int8x8_mat2_2_6 , _ae_int8x8_mat2_2_7 , _ae_int8x8_vec2_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4 ,  _ae_int32x2_acc_5 , _ae_int8x8_mat2_2_8 , _ae_int8x8_mat2_2_9 , _ae_int8x8_mat2_2_10, _ae_int8x8_mat2_2_11, _ae_int8x8_vec2_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6 ,  _ae_int32x2_acc_7 , _ae_int8x8_mat2_2_12, _ae_int8x8_mat2_2_13, _ae_int8x8_mat2_2_14, _ae_int8x8_mat2_2_15, _ae_int8x8_vec2_2); \
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat2_3_0 , _ae_int8x8_mat2_3_1 , _ae_int8x8_mat2_3_2 , _ae_int8x8_mat2_3_3 , _ae_int8x8_vec2_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2,  _ae_int32x2_acc_3, _ae_int8x8_mat2_3_4 , _ae_int8x8_mat2_3_5 , _ae_int8x8_mat2_3_6 , _ae_int8x8_mat2_3_7 , _ae_int8x8_vec2_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_4,  _ae_int32x2_acc_5, _ae_int8x8_mat2_3_8 , _ae_int8x8_mat2_3_9 , _ae_int8x8_mat2_3_10, _ae_int8x8_mat2_3_11, _ae_int8x8_vec2_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_6,  _ae_int32x2_acc_7, _ae_int8x8_mat2_3_12, _ae_int8x8_mat2_3_13, _ae_int8x8_mat2_3_14, _ae_int8x8_mat2_3_15, _ae_int8x8_vec2_3);\

#else

#if 0

#define ROW_UNROLL_OPT 8

#define SETUP_MAT1_8x8_PTR \
        ae_int8x16 * _ae_int8x16_p_mat1_0  = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_1  = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_2  = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_3  = (ae_int8x16 *) &p_mat1[(m_itr+3)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_4  = (ae_int8x16 *) &p_mat1[(m_itr+4)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_5  = (ae_int8x16 *) &p_mat1[(m_itr+5)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_6  = (ae_int8x16 *) &p_mat1[(m_itr+6)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_7  = (ae_int8x16 *) &p_mat1[(m_itr+7)*row_stride1];\

#define SETUP_MAT2_8x8_PTR \
        ae_int8x16 * _ae_int8x16_p_mat2_0  = (ae_int8x16 *) &p_mat2[(m_itr+0)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_1  = (ae_int8x16 *) &p_mat2[(m_itr+1)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_2  = (ae_int8x16 *) &p_mat2[(m_itr+2)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_3  = (ae_int8x16 *) &p_mat2[(m_itr+3)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_4  = (ae_int8x16 *) &p_mat2[(m_itr+4)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_5  = (ae_int8x16 *) &p_mat2[(m_itr+5)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_6  = (ae_int8x16 *) &p_mat2[(m_itr+6)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_7  = (ae_int8x16 *) &p_mat2[(m_itr+7)*row_stride2];\

#define kernel_8x8_mat1_vec1 \
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_0, _ae_int8x8_mat1_3_0,_ae_int8x16_p_mat1_0, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_1, _ae_int8x8_mat1_3_1,_ae_int8x16_p_mat1_1, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_2, _ae_int8x8_mat1_3_2,_ae_int8x16_p_mat1_2, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_3, _ae_int8x8_mat1_3_3,_ae_int8x16_p_mat1_3, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_4, _ae_int8x8_mat1_3_4,_ae_int8x16_p_mat1_4, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_5, _ae_int8x8_mat1_3_5,_ae_int8x16_p_mat1_5, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_6, _ae_int8x8_mat1_3_6,_ae_int8x16_p_mat1_6, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_7, _ae_int8x8_mat1_3_7,_ae_int8x16_p_mat1_7, 16);\
          \
          AE_L8X8X2_IP( _ae_int8x8_mat1_0, _ae_int8x8_mat1_1_0,_ae_int8x16_p_mat1_0, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_1, _ae_int8x8_mat1_1_1,_ae_int8x16_p_mat1_1, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_2, _ae_int8x8_mat1_1_2,_ae_int8x16_p_mat1_2, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_3, _ae_int8x8_mat1_1_3,_ae_int8x16_p_mat1_3, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_4, _ae_int8x8_mat1_1_4,_ae_int8x16_p_mat1_4, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_5, _ae_int8x8_mat1_1_5,_ae_int8x16_p_mat1_5, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_6, _ae_int8x8_mat1_1_6,_ae_int8x16_p_mat1_6, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_7, _ae_int8x8_mat1_1_7,_ae_int8x16_p_mat1_7, 32);\
          \
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_0  , _ae_int8x8_mat1_1  , _ae_int8x8_mat1_2  , _ae_int8x8_mat1_3  , _ae_int8x8_vec1  );\
          AE_MULA8Q8X8(_ae_int32x2_acc_2,  _ae_int32x2_acc_3 , _ae_int8x8_mat1_4  , _ae_int8x8_mat1_5  , _ae_int8x8_mat1_6  , _ae_int8x8_mat1_7  , _ae_int8x8_vec1  );\
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_1_0, _ae_int8x8_mat1_1_1, _ae_int8x8_mat1_1_2, _ae_int8x8_mat1_1_3, _ae_int8x8_vec1_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2,  _ae_int32x2_acc_3 , _ae_int8x8_mat1_1_4, _ae_int8x8_mat1_1_5, _ae_int8x8_mat1_1_6, _ae_int8x8_mat1_1_7, _ae_int8x8_vec1_1);\
          \
          \
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1, _ae_int8x8_mat1_2_0 , _ae_int8x8_mat1_2_1 , _ae_int8x8_mat1_2_2 , _ae_int8x8_mat1_2_3 , _ae_int8x8_vec1_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2,  _ae_int32x2_acc_3, _ae_int8x8_mat1_2_4 , _ae_int8x8_mat1_2_5 , _ae_int8x8_mat1_2_6 , _ae_int8x8_mat1_2_7 , _ae_int8x8_vec1_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1, _ae_int8x8_mat1_3_0 , _ae_int8x8_mat1_3_1 , _ae_int8x8_mat1_3_2 , _ae_int8x8_mat1_3_3 , _ae_int8x8_vec1_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2,  _ae_int32x2_acc_3, _ae_int8x8_mat1_3_4 , _ae_int8x8_mat1_3_5 , _ae_int8x8_mat1_3_6 , _ae_int8x8_mat1_3_7 , _ae_int8x8_vec1_3);\

#define kernel_8x8_mat2_vec2 \
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_0, _ae_int8x8_mat2_3_0,_ae_int8x16_p_mat2_0, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_1, _ae_int8x8_mat2_3_1,_ae_int8x16_p_mat2_1, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_2, _ae_int8x8_mat2_3_2,_ae_int8x16_p_mat2_2, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_3, _ae_int8x8_mat2_3_3,_ae_int8x16_p_mat2_3, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_4, _ae_int8x8_mat2_3_4,_ae_int8x16_p_mat2_4, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_5, _ae_int8x8_mat2_3_5,_ae_int8x16_p_mat2_5, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_6, _ae_int8x8_mat2_3_6,_ae_int8x16_p_mat2_6, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_7, _ae_int8x8_mat2_3_7,_ae_int8x16_p_mat2_7, 16);\
          \
          AE_L8X8X2_IP( _ae_int8x8_mat2_0, _ae_int8x8_mat2_1_0,_ae_int8x16_p_mat2_0, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_1, _ae_int8x8_mat2_1_1,_ae_int8x16_p_mat2_1, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_2, _ae_int8x8_mat2_1_2,_ae_int8x16_p_mat2_2, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_3, _ae_int8x8_mat2_1_3,_ae_int8x16_p_mat2_3, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_4, _ae_int8x8_mat2_1_4,_ae_int8x16_p_mat2_4, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_5, _ae_int8x8_mat2_1_5,_ae_int8x16_p_mat2_5, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_6, _ae_int8x8_mat2_1_6,_ae_int8x16_p_mat2_6, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_7, _ae_int8x8_mat2_1_7,_ae_int8x16_p_mat2_7, 32);\
          \
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat2_0 , _ae_int8x8_mat2_1 , _ae_int8x8_mat2_2 , _ae_int8x8_mat2_3 , _ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 , _ae_int8x8_mat2_4 , _ae_int8x8_mat2_5 , _ae_int8x8_mat2_6 , _ae_int8x8_mat2_7 , _ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat2_1_0 , _ae_int8x8_mat2_1_1 , _ae_int8x8_mat2_1_2 , _ae_int8x8_mat2_1_3 , _ae_int8x8_vec2_1);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2 ,  _ae_int32x2_acc_3 , _ae_int8x8_mat2_1_4 , _ae_int8x8_mat2_1_5 , _ae_int8x8_mat2_1_6 , _ae_int8x8_mat2_1_7 , _ae_int8x8_vec2_1);\
          \
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1, _ae_int8x8_mat2_2_0 , _ae_int8x8_mat2_2_1 , _ae_int8x8_mat2_2_2 , _ae_int8x8_mat2_2_3 , _ae_int8x8_vec2_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2,  _ae_int32x2_acc_3, _ae_int8x8_mat2_2_4 , _ae_int8x8_mat2_2_5 , _ae_int8x8_mat2_2_6 , _ae_int8x8_mat2_2_7 , _ae_int8x8_vec2_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1, _ae_int8x8_mat2_3_0 , _ae_int8x8_mat2_3_1 , _ae_int8x8_mat2_3_2 , _ae_int8x8_mat2_3_3 , _ae_int8x8_vec2_3);\
          AE_MULA8Q8X8(_ae_int32x2_acc_2,  _ae_int32x2_acc_3, _ae_int8x8_mat2_3_4 , _ae_int8x8_mat2_3_5 , _ae_int8x8_mat2_3_6 , _ae_int8x8_mat2_3_7 , _ae_int8x8_vec2_3);\

#else

#define ROW_UNROLL_OPT 4

#define SETUP_MAT1_8b_16(idx) \
  ae_int8x8 _ae_int8x8_mat1_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat1_1_ ## idx; \
  ae_int8x8 _ae_int8x8_mat1_2_ ## idx; \
  ae_int8x8 _ae_int8x8_mat1_3_ ## idx; \

#define SET_UP_MAT1_16_VARIABLES \
        SETUP_MAT1_8b_16(0) ;SETUP_MAT1_8b_16(1) ;SETUP_MAT1_8b_16(2) ;SETUP_MAT1_8b_16(3);\

#define SETUP_MAT2_8b_16(idx) \
  ae_int8x8 _ae_int8x8_mat2_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat2_1_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat2_2_ ## idx ; \
  ae_int8x8 _ae_int8x8_mat2_3_ ## idx ; \

#define SET_UP_MAT2_16_VARIABLES \
        SETUP_MAT2_8b_16(0) ;SETUP_MAT2_8b_16(1) ;SETUP_MAT2_8b_16(2) ;SETUP_MAT2_8b_16(3);\

#define SETUP_MAT1_8x8_PTR \
        ae_int8x16 * _ae_int8x16_p_mat1_0  = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_1  = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_2  = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];\
        ae_int8x16 * _ae_int8x16_p_mat1_3  = (ae_int8x16 *) &p_mat1[(m_itr+3)*row_stride1];\

#define SETUP_MAT2_8x8_PTR \
        ae_int8x16 * _ae_int8x16_p_mat2_0  = (ae_int8x16 *) &p_mat2[(m_itr+0)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_1  = (ae_int8x16 *) &p_mat2[(m_itr+1)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_2  = (ae_int8x16 *) &p_mat2[(m_itr+2)*row_stride2];\
        ae_int8x16 * _ae_int8x16_p_mat2_3  = (ae_int8x16 *) &p_mat2[(m_itr+3)*row_stride2];\

#define kernel_8x8_mat1_vec1 \
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_0, _ae_int8x8_mat1_3_0,_ae_int8x16_p_mat1_0, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_1, _ae_int8x8_mat1_3_1,_ae_int8x16_p_mat1_1, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_2, _ae_int8x8_mat1_3_2,_ae_int8x16_p_mat1_2, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat1_2_3, _ae_int8x8_mat1_3_3,_ae_int8x16_p_mat1_3, 16);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_0, _ae_int8x8_mat1_1_0,_ae_int8x16_p_mat1_0, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_1, _ae_int8x8_mat1_1_1,_ae_int8x16_p_mat1_1, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_2, _ae_int8x8_mat1_1_2,_ae_int8x16_p_mat1_2, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat1_3, _ae_int8x8_mat1_1_3,_ae_int8x16_p_mat1_3, 32);\
          \
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_0  , _ae_int8x8_mat1_1  , _ae_int8x8_mat1_2  , _ae_int8x8_mat1_3  , _ae_int8x8_vec1  );\
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_1_0, _ae_int8x8_mat1_1_1, _ae_int8x8_mat1_1_2, _ae_int8x8_mat1_1_3, _ae_int8x8_vec1_1);\
          \
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1, _ae_int8x8_mat1_2_0 , _ae_int8x8_mat1_2_1 , _ae_int8x8_mat1_2_2 , _ae_int8x8_mat1_2_3 , _ae_int8x8_vec1_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1, _ae_int8x8_mat1_3_0 , _ae_int8x8_mat1_3_1 , _ae_int8x8_mat1_3_2 , _ae_int8x8_mat1_3_3 , _ae_int8x8_vec1_3);\

#define kernel_8x8_mat2_vec2 \
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_0, _ae_int8x8_mat2_3_0,_ae_int8x16_p_mat2_0, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_1, _ae_int8x8_mat2_3_1,_ae_int8x16_p_mat2_1, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_2, _ae_int8x8_mat2_3_2,_ae_int8x16_p_mat2_2, 16);\
          AE_L8X8X2_I ( _ae_int8x8_mat2_2_3, _ae_int8x8_mat2_3_3,_ae_int8x16_p_mat2_3, 16);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_0, _ae_int8x8_mat2_1_0,_ae_int8x16_p_mat2_0, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_1, _ae_int8x8_mat2_1_1,_ae_int8x16_p_mat2_1, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_2, _ae_int8x8_mat2_1_2,_ae_int8x16_p_mat2_2, 32);\
          AE_L8X8X2_IP( _ae_int8x8_mat2_3, _ae_int8x8_mat2_1_3,_ae_int8x16_p_mat2_3, 32);\
          \
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat2_0 , _ae_int8x8_mat2_1 , _ae_int8x8_mat2_2 , _ae_int8x8_mat2_3 , _ae_int8x8_vec2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat2_1_0 , _ae_int8x8_mat2_1_1 , _ae_int8x8_mat2_1_2 , _ae_int8x8_mat2_1_3 , _ae_int8x8_vec2_1);\
          \
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1, _ae_int8x8_mat2_2_0 , _ae_int8x8_mat2_2_1 , _ae_int8x8_mat2_2_2 , _ae_int8x8_mat2_2_3 , _ae_int8x8_vec2_2);\
          AE_MULA8Q8X8(_ae_int32x2_acc_0,  _ae_int32x2_acc_1, _ae_int8x8_mat2_3_0 , _ae_int8x8_mat2_3_1 , _ae_int8x8_mat2_3_2 , _ae_int8x8_mat2_3_3 , _ae_int8x8_vec2_3);\

#endif

#endif

#define TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8 \
        SETUP_ACC_FOR_8bx8b(0); \
        SETUP_ACC_FOR_8bx8b(1); \
        SETUP_VEC1_8b_UNALIGNED; \
        SETUP_MAT1_8b_UNALIGNED(0);\
        SETUP_MAT1_8b_UNALIGNED(1); \
        SETUP_MAT1_8b_UNALIGNED(2); \
        ZER0_8x8_Temp_Variable;\
        int cols_count=cols1-cols1%8;\
        for(c_itr = 0; c_itr < cols_count>>3; c_itr++)\
        { \
          LOAD_VEC1_8b_UNALIGNED;\
          LOAD_ROW_MAT1_8b_UNALIGNED(0);\
          LOAD_ROW_MAT1_8b_UNALIGNED(1);\
          LOAD_ROW_MAT1_8b_UNALIGNED(2);\
          KERNEL_MAT1_VEC1_8b_8b_UNALIGNED;\
        }\
        for(c_itr = cols_count; c_itr < cols1; c_itr++)\
        { \
          LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT; \
          LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);\
          LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(1);\
          LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(2);\
          KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ELEMENT;\
        }\
        SETUP_VEC2_8b_UNALIGNED;\
        SETUP_MAT2_8b_UNALIGNED(0);\
        SETUP_MAT2_8b_UNALIGNED(1);\
        SETUP_MAT2_8b_UNALIGNED(2);\
        cols_count=cols2-cols2%8;\
        for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)\
        { \
          LOAD_VEC2_8b_UNALIGNED;\
          LOAD_ROW_MAT2_8b_UNALIGNED(0);\
          LOAD_ROW_MAT2_8b_UNALIGNED(1);\
          LOAD_ROW_MAT2_8b_UNALIGNED(2);\
          KERNEL_MAT2_VEC2_8b_8b_UNALIGNED; \
        }\
        for(c_itr = cols_count; c_itr < cols2; c_itr++)\
        { \
          LOAD_VEC2_8b_UNALIGNED_SINGLE_ELEMENT; \
          LOAD_ROW_MAT2_8b_UNALIGNED_SINGLE_ELEMENT(0);\
          LOAD_ROW_MAT2_8b_UNALIGNED_SINGLE_ELEMENT(1);\
          LOAD_ROW_MAT2_8b_UNALIGNED_SINGLE_ELEMENT(2);\
          KERNEL_MAT2_VEC2_8b_8b_UNALIGNED_SINGLE_ELEMENT;\
        }\

#define TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8 \
        SETUP_ACC_FOR_8bx8b(0); \
        SETUP_ACC_FOR_8bx8b(1); \
        SETUP_VEC1_8b_UNALIGNED; \
        SETUP_MAT1_8b_UNALIGNED(0);\
        SETUP_MAT1_8b_UNALIGNED(1); \
        SETUP_MAT1_8b_UNALIGNED(2); \
        ZER0_8x8_Temp_Variable;\
        int cols_count=cols1-cols1%8;\
        for(c_itr = 0; c_itr < cols_count>>3; c_itr++)\
        { \
          LOAD_VEC1_8b_UNALIGNED;\
          LOAD_ROW_MAT1_8b_UNALIGNED(0);\
          LOAD_ROW_MAT1_8b_UNALIGNED(1);\
          LOAD_ROW_MAT1_8b_UNALIGNED(2);\
          KERNEL_MAT1_VEC1_8b_8b_UNALIGNED;\
        }\
        for(c_itr = cols_count; c_itr < cols1; c_itr++)\
        { \
          LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT; \
          LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);\
          LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(1);\
          LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(2);\
          KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ELEMENT;\
        }\

#define TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8_SINGLE_ROW \
        SETUP_ACC_FOR_8bx8b(0); \
        SETUP_ACC_FOR_8bx8b(1); \
        SETUP_VEC1_8b_UNALIGNED;\
        SETUP_MAT1_8b_UNALIGNED(0);\
        ZER0_8x8_Temp_Variable;\
        int cols_count=cols1-cols1%8;\
        for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)\
        { \
          LOAD_VEC1_8b_UNALIGNED;\
          LOAD_ROW_MAT1_8b_UNALIGNED(0);\
          KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW;\
        }\
        for(c_itr = cols_count; c_itr < cols1; c_itr++)\
        { \
          LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT;\
          LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);\
          KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_ELEMENT; \
        }\
        SETUP_VEC2_8b_UNALIGNED;\
        SETUP_MAT2_8b_UNALIGNED(0);\
        cols_count=cols2-cols2%8;\
        for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)\
        { \
          LOAD_VEC2_8b_UNALIGNED;\
          LOAD_ROW_MAT2_8b_UNALIGNED(0);\
          KERNEL_MAT2_VEC2_8b_8b_UNALIGNED_SINGLE_ROW;\
        }\
        for(c_itr = cols_count; c_itr < cols2; c_itr++)\
        {   \
          LOAD_VEC2_8b_UNALIGNED_SINGLE_ELEMENT;\
          LOAD_ROW_MAT2_8b_UNALIGNED_SINGLE_ELEMENT(0);\
          KERNEL_MAT2_VEC2_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_ELEMENT;\
        }\

#define TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8_SINGLE_ROW \
        SETUP_ACC_FOR_8bx8b(0); \
        SETUP_ACC_FOR_8bx8b(1); \
        SETUP_VEC1_8b_UNALIGNED;\
        SETUP_MAT1_8b_UNALIGNED(0);\
        ZER0_8x8_Temp_Variable;\
        int cols_count=cols1-cols1%8;\
        for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)\
        { \
          LOAD_VEC1_8b_UNALIGNED;\
          LOAD_ROW_MAT1_8b_UNALIGNED(0);\
          KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW;\
        }\
        for(c_itr = cols_count; c_itr < cols1; c_itr++)\
        { \
          LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT;\
          LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);\
          KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_ELEMENT; \
        }\


WORD32 xa_nn_matXvec_8x8_8(
         WORD8 * __restrict__ p_out,           /* output */
         WORD8 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD8 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD8 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD8 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         WORD8 * __restrict__ p_bias,          /* bias */
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

#define UNROLL_SETUP_ACC        SETUP_ACC_FOR_8bx8b
#define UNROLL_SETUP_MAT1       SETUP_MAT1_8b
#define UNROLL_SETUP_MAT2       SETUP_MAT2_8b
#define SETUP_VEC1              SETUP_VEC1_8b
#define SETUP_VEC2              SETUP_VEC2_8b
#define LOAD_VEC1               LOAD_VEC1_8b
#define LOAD_VEC2               LOAD_VEC2_8b
#define UNROLL_ADD_BIAS_ACC     ADD_BIAS_8b_ACC_FOR_8bx8b
#define SETUP_BIAS              SETUP_BIAS_8b
  acc_shift=acc_shift+32;
  LIMIT_ACC_LSH

  if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && !((unsigned int)p_mat2 & 15) && !((unsigned int)p_vec2 & 15) && (cols1 %32 ==0 ) && (cols2 %32 ==0 ) && row_stride1%16 ==0 && row_stride2 %16 ==0)
  {
    /* All four pointers are non-null */
    SETUP_BIAS;
    m_itr = 0;
    if(rows > ROW_UNROLL_OPT)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL_OPT-1)); m_itr += ROW_UNROLL_OPT)
      {
        SETUP_OUTPUT_STORE_8x8;
#ifdef UNROLL_16
        SETUP_ACC;
#else
        SETUP_ACC_FOR_8bx8b(0);
        SETUP_ACC_FOR_8bx8b(1);
#endif
        SETUP_VEC1_8b_16;
        SET_UP_MAT1_16_VARIABLES;
        SETUP_MAT1_8x8_PTR;
        for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++)
        {
          LOAD_VEC1_8b_16;
          LOAD_VEC1_8b_16_1;
          kernel_8x8_mat1_vec1;
        }
        SETUP_VEC2_8b_16;
        SET_UP_MAT2_16_VARIABLES;
        SETUP_MAT2_8x8_PTR;
        for(c_itr = 0; c_itr < (cols2 >> 5); c_itr++)
        {
          LOAD_VEC2_8b_16;
          LOAD_VEC2_8b_16_1;
          kernel_8x8_mat2_vec2;
        }
#ifdef UNROLL_16
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        SETUP_ACC_64b(2);
        SETUP_ACC_64b(3);
        SETUP_ACC_64b(4);
        SETUP_ACC_64b(5);
        SETUP_ACC_64b(6);
        SETUP_ACC_64b(7);
        ADD_BIAS_ACC;
        STORE_ACC_8bx8b_AT_OUT_8x8x2;
#else
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        ADD_BIAS_8b_ACC_FOR_8bx8b(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b(1);
        ae_int32x2 temp32_1, temp32_2;
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);
        temp32_1 = AE_SLAI32S(temp32_1,24);
        temp32_1 = AE_SRAI32(temp32_1,24);
        temp32_2 = AE_SLAI32S(temp32_2,24);
        temp32_2 = AE_SRAI32(temp32_2,24);
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_1,temp32_1)),output_ptr,sizeof(ae_int8));
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_1,temp32_1)),output_ptr,sizeof(ae_int8));
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_2,temp32_2)),output_ptr,sizeof(ae_int8));
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_2,temp32_2)),output_ptr,sizeof(ae_int8));
#endif
      }
    }
    {
      for(m_itr = (rows & ~(ROW_UNROLL_OPT-1)); m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_8;
        UNROLL_SETUP_ACC(0);
        SETUP_VEC1;
        UNROLL_SETUP_MAT1(0);
        UNROLL_SETUP_ACC(1);
        ZER0_8x8_Temp_Variable;
        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)
        {
          LOAD_VEC1;
          LOAD_ROW_MAT1_8b(0);
          KERNEL_8x8_NOT_UNROLLED_MAT1_VEC1;

        }
        SETUP_VEC2;
        UNROLL_SETUP_MAT2(0);
        for(c_itr = 0; c_itr < (cols2 >> 4); c_itr++)
        {
          LOAD_VEC2;
          LOAD_ROW_MAT2_8b(0);
          KERNEL_8x8_NOT_UNROLLED_MAT2_VEC2;
        }
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_SINGLE;
        STORE_ACC_8bx8b_AT_OUT_8_SINGLE;
      }
    }
  }
  else if(p_mat1 && p_vec1 && p_mat2 && p_vec2)
  {
    SETUP_BIAS_8b_UNALIGNED_SUPPORT;
    m_itr = 0;
    if(rows > 3)
    {
      int row_count = rows - rows%3;
      for(m_itr = 0; m_itr < row_count; m_itr += 3)
      {
        SETUP_OUTPUT_STORE_8_UNALIGNED_SUPPORT;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8;
        SETUP_ACC_64b_8x16(0);
        SETUP_ACC_64b_8x16(1);
        SETUP_ACC_64b_8x16(2);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_1(1);
        STORE_ACC_8bx8b_AT_OUT_8_SINGLE_UNALIGNED(0);
        STORE_ACC_8bx8b_AT_OUT_8_SINGLE_UNALIGNED(1);
        STORE_ACC_8bx8b_AT_OUT_8_SINGLE_UNALIGNED(2);
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_8_UNALIGNED_SUPPORT;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8_SINGLE_ROW;
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        STORE_ACC_8bx8b_AT_OUT_8_SINGLE_UNALIGNED(0);
      }
    }
  }
  else if (p_mat1 && p_vec1 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && (cols1 %32 ==0) && (row_stride1 %16==0))
  {
    /* Only mat1, vec1 are non-null */
    SETUP_BIAS;
    m_itr = 0;
    if(rows > ROW_UNROLL_SINGLE)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL_SINGLE-1)) ; m_itr += ROW_UNROLL_SINGLE)
      {
        SETUP_OUTPUT_STORE_8x8;
#if 1
        UNROLL_SETUP_ACC(0);UNROLL_SETUP_ACC(1);
        SETUP_VEC1_8b_16;
        SETUP_MAT1_8b_16(0) ;SETUP_MAT1_8b_16(1) ;SETUP_MAT1_8b_16(2) ;SETUP_MAT1_8b_16(3);\

        ae_int8x16 * _ae_int8x16_p_mat1_0  = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];
        ae_int8x16 * _ae_int8x16_p_mat1_1  = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];
        ae_int8x16 * _ae_int8x16_p_mat1_2  = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];
        ae_int8x16 * _ae_int8x16_p_mat1_3 = (ae_int8x16 *) &p_mat1[(m_itr+3)*row_stride1];
#else
        SETUP_ACC;
        SETUP_VEC1_8b_16;
        SET_UP_MAT1_16_VARIABLES;
        SETUP_MAT1_8x8_PTR;
#endif
        for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++)
        {
#if 1
          AE_L8X8X2_I(_ae_int8x8_vec1,_ae_int8x8_vec1_1, _ae_int8x16_p_vec1, 16);
          AE_L8X8X2_IP(_ae_int8x8_vec1_2,_ae_int8x8_vec1_3, _ae_int8x16_p_vec1, 4*INCREMENT_IN_BYTES_FOR_WORD8X8 );

          AE_L8X8X2_I(_ae_int8x8_mat1_0, _ae_int8x8_mat1_1_0,_ae_int8x16_p_mat1_0, 16);
          AE_L8X8X2_I( _ae_int8x8_mat1_1, _ae_int8x8_mat1_1_1,_ae_int8x16_p_mat1_1, 16);
          AE_L8X8X2_I( _ae_int8x8_mat1_2, _ae_int8x8_mat1_1_2,_ae_int8x16_p_mat1_2, 16);
          AE_L8X8X2_I( _ae_int8x8_mat1_3, _ae_int8x8_mat1_1_3,_ae_int8x16_p_mat1_3, 16);
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_0 , _ae_int8x8_mat1_1 , _ae_int8x8_mat1_2 , _ae_int8x8_mat1_3 , _ae_int8x8_vec1);
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_1_0 , _ae_int8x8_mat1_1_1 , _ae_int8x8_mat1_1_2 , _ae_int8x8_mat1_1_3 , _ae_int8x8_vec1_1);

          AE_L8X8X2_IP(_ae_int8x8_mat1_2_0, _ae_int8x8_mat1_3_0,_ae_int8x16_p_mat1_0, 4*INCREMENT_IN_BYTES_FOR_WORD8X8);
          AE_L8X8X2_IP( _ae_int8x8_mat1_2_1, _ae_int8x8_mat1_3_1,_ae_int8x16_p_mat1_1, 4*INCREMENT_IN_BYTES_FOR_WORD8X8);
          AE_L8X8X2_IP( _ae_int8x8_mat1_2_2, _ae_int8x8_mat1_3_2,_ae_int8x16_p_mat1_2, 4*INCREMENT_IN_BYTES_FOR_WORD8X8);
          AE_L8X8X2_IP( _ae_int8x8_mat1_2_3, _ae_int8x8_mat1_3_3,_ae_int8x16_p_mat1_3, 4*INCREMENT_IN_BYTES_FOR_WORD8X8);
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_2_0 , _ae_int8x8_mat1_2_1 , _ae_int8x8_mat1_2_2 , _ae_int8x8_mat1_2_3 , _ae_int8x8_vec1_2);
          AE_MULA8Q8X8(_ae_int32x2_acc_0 ,  _ae_int32x2_acc_1 , _ae_int8x8_mat1_3_0 , _ae_int8x8_mat1_3_1 , _ae_int8x8_mat1_3_2 , _ae_int8x8_mat1_3_3 , _ae_int8x8_vec1_3);
#else
          LOAD_VEC1_8b_16;
          LOAD_VEC1_8b_16_1;
          kernel_8x8_mat1_vec1;
#endif
        }
#if 1
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);

        UNROLL_ADD_BIAS_ACC(0);UNROLL_ADD_BIAS_ACC(1);

        ae_int32x2 temp32_1, temp32_2;
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);

        temp32_1 = AE_SLAI32S(temp32_1,24);
        temp32_1 = AE_SRAI32(temp32_1,24);
        temp32_2 = AE_SLAI32S(temp32_2,24);
        temp32_2 = AE_SRAI32(temp32_2,24);
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_1,temp32_1)),output_ptr,sizeof(ae_int8));
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_1,temp32_1)),output_ptr,sizeof(ae_int8));
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_HH(temp32_2,temp32_2)),output_ptr,sizeof(ae_int8));
        AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(AE_SEL32_LL(temp32_2,temp32_2)),output_ptr,sizeof(ae_int8));
#else
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        SETUP_ACC_64b(2);
        SETUP_ACC_64b(3);
        SETUP_ACC_64b(4);
        SETUP_ACC_64b(5);
        SETUP_ACC_64b(6);
        SETUP_ACC_64b(7);
        ADD_BIAS_ACC;
        STORE_ACC_8bx8b_AT_OUT_8x8x2;
#endif
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_8;
        UNROLL_SETUP_ACC(0);
        UNROLL_SETUP_ACC(1);
        SETUP_VEC1;
        UNROLL_SETUP_MAT1(0);
        ZER0_8x8_Temp_Variable;
        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)
        {
          LOAD_VEC1;
          LOAD_ROW_MAT1_8b(0);
          KERNEL_8x8_NOT_UNROLLED_MAT1_VEC1;
        }
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_SINGLE;
        STORE_ACC_8bx8b_AT_OUT_8_SINGLE;
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
    SETUP_BIAS_8b_UNALIGNED_SUPPORT;
    m_itr = 0;
    if(rows > 3)
    {
      int row_count = rows - rows%3;
      for(m_itr = 0; m_itr < row_count ; m_itr += 3)
      {
        SETUP_OUTPUT_STORE_8_UNALIGNED_SUPPORT;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8;
        SETUP_ACC_64b_8x16(0);
        SETUP_ACC_64b_8x16(1);
        SETUP_ACC_64b_8x16(2);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_1(1);
        STORE_ACC_8bx8b_AT_OUT_8_SINGLE_UNALIGNED(0);
        STORE_ACC_8bx8b_AT_OUT_8_SINGLE_UNALIGNED(1);
        STORE_ACC_8bx8b_AT_OUT_8_SINGLE_UNALIGNED(2);
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_8_UNALIGNED_SUPPORT;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8_SINGLE_ROW;
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        STORE_ACC_8bx8b_AT_OUT_8_SINGLE_UNALIGNED(0);
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
#undef SETUP_VEC1
#undef SETUP_VEC2
#undef UNROLL_LOAD_MAT1
#undef UNROLL_LOAD_MAT2
#undef UNROLL_ADD_BIAS_ACC
#undef LOAD_VEC1
#undef LOAD_VEC2
#undef SETUP_BIAS
  return 0;
}
WORD32 xa_nn_matXvec_8x8_16(
         WORD16 * __restrict__ p_out,           /* output */
         WORD8 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD8 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD8 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD8 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         WORD8 * __restrict__ p_bias,          /* bias */
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

#define UNROLL_SETUP_ACC        SETUP_ACC_FOR_8bx8b
#define UNROLL_SETUP_MAT1       SETUP_MAT1_8b
#define UNROLL_SETUP_MAT2       SETUP_MAT2_8b
#define SETUP_VEC1              SETUP_VEC1_8b
#define SETUP_VEC2              SETUP_VEC2_8b
#define LOAD_VEC1               LOAD_VEC1_8b
#define LOAD_VEC2               LOAD_VEC2_8b
#define SETUP_BIAS              SETUP_BIAS_8b

  acc_shift=acc_shift+32;
  LIMIT_ACC_LSH
  if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && !((unsigned int)p_mat2 & 15) && !((unsigned int)p_vec2 & 15) && (cols1 %32 ==0 ) && (cols2 %32 ==0 ) && row_stride1%16 ==0 && row_stride2 %16 ==0)
  {
    /* All four pointers are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL_OPT)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL_OPT-1)); m_itr += ROW_UNROLL_OPT)
      {
        SETUP_OUTPUT_STORE_16x4;
#ifdef UNROLL_16
        SETUP_ACC;
#else
       SETUP_ACC_FOR_8bx8b(0);
       SETUP_ACC_FOR_8bx8b(1);
#endif
        SETUP_VEC1_8b_16;
        SET_UP_MAT1_16_VARIABLES;
        SETUP_MAT1_8x8_PTR;
        for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++)
        {
          LOAD_VEC1_8b_16;
          LOAD_VEC1_8b_16_1;
          kernel_8x8_mat1_vec1;
        }
        SETUP_VEC2_8b_16;
        SET_UP_MAT2_16_VARIABLES;
        SETUP_MAT2_8x8_PTR;
        for(c_itr = 0; c_itr < (cols2 >> 5); c_itr++)
        {
          LOAD_VEC2_8b_16;
          LOAD_VEC2_8b_16_1;
          kernel_8x8_mat2_vec2;
        }
#ifdef UNROLL_16
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        SETUP_ACC_64b(2);
        SETUP_ACC_64b(3);
        SETUP_ACC_64b(4);
        SETUP_ACC_64b(5);
        SETUP_ACC_64b(6);
        SETUP_ACC_64b(7);
        ADD_BIAS_ACC;
        STORE_ACC_8bx8b_AT_OUT_16x4x2;
#else
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        ADD_BIAS_8b_ACC_FOR_8bx8b(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b(1);
        ae_int32x2 temp32_1, temp32_2;
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);
        temp32_1 = AE_SLAI32S(temp32_1, 16);
        temp32_1 = AE_SRAI32(temp32_1,16);
        temp32_2 = AE_SLAI32S(temp32_2, 16);
        temp32_2 = AE_SRAI32(temp32_2,16);
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_1,temp32_1)),output_ptr,sizeof(ae_int16));
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_1,temp32_1)),output_ptr,sizeof(ae_int16));
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_2,temp32_2)),output_ptr,sizeof(ae_int16));
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_2,temp32_2)),output_ptr,sizeof(ae_int16));

#endif
      }
    }
    {
      for(m_itr = (rows & ~(ROW_UNROLL_OPT-1)); m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_16;
        UNROLL_SETUP_ACC(0);
        UNROLL_SETUP_ACC(1);
        SETUP_VEC1;
        UNROLL_SETUP_MAT1(0);
        ZER0_8x8_Temp_Variable;
        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)
        {
          LOAD_VEC1;
          LOAD_ROW_MAT1_8b(0);
          KERNEL_8x8_NOT_UNROLLED_MAT1_VEC1;
        }
        SETUP_VEC2; UNROLL_SETUP_MAT2(0);
        for(c_itr = 0; c_itr < (cols2 >> 4); c_itr++)
        {
          LOAD_VEC2;
          LOAD_ROW_MAT2_8b(0);
          KERNEL_8x8_NOT_UNROLLED_MAT2_VEC2;
        }
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_SINGLE;
        STORE_ACC_8bx8b_AT_OUT_16_SINGLE;
      }
    }
  }
  else if(p_mat1 && p_vec1 && p_mat2 && p_vec2)
  {
    SETUP_BIAS_8b_UNALIGNED_SUPPORT;
    if(rows > 3)
    {
      int row_count = rows - rows%3;
      for(m_itr = 0; m_itr < row_count; m_itr += 3)
      {
        SETUP_OUTPUT_STORE_16_UNALIGNED_SUPPORT;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8;
        SETUP_ACC_64b_8x16(0);
        SETUP_ACC_64b_8x16(1);
        SETUP_ACC_64b_8x16(2);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_1(1);
        STORE_ACC_8bx8b_AT_OUT_16_SINGLE_UNALIGNED(0);
        STORE_ACC_8bx8b_AT_OUT_16_SINGLE_UNALIGNED(1);
        STORE_ACC_8bx8b_AT_OUT_16_SINGLE_UNALIGNED(2);
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_16_UNALIGNED_SUPPORT;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8_SINGLE_ROW;
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        STORE_ACC_8bx8b_AT_OUT_16_SINGLE_UNALIGNED(0);
      }
    }
  }
  else if (p_mat1 && p_vec1 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && ((unsigned int)cols1%32==0) && (row_stride1%16 ==0))
  {
    /* Only mat1, vec1 are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL_OPT)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL_OPT-1)) ; m_itr += ROW_UNROLL_OPT)
      {
        SETUP_OUTPUT_STORE_16x4;
#ifdef UNROLL_16
        SETUP_ACC;
#else
       SETUP_ACC_FOR_8bx8b(0);
       SETUP_ACC_FOR_8bx8b(1);
#endif
        SETUP_VEC1_8b_16;
        SET_UP_MAT1_16_VARIABLES;
        SETUP_MAT1_8x8_PTR;
        for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++)
        {
          LOAD_VEC1_8b_16;
          LOAD_VEC1_8b_16_1;
          kernel_8x8_mat1_vec1;
        }
#ifdef UNROLL_16
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        SETUP_ACC_64b(2);
        SETUP_ACC_64b(3);
        SETUP_ACC_64b(4);
        SETUP_ACC_64b(5);
        SETUP_ACC_64b(6);
        SETUP_ACC_64b(7);
        ADD_BIAS_ACC;
        STORE_ACC_8bx8b_AT_OUT_16x4x2;
#else
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        ADD_BIAS_8b_ACC_FOR_8bx8b(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b(1);
        ae_int32x2 temp32_1, temp32_2;
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);
        temp32_1 = AE_SLAI32S(temp32_1, 16);
        temp32_1 = AE_SRAI32(temp32_1,16);
        temp32_2 = AE_SLAI32S(temp32_2, 16);
        temp32_2 = AE_SRAI32(temp32_2,16);
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_1,temp32_1)),output_ptr,sizeof(ae_int16));
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_1,temp32_1)),output_ptr,sizeof(ae_int16));
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_HH(temp32_2,temp32_2)),output_ptr,sizeof(ae_int16));
        AE_S16_0_IP(AE_MOVINT16X4_FROMINT32X2(AE_SEL32_LL(temp32_2,temp32_2)),output_ptr,sizeof(ae_int16));

#endif
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_16;
        UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
        UNROLL_SETUP_ACC(1);
        ZER0_8x8_Temp_Variable;
        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)
        {
          LOAD_VEC1;
          LOAD_ROW_MAT1_8b(0);
          KERNEL_8x8_NOT_UNROLLED_MAT1_VEC1;
        }
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_SINGLE;
        STORE_ACC_8bx8b_AT_OUT_16_SINGLE;
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
    SETUP_BIAS_8b_UNALIGNED_SUPPORT;
    if(rows > 3)
    {
      int row_count = rows - rows%3;
      for(m_itr = 0; m_itr < row_count; m_itr += 3)
      {
        SETUP_OUTPUT_STORE_16_UNALIGNED_SUPPORT;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8;
        SETUP_ACC_64b_8x16(0);
        SETUP_ACC_64b_8x16(1);
        SETUP_ACC_64b_8x16(2);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_1(1);
        STORE_ACC_8bx8b_AT_OUT_16_SINGLE_UNALIGNED(0);
        STORE_ACC_8bx8b_AT_OUT_16_SINGLE_UNALIGNED(1);
        STORE_ACC_8bx8b_AT_OUT_16_SINGLE_UNALIGNED(2);
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_16_UNALIGNED_SUPPORT;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8_SINGLE_ROW;
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        STORE_ACC_8bx8b_AT_OUT_16_SINGLE_UNALIGNED(0);
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
#undef SETUP_VEC1
#undef UNROLL_LOAD_MAT1
#undef UNROLL_LOAD_MAT2
#undef SETUP_VEC2
#undef LOAD_VEC1
#undef UNROLL_ADD_BIAS_ACC
#undef LOAD_VEC2
#undef SETUP_BIAS
  return 0;
}

WORD32 xa_nn_matXvec_8x8_32(
         WORD32 * __restrict__ p_out,           /* output */
         WORD8 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD8 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD8 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD8 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         WORD8 * __restrict__ p_bias,          /* bias */
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

#define UNROLL_SETUP_ACC        SETUP_ACC_FOR_8bx8b
#define UNROLL_SETUP_MAT1       SETUP_MAT1_8b
#define UNROLL_SETUP_MAT2       SETUP_MAT2_8b
#define SETUP_VEC1              SETUP_VEC1_8b
#define SETUP_VEC2              SETUP_VEC2_8b
#define LOAD_VEC1               LOAD_VEC1_8b
#define LOAD_VEC2               LOAD_VEC2_8b
#define SETUP_BIAS              SETUP_BIAS_8b

  acc_shift=acc_shift+32;
  LIMIT_ACC_LSH

  if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && !((unsigned int)p_mat2 & 15) && !((unsigned int)p_vec2 & 15) && (cols1 %32 ==0 ) && (cols2 %32 ==0 ) && row_stride1%16 ==0 && row_stride2 %16 ==0)
  {
    /* All four pointers are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL_OPT)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL_OPT-1)); m_itr += ROW_UNROLL_OPT)
      {
        SETUP_OUTPUT_STORE_32x4;
#ifdef UNROLL_16
        SETUP_ACC;
#else
       SETUP_ACC_FOR_8bx8b(0);
       SETUP_ACC_FOR_8bx8b(1);
#endif
        SETUP_VEC1_8b_16;
        SET_UP_MAT1_16_VARIABLES;
        SETUP_MAT1_8x8_PTR;
        for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++)
        {
          LOAD_VEC1_8b_16;
          LOAD_VEC1_8b_16_1;
          kernel_8x8_mat1_vec1;
        }
        SETUP_VEC2_8b_16;
        SET_UP_MAT2_16_VARIABLES;
        SETUP_MAT2_8x8_PTR;
        for(c_itr = 0; c_itr < (cols2 >> 5); c_itr++)
        {
          LOAD_VEC2_8b_16;
          LOAD_VEC2_8b_16_1;
          kernel_8x8_mat2_vec2;
        }
#ifdef UNROLL_16
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        SETUP_ACC_64b(2);
        SETUP_ACC_64b(3);
        SETUP_ACC_64b(4);
        SETUP_ACC_64b(5);
        SETUP_ACC_64b(6);
        SETUP_ACC_64b(7);
        ADD_BIAS_ACC;
        STORE_ACC_8bx8b_AT_OUT_32x4x2;
#else
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        ADD_BIAS_8b_ACC_FOR_8bx8b(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b(1);
        ae_int32x2 temp32_1, temp32_2;
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);
        AE_S32X2X2_IP(temp32_1,temp32_2, output_ptr, sizeof(ae_int32x4));
#endif
      }
    }
    {
      for(m_itr = (rows & ~(ROW_UNROLL_OPT-1)); m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_32;
        UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
        UNROLL_SETUP_ACC(1);
        ZER0_8x8_Temp_Variable;
        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)
        {
          LOAD_VEC1;LOAD_ROW_MAT1_8b(0);
          KERNEL_8x8_NOT_UNROLLED_MAT1_VEC1;
        }
        SETUP_VEC2; UNROLL_SETUP_MAT2(0);
        for(c_itr = 0; c_itr < (cols2 >> 4); c_itr++)
        {
          LOAD_VEC2; LOAD_ROW_MAT2_8b(0);
          KERNEL_8x8_NOT_UNROLLED_MAT2_VEC2;
        }
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_SINGLE;
        STORE_ACC_8bx8b_AT_OUT_32_SINGLE;
      }
    }
  }
  else if(p_mat1 && p_vec1 && p_mat2 && p_vec2)
  {
    SETUP_BIAS_8b_UNALIGNED_SUPPORT;
    if(rows > 2)
    {
      int row_count = rows - rows%3;
      for(m_itr = 0; m_itr < row_count; m_itr += 3)
      {
        SETUP_OUTPUT_STORE_32;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8;
        SETUP_ACC_64b_8x16(0);
        SETUP_ACC_64b_8x16(1);
        SETUP_ACC_64b_8x16(2);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_1(1);
        STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
        STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(1);
        STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(2);
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_32;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8_SINGLE_ROW;
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
      }
    }
  }
  else if (p_mat1 && p_vec1 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && (cols1 % 32 == 0) && (row_stride1 %16 ==0))
  {
    /* Only mat1, vec1 are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL_OPT)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL_OPT-1)) ; m_itr += ROW_UNROLL_OPT)
      {
        SETUP_OUTPUT_STORE_32x4;
#ifdef UNROLL_16
        SETUP_ACC;
#else
       SETUP_ACC_FOR_8bx8b(0);
       SETUP_ACC_FOR_8bx8b(1);
#endif
        SETUP_VEC1_8b_16;
        SET_UP_MAT1_16_VARIABLES;
        SETUP_MAT1_8x8_PTR;
        for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++)
        {
          LOAD_VEC1_8b_16;
          LOAD_VEC1_8b_16_1;
          kernel_8x8_mat1_vec1;
        }
#ifdef UNROLL_16
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        SETUP_ACC_64b(2);
        SETUP_ACC_64b(3);
        SETUP_ACC_64b(4);
        SETUP_ACC_64b(5);
        SETUP_ACC_64b(6);
        SETUP_ACC_64b(7);
        ADD_BIAS_ACC;
        STORE_ACC_8bx8b_AT_OUT_32x4x2;
#else
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        ADD_BIAS_8b_ACC_FOR_8bx8b(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b(1);
        ae_int32x2 temp32_1, temp32_2;
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);
        AE_S32X2X2_IP(temp32_1,temp32_2, output_ptr, sizeof(ae_int32x4));
#endif
      }
    }
    {
      for(m_itr = (rows & ~(ROW_UNROLL_OPT-1)); m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_32;
        UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
        UNROLL_SETUP_ACC(1);
        ZER0_8x8_Temp_Variable;
        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)
        {
          LOAD_VEC1;LOAD_ROW_MAT1_8b(0);
          KERNEL_8x8_NOT_UNROLLED_MAT1_VEC1;
        }
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_SINGLE;
        STORE_ACC_8bx8b_AT_OUT_32_SINGLE;
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
    SETUP_BIAS_8b_UNALIGNED_SUPPORT;
    if(rows > 3)
    {
      int row_count = rows - rows%3;
      for(m_itr = 0; m_itr < row_count; m_itr += 3)
      {
        SETUP_OUTPUT_STORE_32;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8;
        SETUP_ACC_64b_8x16(0);
        SETUP_ACC_64b_8x16(1);
        SETUP_ACC_64b_8x16(2);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_1(1);
        STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
        STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(1);
        STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(2);
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_32;
        TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8_SINGLE_ROW;
        SETUP_ACC_64b_8x16(0);
        ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT(0);
        STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
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
#undef UNROLL_LOAD_MAT1
#undef UNROLL_LOAD_MAT2
#undef SETUP_VEC1
#undef SETUP_VEC2
#undef LOAD_VEC1
#undef LOAD_VEC2
#undef UNROLL_ADD_BIAS_ACC
#undef SETUP_BIAS
  return 0;
}

WORD32 xa_nn_matXvec_8x8_8_tanh(
         WORD8 * __restrict__ p_out,      /* output */
         WORD8 * __restrict__ p_mat1,     /* matrix1: rows x cols1 */
         WORD8 * __restrict__ p_mat2,     /* matrix2: rows x cols2 */
         WORD8 * __restrict__ p_vec1,     /* vec1: cols1 x 1 */
         WORD8 * __restrict__ p_vec2,     /* vec2: cols2 x 1 */
         VOID   * __restrict__ p_bias,    /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,              /* row stride for matrix1 */
         WORD32 row_stride2,              /* row stride for matrix2 */
         WORD32 acc_shift,                  /* out accumulator shift amount */
         WORD32 bias_shift,                 /* bias shift amount */
         WORD32 bias_precision,           /* 8 or 32 */
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
    case 8:
      {
        err = xa_nn_matXvec_8x8_32(
        ((WORD32 *)p_scratch),       /* output stored in scratch*/
        p_mat1,          /* matrix1: rows x cols1 */
        p_mat2,          /* matrix2: rows x cols2 */
        p_vec1,          /* vec1: cols1 x 1 */
        p_vec2,          /* vec2: cols2 x 1 */
        ((WORD8 *)p_bias),          /* bias */
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
    case 32:
      {
#define UNROLL_SETUP_ACC        SETUP_ACC_FOR_8bx8b
#define UNROLL_SETUP_MAT1       SETUP_MAT1_8b
#define UNROLL_SETUP_MAT2       SETUP_MAT2_8b
#define SETUP_VEC1              SETUP_VEC1_8b
#define SETUP_VEC2              SETUP_VEC2_8b
#define LOAD_VEC1               LOAD_VEC1_8b
#define LOAD_VEC2               LOAD_VEC2_8b
#define SETUP_BIAS              SETUP_BIAS_32b
        acc_shift=acc_shift+32;
        LIMIT_ACC_LSH
        if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && !((unsigned int)p_mat2 & 15) && !((unsigned int)p_vec2 & 15) && (cols1 %32 ==0 ) && (cols2 %32 ==0 ) && row_stride1%16 ==0 && row_stride2 %16 ==0)
        {
            /* All four pointers are non-null */
            SETUP_BIAS;
            if(rows > ROW_UNROLL_OPT)
            {
              for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL_OPT-1)); m_itr += ROW_UNROLL_OPT)
              {
                SETUP_OUTPUT_STORE_32x4_SCRATCH;
#ifdef UNROLL_16
        SETUP_ACC;
#else
       SETUP_ACC_FOR_8bx8b(0);
       SETUP_ACC_FOR_8bx8b(1);
#endif
                SETUP_VEC1_8b_16;
                SET_UP_MAT1_16_VARIABLES;
                SETUP_MAT1_8x8_PTR;
                for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++)
                {
                  LOAD_VEC1_8b_16;
                  LOAD_VEC1_8b_16_1;
                  kernel_8x8_mat1_vec1;
                }
                SETUP_VEC2_8b_16;
                SET_UP_MAT2_16_VARIABLES;
                SETUP_MAT2_8x8_PTR;
                for(c_itr = 0; c_itr < (cols2 >> 5); c_itr++)
                {
                  LOAD_VEC2_8b_16;
                  LOAD_VEC2_8b_16_1;
                  kernel_8x8_mat2_vec2;
                }
#ifdef UNROLL_16
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        SETUP_ACC_64b(2);
        SETUP_ACC_64b(3);
        SETUP_ACC_64b(4);
        SETUP_ACC_64b(5);
        SETUP_ACC_64b(6);
        SETUP_ACC_64b(7);
        ADD_BIAS_ACC;
        STORE_ACC_8bx8b_AT_OUT_32x4x2;
#else
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        ADD_BIAS_32b_ACC_FOR_8bx8b(0);
        ADD_BIAS_32b_ACC_FOR_8bx8b(1);
        ae_int32x2 temp32_1, temp32_2;
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);
        AE_S32X2X2_IP(temp32_1,temp32_2, output_ptr, sizeof(ae_int32x4));
#endif

              }
            }
            {
              //SETUP_BIAS_32b_SINGLE;
              for(m_itr = (rows & ~(ROW_UNROLL_OPT-1)); m_itr < rows; m_itr++)
              {
                SETUP_OUTPUT_STORE_32_SCRATCH;
                UNROLL_SETUP_ACC(0);
                SETUP_VEC1;
                UNROLL_SETUP_MAT1(0);
                UNROLL_SETUP_ACC(1);
                ZER0_8x8_Temp_Variable;
                for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)
                {
                  LOAD_VEC1;
                  LOAD_ROW_MAT1_8b(0);
                  KERNEL_8x8_NOT_UNROLLED_MAT1_VEC1;

                }
                SETUP_VEC2;
                UNROLL_SETUP_MAT2(0);
                for(c_itr = 0; c_itr < (cols2 >> 4); c_itr++)
                {
                  LOAD_VEC2;
                  LOAD_ROW_MAT2_8b(0);
                  KERNEL_8x8_NOT_UNROLLED_MAT2_VEC2;
                }
                SETUP_ACC_64b_8x16(0);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE_EXTRA_ROW(0);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE;
              }
            }

        }
          else if(p_mat1 && p_vec1 && p_mat2 && p_vec2)
          {
            SETUP_BIAS_32b_SINGLE;
            if(rows > 3)
            {
              int row_count = rows - rows%3;
              for(m_itr = 0; m_itr < row_count; m_itr += 3)
              {
                SETUP_OUTPUT_STORE_32_SCRATCH;
                TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8;
                SETUP_ACC_64b_8x16(0);
                SETUP_ACC_64b_8x16(1);
                SETUP_ACC_64b_8x16(2);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE(0);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE_1(1);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(1);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(2);
              }
            }
            {
              for(; m_itr < rows; m_itr++)
              {
                SETUP_OUTPUT_STORE_32_SCRATCH;
                TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8_SINGLE_ROW;
                SETUP_ACC_64b_8x16(0);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE(0);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
              }
            }
          }
        else if (p_mat1 && p_vec1 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && cols1%32 ==0 && row_stride1 % 16 == 0)
        {
          /* Only mat1, vec1 are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL_OPT-1)) ; m_itr += ROW_UNROLL_OPT)
            {
              SETUP_OUTPUT_STORE_32x4_SCRATCH;
#ifdef UNROLL_16
        SETUP_ACC;
#else
       SETUP_ACC_FOR_8bx8b(0);
       SETUP_ACC_FOR_8bx8b(1);
#endif
              SETUP_VEC1_8b_16;
              SET_UP_MAT1_16_VARIABLES;
              SETUP_MAT1_8x8_PTR;
              for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++)
              {
                  LOAD_VEC1_8b_16;
                  LOAD_VEC1_8b_16_1;
                  kernel_8x8_mat1_vec1;
              }
#ifdef UNROLL_16
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        SETUP_ACC_64b(2);
        SETUP_ACC_64b(3);
        SETUP_ACC_64b(4);
        SETUP_ACC_64b(5);
        SETUP_ACC_64b(6);
        SETUP_ACC_64b(7);
        ADD_BIAS_ACC;
        STORE_ACC_8bx8b_AT_OUT_32x4x2;
#else
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        ADD_BIAS_32b_ACC_FOR_8bx8b(0);
        ADD_BIAS_32b_ACC_FOR_8bx8b(1);
        ae_int32x2 temp32_1, temp32_2;
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);
        AE_S32X2X2_IP(temp32_1,temp32_2, output_ptr, sizeof(ae_int32x4));
#endif
            }
          }
          {
            SETUP_BIAS_32b_SINGLE;
            for(m_itr = (rows & ~(ROW_UNROLL-1)); m_itr < rows; m_itr++)
            {
              SETUP_OUTPUT_STORE_32_SCRATCH;
              UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
              UNROLL_SETUP_ACC(1);
              ZER0_8x8_Temp_Variable;
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                  LOAD_VEC1;
                  LOAD_ROW_MAT1_8b(0);
                  KERNEL_8x8_NOT_UNROLLED_MAT1_VEC1;
              }
              SETUP_ACC_64b_8x16(0);
              ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE(0);
              STORE_ACC_8bx8b_AT_OUT_32_SINGLE;
            }
          }
        }
          else if(p_mat1 && p_vec1)
          {
            SETUP_BIAS_32b_SINGLE;
            if(rows > 3)
            {
              int row_count = rows - rows%3;
              for(m_itr = 0; m_itr < row_count; m_itr += 3)
              {
                SETUP_OUTPUT_STORE_32_SCRATCH;
                TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8;
                SETUP_ACC_64b_8x16(0);
                SETUP_ACC_64b_8x16(1);
                SETUP_ACC_64b_8x16(2);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE(0);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE_1(1);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(1);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(2);
              }
            }
            {
              for(; m_itr < rows; m_itr++)
              {
                SETUP_OUTPUT_STORE_32_SCRATCH;
                TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8_SINGLE_ROW;
                SETUP_ACC_64b_8x16(0);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE(0);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
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
  }

#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef SETUP_VEC1
#undef SETUP_VEC2
#undef UNROLL_LOAD_MAT1
#undef UNROLL_LOAD_MAT2
#undef LOAD_VEC1
#undef LOAD_VEC2
  xa_nn_vec_tanh_32_8((pWORD8) p_out, (pWORD32) p_scratch, rows);

  return 0;
}

WORD32 xa_nn_matXvec_8x8_8_sigmoid(
         WORD8 * __restrict__ p_out,      /* output */
         WORD8 * __restrict__ p_mat1,     /* matrix1: rows x cols1 */
         WORD8 * __restrict__ p_mat2,     /* matrix2: rows x cols2 */
         WORD8 * __restrict__ p_vec1,     /* vec1: cols1 x 1 */
         WORD8 * __restrict__ p_vec2,     /* vec2: cols2 x 1 */
         VOID   * __restrict__ p_bias,    /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,              /* row stride for matrix1 */
         WORD32 row_stride2,              /* row stride for matrix2 */
         WORD32 acc_shift,                  /* out accumulator shift amount */
         WORD32 bias_shift,                 /* bias shift amount */
         WORD32 bias_precision,           /* 8 or 32 */
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
    case 8:
      {
        err = xa_nn_matXvec_8x8_32(
        ((WORD32 *)p_scratch),       /* output stored in scratch*/
        p_mat1,          /* matrix1: rows x cols1 */
        p_mat2,          /* matrix2: rows x cols2 */
        p_vec1,          /* vec1: cols1 x 1 */
        p_vec2,          /* vec2: cols2 x 1 */
        ((WORD8 *)p_bias),          /* bias */
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
    case 32:
      {
#define UNROLL_SETUP_ACC        SETUP_ACC_FOR_8bx8b
#define UNROLL_SETUP_MAT1       SETUP_MAT1_8b
#define UNROLL_SETUP_MAT2       SETUP_MAT2_8b
#define SETUP_VEC1              SETUP_VEC1_8b
#define SETUP_VEC2              SETUP_VEC2_8b
#define LOAD_VEC1               LOAD_VEC1_8b
#define LOAD_VEC2               LOAD_VEC2_8b
#define SETUP_BIAS              SETUP_BIAS_32b
        acc_shift=acc_shift+32;
        LIMIT_ACC_LSH
        if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && !((unsigned int)p_mat2 & 15) && !((unsigned int)p_vec2 & 15) && (cols1 %32 ==0 ) && (cols2 %32 ==0 ) && row_stride1%16 ==0 && row_stride2 %16 ==0)
        {
            /* All four pointers are non-null */
            //printf("Running inside\n");
            SETUP_BIAS;
            if(rows > ROW_UNROLL_OPT)
            {
              for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL_OPT-1)); m_itr += ROW_UNROLL_OPT)
              {
                SETUP_OUTPUT_STORE_32x4_SCRATCH;
#ifdef UNROLL_16
        SETUP_ACC;
#else
       SETUP_ACC_FOR_8bx8b(0);
       SETUP_ACC_FOR_8bx8b(1);
#endif
                SETUP_VEC1_8b_16;
                SET_UP_MAT1_16_VARIABLES;
                SETUP_MAT1_8x8_PTR;
                for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++)
                {
                  LOAD_VEC1_8b_16;
                  LOAD_VEC1_8b_16_1;
                  kernel_8x8_mat1_vec1;
                }
                SETUP_VEC2_8b_16;
                SET_UP_MAT2_16_VARIABLES;
                SETUP_MAT2_8x8_PTR;
                for(c_itr = 0; c_itr < (cols2 >> 5); c_itr++)
                {
                  LOAD_VEC2_8b_16;
                  LOAD_VEC2_8b_16_1;
                  kernel_8x8_mat2_vec2;
                }
#ifdef UNROLL_16
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        SETUP_ACC_64b(2);
        SETUP_ACC_64b(3);
        SETUP_ACC_64b(4);
        SETUP_ACC_64b(5);
        SETUP_ACC_64b(6);
        SETUP_ACC_64b(7);
        ADD_BIAS_ACC;
        STORE_ACC_8bx8b_AT_OUT_32x4x2;
#else
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        ADD_BIAS_32b_ACC_FOR_8bx8b(0);
        ADD_BIAS_32b_ACC_FOR_8bx8b(1);
        ae_int32x2 temp32_1, temp32_2;
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);
        AE_S32X2X2_IP(temp32_1,temp32_2, output_ptr, sizeof(ae_int32x4));
#endif

              }
            }
            {
              //SETUP_BIAS_32b_SINGLE;
              for(; m_itr < rows; m_itr++)
              {
                SETUP_OUTPUT_STORE_32_SCRATCH;
                UNROLL_SETUP_ACC(0);
                SETUP_VEC1;
                UNROLL_SETUP_MAT1(0);
                UNROLL_SETUP_ACC(1);
                ZER0_8x8_Temp_Variable;
                for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)
                {
                  LOAD_VEC1;
                  LOAD_ROW_MAT1_8b(0);
                  KERNEL_8x8_NOT_UNROLLED_MAT1_VEC1;

                }
                SETUP_VEC2;
                UNROLL_SETUP_MAT2(0);
                for(c_itr = 0; c_itr < (cols2 >> 4); c_itr++)
                {
                  LOAD_VEC2;
                  LOAD_ROW_MAT2_8b(0);
                  KERNEL_8x8_NOT_UNROLLED_MAT2_VEC2;
                }
                SETUP_ACC_64b_8x16(0);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE_EXTRA_ROW(0);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE;
              }
            }

        }
        else if(p_mat1 && p_vec1 && p_mat2 && p_vec2)
          {
            SETUP_BIAS_32b_SINGLE;
            if(rows > 3)
            {
              int row_count = rows - rows%3;
              for(m_itr = 0; m_itr < row_count; m_itr += 3)
              {
                SETUP_OUTPUT_STORE_32_SCRATCH;
                TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8;
                SETUP_ACC_64b_8x16(0);
                SETUP_ACC_64b_8x16(1);
                SETUP_ACC_64b_8x16(2);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE(0);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE_1(1);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(1);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(2);
              }
            }
            {
              for(; m_itr < rows; m_itr++)
              {
                SETUP_OUTPUT_STORE_32_SCRATCH;
                TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_mat2xvec2_8x8_SINGLE_ROW;
                SETUP_ACC_64b_8x16(0);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE(0);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
              }
            }
          }
        else if (p_mat1 && p_vec1 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && (cols1 %32 ==0) && row_stride1 % 16 == 0)
        {
          /* Only mat1, vec1 are non-null */
          SETUP_BIAS;
          if(rows > ROW_UNROLL_OPT)
          {
            for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL_OPT-1)) ; m_itr += ROW_UNROLL_OPT)
            {
              SETUP_OUTPUT_STORE_32x4_SCRATCH;
#ifdef UNROLL_16
        SETUP_ACC;
#else
       SETUP_ACC_FOR_8bx8b(0);
       SETUP_ACC_FOR_8bx8b(1);
#endif
              SETUP_VEC1_8b_16;
              SET_UP_MAT1_16_VARIABLES;
              SETUP_MAT1_8x8_PTR;
              for(c_itr = 0; c_itr < (cols1 >> 5); c_itr++)
              {
                  LOAD_VEC1_8b_16;
                  LOAD_VEC1_8b_16_1;
                  kernel_8x8_mat1_vec1;
              }
#ifdef UNROLL_16
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        SETUP_ACC_64b(2);
        SETUP_ACC_64b(3);
        SETUP_ACC_64b(4);
        SETUP_ACC_64b(5);
        SETUP_ACC_64b(6);
        SETUP_ACC_64b(7);
        ADD_BIAS_ACC;
        STORE_ACC_8bx8b_AT_OUT_32x4x2;
#else
        SETUP_ACC_64b(0);
        SETUP_ACC_64b(1);
        ADD_BIAS_32b_ACC_FOR_8bx8b(0);
        ADD_BIAS_32b_ACC_FOR_8bx8b(1);
        ae_int32x2 temp32_1, temp32_2;
        _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0 ,acc_shift);
        _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1 ,acc_shift);
        _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0 ,acc_shift);
        _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1 ,acc_shift);
        temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
        temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);
        AE_S32X2X2_IP(temp32_1,temp32_2, output_ptr, sizeof(ae_int32x4));
#endif
            }
          }
          {
            SETUP_BIAS_32b_SINGLE;
            for(m_itr = (rows & ~(ROW_UNROLL-1)); m_itr < rows; m_itr++)
            {
              SETUP_OUTPUT_STORE_32_SCRATCH;
              UNROLL_SETUP_ACC(0); SETUP_VEC1; UNROLL_SETUP_MAT1(0);
              UNROLL_SETUP_ACC(1);
              ZER0_8x8_Temp_Variable;
              for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
              {
                  LOAD_VEC1;
                  LOAD_ROW_MAT1_8b(0);
                  KERNEL_8x8_NOT_UNROLLED_MAT1_VEC1;
              }
              SETUP_ACC_64b_8x16(0);
              ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE(0);
              STORE_ACC_8bx8b_AT_OUT_32_SINGLE;
            }
          }
        }
          else if(p_mat1 && p_vec1)
          {
            SETUP_BIAS_32b_SINGLE;
            if(rows > 3)
            {
              int row_count = rows - rows%3;
              for(m_itr = 0; m_itr < row_count; m_itr += 3)
              {
                SETUP_OUTPUT_STORE_32_SCRATCH;
                TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8;
                SETUP_ACC_64b_8x16(0);
                SETUP_ACC_64b_8x16(1);
                SETUP_ACC_64b_8x16(2);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE(0);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE_1(1);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(1);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(2);
              }
            }
            {
              for(; m_itr < rows; m_itr++)
              {
                SETUP_OUTPUT_STORE_32_SCRATCH;
                TEMPLATE_UNALIGNED_MULTIPLICATION_mat1xvec1_8x8_SINGLE_ROW;
                SETUP_ACC_64b_8x16(0);
                ADD_BIAS_32b_ACC_FOR_8bx8b_SINGLE(0);
                STORE_ACC_8bx8b_AT_OUT_32_SINGLE_UNALIGNED(0);
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
  }


#undef UNROLL_SETUP_ACC
#undef UNROLL_SETUP_MAT1
#undef UNROLL_SETUP_MAT2
#undef SETUP_VEC1
#undef SETUP_VEC2
#undef UNROLL_LOAD_MAT1
#undef UNROLL_LOAD_MAT2
#undef LOAD_VEC1
#undef LOAD_VEC2

  xa_nn_vec_sigmoid_32_8((pWORD8) p_out, (pWORD32) p_scratch, rows);


  return 0;
}
