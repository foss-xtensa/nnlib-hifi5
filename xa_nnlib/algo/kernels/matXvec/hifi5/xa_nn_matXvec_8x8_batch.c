/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
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
#include <xtensa/hal.h>

#define ROW_UNROLL 8
#include "xa_nnlib_common_macros_hifi5.h"

WORD32 xa_nn_matXvec_batch_8x8_32(

         WORD32 ** __restrict__ p_out,          /* array of output pointers */
         WORD8 *  __restrict__ p_mat1,         /* matrix1: rows x cols1 */
         WORD8 ** __restrict__ p_vec1,         /* vec1: cols1 x 1 */
         WORD8 *  __restrict__ p_bias,         /* bias TBD: Need array? */
         WORD32 rows,
         WORD32 cols1,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift,                       /* bias shift amount */
         WORD32 vec_count)                      /* number of vectors: 2, 4, 2n */
{
  int i;
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  for(i = 0; i < vec_count; i++)
  {
    XA_NNLIB_ARG_CHK_PTR(p_out[i], -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec1[i], -1);
  }
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32 *), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD8 *), -1);
  int isAlignedvec = 1;
  for(i = 0; i < vec_count; i++)
  {
    XA_NNLIB_ARG_CHK_ALIGN(p_out[i], sizeof(WORD32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec1[i],sizeof(WORD8), -1);
    isAlignedvec &= (((unsigned int)p_vec1[i] & 15) == 0);
  }
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec_count <= 0), -1);
  
  /* Iterators used in for loops */
  int m_itr, c_itr, vec_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  vec_itr = 0;

#define VEC_UNROLL 2
#define SETUP_BIAS                          SETUP_BIAS_8b
  acc_shift = acc_shift + 32;
  if (((unsigned int)p_mat1 & 15) == 0 && isAlignedvec && ((cols1&15) == 0) && ((row_stride1&15) == 0))
  {
    if(rows > ROW_UNROLL)
    {
      if(vec_count > VEC_UNROLL)
      {
        for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
        {
          SETUP_BIAS;
          int loop_count=rows -rows%ROW_UNROLL;
          for(m_itr = 0; m_itr < loop_count; m_itr += ROW_UNROLL)
          {

            ae_int8x16 * _ae_int8x16_p_vec1 = (ae_int8x16 *) (p_vec1[vec_itr]);
            ae_int8x16 * _ae_int8x16_p_vec2 = (ae_int8x16 *) (p_vec1[vec_itr+1]);
            ae_int32 *output_ptr  =(ae_int32 *)(p_out[vec_itr]+m_itr);
            ae_int32 *output_ptr1 =(ae_int32 *)(p_out[vec_itr+1]+m_itr);

            ae_int32x2 _ae_int32x2_acc_row0_vec1 = ZERO32;
            ae_int32x2 _ae_int32x2_acc_row1_vec1 = ZERO32;
            ae_int32x2 _ae_int32x2_acc_row2_vec1 = ZERO32;
            ae_int32x2 _ae_int32x2_acc_row3_vec1 = ZERO32;
            ae_int32x2 _ae_int32x2_acc_row0_vec2 = ZERO32;
            ae_int32x2 _ae_int32x2_acc_row1_vec2 = ZERO32;
            ae_int32x2 _ae_int32x2_acc_row2_vec2 = ZERO32;
            ae_int32x2 _ae_int32x2_acc_row3_vec2 = ZERO32;

            ae_int8x8 _ae_int8x8_vec1;
            ae_int8x8 _ae_int8x8_vec1_1;
            ae_int8x8 _ae_int8x8_vec2;
            ae_int8x8 _ae_int8x8_vec2_1;

            SETUP_MAT1_8b_16_BATCH(0);
            SETUP_MAT1_8b_16_BATCH(1);
            SETUP_MAT1_8b_16_BATCH(2);
            SETUP_MAT1_8b_16_BATCH(3);
            SETUP_MAT1_8b_16_BATCH(4);
            SETUP_MAT1_8b_16_BATCH(5);
            SETUP_MAT1_8b_16_BATCH(6);
            SETUP_MAT1_8b_16_BATCH(7);

            ae_int8x16 * _ae_int8x16_p_mat1_0  = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];
#pragma loop_count min=1
            for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)
            {

              AE_L8X8X2_IP (_ae_int8x8_vec1,_ae_int8x8_vec1_1, _ae_int8x16_p_vec1, 1*sizeof(ae_int8x16));
              AE_L8X8X2_IP (_ae_int8x8_vec2,_ae_int8x8_vec2_1, _ae_int8x16_p_vec2, 1*sizeof(ae_int8x16));

              AE_L8X8X2_XP ( _ae_int8x8_mat1_0, _ae_int8x8_mat1_1_0,_ae_int8x16_p_mat1_0, sizeof(WORD8)*row_stride1);
              AE_L8X8X2_XP ( _ae_int8x8_mat1_1, _ae_int8x8_mat1_1_1,_ae_int8x16_p_mat1_0, sizeof(WORD8)*row_stride1);
              AE_L8X8X2_XP ( _ae_int8x8_mat1_2, _ae_int8x8_mat1_1_2,_ae_int8x16_p_mat1_0, sizeof(WORD8)*row_stride1);
              AE_L8X8X2_XP ( _ae_int8x8_mat1_3, _ae_int8x8_mat1_1_3,_ae_int8x16_p_mat1_0, sizeof(WORD8)*row_stride1);

              AE_MULA8Q8X8(_ae_int32x2_acc_row0_vec1 , _ae_int32x2_acc_row1_vec1 , _ae_int8x8_mat1_0 , _ae_int8x8_mat1_1 , _ae_int8x8_mat1_2 , _ae_int8x8_mat1_3 , _ae_int8x8_vec1);
              AE_MULA8Q8X8(_ae_int32x2_acc_row0_vec2 , _ae_int32x2_acc_row1_vec2 , _ae_int8x8_mat1_0 , _ae_int8x8_mat1_1 , _ae_int8x8_mat1_2 , _ae_int8x8_mat1_3 , _ae_int8x8_vec2);

              AE_MULA8Q8X8(_ae_int32x2_acc_row0_vec1 , _ae_int32x2_acc_row1_vec1 , _ae_int8x8_mat1_1_0 , _ae_int8x8_mat1_1_1 , _ae_int8x8_mat1_1_2 , _ae_int8x8_mat1_1_3 , _ae_int8x8_vec1_1);
              AE_MULA8Q8X8(_ae_int32x2_acc_row0_vec2 , _ae_int32x2_acc_row1_vec2 , _ae_int8x8_mat1_1_0 , _ae_int8x8_mat1_1_1 , _ae_int8x8_mat1_1_2 , _ae_int8x8_mat1_1_3 , _ae_int8x8_vec2_1);

              AE_L8X8X2_XP ( _ae_int8x8_mat1_4, _ae_int8x8_mat1_1_4,_ae_int8x16_p_mat1_0, sizeof(WORD8)*row_stride1);
              AE_L8X8X2_XP ( _ae_int8x8_mat1_5, _ae_int8x8_mat1_1_5,_ae_int8x16_p_mat1_0, sizeof(WORD8)*row_stride1);
              AE_L8X8X2_XP ( _ae_int8x8_mat1_6, _ae_int8x8_mat1_1_6,_ae_int8x16_p_mat1_0, sizeof(WORD8)*row_stride1);
              AE_L8X8X2_XP ( _ae_int8x8_mat1_7, _ae_int8x8_mat1_1_7,_ae_int8x16_p_mat1_0, sizeof(WORD8)*(16-7*row_stride1));

              AE_MULA8Q8X8(_ae_int32x2_acc_row2_vec1,  _ae_int32x2_acc_row3_vec1 , _ae_int8x8_mat1_4 , _ae_int8x8_mat1_5 , _ae_int8x8_mat1_6 , _ae_int8x8_mat1_7 , _ae_int8x8_vec1);
              AE_MULA8Q8X8(_ae_int32x2_acc_row2_vec2,  _ae_int32x2_acc_row3_vec2 , _ae_int8x8_mat1_4 , _ae_int8x8_mat1_5 , _ae_int8x8_mat1_6 , _ae_int8x8_mat1_7 , _ae_int8x8_vec2);

              AE_MULA8Q8X8(_ae_int32x2_acc_row2_vec1,  _ae_int32x2_acc_row3_vec1 , _ae_int8x8_mat1_1_4 , _ae_int8x8_mat1_1_5 , _ae_int8x8_mat1_1_6 , _ae_int8x8_mat1_1_7 , _ae_int8x8_vec1_1);
              AE_MULA8Q8X8(_ae_int32x2_acc_row2_vec2,  _ae_int32x2_acc_row3_vec2 , _ae_int8x8_mat1_1_4 , _ae_int8x8_mat1_1_5 , _ae_int8x8_mat1_1_6 , _ae_int8x8_mat1_1_7 , _ae_int8x8_vec2_1);
            }
            ae_int32x2 temp32_1, temp32_2, temp32_3, temp32_4;
            ae_int64 _ae_int64_acc_0, _ae_int64_acc_1_0;
            ae_int64 _ae_int64_acc_1, _ae_int64_acc_1_1;
            ae_int64 _ae_int64_acc_2, _ae_int64_acc_1_2;
            ae_int64 _ae_int64_acc_3, _ae_int64_acc_1_3;

            AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);
            AE_L8_IP(_ae_int8_bias_1,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);
            sat_1 = AE_MOVINT64_FROMINT8X8(_ae_int8_bias);
            sat_1 = AE_SRAI64(sat_1, 56);
            sat_1 = AE_SLAA64S(sat_1, bias_shift);
            sat_2 = AE_MOVINT64_FROMINT8X8(_ae_int8_bias_1);
            sat_2 = AE_SRAI64(sat_2, 56);
            sat_2 = AE_SLAA64S(sat_2, bias_shift);

            AE_ADDW32(_ae_int64_acc_0, _ae_int64_acc_1_0, _ae_int32x2_acc_row0_vec1, ZERO32);
            _ae_int64_acc_0 = AE_ADD64S(_ae_int64_acc_0,sat_1);
            _ae_int64_acc_1_0 = AE_ADD64S(_ae_int64_acc_1_0,sat_2);

            AE_ADDW32(_ae_int64_acc_1, _ae_int64_acc_1_1, _ae_int32x2_acc_row0_vec2, ZERO32);
            _ae_int64_acc_1 = AE_ADD64S(_ae_int64_acc_1,sat_1);
            _ae_int64_acc_1_1 = AE_ADD64S(_ae_int64_acc_1_1,sat_2);

            AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);
            AE_L8_IP(_ae_int8_bias_1,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);
            sat_1 = AE_MOVINT64_FROMINT8X8(_ae_int8_bias);
            sat_1 = AE_SRAI64(sat_1, 56);
            sat_1 = AE_SLAA64S(sat_1, bias_shift);
            sat_2 = AE_MOVINT64_FROMINT8X8(_ae_int8_bias_1);
            sat_2 = AE_SRAI64(sat_2, 56);
            sat_2 = AE_SLAA64S(sat_2, bias_shift);

            AE_ADDW32(_ae_int64_acc_2, _ae_int64_acc_1_2, _ae_int32x2_acc_row1_vec1, ZERO32);
            _ae_int64_acc_2 = AE_ADD64S(_ae_int64_acc_2,sat_1);
            _ae_int64_acc_1_2 = AE_ADD64S(_ae_int64_acc_1_2,sat_2);

            AE_ADDW32(_ae_int64_acc_3, _ae_int64_acc_1_3, _ae_int32x2_acc_row1_vec2, ZERO32);
            _ae_int64_acc_3 = AE_ADD64S(_ae_int64_acc_3,sat_1);
            _ae_int64_acc_1_3 = AE_ADD64S(_ae_int64_acc_1_3,sat_2);

            _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0,acc_shift);
            _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1,acc_shift);
            _ae_int64_acc_2 = AE_SLAA64S(_ae_int64_acc_2,acc_shift);
            _ae_int64_acc_3 = AE_SLAA64S(_ae_int64_acc_3,acc_shift);

            _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0,acc_shift);
            _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1,acc_shift);
            _ae_int64_acc_1_2 = AE_SLAA64S(_ae_int64_acc_1_2,acc_shift);
            _ae_int64_acc_1_3 = AE_SLAA64S(_ae_int64_acc_1_3,acc_shift);

            temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
            temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);
            temp32_3  = AE_ROUND32X2F64SSYM(_ae_int64_acc_2, _ae_int64_acc_1_2);
            temp32_4  = AE_ROUND32X2F64SSYM(_ae_int64_acc_3, _ae_int64_acc_1_3);

            AE_S32_H_I(temp32_1,output_ptr,0);
            AE_S32_L_I(temp32_1,output_ptr,4);
            AE_S32_H_I(temp32_3,output_ptr,8);
            AE_S32_L_I(temp32_3,output_ptr,12);
            AE_S32_H_I(temp32_2,output_ptr1,0);
            AE_S32_L_I(temp32_2,output_ptr1,4);
            AE_S32_H_I(temp32_4,output_ptr1,8);
            AE_S32_L_I(temp32_4,output_ptr1,12);

            AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);
            AE_L8_IP(_ae_int8_bias_1,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);
            sat_1 = AE_MOVINT64_FROMINT8X8(_ae_int8_bias);
            sat_1 = AE_SRAI64(sat_1, 56);
            sat_1 = AE_SLAA64S(sat_1, bias_shift);
            sat_2 = AE_MOVINT64_FROMINT8X8(_ae_int8_bias_1);
            sat_2 = AE_SRAI64(sat_2, 56);
            sat_2 = AE_SLAA64S(sat_2, bias_shift);

            AE_ADDW32(_ae_int64_acc_0, _ae_int64_acc_1_0, _ae_int32x2_acc_row2_vec1, ZERO32);
            _ae_int64_acc_0 = AE_ADD64S(_ae_int64_acc_0,sat_1);
            _ae_int64_acc_1_0 = AE_ADD64S(_ae_int64_acc_1_0,sat_2);

            AE_ADDW32(_ae_int64_acc_1, _ae_int64_acc_1_1, _ae_int32x2_acc_row2_vec2, ZERO32);
            _ae_int64_acc_1 = AE_ADD64S(_ae_int64_acc_1,sat_1);
            _ae_int64_acc_1_1 = AE_ADD64S(_ae_int64_acc_1_1,sat_2);

            AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);
            AE_L8_IP(_ae_int8_bias_1,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);
            sat_1 = AE_MOVINT64_FROMINT8X8(_ae_int8_bias);
            sat_1 = AE_SRAI64(sat_1, 56);
            sat_1 = AE_SLAA64S(sat_1, bias_shift);
            sat_2 = AE_MOVINT64_FROMINT8X8(_ae_int8_bias_1);
            sat_2 = AE_SRAI64(sat_2, 56);
            sat_2 = AE_SLAA64S(sat_2, bias_shift);

            AE_ADDW32(_ae_int64_acc_2, _ae_int64_acc_1_2, _ae_int32x2_acc_row3_vec1, ZERO32);
            _ae_int64_acc_2 = AE_ADD64S(_ae_int64_acc_2,sat_1);
            _ae_int64_acc_1_2 = AE_ADD64S(_ae_int64_acc_1_2,sat_2);

            AE_ADDW32(_ae_int64_acc_3, _ae_int64_acc_1_3, _ae_int32x2_acc_row3_vec2, ZERO32);
            _ae_int64_acc_3 = AE_ADD64S(_ae_int64_acc_3,sat_1);
            _ae_int64_acc_1_3 = AE_ADD64S(_ae_int64_acc_1_3,sat_2);

            _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0,acc_shift);
            _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1,acc_shift);
            _ae_int64_acc_2 = AE_SLAA64S(_ae_int64_acc_2,acc_shift);
            _ae_int64_acc_3 = AE_SLAA64S(_ae_int64_acc_3,acc_shift);

            _ae_int64_acc_1_0 = AE_SLAA64S(_ae_int64_acc_1_0,acc_shift);
            _ae_int64_acc_1_1 = AE_SLAA64S(_ae_int64_acc_1_1,acc_shift);
            _ae_int64_acc_1_2 = AE_SLAA64S(_ae_int64_acc_1_2,acc_shift);
            _ae_int64_acc_1_3 = AE_SLAA64S(_ae_int64_acc_1_3,acc_shift);

            temp32_1  = AE_ROUND32X2F64SSYM(_ae_int64_acc_0, _ae_int64_acc_1_0);
            temp32_2  = AE_ROUND32X2F64SSYM(_ae_int64_acc_1, _ae_int64_acc_1_1);
            temp32_3  = AE_ROUND32X2F64SSYM(_ae_int64_acc_2, _ae_int64_acc_1_2);
            temp32_4  = AE_ROUND32X2F64SSYM(_ae_int64_acc_3, _ae_int64_acc_1_3);

            AE_S32_H_I(temp32_1,output_ptr ,16);
            AE_S32_L_I(temp32_1,output_ptr ,20);
            AE_S32_H_I(temp32_3,output_ptr ,24);
            AE_S32_L_I(temp32_3,output_ptr ,28);
            AE_S32_H_I(temp32_2,output_ptr1,16);
            AE_S32_L_I(temp32_2,output_ptr1,20);
            AE_S32_H_I(temp32_4,output_ptr1,24);
            AE_S32_L_I(temp32_4,output_ptr1,28);
          }

          for(m_itr =loop_count; m_itr < rows; m_itr++)
          {
            ae_int32x2 _ae_int32x2_acc_row0_vec12 = ZERO32;
            ae_int32x2 _ae_int32x2_tmp = ZERO32;

            ae_int8x8 * _ae_int8x8_p_vec1 = (ae_int8x8 *) (p_vec1[vec_itr]);
            ae_int8x8 * _ae_int8x8_p_vec2 = (ae_int8x8 *) (p_vec1[vec_itr+1]);
            ae_int32 *output_ptr  =(ae_int32 *)(p_out[vec_itr]+m_itr);
            ae_int32 *output_ptr1 =(ae_int32 *)(p_out[vec_itr+1]+m_itr);
            ae_int8x8 * _ae_int8x8_p_mat1_0  = (ae_int8x8 *) &p_mat1[m_itr*row_stride1];
            ae_int8x8 _ae_int8x8_mat1_0,_ae_int8x8_vec1,_ae_int8x8_vec2;
            ZER0_8x8_Temp_Variable;
            for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
            {
              AE_L8X8_IP(_ae_int8x8_mat1_0,_ae_int8x8_p_mat1_0, 8);
              AE_L8X8_IP(_ae_int8x8_vec1, _ae_int8x8_p_vec1, 8);
              AE_L8X8_IP(_ae_int8x8_vec2, _ae_int8x8_p_vec2, 8);
              AE_MULA8Q8X8(_ae_int32x2_tmp, _ae_int32x2_acc_row0_vec12, zero_temp, zero_temp, _ae_int8x8_vec1, _ae_int8x8_vec2, _ae_int8x8_mat1_0);
            }

            ae_int32x2 temp32_1, temp32_2;
            ae_int64 _ae_int64_acc_0;
            ae_int64 _ae_int64_acc_1;
            AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);
            sat_1 = AE_MOVINT64_FROMINT8X8(_ae_int8_bias);
            sat_1 = AE_SRAI64(sat_1,56);
            sat_1 = AE_SLAA64S(sat_1,bias_shift);
            AE_ADDW32(_ae_int64_acc_0, _ae_int64_acc_1, _ae_int32x2_acc_row0_vec12,ZERO32);
            _ae_int64_acc_0 = AE_ADD64S(_ae_int64_acc_0,sat_1);
            _ae_int64_acc_1 = AE_ADD64S(_ae_int64_acc_1,sat_1);

            _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0,acc_shift);
            _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1,acc_shift);

            temp32_1  = AE_ROUND32X2F64SSYM(0, _ae_int64_acc_0);
            temp32_2  = AE_ROUND32X2F64SSYM(0, _ae_int64_acc_1);

            AE_S32_L_I(temp32_1,output_ptr,0*sizeof(ae_int32));
            AE_S32_L_I(temp32_2,output_ptr1,0*sizeof(ae_int32));
          }
        }
      }
      {
        for(; vec_itr < vec_count; vec_itr++)
        {
          xa_nn_matXvec_8x8_32(p_out[vec_itr],p_mat1,NULL,p_vec1[vec_itr],NULL,p_bias,rows,cols1,0,row_stride1,0,acc_shift,bias_shift);
        }
      }
    }
    else
    {
      if(vec_count > VEC_UNROLL)
      {
        for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
        {
          ae_int64 sat_1;
          ae_int8x8 _ae_int8_bias;
          ae_int8 *_ae_int8_p_bias = (ae_int8 *) p_bias;
          for(m_itr =0; m_itr < rows; m_itr++)
          {
            ae_int32x2 _ae_int32x2_acc_row0_vec12 = ZERO32;
            ae_int32x2 _ae_int32x2_acc_tmp = ZERO32;

            ae_int8x8 * _ae_int8x8_p_vec1 = (ae_int8x8 *) (p_vec1[vec_itr]);
            ae_int8x8 * _ae_int8x8_p_vec2 = (ae_int8x8 *) (p_vec1[vec_itr+1]);
            ae_int32 *output_ptr  =(ae_int32 *)(p_out[vec_itr]+m_itr);
            ae_int32 *output_ptr1 =(ae_int32 *)(p_out[vec_itr+1]+m_itr);
            ae_int8x8 * _ae_int8x8_p_mat1_0  = (ae_int8x8 *) &p_mat1[m_itr*row_stride1];
            ae_int8x8 _ae_int8x8_mat1_0,_ae_int8x8_vec1,_ae_int8x8_vec2;
            ZER0_8x8_Temp_Variable;
            for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
            {
              AE_L8X8_IP(_ae_int8x8_mat1_0,_ae_int8x8_p_mat1_0, 8);
              AE_L8X8_IP(_ae_int8x8_vec1, _ae_int8x8_p_vec1, 8);
              AE_L8X8_IP(_ae_int8x8_vec2, _ae_int8x8_p_vec2, 8);
              AE_MULA8Q8X8(_ae_int32x2_acc_tmp, _ae_int32x2_acc_row0_vec12, zero_temp, zero_temp, _ae_int8x8_vec1, _ae_int8x8_vec2, _ae_int8x8_mat1_0);
            }

            ae_int32x2 temp32_1, temp32_2;
            ae_int64 _ae_int64_acc_0;
            ae_int64 _ae_int64_acc_1;
            AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);
            sat_1 = AE_MOVINT64_FROMINT8X8(_ae_int8_bias);
            sat_1 = AE_SRAI64(sat_1,56);
            sat_1 = AE_SLAA64S(sat_1,bias_shift);
            AE_ADDW32(_ae_int64_acc_0, _ae_int64_acc_1, _ae_int32x2_acc_row0_vec12, ZERO32);
            _ae_int64_acc_0 = AE_ADD64S(_ae_int64_acc_0,sat_1);
            _ae_int64_acc_1 = AE_ADD64S(_ae_int64_acc_1,sat_1);

            _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0,acc_shift);
            _ae_int64_acc_1 = AE_SLAA64S(_ae_int64_acc_1,acc_shift);

            temp32_1  = AE_ROUND32X2F64SSYM(0, _ae_int64_acc_0);
            temp32_2  = AE_ROUND32X2F64SSYM(0, _ae_int64_acc_1);

            AE_S32_L_I(temp32_1,output_ptr,0*sizeof(ae_int32));
            AE_S32_L_I(temp32_2,output_ptr1,0*sizeof(ae_int32));

          }
        }
      }
      {
        for(; vec_itr < vec_count; vec_itr++)
        {
          ae_int64 sat_1;
          ae_int8x8 _ae_int8_bias;
          ae_int8 *_ae_int8_p_bias = (ae_int8 *) p_bias;
          for(m_itr =0; m_itr < rows; m_itr++)
          {
            ae_int32x2 _ae_int32x2_acc_row0_vec1 = ZERO32;
            ae_int32x2 _ae_int32x2_acc_row0_vec2 = ZERO32;

            ae_int8x8 * _ae_int8x8_p_vec1 = (ae_int8x8 *) (p_vec1[vec_itr]);
            ae_int32 *output_ptr  =(ae_int32 *)(p_out[vec_itr]+m_itr);
            ae_int8x8 * _ae_int8x8_p_mat1_0  = (ae_int8x8 *) &p_mat1[m_itr*row_stride1];
            ae_int8x8 _ae_int8x8_mat1_0,_ae_int8x8_vec1;
            ZER0_8x8_Temp_Variable;
            for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
            {
              AE_L8X8_IP(_ae_int8x8_mat1_0,_ae_int8x8_p_mat1_0, 8);
              AE_L8X8_IP(_ae_int8x8_vec1, _ae_int8x8_p_vec1, 8);
              AE_MULA8Q8X8(_ae_int32x2_acc_row0_vec2,_ae_int32x2_acc_row0_vec1 ,zero_temp,zero_temp ,zero_temp,_ae_int8x8_mat1_0 , _ae_int8x8_vec1);

            }

            ae_int32x2 temp32_1;
            ae_int64 _ae_int64_acc_0, _ae_int64_acc_1;
            AE_L8_IP(_ae_int8_bias,_ae_int8_p_bias,INCREMENT_IN_BYTES_FOR_WORD8);
            sat_1 = AE_MOVINT64_FROMINT8X8(_ae_int8_bias);
            sat_1 = AE_SRAA64(sat_1, 56);
            sat_1 = AE_SLAA64S(sat_1,bias_shift);
            AE_ADDW32(_ae_int64_acc_0, _ae_int64_acc_1, _ae_int32x2_acc_row0_vec1, ZERO32);
            _ae_int64_acc_0 = AE_ADD64S(_ae_int64_acc_0,sat_1);
            _ae_int64_acc_0 = AE_SLAA64S(_ae_int64_acc_0,acc_shift);
            temp32_1  = AE_ROUND32X2F64SSYM(0, _ae_int64_acc_0);
            AE_S32_L_I(temp32_1,output_ptr,0*sizeof(ae_int32));
          }
        }
      }
    }
  }
#undef SETUP_BIAS
#define SETUP_BIAS                          SETUP_BIAS_8b_UNALIGNED_SUPPORT
  else if (p_mat1 && p_vec1 && (cols1 > 0) && (row_stride1 > 0))
  {
    ZER0_8x8_Temp_Variable;
    if(rows > 2 && vec_count > VEC_UNROLL)
    {
      for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
      {
        SETUP_BIAS;
        for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
        {
          SETUP_ACC_FOR_8bx8b(0);
          SETUP_ACC_FOR_8bx8b(1);
          SETUP_ACC_FOR_8bx8b(2);
          SETUP_ACC_FOR_8bx8b(3);
          SETUP_MAT1_8b_UNALIGNED(0);
          SETUP_MAT1_8b_UNALIGNED(1);

          SETUP_VEC1_8b_UNALIGNED_BATCH(0);
          SETUP_VEC1_8b_UNALIGNED_BATCH(1);
          int cols_count=cols1-cols1%8;
          for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
          {
            LOAD_ROW_MAT1_8b_UNALIGNED(0);
            LOAD_ROW_MAT1_8b_UNALIGNED(1);
            LOAD_VEC1_8b_UNALIGNED_BATCH(0);
            LOAD_VEC1_8b_UNALIGNED_BATCH(1);
            KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_BATCH;
          }
          for(c_itr = cols_count; c_itr < cols1; c_itr++)
          {
            LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);
            LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(1);
            LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT_BATCH(0);
            LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT_BATCH(1);
            KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ELEMENT_BATCH;
          }
          SETUP_ACC_64b_8x16(0);
          SETUP_ACC_64b_8x16(1);
          SETUP_ACC_64b_8x16(2);
          SETUP_ACC_64b_8x16(3);
          ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_BATCH(0,2);
          ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_BATCH(1,3);
          STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(0,0,0);
          STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(1,0,1);
          STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(0,1,2);
          STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(1,1,3);
        }
        // remaining 1 row
        {
          for(; m_itr < rows; m_itr++)
          {
            SETUP_ACC_FOR_8bx8b(0);
            SETUP_ACC_FOR_8bx8b(1);
            SETUP_ACC_FOR_8bx8b(2);
            SETUP_ACC_FOR_8bx8b(3);
            SETUP_MAT1_8b_UNALIGNED(0);
            SETUP_VEC1_8b_UNALIGNED_BATCH(0);
            SETUP_VEC1_8b_UNALIGNED_BATCH(1);
            int cols_count=cols1-cols1%8;
            for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
            {
              LOAD_ROW_MAT1_8b_UNALIGNED(0);
              LOAD_VEC1_8b_UNALIGNED_BATCH(0);
              LOAD_VEC1_8b_UNALIGNED_BATCH(1);
              KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_BATCH;
            }
            for(c_itr = cols_count; c_itr < cols1; c_itr++)
            {
              LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);
              LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT_BATCH(0);
              LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT_BATCH(1);
              KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_ELEMENT_BATCH;
            }
            SETUP_ACC_64b_8x16(0);
            SETUP_ACC_64b_8x16(2);
            ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_BATCH(0,2);
            STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(0,0,0);
            STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(0,1,2);

          }
        }
      }
      // Remaining one vector
      for (; vec_itr < vec_count ; vec_itr++)
      {
        SETUP_BIAS;
        m_itr = 0;
        if(rows > 2)
        {
          for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
          {
            SETUP_ACC_FOR_8bx8b(0);
            SETUP_ACC_FOR_8bx8b(1);
            SETUP_ACC_FOR_8bx8b(2);
            SETUP_ACC_FOR_8bx8b(3);
            SETUP_MAT1_8b_UNALIGNED(0);
            SETUP_MAT1_8b_UNALIGNED(1);
            SETUP_VEC1_8b_UNALIGNED_BATCH(0);
            int cols_count=cols1-cols1%8;
            for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
            {
              LOAD_ROW_MAT1_8b_UNALIGNED(0);
              LOAD_ROW_MAT1_8b_UNALIGNED(1);
              LOAD_VEC1_8b_UNALIGNED_BATCH(0);
              KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_BATCH_SINGLE_ROW;
            }

            for(c_itr = cols_count; c_itr < cols1; c_itr++)
            {
              LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);
              LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(1);
              LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT_BATCH(0);
              KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ELEMENT_BATCH_SINGLE_ROW;
            }
            SETUP_ACC_64b_8x16(0);
            SETUP_ACC_64b_8x16(1);
            SETUP_ACC_64b_8x16(2);
            SETUP_ACC_64b_8x16(3);
            ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_BATCH(0,2);
            ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_BATCH(1,3);
            STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(0,0,0);
            STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(1,0,1);
          }
        }
        {
          for(; m_itr < rows; m_itr++)
          {
            SETUP_ACC_FOR_8bx8b(0);
            SETUP_ACC_FOR_8bx8b(1);
            SETUP_ACC_FOR_8bx8b(2);
            SETUP_MAT1_8b_UNALIGNED(0);
            SETUP_VEC1_8b_UNALIGNED_BATCH(0);
            int cols_count=cols1-cols1%8;
            for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
            {
              LOAD_ROW_MAT1_8b_UNALIGNED(0);
              LOAD_VEC1_8b_UNALIGNED_BATCH(0);
              KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_VECTOR_BATCH;
            }
            for(c_itr = cols_count; c_itr < cols1; c_itr++)
            {
              LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);
              LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT_BATCH(0);
              KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_ELEMENT_BATCH_SINGLE_VECTOR;
            }
            SETUP_ACC_64b_8x16(0);
            SETUP_ACC_64b_8x16(2);
            ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_BATCH(0,2);
            STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(0,0,0);
          }
        }
      }

    }
    else
    {
      for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
      {
        SETUP_BIAS;
        m_itr = 0;
        if(rows > 2)
        {
          for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
          {
            SETUP_ACC_FOR_8bx8b(0);
            SETUP_ACC_FOR_8bx8b(1);
            SETUP_ACC_FOR_8bx8b(2);
            SETUP_ACC_FOR_8bx8b(3);
            SETUP_MAT1_8b_UNALIGNED(0);
            SETUP_MAT1_8b_UNALIGNED(1);
            SETUP_VEC1_8b_UNALIGNED_BATCH(0);
            int cols_count=cols1-cols1%8;
            for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
            {
              LOAD_ROW_MAT1_8b_UNALIGNED(0);
              LOAD_ROW_MAT1_8b_UNALIGNED(1);
              LOAD_VEC1_8b_UNALIGNED_BATCH(0);
              KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_BATCH_SINGLE_ROW;
            }
            for(c_itr = cols_count; c_itr < cols1; c_itr++)
            {
              LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);
              LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(1);
              LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT_BATCH(0);
              KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ELEMENT_BATCH_SINGLE_ROW;
            }
            SETUP_ACC_64b_8x16(0);
            SETUP_ACC_64b_8x16(1);
            SETUP_ACC_64b_8x16(2);
            SETUP_ACC_64b_8x16(3);
            ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_BATCH(0,2);
            ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_BATCH(1,3);
            STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(0,0,0);
            STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(1,0,1);

          }
        }
        {
          for(; m_itr < rows; m_itr++)
          {
            SETUP_ACC_FOR_8bx8b(0);
            SETUP_ACC_FOR_8bx8b(1);
            SETUP_ACC_FOR_8bx8b(2);
            SETUP_MAT1_8b_UNALIGNED(0);
            SETUP_VEC1_8b_UNALIGNED_BATCH(0);
            int cols_count=cols1-cols1%8;
            for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
            {
              LOAD_ROW_MAT1_8b_UNALIGNED(0);

              LOAD_VEC1_8b_UNALIGNED_BATCH(0);
              KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_VECTOR_BATCH;
            }
            for(c_itr = cols_count; c_itr < cols1; c_itr++)
            {
              LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);
              LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT_BATCH(0);
              KERNEL_MAT1_VEC1_8b_8b_UNALIGNED_SINGLE_ROW_SINGLE_ELEMENT_BATCH_SINGLE_VECTOR;
            }
            SETUP_ACC_64b_8x16(0);
            SETUP_ACC_64b_8x16(2);
            ADD_BIAS_8b_ACC_FOR_8bx8b_UNALIGNED_SUPPORT_BATCH(0,2);
            STORE_ACC_BATCH_8bx8b_AT_OUT_32b_UNALIGNED_SUPPORT(0,0,0);

          }
        }

      }
    }
  }
  else
  {
    return -1;
  }

#undef SETUP_BIAS
#undef LOAD_BIAS
#undef ROW_UNROLL
  return 0;
}


#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
#define MATXVEC_ALIGNED_VEC_8x8_ASYM16S(cols, ptr_vec) \
  for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4) \
  { \
    ae_int32x2 d_acc0_0 = AE_ZERO32(); \
    ae_int32x2 d_acc1_0 = AE_ZERO32(); \
    ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01; \
    ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1]; \
    ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0); \
    ae_int8x8 _ae8x8_mat1_10, _ae8x8_mat1_11; \
    ae_int8x16 *_ae8x16_p_mat1_1 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_0 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_1 = AE_LA128_PP(_ae8x16_p_mat1_1); \
    ae_int8x8 _ae8x8_mat1_20, _ae8x8_mat1_21; \
    ae_int8x16 *_ae8x16_p_mat1_2 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_1 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_2 = AE_LA128_PP(_ae8x16_p_mat1_2); \
    ae_int8x8 _ae8x8_mat1_30, _ae8x8_mat1_31; \
    ae_int8x16 *_ae8x16_p_mat1_3 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_2 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_3 = AE_LA128_PP(_ae8x16_p_mat1_3); \
    p_local_vec = (ae_int8x16 *)ptr_vec; \
    int cols_count=cols; \
    for(c_itr = 0; c_itr < cols_count>>4; c_itr++) \
    { \
      AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0); \
      AE_LA8X8X2_IP(_ae8x8_mat1_10, _ae8x8_mat1_11, _align_ae8x16_p_mat1_1, _ae8x16_p_mat1_1); \
      AE_LA8X8X2_IP(_ae8x8_mat1_20, _ae8x8_mat1_21, _align_ae8x16_p_mat1_2, _ae8x16_p_mat1_2); \
      AE_LA8X8X2_IP(_ae8x8_mat1_30, _ae8x8_mat1_31, _align_ae8x16_p_mat1_3, _ae8x16_p_mat1_3); \
      AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 16); \
      AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_00); \
      AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_31, _ae8x8_vec1_01); \
    } \
    { \
      d_acc0_0 = AE_ADD32S(d_acc0_0, AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1])); \
      d_acc1_0 = AE_ADD32S(d_acc1_0, AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3])); \
    } \
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift); \
    d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias)); \
    d_acc1_0 = AE_ADD32S(d_acc1_0, AE_MOVDA32(out_zero_bias)); \
    { \
      ae_int32x2 out0, out1; \
      out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]); \
      out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]); \
      d_acc0_0 = AE_ADD32S(d_acc0_0, out0); \
      d_acc1_0 = AE_ADD32S(d_acc1_0, out1); \
    } \
    ae_int16x4 _ae_int16x4_out; \
    _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = (_ae_int16x4_out); \
  } \
_Pragma("no_unroll") \
_Pragma("loop_count max=3") \
  for(m_itr = (rows&(~3)); m_itr < rows; m_itr++) \
  { \
    ae_int32x2 d_acc0_0; \
    ae_int64 d64_acc0 = AE_ZERO64(); \
    ae_int64 d64_acc1 = AE_ZERO64(); \
    ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01; \
    ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1]; \
    ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0); \
    p_local_vec = (ae_int8x16 *)ptr_vec; \
    int cols_count=cols; \
    for(c_itr = 0; c_itr < cols_count>>4; c_itr++) \
    { \
      AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0); \
      AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 16); \
      AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_00, _ae8x8_vec1_00); \
      AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_01, _ae8x8_vec1_01); \
    } \
    d64_acc0 = AE_ADD64(d64_acc0, d64_acc1); \
    d_acc0_0 = AE_SAT32X2(d64_acc0, d64_acc0); \
    { \
      d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]); \
    } \
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift); \
    d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias)); \
    { \
      ae_int32x2 out0; \
      out0 = AE_MOVDA32(p_out[vec_itr*rows+m_itr]); \
      d_acc0_0 = AE_ADD32S(d_acc0_0, out0); \
    } \
    ae_int16x4 _ae_int16x4_out; \
    _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = _ae_int16x4_out; \
  }
#else
#define MATXVEC_ALIGNED_VEC_8x8_ASYM16S(cols, ptr_vec) \
  for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4) \
  { \
    ae_int32x2 d_acc0_0 = AE_ZERO32(); \
    ae_int32x2 d_acc1_0 = AE_ZERO32(); \
    ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01; \
    ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1]; \
    ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0); \
    ae_int8x8 _ae8x8_mat1_10, _ae8x8_mat1_11; \
    ae_int8x16 *_ae8x16_p_mat1_1 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_0 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_1 = AE_LA128_PP(_ae8x16_p_mat1_1); \
    ae_int8x8 _ae8x8_mat1_20, _ae8x8_mat1_21; \
    ae_int8x16 *_ae8x16_p_mat1_2 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_1 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_2 = AE_LA128_PP(_ae8x16_p_mat1_2); \
    ae_int8x8 _ae8x8_mat1_30, _ae8x8_mat1_31; \
    ae_int8x16 *_ae8x16_p_mat1_3 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_2 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_3 = AE_LA128_PP(_ae8x16_p_mat1_3); \
    p_local_vec = (ae_int8x16 *)ptr_vec; \
    int cols_count=cols; \
    for(c_itr = 0; c_itr < cols_count>>4; c_itr++) \
    { \
      AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0); \
      AE_LA8X8X2_IP(_ae8x8_mat1_10, _ae8x8_mat1_11, _align_ae8x16_p_mat1_1, _ae8x16_p_mat1_1); \
      AE_LA8X8X2_IP(_ae8x8_mat1_20, _ae8x8_mat1_21, _align_ae8x16_p_mat1_2, _ae8x16_p_mat1_2); \
      AE_LA8X8X2_IP(_ae8x8_mat1_30, _ae8x8_mat1_31, _align_ae8x16_p_mat1_3, _ae8x16_p_mat1_3); \
      AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 16); \
      AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_00); \
      AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_31, _ae8x8_vec1_01); \
    } \
    { \
      d_acc0_0 = AE_ADD32S(d_acc0_0, AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1])); \
      d_acc1_0 = AE_ADD32S(d_acc1_0, AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3])); \
    } \
    MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift); \
    d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias)); \
    d_acc1_0 = AE_ADD32S(d_acc1_0, AE_MOVDA32(out_zero_bias)); \
    { \
      ae_int32x2 out0, out1; \
      out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]); \
      out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]); \
      d_acc0_0 = AE_ADD32S(d_acc0_0, out0); \
      d_acc1_0 = AE_ADD32S(d_acc1_0, out1); \
    } \
    ae_int16x4 _ae_int16x4_out; \
    _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = (_ae_int16x4_out); \
  } \
_Pragma("no_unroll") \
_Pragma("loop_count max=3") \
  for(m_itr = (rows&(~3)); m_itr < rows; m_itr++) \
  { \
    ae_int32x2 d_acc0_0; \
    ae_int64 d64_acc0 = AE_ZERO64(); \
    ae_int64 d64_acc1 = AE_ZERO64(); \
    ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01; \
    ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1]; \
    ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0); \
    p_local_vec = (ae_int8x16 *)ptr_vec; \
    int cols_count=cols; \
    for(c_itr = 0; c_itr < cols_count>>4; c_itr++) \
    { \
      AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0); \
      AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 16); \
      AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_00, _ae8x8_vec1_00); \
      AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_01, _ae8x8_vec1_01); \
    } \
    d64_acc0 = AE_ADD64(d64_acc0, d64_acc1); \
    d_acc0_0 = AE_SAT32X2(d64_acc0, d64_acc0); \
    { \
      d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]); \
    } \
    MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift); \
    d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias)); \
    { \
      ae_int32x2 out0; \
      out0 = AE_MOVDA32(p_out[vec_itr*rows+m_itr]); \
      d_acc0_0 = AE_ADD32S(d_acc0_0, out0); \
    } \
    ae_int16x4 _ae_int16x4_out; \
    _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = _ae_int16x4_out; \
  }
#endif

WORD32 xa_nn_matXvec_acc_batch_sym8sx8_asym16s(
         WORD16 * __restrict__ p_out,           /* output pointer */
         const WORD8 *  __restrict__ p_mat1,    /* matrix1: rows x cols1 */
         const WORD8 * __restrict__ p_vec1,     /* vec1: cols1 x vec_count */
         const WORD32 *  __restrict__ p_bias,   /* bias: rows x 1 */
         WORD32 rows,
         WORD32 cols1,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 out_multiplier,                 /* out multiplier for quantization */
         WORD32 out_shift,                      /* out shift for quantization */
         WORD32 out_zero_bias,						          /* out zero bias for quantization */
         WORD32 vec_count)                      /* number of vectors */
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -32768 || out_zero_bias > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((vec_count < 0), -1);

  /* Iterators used in for loops */
  int m_itr, c_itr, vec_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  vec_itr = 0;
  int left_shift, right_shift;

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  right_shift = out_shift;
#if XCHAL_HAVE_HIFI5S
  left_shift = 31 - left_shift;
  left_shift = (left_shift << 16) | left_shift;
#endif  
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift > 0 ? out_shift : 0;
  right_shift = out_shift < 0 ? -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  if(cols1 > 32 && cols1 <= 48)
  {
    for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
    {
      m_itr = 0;
      ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
      ae_int8x8 _ae8x8_vec1_02, _ae8x8_vec1_03;
      ae_int8x8 _ae8x8_vec1_04, _ae8x8_vec1_05;
      ae_int8x16 *_ae8x16_p_vec1_0;
      ae_valignx2 _align_ae8x16_p_vec1_0;

      _ae8x16_p_vec1_0 = (ae_int8x16 *)&p_vec1[vec_itr*cols1];
      _align_ae8x16_p_vec1_0 = AE_LA128_PP(_ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_02, _ae8x8_vec1_03, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LAV8X8X2_XP(_ae8x8_vec1_04, _ae8x8_vec1_05, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0, cols1-32);
      ae_int8x16 local_vec[3];
      ae_int8x16 *p_local_vec = (ae_int8x16 *)local_vec;
      AE_S8X8X2_I(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 0);
      AE_S8X8X2_I(_ae8x8_vec1_02, _ae8x8_vec1_03, p_local_vec, 16);
      AE_S8X8X2_I(_ae8x8_vec1_04, _ae8x8_vec1_05, p_local_vec, 32);
      
      MATXVEC_ALIGNED_VEC_8x8_ASYM16S(48, local_vec);
    }
  }
  else if(cols1 > 48 && cols1 <= 64)
  {
    for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
    {
      m_itr = 0;
      ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
      ae_int8x8 _ae8x8_vec1_02, _ae8x8_vec1_03;
      ae_int8x8 _ae8x8_vec1_04, _ae8x8_vec1_05;
      ae_int8x8 _ae8x8_vec1_06, _ae8x8_vec1_07;
      ae_int8x16 *_ae8x16_p_vec1_0;
      ae_valignx2 _align_ae8x16_p_vec1_0;

      _ae8x16_p_vec1_0 = (ae_int8x16 *)&p_vec1[vec_itr*cols1];
      _align_ae8x16_p_vec1_0 = AE_LA128_PP(_ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_02, _ae8x8_vec1_03, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_04, _ae8x8_vec1_05, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LAV8X8X2_XP(_ae8x8_vec1_06, _ae8x8_vec1_07, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0, cols1-48);
      ae_int8x16 local_vec[4];
      ae_int8x16 *p_local_vec = (ae_int8x16 *)local_vec;
      AE_S8X8X2_I(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 0);
      AE_S8X8X2_I(_ae8x8_vec1_02, _ae8x8_vec1_03, p_local_vec, 16);
      AE_S8X8X2_I(_ae8x8_vec1_04, _ae8x8_vec1_05, p_local_vec, 32);
      AE_S8X8X2_X(_ae8x8_vec1_06, _ae8x8_vec1_07, p_local_vec, 48);
      
      MATXVEC_ALIGNED_VEC_8x8_ASYM16S(64, local_vec);
    }
  }
  else if(cols1 > 64 && cols1 <= 80)
  {
    for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
    {
      m_itr = 0;
      ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
      ae_int8x8 _ae8x8_vec1_02, _ae8x8_vec1_03;
      ae_int8x8 _ae8x8_vec1_04, _ae8x8_vec1_05;
      ae_int8x8 _ae8x8_vec1_06, _ae8x8_vec1_07;
      ae_int8x8 _ae8x8_vec1_08, _ae8x8_vec1_09;
      ae_int8x16 *_ae8x16_p_vec1_0;
      ae_valignx2 _align_ae8x16_p_vec1_0;

      _ae8x16_p_vec1_0 = (ae_int8x16 *)&p_vec1[vec_itr*cols1];
      _align_ae8x16_p_vec1_0 = AE_LA128_PP(_ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_02, _ae8x8_vec1_03, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_04, _ae8x8_vec1_05, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_06, _ae8x8_vec1_07, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LAV8X8X2_XP(_ae8x8_vec1_08, _ae8x8_vec1_09, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0, cols1-64);
      ae_int8x16 local_vec[5];
      ae_int8x16 *p_local_vec = (ae_int8x16 *)local_vec;
      AE_S8X8X2_I(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 0);
      AE_S8X8X2_I(_ae8x8_vec1_02, _ae8x8_vec1_03, p_local_vec, 16);
      AE_S8X8X2_I(_ae8x8_vec1_04, _ae8x8_vec1_05, p_local_vec, 32);
      AE_S8X8X2_X(_ae8x8_vec1_06, _ae8x8_vec1_07, p_local_vec, 48);
      AE_S8X8X2_X(_ae8x8_vec1_08, _ae8x8_vec1_09, p_local_vec, 64);

      MATXVEC_ALIGNED_VEC_8x8_ASYM16S(80, local_vec);
    }
  }
  else if((cols1&15) == 0 && cols1 == row_stride1 && ((unsigned)p_mat1&15) == 0 && ((unsigned)p_vec1&15) == 0)
  {
    vec_itr = 0;
    if(vec_count&1)
    {
      xthal_dcache_block_prefetch_for_read((void *)p_mat1, row_stride1*4);

      ae_int32x4 *pae_bias;
      ae_valignx2 b_a;
      pae_bias = (ae_int32x4 *)p_bias;
      b_a = AE_LA128_PP(pae_bias);
      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];
        ae_int8x8 _ae8x8_mat1_10, _ae8x8_mat1_11;
        ae_int8x16 *_ae8x16_p_mat1_1 = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];
        ae_int8x8 _ae8x8_mat1_20, _ae8x8_mat1_21;
        ae_int8x16 *_ae8x16_p_mat1_2 = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];
        ae_int8x8 _ae8x8_mat1_30, _ae8x8_mat1_31;
        ae_int8x16 *_ae8x16_p_mat1_3 = (ae_int8x16 *) &p_mat1[(m_itr+3)*row_stride1];

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0 = (ae_int8x16 *) p_vec1;

        {
          xthal_dcache_block_prefetch_for_read((void *)((char *)_ae8x16_p_mat1_3 + row_stride1), row_stride1*4);
        }
#pragma loop_count min=1
        for(c_itr = 0; c_itr < (cols1>>4); c_itr++)
        {
          AE_L8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _ae8x16_p_mat1_0, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_10, _ae8x8_mat1_11, _ae8x16_p_mat1_1, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_20, _ae8x8_mat1_21, _ae8x16_p_mat1_2, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_30, _ae8x8_mat1_31, _ae8x16_p_mat1_3, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _ae8x16_p_vec1_0, 16);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_31, _ae8x8_vec1_01);
        }

        {
          ae_int32x2 d_bias01, d_bias23;
          AE_LA32X2X2_IP(d_bias01, d_bias23, b_a, pae_bias);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        d_acc1_0 = AE_ADD32S(d_acc1_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[m_itr], p_out[m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[m_itr+2], p_out[m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
        *(ae_int16 *)&p_out[m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[m_itr+3] = (_ae_int16x4_out);
      }

#pragma no_unroll
#pragma loop_count max=3
      for(m_itr = (rows & (~3)); m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0;
        ae_int64 d64_acc0 = AE_ZERO64();
        ae_int64 d64_acc1 = AE_ZERO64();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0 = (ae_int8x16 *) p_vec1;

#pragma loop_count min=1
        for(c_itr = 0; c_itr < (cols1>>4); c_itr++)
        {
          AE_L8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _ae8x16_p_mat1_0, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _ae8x16_p_vec1_0, 16);
          AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_00, _ae8x8_vec1_00);
          AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_01, _ae8x8_vec1_01);
        }
        d64_acc0 = AE_ADD64(d64_acc0, d64_acc1);
        d_acc0_0 = AE_SAT32X2(d64_acc0, d64_acc0);
        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32(p_out[m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
      }
    }
    for (vec_itr = (vec_count&1); vec_itr < vec_count; vec_itr+=2)
    {
      xthal_dcache_block_prefetch_for_read((void *)p_mat1, row_stride1*4);
      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();
        ae_int32x2 d_acc0_1 = AE_ZERO32();
        ae_int32x2 d_acc1_1 = AE_ZERO32();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];
        ae_int8x8 _ae8x8_mat1_10, _ae8x8_mat1_11;
        ae_int8x16 *_ae8x16_p_mat1_1 = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];
        ae_int8x8 _ae8x8_mat1_20, _ae8x8_mat1_21;
        ae_int8x16 *_ae8x16_p_mat1_2 = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];
        ae_int8x8 _ae8x8_mat1_30, _ae8x8_mat1_31;
        ae_int8x16 *_ae8x16_p_mat1_3 = (ae_int8x16 *) &p_mat1[(m_itr+3)*row_stride1];

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0 = (ae_int8x16 *) &p_vec1[(vec_itr+0)*cols1];
        ae_int8x8 _ae8x8_vec1_10, _ae8x8_vec1_11;
        ae_int8x16 *_ae8x16_p_vec1_1 = (ae_int8x16 *) &p_vec1[(vec_itr+1)*cols1];

        {
          xthal_dcache_block_prefetch_for_read((void *)((char *)_ae8x16_p_mat1_3 + row_stride1), row_stride1*4);
        }
        for(c_itr = 0; c_itr < (cols1>>4); c_itr++)
        {
          AE_L8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _ae8x16_p_mat1_0, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_10, _ae8x8_mat1_11, _ae8x16_p_mat1_1, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_20, _ae8x8_mat1_21, _ae8x16_p_mat1_2, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_30, _ae8x8_mat1_31, _ae8x16_p_mat1_3, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _ae8x16_p_vec1_0, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_10, _ae8x8_vec1_11, _ae8x16_p_vec1_1, 16);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_31, _ae8x8_vec1_01);
          AE_MULA8Q8X8(d_acc0_1, d_acc1_1, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_10);
          AE_MULA8Q8X8(d_acc0_1, d_acc1_1, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_31, _ae8x8_vec1_11);
        }

        {
          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
          d_acc0_1 = AE_ADD32S(d_acc0_1, d_bias01);
          d_acc1_1 = AE_ADD32S(d_acc1_1, d_bias23);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        d_acc1_0 = AE_ADD32S(d_acc1_0, AE_MOVDA32(out_zero_bias));
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_1 = AE_ADD32S(d_acc0_1, AE_MOVDA32(out_zero_bias));
        d_acc1_1 = AE_ADD32S(d_acc1_1, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
          out0 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr+2], p_out[(vec_itr+1)*rows+m_itr+3]);
          d_acc0_1 = AE_ADD32S(d_acc0_1, out0);
          d_acc1_1 = AE_ADD32S(d_acc1_1, out1);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = (_ae_int16x4_out);
        
        _ae_int16x4_out = AE_SAT16X4(d_acc0_1, d_acc1_1);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+3] = (_ae_int16x4_out);
      }
      
#pragma no_unroll
      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc0_1 = AE_ZERO32();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0 = (ae_int8x16 *) &p_vec1[(vec_itr+0)*cols1];
        ae_int8x8 _ae8x8_vec1_10, _ae8x8_vec1_11;
        ae_int8x16 *_ae8x16_p_vec1_1 = (ae_int8x16 *) &p_vec1[(vec_itr+1)*cols1];

        for(c_itr = 0; c_itr < (cols1>>4); c_itr++)
        {
          AE_L8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _ae8x16_p_mat1_0, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _ae8x16_p_vec1_0, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_10, _ae8x8_vec1_11, _ae8x16_p_vec1_1, 16);
          AE_MULA8Q8X8(d_acc0_0, d_acc0_1, _ae8x8_vec1_00, _ae8x8_vec1_10, _ae8x8_vec1_00, _ae8x8_vec1_10, _ae8x8_mat1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc0_1, _ae8x8_vec1_01, _ae8x8_vec1_11, _ae8x8_vec1_01, _ae8x8_vec1_11, _ae8x8_mat1_01);
        }

        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
      }
    }
  }
  else
  {
    for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
    {
      xthal_dcache_block_prefetch_for_read((void *)p_mat1, row_stride1*3);
      for(m_itr = 0; m_itr < (rows - (3-1)); m_itr += 3)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];
        ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0);
        ae_int8x8 _ae8x8_mat1_10, _ae8x8_mat1_11;
        ae_int8x16 *_ae8x16_p_mat1_1 = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];
        ae_valignx2 _align_ae8x16_p_mat1_1 = AE_LA128_PP(_ae8x16_p_mat1_1);
        ae_int8x8 _ae8x8_mat1_20, _ae8x8_mat1_21;
        ae_int8x16 *_ae8x16_p_mat1_2 = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];
        ae_valignx2 _align_ae8x16_p_mat1_2 = AE_LA128_PP(_ae8x16_p_mat1_2);

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0;
        ae_valignx2 _align_ae8x16_p_vec1_0;

        _ae8x16_p_vec1_0 = (ae_int8x16 *)&p_vec1[vec_itr*cols1];
        _align_ae8x16_p_vec1_0 = AE_LA128_PP(_ae8x16_p_vec1_0);

        {
          xthal_dcache_block_prefetch_for_read((void *)((char *)_ae8x16_p_mat1_2 + row_stride1), row_stride1*3);
        }
        int cols_count=cols1&(~15);
        for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
        {
          AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0);
          AE_LA8X8X2_IP(_ae8x8_mat1_10, _ae8x8_mat1_11, _align_ae8x16_p_mat1_1, _ae8x16_p_mat1_1);
          AE_LA8X8X2_IP(_ae8x8_mat1_20, _ae8x8_mat1_21, _align_ae8x16_p_mat1_2, _ae8x16_p_mat1_2);
          AE_LA8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_20, _ae8x8_vec1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_21, _ae8x8_vec1_01);
        }

        {
          AE_LAV8X8X2_XP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0, cols1 - cols_count);
          AE_LAV8X8X2_XP(_ae8x8_mat1_10, _ae8x8_mat1_11, _align_ae8x16_p_mat1_1, _ae8x16_p_mat1_1, cols1 - cols_count);
          AE_LAV8X8X2_XP(_ae8x8_mat1_20, _ae8x8_mat1_21, _align_ae8x16_p_mat1_2, _ae8x16_p_mat1_2, cols1 - cols_count);
          AE_LAV8X8X2_XP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0, cols1 - cols_count);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_20, _ae8x8_vec1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_21, _ae8x8_vec1_01);
        }
        {
          ae_int32x2 d_bias01, d_bias22;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias22 = *(ae_int32 *)&p_bias[m_itr + 2];
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias22);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc1_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc1_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        d_acc1_0 = AE_ADD32S(d_acc1_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32(p_out[vec_itr*rows+m_itr+2]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = (_ae_int16x4_out);
      }

      m_itr = (rows / 3) * 3;
      if(m_itr < rows)
      {
        int m_itr_1 = m_itr + 1 > rows - 1 ? rows - 1 : m_itr + 1;
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[m_itr*row_stride1];
        ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0);
        ae_int8x8 _ae8x8_mat1_10, _ae8x8_mat1_11;
        ae_int8x16 *_ae8x16_p_mat1_1 = (ae_int8x16 *) &p_mat1[m_itr_1*row_stride1];
        ae_valignx2 _align_ae8x16_p_mat1_1 = AE_LA128_PP(_ae8x16_p_mat1_1);

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0;
        ae_valignx2 _align_ae8x16_p_vec1_0;

        _ae8x16_p_vec1_0 = (ae_int8x16 *)&p_vec1[vec_itr*cols1];
        _align_ae8x16_p_vec1_0 = AE_LA128_PP(_ae8x16_p_vec1_0);

        int cols_count=cols1&(~15);
        for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
        {
          AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0);
          AE_LA8X8X2_IP(_ae8x8_mat1_10, _ae8x8_mat1_11, _align_ae8x16_p_mat1_1, _ae8x16_p_mat1_1);
          AE_LA8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_vec1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_vec1_01);
        }

        {
          AE_LAV8X8X2_XP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0, cols1 - cols_count);
          AE_LAV8X8X2_XP(_ae8x8_mat1_10, _ae8x8_mat1_11, _align_ae8x16_p_mat1_1, _ae8x16_p_mat1_1, cols1 - cols_count);
          AE_LAV8X8X2_XP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0, cols1 - cols_count);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_vec1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_vec1_01);
        }
        {
          ae_int32x2 d_bias01;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr], *(ae_int32 *)&p_bias[m_itr_1]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, AE_MOVDA32(out_zero_bias));
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr_1]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr_1] = (_ae_int16x4_out);
      }
    }
  }

  return 0;
}

#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
#define MATXVEC_ALIGNED_VEC_8x8_ASYM16S_HU(cols, ptr_vec) \
  for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4) \
  { \
    ae_int32x2 d_acc0_0 = AE_ZERO32(); \
    ae_int32x2 d_acc1_0 = AE_ZERO32(); \
    ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01; \
    ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1]; \
    ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0); \
    ae_int8x8 _ae8x8_mat1_10, _ae8x8_mat1_11; \
    ae_int8x16 *_ae8x16_p_mat1_1 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_0 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_1 = AE_LA128_PP(_ae8x16_p_mat1_1); \
    ae_int8x8 _ae8x8_mat1_20, _ae8x8_mat1_21; \
    ae_int8x16 *_ae8x16_p_mat1_2 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_1 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_2 = AE_LA128_PP(_ae8x16_p_mat1_2); \
    ae_int8x8 _ae8x8_mat1_30, _ae8x8_mat1_31; \
    ae_int8x16 *_ae8x16_p_mat1_3 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_2 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_3 = AE_LA128_PP(_ae8x16_p_mat1_3); \
    p_local_vec = (ae_int8x16 *)ptr_vec; \
    int cols_count=cols; \
    for(c_itr = 0; c_itr < cols_count>>4; c_itr++) \
    { \
      AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0); \
      AE_LA8X8X2_IP(_ae8x8_mat1_10, _ae8x8_mat1_11, _align_ae8x16_p_mat1_1, _ae8x16_p_mat1_1); \
      AE_LA8X8X2_IP(_ae8x8_mat1_20, _ae8x8_mat1_21, _align_ae8x16_p_mat1_2, _ae8x16_p_mat1_2); \
      AE_LA8X8X2_IP(_ae8x8_mat1_30, _ae8x8_mat1_31, _align_ae8x16_p_mat1_3, _ae8x16_p_mat1_3); \
      AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 16); \
      AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_00); \
      AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_31, _ae8x8_vec1_01); \
    } \
    { \
      d_acc0_0 = AE_ADD32S(d_acc0_0, AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1])); \
      d_acc1_0 = AE_ADD32S(d_acc1_0, AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3])); \
    } \
    MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift); \
    d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb); \
    d_acc1_0 = AE_ADD32S(d_acc1_0, d_outzb); \
    ae_int16x4 _ae_int16x4_out; \
    _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0); \
    AE_CVTA32X4F16(d_acc0_0, d_acc1_0, _ae_int16x4_out, 1); \
    { \
      ae_int16x4 *p_16x4_out, d_16x4_out; \
      ae_valign out_align; \
      p_16x4_out = (ae_int16x4 *)(&p_out[vec_itr*rows+m_itr]); \
      out_align = AE_LA64_PP(p_16x4_out); \
      AE_LA16X4_IP(d_16x4_out, out_align, p_16x4_out); \
      AE_ACCW16(d_acc0_0, d_acc1_0, d_16x4_out, AE_ZERO16()); \
      _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0); \
    } \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = (_ae_int16x4_out); \
  } \
_Pragma("no_unroll") \
_Pragma("loop_count max=3") \
  for(m_itr = (rows&(~3)); m_itr < rows; m_itr++) \
  { \
    ae_int32x2 d_acc0_0, d_acc1_0; \
    ae_int64 d64_acc0 = AE_ZERO64(); \
    ae_int64 d64_acc1 = AE_ZERO64(); \
    ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01; \
    ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1]; \
    ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0); \
    p_local_vec = (ae_int8x16 *)ptr_vec; \
    int cols_count=cols; \
    for(c_itr = 0; c_itr < cols_count>>4; c_itr++) \
    { \
      AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0); \
      AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 16); \
      AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_00, _ae8x8_vec1_00); \
      AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_01, _ae8x8_vec1_01); \
    } \
    d64_acc0 = AE_ADD64(d64_acc0, d64_acc1); \
    d_acc0_0 = AE_SAT32X2(d64_acc0, d64_acc0); \
    { \
      d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]); \
    } \
    MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift); \
    d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb); \
    ae_int16x4 _ae_int16x4_out; \
    _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0); \
    AE_CVTA32X4F16(d_acc0_0, d_acc1_0, _ae_int16x4_out, 1); \
    { \
      ae_int32x2 out0; \
      out0 = AE_MOVDA32(p_out[vec_itr*rows+m_itr]); \
      d_acc0_0 = AE_ADD32S(d_acc0_0, out0); \
      _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0); \
    } \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = _ae_int16x4_out; \
  }
#else
#define MATXVEC_ALIGNED_VEC_8x8_ASYM16S_HU(cols, ptr_vec) \
  for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4) \
  { \
    ae_int32x2 d_acc0_0 = AE_ZERO32(); \
    ae_int32x2 d_acc1_0 = AE_ZERO32(); \
    ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01; \
    ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1]; \
    ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0); \
    ae_int8x8 _ae8x8_mat1_10, _ae8x8_mat1_11; \
    ae_int8x16 *_ae8x16_p_mat1_1 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_0 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_1 = AE_LA128_PP(_ae8x16_p_mat1_1); \
    ae_int8x8 _ae8x8_mat1_20, _ae8x8_mat1_21; \
    ae_int8x16 *_ae8x16_p_mat1_2 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_1 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_2 = AE_LA128_PP(_ae8x16_p_mat1_2); \
    ae_int8x8 _ae8x8_mat1_30, _ae8x8_mat1_31; \
    ae_int8x16 *_ae8x16_p_mat1_3 = (ae_int8x16 *)((char *)_ae8x16_p_mat1_2 + row_stride1); \
    ae_valignx2 _align_ae8x16_p_mat1_3 = AE_LA128_PP(_ae8x16_p_mat1_3); \
    p_local_vec = (ae_int8x16 *)ptr_vec; \
    int cols_count=cols; \
    for(c_itr = 0; c_itr < cols_count>>4; c_itr++) \
    { \
      AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0); \
      AE_LA8X8X2_IP(_ae8x8_mat1_10, _ae8x8_mat1_11, _align_ae8x16_p_mat1_1, _ae8x16_p_mat1_1); \
      AE_LA8X8X2_IP(_ae8x8_mat1_20, _ae8x8_mat1_21, _align_ae8x16_p_mat1_2, _ae8x16_p_mat1_2); \
      AE_LA8X8X2_IP(_ae8x8_mat1_30, _ae8x8_mat1_31, _align_ae8x16_p_mat1_3, _ae8x16_p_mat1_3); \
      AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 16); \
      AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_00); \
      AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_31, _ae8x8_vec1_01); \
    } \
    { \
      d_acc0_0 = AE_ADD32S(d_acc0_0, AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1])); \
      d_acc1_0 = AE_ADD32S(d_acc1_0, AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3])); \
    } \
    MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift); \
    d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb); \
    d_acc1_0 = AE_ADD32S(d_acc1_0, d_outzb); \
    ae_int16x4 _ae_int16x4_out; \
    _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0); \
    AE_CVTA32X4F16(d_acc0_0, d_acc1_0, _ae_int16x4_out, 1); \
    { \
      ae_int16x4 *p_16x4_out, d_16x4_out; \
      ae_valign out_align; \
      p_16x4_out = (ae_int16x4 *)(&p_out[vec_itr*rows+m_itr]); \
      out_align = AE_LA64_PP(p_16x4_out); \
      AE_LA16X4_IP(d_16x4_out, out_align, p_16x4_out); \
      AE_ACCW16(d_acc0_0, d_acc1_0, d_16x4_out, AE_ZERO16()); \
      _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0); \
    } \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out); \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = (_ae_int16x4_out); \
  } \
_Pragma("no_unroll") \
_Pragma("loop_count max=3") \
  for(m_itr = (rows&(~3)); m_itr < rows; m_itr++) \
  { \
    ae_int32x2 d_acc0_0, d_acc1_0; \
    ae_int64 d64_acc0 = AE_ZERO64(); \
    ae_int64 d64_acc1 = AE_ZERO64(); \
    ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01; \
    ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1]; \
    ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0); \
    p_local_vec = (ae_int8x16 *)ptr_vec; \
    int cols_count=cols; \
    for(c_itr = 0; c_itr < cols_count>>4; c_itr++) \
    { \
      AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0); \
      AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 16); \
      AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_00, _ae8x8_vec1_00); \
      AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_01, _ae8x8_vec1_01); \
    } \
    d64_acc0 = AE_ADD64(d64_acc0, d64_acc1); \
    d_acc0_0 = AE_SAT32X2(d64_acc0, d64_acc0); \
    { \
      d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]); \
    } \
    MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift); \
    d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb); \
    ae_int16x4 _ae_int16x4_out; \
    _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0); \
    AE_CVTA32X4F16(d_acc0_0, d_acc1_0, _ae_int16x4_out, 1); \
    { \
      ae_int32x2 out0; \
      out0 = AE_MOVDA32(p_out[vec_itr*rows+m_itr]); \
      d_acc0_0 = AE_ADD32S(d_acc0_0, out0); \
      _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0); \
    } \
    *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = _ae_int16x4_out; \
  }
#endif
WORD32 xa_nn_matXvec_acc_batch_sym8sx8_asym16s_hU(
         WORD16 * __restrict__ p_out,           /* output pointer */
         const WORD8 *  __restrict__ p_mat1,    /* matrix1: rows x cols1 */
         const WORD8 * __restrict__ p_vec1,     /* vec1: cols1 x vec_count */
         const WORD32 *  __restrict__ p_bias,   /* bias: rows x 1 */
         WORD32 rows,
         WORD32 cols1,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 out_multiplier,                 /* out multiplier for quantization */
         WORD32 out_shift,                      /* out shift for quantization */
         WORD32 out_zero_bias,						          /* out zero bias for quantization */
         WORD32 vec_count)                      /* number of vectors */
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -32768 || out_zero_bias > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((vec_count < 0), -1);

  /* Iterators used in for loops */
  int m_itr, c_itr, vec_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  vec_itr = 0;
  int left_shift, right_shift;
  ae_int32x2 d_outzb = AE_MOVDA32(out_zero_bias);

  out_shift = out_shift - 1;
  d_outzb = AE_SRAI32(d_outzb, 1);

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
#if XCHAL_HAVE_HIFI5S
  left_shift = 31 - left_shift;
  left_shift = (left_shift << 16) | left_shift;
#endif  
  right_shift = out_shift;
  /* Single rounding macro doesn't need two shifts so this is not used */
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift > 0 ? out_shift : 0;
  right_shift = out_shift < 0 ? -out_shift : 0;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  ae_int32x2 min_word16 = AE_MOVDA32(-32768);
  ae_int32x2 max_word16 = AE_MOVDA32(32767);

  if(cols1 > 32 && cols1 <= 48)
  {
    for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
    {
      m_itr = 0;
      ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
      ae_int8x8 _ae8x8_vec1_02, _ae8x8_vec1_03;
      ae_int8x8 _ae8x8_vec1_04, _ae8x8_vec1_05;
      ae_int8x16 *_ae8x16_p_vec1_0;
      ae_valignx2 _align_ae8x16_p_vec1_0;

      _ae8x16_p_vec1_0 = (ae_int8x16 *)&p_vec1[vec_itr*cols1];
      _align_ae8x16_p_vec1_0 = AE_LA128_PP(_ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_02, _ae8x8_vec1_03, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LAV8X8X2_XP(_ae8x8_vec1_04, _ae8x8_vec1_05, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0, cols1-32);
      ae_int8x16 local_vec[3];
      ae_int8x16 *p_local_vec = (ae_int8x16 *)local_vec;
      AE_S8X8X2_I(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 0);
      AE_S8X8X2_I(_ae8x8_vec1_02, _ae8x8_vec1_03, p_local_vec, 16);
      AE_S8X8X2_I(_ae8x8_vec1_04, _ae8x8_vec1_05, p_local_vec, 32);
      
      MATXVEC_ALIGNED_VEC_8x8_ASYM16S_HU(48, local_vec);
    }
  }
  else if(cols1 > 48 && cols1 <= 64)
  {
    for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
    {
      m_itr = 0;
      ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
      ae_int8x8 _ae8x8_vec1_02, _ae8x8_vec1_03;
      ae_int8x8 _ae8x8_vec1_04, _ae8x8_vec1_05;
      ae_int8x8 _ae8x8_vec1_06, _ae8x8_vec1_07;
      ae_int8x16 *_ae8x16_p_vec1_0;
      ae_valignx2 _align_ae8x16_p_vec1_0;

      _ae8x16_p_vec1_0 = (ae_int8x16 *)&p_vec1[vec_itr*cols1];
      _align_ae8x16_p_vec1_0 = AE_LA128_PP(_ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_02, _ae8x8_vec1_03, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_04, _ae8x8_vec1_05, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LAV8X8X2_XP(_ae8x8_vec1_06, _ae8x8_vec1_07, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0, cols1-48);
      ae_int8x16 local_vec[4];
      ae_int8x16 *p_local_vec = (ae_int8x16 *)local_vec;
      AE_S8X8X2_I(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 0);
      AE_S8X8X2_I(_ae8x8_vec1_02, _ae8x8_vec1_03, p_local_vec, 16);
      AE_S8X8X2_I(_ae8x8_vec1_04, _ae8x8_vec1_05, p_local_vec, 32);
      AE_S8X8X2_X(_ae8x8_vec1_06, _ae8x8_vec1_07, p_local_vec, 48);
      
      MATXVEC_ALIGNED_VEC_8x8_ASYM16S_HU(64, local_vec);
    }
  }
  else if(cols1 > 64 && cols1 <= 80)
  {
    for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
    {
      m_itr = 0;
      ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
      ae_int8x8 _ae8x8_vec1_02, _ae8x8_vec1_03;
      ae_int8x8 _ae8x8_vec1_04, _ae8x8_vec1_05;
      ae_int8x8 _ae8x8_vec1_06, _ae8x8_vec1_07;
      ae_int8x8 _ae8x8_vec1_08, _ae8x8_vec1_09;
      ae_int8x16 *_ae8x16_p_vec1_0;
      ae_valignx2 _align_ae8x16_p_vec1_0;

      _ae8x16_p_vec1_0 = (ae_int8x16 *)&p_vec1[vec_itr*cols1];
      _align_ae8x16_p_vec1_0 = AE_LA128_PP(_ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_02, _ae8x8_vec1_03, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_04, _ae8x8_vec1_05, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LA8X8X2_IP(_ae8x8_vec1_06, _ae8x8_vec1_07, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
      AE_LAV8X8X2_XP(_ae8x8_vec1_08, _ae8x8_vec1_09, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0, cols1-64);
      ae_int8x16 local_vec[5];
      ae_int8x16 *p_local_vec = (ae_int8x16 *)local_vec;
      AE_S8X8X2_I(_ae8x8_vec1_00, _ae8x8_vec1_01, p_local_vec, 0);
      AE_S8X8X2_I(_ae8x8_vec1_02, _ae8x8_vec1_03, p_local_vec, 16);
      AE_S8X8X2_I(_ae8x8_vec1_04, _ae8x8_vec1_05, p_local_vec, 32);
      AE_S8X8X2_X(_ae8x8_vec1_06, _ae8x8_vec1_07, p_local_vec, 48);
      AE_S8X8X2_X(_ae8x8_vec1_08, _ae8x8_vec1_09, p_local_vec, 64);

      MATXVEC_ALIGNED_VEC_8x8_ASYM16S_HU(80, local_vec);
    }
  }
  else if((cols1&15) == 0 && cols1 == row_stride1 && ((unsigned)p_mat1&15) == 0 && ((unsigned)p_vec1&15) == 0)
  {
    vec_itr = 0;
    if(vec_count&1)
    {
      ae_int32x4 *pae_bias;
      ae_valignx2 b_a;
      pae_bias = (ae_int32x4 *)p_bias;
      b_a = AE_LA128_PP(pae_bias);
      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];
        ae_int8x8 _ae8x8_mat1_10, _ae8x8_mat1_11;
        ae_int8x16 *_ae8x16_p_mat1_1 = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];
        ae_int8x8 _ae8x8_mat1_20, _ae8x8_mat1_21;
        ae_int8x16 *_ae8x16_p_mat1_2 = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];
        ae_int8x8 _ae8x8_mat1_30, _ae8x8_mat1_31;
        ae_int8x16 *_ae8x16_p_mat1_3 = (ae_int8x16 *) &p_mat1[(m_itr+3)*row_stride1];

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0 = (ae_int8x16 *) p_vec1;

#pragma loop_count min=1
        for(c_itr = 0; c_itr < (cols1>>4); c_itr++)
        {
          AE_L8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _ae8x16_p_mat1_0, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_10, _ae8x8_mat1_11, _ae8x16_p_mat1_1, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_20, _ae8x8_mat1_21, _ae8x16_p_mat1_2, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_30, _ae8x8_mat1_31, _ae8x16_p_mat1_3, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _ae8x16_p_vec1_0, 16);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_31, _ae8x8_vec1_01);
        }

        {
          ae_int32x2 d_bias01, d_bias23;
          AE_LA32X2X2_IP(d_bias01, d_bias23, b_a, pae_bias);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        d_acc1_0 = AE_ADD32S(d_acc1_0, d_outzb);
        AE_MINMAX32(d_acc0_0, min_word16, max_word16);
        AE_MINMAX32(d_acc1_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        d_acc1_0 = AE_SLAI32(d_acc1_0, 1);
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[m_itr], p_out[m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[m_itr+2], p_out[m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
        *(ae_int16 *)&p_out[m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[m_itr+3] = (_ae_int16x4_out);
      }

#pragma no_unroll
#pragma loop_count max=3
      for(m_itr = (rows & (~3)); m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0;
        ae_int64 d64_acc0 = AE_ZERO64();
        ae_int64 d64_acc1 = AE_ZERO64();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0 = (ae_int8x16 *) p_vec1;

#pragma loop_count min=1
        for(c_itr = 0; c_itr < (cols1>>4); c_itr++)
        {
          AE_L8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _ae8x16_p_mat1_0, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _ae8x16_p_vec1_0, 16);
          AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_00, _ae8x8_vec1_00);
          AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_01, _ae8x8_vec1_01);
        }
        d64_acc0 = AE_ADD64(d64_acc0, d64_acc1);
        d_acc0_0 = AE_SAT32X2(d64_acc0, d64_acc0);
        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        AE_MINMAX32(d_acc0_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32(p_out[m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
      }
      vec_itr++;
    }
    for (; vec_itr < vec_count; vec_itr+=2)
    {
      for(m_itr = 0; m_itr < (rows - (4-1)); m_itr += 4)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();
        ae_int32x2 d_acc0_1 = AE_ZERO32();
        ae_int32x2 d_acc1_1 = AE_ZERO32();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];
        ae_int8x8 _ae8x8_mat1_10, _ae8x8_mat1_11;
        ae_int8x16 *_ae8x16_p_mat1_1 = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];
        ae_int8x8 _ae8x8_mat1_20, _ae8x8_mat1_21;
        ae_int8x16 *_ae8x16_p_mat1_2 = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];
        ae_int8x8 _ae8x8_mat1_30, _ae8x8_mat1_31;
        ae_int8x16 *_ae8x16_p_mat1_3 = (ae_int8x16 *) &p_mat1[(m_itr+3)*row_stride1];

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0 = (ae_int8x16 *) &p_vec1[(vec_itr+0)*cols1];
        ae_int8x8 _ae8x8_vec1_10, _ae8x8_vec1_11;
        ae_int8x16 *_ae8x16_p_vec1_1 = (ae_int8x16 *) &p_vec1[(vec_itr+1)*cols1];

        for(c_itr = 0; c_itr < (cols1>>4); c_itr++)
        {
          AE_L8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _ae8x16_p_mat1_0, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_10, _ae8x8_mat1_11, _ae8x16_p_mat1_1, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_20, _ae8x8_mat1_21, _ae8x16_p_mat1_2, 16);
          AE_L8X8X2_IP(_ae8x8_mat1_30, _ae8x8_mat1_31, _ae8x16_p_mat1_3, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _ae8x16_p_vec1_0, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_10, _ae8x8_vec1_11, _ae8x16_p_vec1_1, 16);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_31, _ae8x8_vec1_01);
          AE_MULA8Q8X8(d_acc0_1, d_acc1_1, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_30, _ae8x8_vec1_10);
          AE_MULA8Q8X8(d_acc0_1, d_acc1_1, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_31, _ae8x8_vec1_11);
        }

        {
          ae_int32x2 d_bias01, d_bias23;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias23 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 2], *(ae_int32 *)&p_bias[m_itr + 3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias23);
          d_acc0_1 = AE_ADD32S(d_acc0_1, d_bias01);
          d_acc1_1 = AE_ADD32S(d_acc1_1, d_bias23);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2X2_OUT32_HIFI5S(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_0, d_acc1_0, d_acc0_0, d_acc1_0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2X2_OUT32(d_acc0_1, d_acc1_1, d_acc0_1, d_acc1_1, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        d_acc1_0 = AE_ADD32S(d_acc1_0, d_outzb);
        d_acc0_1 = AE_ADD32S(d_acc0_1, d_outzb);
        d_acc1_1 = AE_ADD32S(d_acc1_1, d_outzb);
        AE_MINMAX32(d_acc0_0, min_word16, max_word16);
        AE_MINMAX32(d_acc1_0, min_word16, max_word16);
        AE_MINMAX32(d_acc0_1, min_word16, max_word16);
        AE_MINMAX32(d_acc1_1, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        d_acc1_0 = AE_SLAI32(d_acc1_0, 1);
        d_acc0_1 = AE_SLAI32(d_acc0_1, 1);
        d_acc1_1 = AE_SLAI32(d_acc1_1, 1);
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr+2], p_out[vec_itr*rows+m_itr+3]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
          out0 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr+1]);
          out1 = AE_MOVDA32X2(p_out[(vec_itr+1)*rows+m_itr+2], p_out[(vec_itr+1)*rows+m_itr+3]);
          d_acc0_1 = AE_ADD32S(d_acc0_1, out0);
          d_acc1_1 = AE_ADD32S(d_acc1_1, out1);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+3] = (_ae_int16x4_out);
        
        _ae_int16x4_out = AE_SAT16X4(d_acc0_1, d_acc1_1);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+2] = AE_SEL16_4321(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr+3] = (_ae_int16x4_out);
      }
      
#pragma no_unroll
      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc0_1 = AE_ZERO32();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0 = (ae_int8x16 *) &p_vec1[(vec_itr+0)*cols1];
        ae_int8x8 _ae8x8_vec1_10, _ae8x8_vec1_11;
        ae_int8x16 *_ae8x16_p_vec1_1 = (ae_int8x16 *) &p_vec1[(vec_itr+1)*cols1];

        for(c_itr = 0; c_itr < (cols1>>4); c_itr++)
        {
          AE_L8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _ae8x16_p_mat1_0, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _ae8x16_p_vec1_0, 16);
          AE_L8X8X2_IP(_ae8x8_vec1_10, _ae8x8_vec1_11, _ae8x16_p_vec1_1, 16);
          AE_MULA8Q8X8(d_acc0_0, d_acc0_1, _ae8x8_vec1_00, _ae8x8_vec1_10, _ae8x8_vec1_00, _ae8x8_vec1_10, _ae8x8_mat1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc0_1, _ae8x8_vec1_01, _ae8x8_vec1_11, _ae8x8_vec1_01, _ae8x8_vec1_11, _ae8x8_mat1_01);
        }

        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        AE_MINMAX32(d_acc0_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[(vec_itr+1)*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        
        *(ae_int16 *)&p_out[(vec_itr+1)*rows+m_itr] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
      }
    }
  }
  else
  {
    for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
    {
      for(m_itr = 0; m_itr < (rows - (3-1)); m_itr += 3)
      {
        ae_int32x2 d_acc0_0 = AE_ZERO32();
        ae_int32x2 d_acc1_0 = AE_ZERO32();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];
        ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0);
        ae_int8x8 _ae8x8_mat1_10, _ae8x8_mat1_11;
        ae_int8x16 *_ae8x16_p_mat1_1 = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];
        ae_valignx2 _align_ae8x16_p_mat1_1 = AE_LA128_PP(_ae8x16_p_mat1_1);
        ae_int8x8 _ae8x8_mat1_20, _ae8x8_mat1_21;
        ae_int8x16 *_ae8x16_p_mat1_2 = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];
        ae_valignx2 _align_ae8x16_p_mat1_2 = AE_LA128_PP(_ae8x16_p_mat1_2);

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0;
        ae_valignx2 _align_ae8x16_p_vec1_0;

        _ae8x16_p_vec1_0 = (ae_int8x16 *)&p_vec1[vec_itr*cols1];
        _align_ae8x16_p_vec1_0 = AE_LA128_PP(_ae8x16_p_vec1_0);

        int cols_count=cols1&(~15);
        for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
        {
          AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0);
          AE_LA8X8X2_IP(_ae8x8_mat1_10, _ae8x8_mat1_11, _align_ae8x16_p_mat1_1, _ae8x16_p_mat1_1);
          AE_LA8X8X2_IP(_ae8x8_mat1_20, _ae8x8_mat1_21, _align_ae8x16_p_mat1_2, _ae8x16_p_mat1_2);
          AE_LA8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_20, _ae8x8_vec1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_21, _ae8x8_vec1_01);
        }

        if(cols_count < cols1)
        {
          AE_LAV8X8X2_XP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0, cols1 - cols_count);
          AE_LAV8X8X2_XP(_ae8x8_mat1_10, _ae8x8_mat1_11, _align_ae8x16_p_mat1_1, _ae8x16_p_mat1_1, cols1 - cols_count);
          AE_LAV8X8X2_XP(_ae8x8_mat1_20, _ae8x8_mat1_21, _align_ae8x16_p_mat1_2, _ae8x16_p_mat1_2, cols1 - cols_count);
          AE_LAV8X8X2_XP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0, cols1 - cols_count);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_00, _ae8x8_mat1_10, _ae8x8_mat1_20, _ae8x8_mat1_20, _ae8x8_vec1_00);
          AE_MULA8Q8X8(d_acc0_0, d_acc1_0, _ae8x8_mat1_01, _ae8x8_mat1_11, _ae8x8_mat1_21, _ae8x8_mat1_21, _ae8x8_vec1_01);
        }
        {
          ae_int32x2 d_bias01, d_bias22;
          d_bias01 = AE_SEL32_LL(*(ae_int32 *)&p_bias[m_itr + 0], *(ae_int32 *)&p_bias[m_itr + 1]);
          d_bias22 = *(ae_int32 *)&p_bias[m_itr + 2];
          d_acc0_0 = AE_ADD32S(d_acc0_0, d_bias01);
          d_acc1_0 = AE_ADD32S(d_acc1_0, d_bias22);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc1_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc1_0, d_acc1_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        d_acc1_0 = AE_ADD32S(d_acc1_0, d_outzb);
        AE_MINMAX32(d_acc0_0, min_word16, max_word16);
        AE_MINMAX32(d_acc1_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        d_acc1_0 = AE_SLAI32(d_acc1_0, 1);
        {
          ae_int32x2 out0, out1;
          out0 = AE_MOVDA32X2(p_out[vec_itr*rows+m_itr], p_out[vec_itr*rows+m_itr+1]);
          out1 = AE_MOVDA32(p_out[vec_itr*rows+m_itr+2]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
          d_acc1_0 = AE_ADD32S(d_acc1_0, out1);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc1_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = AE_SEL16_6543(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+1] = AE_SEL16_5432(_ae_int16x4_out, _ae_int16x4_out);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr+2] = (_ae_int16x4_out);
      }
      
      
      for(; m_itr < rows; m_itr++)
      {
        ae_int32x2 d_acc0_0;
        ae_int64 d64_acc0 = AE_ZERO64();
        ae_int64 d64_acc1 = AE_ZERO64();
        ae_int8x8 _ae8x8_mat1_00, _ae8x8_mat1_01;
        ae_int8x16 *_ae8x16_p_mat1_0 = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];
        ae_valignx2 _align_ae8x16_p_mat1_0 = AE_LA128_PP(_ae8x16_p_mat1_0);

        ae_int8x8 _ae8x8_vec1_00, _ae8x8_vec1_01;
        ae_int8x16 *_ae8x16_p_vec1_0;
        ae_valignx2 _align_ae8x16_p_vec1_0;

        _ae8x16_p_vec1_0 = (ae_int8x16 *)&p_vec1[vec_itr*cols1];
        _align_ae8x16_p_vec1_0 = AE_LA128_PP(_ae8x16_p_vec1_0);

        int cols_count=cols1&(~15);
        for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
        {
          AE_LA8X8X2_IP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0);
          AE_LA8X8X2_IP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0);
          AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_00, _ae8x8_vec1_00);
          AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_01, _ae8x8_vec1_01);
        }

        if(cols_count < cols1)
        {
          AE_LAV8X8X2_XP(_ae8x8_mat1_00, _ae8x8_mat1_01, _align_ae8x16_p_mat1_0, _ae8x16_p_mat1_0, cols1 - cols_count);
          AE_LAV8X8X2_XP(_ae8x8_vec1_00, _ae8x8_vec1_01, _align_ae8x16_p_vec1_0, _ae8x16_p_vec1_0, cols1 - cols_count);
          AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_00, _ae8x8_vec1_00);
          AE_MULAAAA2Q8(d64_acc0, d64_acc1, _ae8x8_mat1_01, _ae8x8_vec1_01);
        }
        d64_acc0 = AE_ADD64(d64_acc0, d64_acc1);
        d_acc0_0 = AE_SAT32X2(d64_acc0, d64_acc0);
        {
          d_acc0_0 = AE_ADD32S(d_acc0_0, *(ae_int32 *)&p_bias[m_itr + 0]);
        }
#if (XCHAL_HAVE_HIFI5S && TFLITE_SINGLE_ROUNDING)
        MPY_BY_QUANT_MULT_X2_OUT32_HIFI5S(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#else        
        MPY_BY_QUANT_MULT_X2_OUT32(d_acc0_0, d_acc0_0, out_multiplier, left_shift, right_shift);
#endif        
        d_acc0_0 = AE_ADD32S(d_acc0_0, d_outzb);
        AE_MINMAX32(d_acc0_0, min_word16, max_word16);
        d_acc0_0 = AE_SLAI32(d_acc0_0, 1);
        {
          ae_int32x2 out0;
          out0 = AE_MOVDA32(p_out[vec_itr*rows+m_itr]);
          d_acc0_0 = AE_ADD32S(d_acc0_0, out0);
        }
        ae_int16x4 _ae_int16x4_out;
        _ae_int16x4_out = AE_SAT16X4(d_acc0_0, d_acc0_0);
        *(ae_int16 *)&p_out[vec_itr*rows+m_itr] = _ae_int16x4_out;
      }
    }
  }

  return 0;
}
