/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
  /* Iterators used in for loops */
  int m_itr, c_itr, vec_itr;
  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  vec_itr = 0;

  if (!p_bias)
  {
    return -1;
  }

#define VEC_UNROLL 2
#define SETUP_BIAS                          SETUP_BIAS_8b
#define LOAD_BIAS                           LOAD_BIAS_8b_FOR_8bx8b
  // ae_int32x2 _ae_int32x2_sat_bias;
  // ae_int32x2 sat_1,sat_2;
  acc_shift = acc_shift + 32;
  if (p_mat1 && p_vec1 && ((cols1&15) == 0) && ((row_stride1&15) == 0)
      && (cols1 > 0) && (row_stride1 > 0))
  {
    if(rows > ROW_UNROLL)
    {
      if(vec_count > VEC_UNROLL)
      {
        for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
        {
          // ae_int8x8 _ae_int8_bias;
          // ae_int8x8 _ae_int8_bias_1;
          // ae_int8 *_ae_int8_p_bias = (ae_int8 *) p_bias;
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
            // AE_S32X2_I(temp32_1, output_1ptr, 0*sizeof(ae_int32x2));
            // AE_S32X2_I(temp32_3, output_ptr, 1*sizeof(ae_int32x2));
            // AE_S32X2_I(temp32_2, output_ptr1, 0*sizeof(ae_int32x2));
            // AE_S32X2_I(temp32_4, output_ptr1, 1*sizeof(ae_int32x2));

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
            // AE_S32X2_I(temp32_1, output_1ptr, 0*sizeof(ae_int32x2));

            // AE_S32X2_I(temp32_1, output_ptr, 2*sizeof(ae_int32x2));
            // AE_S32X2_I(temp32_3, output_ptr, 3*sizeof(ae_int32x2));
            // AE_S32X2_I(temp32_2, output_ptr1, 2*sizeof(ae_int32x2));
            // AE_S32X2_I(temp32_4, output_ptr1, 3*sizeof(ae_int32x2));
            //AE_S32X2X2_I(temp32_1,temp32_3, output_ptr, 1*sizeof(ae_int32x4));
            //AE_S32X2X2_I(temp32_2,temp32_4, output_ptr1, 1*sizeof(ae_int32x4));
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

