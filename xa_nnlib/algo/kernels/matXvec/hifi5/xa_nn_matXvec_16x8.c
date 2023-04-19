/*******************************************************************************
* Copyright (c) 2018-2023 Cadence Design Systems, Inc.
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

WORD32 xa_nn_matXvec_16x8_16(
         WORD16 * __restrict__ p_out,  /* output */
         WORD16 * __restrict__ p_mat1, /* matrix1: rows x cols1 */
         WORD16 * __restrict__ p_mat2, /* matrix2: rows x cols2 */
         WORD8 * __restrict__ p_vec1,  /* vec1: cols1 x 1 */
         WORD8 * __restrict__ p_vec2,  /* vec2: cols2 x 1 */
         WORD16 * __restrict__ p_bias, /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,           /* row stride for matrix1 */
         WORD32 row_stride2,           /* row stride for matrix2 */
         WORD32 acc_shift,               /* out accumulator shift amount */
         WORD32 bias_shift)              /* bias shift amount */
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
#define UNROLL_SETUP_MAT1       SETUP_MAT1_16b
#define UNROLL_SETUP_MAT2       SETUP_MAT2_16b
#define UNROLL_STORE_ACC        STORE_ACC_16bx8b_AT_OUT_16b
#define SETUP_VEC1              SETUP_VEC1_8b_SINGLE
#define SETUP_VEC2              SETUP_VEC2_8b_SINGLE
#define UNROLL_LOAD_MAT1        LOAD_ROW_MAT1_16b
#define UNROLL_LOAD_MAT2        LOAD_ROW_MAT2_16b
#define LOAD_VEC1               LOAD_VEC1_8b_SINGLE
#define LOAD_VEC2               LOAD_VEC2_8b_SINGLE
#define SETUP_BIAS              SETUP_BIAS_16b
#define UNROLL_ADD_BIAS_ACC     ADD_BIAS_16b_ACC_FOR_16bx8b

  acc_shift=32+acc_shift;

  if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && cols1 % 8 ==0 && cols2 %8 ==0)
  {
    /* All four pointers are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
      {
        SETUP_OUTPUT_STORE_16x4;
        SETUP_ACC;
        SETUP_MAT1;
        SETUP_VEC1;
        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
        {
            LOAD_VEC1;
            LOAD_MAT1;
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(0,1);
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(2,3);
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(4,5);
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(6,7);
        }
        SETUP_MAT2;
        SETUP_VEC2_8b_SINGLE;
        for(c_itr = 0; c_itr < (cols2 >> 3); c_itr++)
        {
            LOAD_VEC2;
            LOAD_MAT2;
            KERNEL_MAT2_16_VEC2_8_TWO_ACCUMULATOR(0,1);
            KERNEL_MAT2_16_VEC2_8_TWO_ACCUMULATOR(2,3);
            KERNEL_MAT2_16_VEC2_8_TWO_ACCUMULATOR(4,5);
            KERNEL_MAT2_16_VEC2_8_TWO_ACCUMULATOR(6,7);
        }
        ADD_BIAS_ACC;
        STORE_ACC_8bx16b_AT_OUT_16x4(0,1,2,3);
        STORE_ACC_8bx16b_AT_OUT_16x4(4,5,6,7);
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        UNROLL_SETUP_ACC(0);
        UNROLL_SETUP_ACC(1);
        SETUP_VEC1;
        UNROLL_SETUP_MAT1(0);
        ae_int16x4 _ae_int16x4_mat1_1 = AE_ZERO16();
        ae_int16x4 _ae_int16x4_mat1_1_1 = AE_ZERO16();
        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
        {
          LOAD_VEC1;
          UNROLL_LOAD_MAT1(0);
          KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(0,1);
        }

        SETUP_VEC2;
        UNROLL_SETUP_MAT2(0);
        ae_int16x4 _ae_int16x4_mat2_1 = AE_ZERO16();
        ae_int16x4 _ae_int16x4_mat2_1_1 = AE_ZERO16();
        for(c_itr = 0; c_itr < (cols2 >> 3); c_itr++)
        {
          LOAD_VEC2;
          UNROLL_LOAD_MAT2(0);
          KERNEL_MAT2_16_VEC2_8_TWO_ACCUMULATOR(0,1);
        }
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_STORE_ACC(0);
      }
    }
  }
  else if(p_mat1 && p_vec1 && p_mat2 && p_vec2)
  {
    SETUP_BIAS;
    if(rows > 2)
    {
      for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
      {
        SETUP_OUTPUT_STORE_16;
        UNROLL_SETUP_ACC(0);
        UNROLL_SETUP_ACC(1);
        SETUP_MAT1_16b_UNALIGNED_16x8(0);
        SETUP_MAT1_16b_UNALIGNED_16x8(1);
        SETUP_VEC1_8b_UNALIGNED;
        int cols_count=cols1-cols1%8;
        for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)
        {
            LOAD_VEC1_8b_UNALIGNED;
            LOAD_ROW_MAT1_16b_UNALIGNED_16x8(0);
            LOAD_ROW_MAT1_16b_UNALIGNED_16x8(1);
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(0,1);
        }
        for(c_itr=cols_count;c_itr<cols1;c_itr++)
        {
            LOAD_ROW_MAT1_16b_UNALIGNED_SINGLE_ELEMENT_16x8(0);
            LOAD_ROW_MAT1_16b_UNALIGNED_SINGLE_ELEMENT_16x8(1);
            LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT;
            KERNEL_MAT1_VEC1_16b_8b_UNALIGNED_SINGLE_ELEMENT;
        }
        SETUP_MAT2_16b_UNALIGNED_16x8(0);
        SETUP_MAT2_16b_UNALIGNED_16x8(1);
        SETUP_VEC2_8b_UNALIGNED;
        cols_count=cols2-cols2%8;
        for(c_itr = 0; c_itr < (cols2 >> 3); c_itr++)
        {
            LOAD_VEC2_8b_UNALIGNED;
            LOAD_ROW_MAT2_16b_UNALIGNED_16x8(0);
            LOAD_ROW_MAT2_16b_UNALIGNED_16x8(1);
            KERNEL_MAT2_16_VEC2_8_TWO_ACCUMULATOR(0,1);
        }
        for(c_itr=cols_count;c_itr<cols2;c_itr++)
        {
            LOAD_ROW_MAT2_16b_UNALIGNED_SINGLE_ELEMENT_16x8(0);
            LOAD_ROW_MAT2_16b_UNALIGNED_SINGLE_ELEMENT_16x8(1);
            LOAD_VEC2_8b_UNALIGNED_SINGLE_ELEMENT;
            KERNEL_MAT2_VEC2_16b_8b_UNALIGNED_SINGLE_ELEMENT;
        }
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_ADD_BIAS_ACC(1);
        UNROLL_STORE_ACC(0);
        UNROLL_STORE_ACC(1);
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_16;
        UNROLL_SETUP_ACC(0);
        UNROLL_SETUP_ACC(1);
        SETUP_MAT1_16b_UNALIGNED_16x8(0);
        SETUP_VEC1_8b_UNALIGNED;
        int cols_count=cols1-cols1%8;
        ae_int16x4 _ae_int16x4_mat1_1 = AE_ZERO16();
        ae_int16x4 _ae_int16x4_mat1_1_1 = AE_ZERO16();

        for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)
        {
            LOAD_VEC1_8b_UNALIGNED;
            LOAD_ROW_MAT1_16b_UNALIGNED_16x8(0);
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(0,1);
        }
        for(c_itr=cols_count;c_itr<cols1;c_itr++)
        {
            LOAD_ROW_MAT1_16b_UNALIGNED_SINGLE_ELEMENT_16x8(0);
            LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT;
            KERNEL_MAT1_VEC1_16b_8b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW;
        }
        SETUP_MAT2_16b_UNALIGNED_16x8(0);
        SETUP_VEC2_8b_UNALIGNED;
        cols_count=cols2-cols2%8;
        ae_int16x4 _ae_int16x4_mat2_1 = AE_ZERO16();
        ae_int16x4 _ae_int16x4_mat2_1_1 = AE_ZERO16();
        for(c_itr = 0; c_itr < (cols2 >> 3); c_itr++)
        {
            LOAD_VEC2_8b_UNALIGNED;
            LOAD_ROW_MAT2_16b_UNALIGNED_16x8(0);
            KERNEL_MAT2_16_VEC2_8_TWO_ACCUMULATOR(0,1);
        }
        for(c_itr=cols_count;c_itr<cols2;c_itr++)
        {
            LOAD_ROW_MAT2_16b_UNALIGNED_SINGLE_ELEMENT_16x8(0);
            LOAD_VEC2_8b_UNALIGNED_SINGLE_ELEMENT;
            KERNEL_MAT2_VEC2_16b_8b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW;
        }
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_STORE_ACC(0);
      }
    }
  }
  else if (p_mat1 && p_vec1 && cols1%8==0)
  {
    /* Only mat1, vec1 are non-null */
    SETUP_BIAS;
    if(rows > ROW_UNROLL)
    {
      for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)) ; m_itr += ROW_UNROLL)
      {
        SETUP_OUTPUT_STORE_16x4;
        SETUP_ACC;
        SETUP_MAT1;
        SETUP_VEC1;
        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
        {
            LOAD_VEC1;
            LOAD_MAT1;
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(0,1);
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(2,3);
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(4,5);
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(6,7);
        }
        ADD_BIAS_ACC;
        STORE_ACC_8bx16b_AT_OUT_16x4(0,1,2,3);
        STORE_ACC_8bx16b_AT_OUT_16x4(4,5,6,7);
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        UNROLL_SETUP_ACC(0);
        UNROLL_SETUP_ACC(1);
        SETUP_VEC1;
        UNROLL_SETUP_MAT1(0);
        ae_int16x4 _ae_int16x4_mat1_1 = AE_ZERO16();
        ae_int16x4 _ae_int16x4_mat1_1_1 = AE_ZERO16();
        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
        {
          LOAD_VEC1;
          UNROLL_LOAD_MAT1(0);
          KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(0,1);
        }
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_STORE_ACC(0);
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
    SETUP_BIAS;
    if(rows > 2)
    {
      for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
      {
        SETUP_OUTPUT_STORE_16;
        UNROLL_SETUP_ACC(0);
        UNROLL_SETUP_ACC(1);
        SETUP_MAT1_16b_UNALIGNED_16x8(0);
        SETUP_MAT1_16b_UNALIGNED_16x8(1);
        SETUP_VEC1_8b_UNALIGNED;
        int cols_count=cols1-cols1%8;
        for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)
        {
            LOAD_VEC1_8b_UNALIGNED;
            LOAD_ROW_MAT1_16b_UNALIGNED_16x8(0);
            LOAD_ROW_MAT1_16b_UNALIGNED_16x8(1);
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(0,1);
        }
        for(c_itr=cols_count;c_itr<cols1;c_itr++)
        {
            LOAD_ROW_MAT1_16b_UNALIGNED_SINGLE_ELEMENT_16x8(0);
            LOAD_ROW_MAT1_16b_UNALIGNED_SINGLE_ELEMENT_16x8(1);
            LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT;
            KERNEL_MAT1_VEC1_16b_8b_UNALIGNED_SINGLE_ELEMENT;
        }
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_ADD_BIAS_ACC(1);
        UNROLL_STORE_ACC(0);
        UNROLL_STORE_ACC(1);
      }
    }
    {
      for(; m_itr < rows; m_itr++)
      {
        SETUP_OUTPUT_STORE_16;
        UNROLL_SETUP_ACC(0);
        UNROLL_SETUP_ACC(1);
        SETUP_MAT1_16b_UNALIGNED_16x8(0);
        SETUP_VEC1_8b_UNALIGNED;
        int cols_count=cols1-cols1%8;
        ae_int16x4 _ae_int16x4_mat1_1 = AE_ZERO16();
        ae_int16x4 _ae_int16x4_mat1_1_1 = AE_ZERO16();

        for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)
        {
            LOAD_VEC1_8b_UNALIGNED;
            LOAD_ROW_MAT1_16b_UNALIGNED_16x8(0);
            KERNEL_MAT1_16_VEC1_8_TWO_ACCUMULATOR(0,1);
        }
        for(c_itr=cols_count;c_itr<cols1;c_itr++)
        {
            LOAD_ROW_MAT1_16b_UNALIGNED_SINGLE_ELEMENT_16x8(0);
            LOAD_VEC1_8b_UNALIGNED_SINGLE_ELEMENT;
            KERNEL_MAT1_VEC1_16b_8b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW;
        }
        UNROLL_ADD_BIAS_ACC(0);
        UNROLL_STORE_ACC(0);
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
#undef UNROLL_KERNEL_MAT1_VEC1
#undef UNROLL_KERNEL_MAT2_VEC2
#undef UNROLL_STORE_ACC
#undef SETUP_VEC1
#undef SETUP_VEC2
#undef LOAD_VEC1
#undef LOAD_VEC2
#undef SETUP_BIAS
#undef UNROLL_ADD_BIAS_ACC

  return 0;
}
