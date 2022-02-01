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
#include "common_fpu.h"
#include "xa_nnlib_common.h"

#if HAVE_VFPU

#if defined(CUST_UNROLL) && (CUST_UNROLL != 0)
#define UNROLL_S CUST_UNROLL
#else
#define UNROLL_S  8 /// Optimal unroll
#endif

#define SETUP_ROW_S(N) \
  xtfloatx2 accu1_ ##N;\
  xtfloatx2 accu1_1_ ##N;\
  xtfloatx4 *p_mat1_ ##N = (xtfloatx4*)&p_mat[(row+N)*cols]; \
  accu1_ ##N = (xtfloatx2)0.0f; \
  accu1_1_ ##N = (xtfloatx2)0.0f;

#define KERNEL_ROW_S(N) \
{ \
  xtfloatx2 temp_in1; \
  xtfloatx2 temp_in2; \
  AE_LSX2X2_IP(temp_in1,temp_in2, p_mat1_ ##N, 16); \
  MADD_SX2X2(accu1_ ##N, accu1_1_ ##N, temp_src1, temp_src2, temp_in1, temp_in2);\
}

#define KERNEL_ROW_S_I(N) \
{ \
  xtfloatx2 temp_in1; \
  xtfloatx2 temp_in2; \
  AE_LSX2X2_IP(temp_in1,temp_in2, p_mat1_ ##N, 16); \
  AE_LSX2XC(temp_src1,p_src1, 8); \
  AE_LSX2XC(temp_src2,p_src1, 8); \
  MADD_SX2X2(accu1_ ##N, accu1_1_ ##N, temp_src1, temp_src2, temp_in1, temp_in2);\
}

#define STORE_ROW_S(N) \
  accu1_ ##N =(accu1_ ##N + accu1_1_ ##N);\
  xtfloat raccu1_ ##N = RADD_SX2(accu1_ ##N); \
  xtfloat bias_ ##N = p_bias[row+N]; \
  p_out[(row+N)*out_offset] = ADD_S(raccu1_ ##N , bias_ ##N);

#if (UNROLL_S == 1)
#define SETUP_S SETUP_ROW_S(0)
#define KERNEL_S KERNEL_ROW_S_I(0)
#define STORE_S STORE_ROW_S(0)

#elif (UNROLL_S == 2)
#define SETUP_S  SETUP_ROW_S(0)  SETUP_ROW_S(1)
#define KERNEL_S KERNEL_ROW_S_I(0) KERNEL_ROW_S(1)
#define STORE_S  STORE_ROW_S(0)  STORE_ROW_S(1)

#elif (UNROLL_S == 4)
#define SETUP_S  SETUP_ROW_S(0)  SETUP_ROW_S(1)  SETUP_ROW_S(2)  SETUP_ROW_S(3)
#define KERNEL_S KERNEL_ROW_S_I(0) KERNEL_ROW_S(1) KERNEL_ROW_S(2) KERNEL_ROW_S(3)
#define STORE_S  STORE_ROW_S(0)  STORE_ROW_S(1)  STORE_ROW_S(2)  STORE_ROW_S(3)
#elif (UNROLL_S == 8)
#define SETUP_S   SETUP_ROW_S(0)  SETUP_ROW_S(1)  SETUP_ROW_S(2)  SETUP_ROW_S(3)  SETUP_ROW_S(4)  SETUP_ROW_S(5)  SETUP_ROW_S(6)  SETUP_ROW_S(7)
#define KERNEL_S KERNEL_ROW_S_I(0) KERNEL_ROW_S(1) KERNEL_ROW_S(2) KERNEL_ROW_S(3) KERNEL_ROW_S(4) KERNEL_ROW_S(5) KERNEL_ROW_S(6) KERNEL_ROW_S(7)
#define STORE_S   STORE_ROW_S(0)  STORE_ROW_S(1)  STORE_ROW_S(2)  STORE_ROW_S(3)  STORE_ROW_S(4)  STORE_ROW_S(5)  STORE_ROW_S(6)  STORE_ROW_S(7)

#endif

WORD32 xa_nn_matXvec_f32_circ_nb(
  FLOAT32 * __restrict__ p_out,
  FLOAT32 * __restrict__ p_mat,
  FLOAT32 * __restrict__ p_vec,
  FLOAT32 * __restrict__ p_bias,
  WORD32 rows,
  WORD32 cols,
  WORD32 out_offset)
{
  WORD32 row, col;
  if ((NULL == p_out) || (NULL == p_mat) || (NULL == p_vec))
  {
    return -1;
  }

  if ((0 >= rows ) || (0 >= cols ) || (cols & 0x1))
  {
    return -2;
  }

  row = 0;
  if(cols %4 ==0)
  {
      xtfloatx2 temp_src1;
      xtfloatx2 temp_src2;
      if(rows >=UNROLL_S)
      {
        for (row = 0; row < ( rows & ~(UNROLL_S-1)) ; row+=UNROLL_S)
        {
          xtfloatx2 *p_src1 = (xtfloatx2 *)p_vec;
          SETUP_S;
          for (col = 0; col < (cols>>2); col++)
          {
              KERNEL_S;
          }
          STORE_S;
        }
      }
      // Handle remaining rows
      for (; row < rows ; row++)
      {
        xtfloatx2 *p_src1 = (xtfloatx2*)p_vec;
        xtfloatx2 accu1_0;
        xtfloatx2 temp_in1;
        xtfloatx2 temp_in2;
        xtfloatx2 *p_mat1_0 = (xtfloatx2*)&p_mat[(row)*cols];
        accu1_0 = (xtfloatx2)0.0f;
        for (col = 0; col < (cols>>2); col++)
        {
            AE_LSX2IP(temp_in1,p_mat1_0,8);
            AE_LSX2XC(temp_src1,p_src1,8);
            MADD_SX2(accu1_0,temp_src1,temp_in1);
            AE_LSX2IP(temp_in2,p_mat1_0,8);
            AE_LSX2XC(temp_src2,p_src1,8);
            MADD_SX2(accu1_0,temp_src2,temp_in2);
        }
        xtfloat raccu1_0 = RADD_SX2(accu1_0);
        xtfloat bias_0 = p_bias[row];
        p_out[(row)*out_offset] = ADD_S(raccu1_0 , bias_0);
      }
  }
  else
  {
      // Support of non-multiples of 4 cols
      xtfloat temp_src1;
      xtfloat temp_src2;
      for (row = 0; row < rows ; row++)
      {
        xtfloat *p_src1 = (xtfloat*)p_vec;
        xtfloat accu1_0;
        xtfloat temp_in1;
        xtfloat temp_in2;
        xtfloat *p_mat1_0 = (xtfloat*)&p_mat[(row)*cols];
        accu1_0 = (xtfloat)0.0f;
        for (col = 0; col < cols>>1; col++)
        {
            AE_LSIP(temp_in1,p_mat1_0,4);
            AE_LSXC(temp_src1,p_src1,4);
            MADD_S(accu1_0,temp_src1,temp_in1);
            AE_LSIP(temp_in2,p_mat1_0,4);
            AE_LSXC(temp_src2,p_src1,4);
            MADD_S(accu1_0,temp_src2,temp_in2);
        }
        xtfloat bias_0 = p_bias[row];
        p_out[(row)*out_offset] = ADD_S(accu1_0 , bias_0);
      }
  }
  return 0;
}

#endif /* HAVE_VFPU */

