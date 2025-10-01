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

#define SW_MOVDA32(a) AE_MOVDA32X2(a, a)
#define SW_SLAA64S_INT64_INT64(inp1, bias_shift) AE_MOVINT64_FROMF64(AE_SLAA64S(AE_MOVF64_FROMINT64(inp1), bias_shift))

#define ZERO64   AE_MOVINT64_FROMINT32X2(SW_MOVDA32(0))

#if defined(CUST_UNROLL) && (CUST_UNROLL != 0)
#define UNROLL_S CUST_UNROLL
#else
#define UNROLL_S  8 /// Optimal unroll
#endif

/**************************** Multiple of 16 ***********************************************************/
#define SETUP_ROW_S_16(N) \
  ae_int64 accu1_ ##N;\
  ae_int8x16 *p_mat1_ ##N = (ae_int8x16*)(&p_mat[(row+N)*cols]); \
  accu1_ ##N = ZERO64;\

#define KERNEL_ROW_S_16(N) \
  ae_int8x8 temp_in1_ ## N; \
  ae_int8x8 temp_in2_ ## N; \
  AE_L8X8X2_IP(temp_in1_ ## N, temp_in2_ ## N, p_mat1_ ##N, 16); \

#define KERNEL_ROW_S_I_16(N) \
  ae_int8x8 temp_in1_ ## N; \
  ae_int8x8 temp_in2_ ## N; \
  AE_L8X8X2_IP(temp_in1_ ## N, temp_in2_ ## N, p_mat1_ ##N, 16); \
  AE_L16X4_XC(temp_src1, p_src1, 8); \
  AE_L16X4_XC(temp_src1_1, p_src1, 8); \
  AE_L16X4_XC(temp_src1_2, p_src1, 8); \
  AE_L16X4_XC(temp_src1_3, p_src1, 8); \

#define STORE_ROW_S_16(N) \
  ae_int64 temp1_ ##N = AE_SRAI64(AE_MOVINT64_FROMINT16X4(AE_MOVDA16((WORD32)p_bias[row+N])),48);            \
  temp1_ ##N = SW_SLAA64S_INT64_INT64(temp1_ ##N , bias_shift);\
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N);\
  accu1_ ##N = SW_SLAA64S_INT64_INT64(accu1_ ##N , acc_shift);\
  p_out[(row+N)*out_offset] =AE_MOVINT16_FROMF32X2(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(AE_MOVF64_FROMINT64(accu1_ ##N)),16),-16));

/**************************** Multiple of 8 ***********************************************************/
#define SETUP_ROW_S_8(N) \
  ae_int64 accu1_ ##N;\
  ae_int8x8 *p_mat1_ ##N = (ae_int8x8*)(&p_mat[(row+N)*cols]); \
  accu1_ ##N = ZERO64;\

#define KERNEL_ROW_S_8(N) \
  ae_int8x8 temp_in1_ ## N; \
  AE_L8X8_IP(temp_in1_ ## N, p_mat1_ ##N, 8); \

#define KERNEL_ROW_S_I_8(N) \
  ae_int8x8 temp_in1_ ## N; \
  AE_L8X8_IP(temp_in1_ ## N, p_mat1_ ##N, 8); \
  AE_L16X4_XC(temp_src1, p_src1, 8); \
  AE_L16X4_XC(temp_src1_1, p_src1, 8); \

#define STORE_ROW_S_8(N) \
  ae_int64 temp1_ ##N = AE_SRAI64(AE_MOVINT64_FROMINT16X4(AE_MOVDA16((WORD32)p_bias[row+N])),48);            \
  temp1_ ##N = SW_SLAA64S_INT64_INT64(temp1_ ##N , bias_shift);\
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N);\
  accu1_ ##N = SW_SLAA64S_INT64_INT64(accu1_ ##N , acc_shift);\
  p_out[(row+N)*out_offset] =AE_MOVINT16_FROMF32X2(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(AE_MOVF64_FROMINT64(accu1_ ##N)),16),-16));

/**************************** Multiple of 4 ***********************************************************/

#define SETUP_ROW_S(N) \
  ae_int64 accu1_ ##N;\
  WORD8 *p_mat1_ ##N = &p_mat[(row+N)*cols]; \
  accu1_ ##N = ZERO64;\

#define KERNEL_ROW_S(N) \
{ \
  ae_f16x4 temp_in1; \
  AE_L8X4F_IP(temp_in1, p_mat1_ ##N, 4); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, AE_MOVINT16X4_FROMF16X4(temp_in1));\
}

#define KERNEL_ROW_S_I(N) \
{ \
  ae_f16x4 temp_in1; \
  AE_L8X4F_IP(temp_in1, p_mat1_ ##N, 4); \
  AE_L16X4_XC(temp_src1, p_src1, 8); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, AE_MOVINT16X4_FROMF16X4(temp_in1));\
}

#define STORE_ROW_S(N) \
  accu1_ ##N = SW_SLAA64S_INT64_INT64(accu1_ ##N , -8);\
  ae_int64 temp1_ ##N = AE_SRAI64(AE_MOVINT64_FROMINT16X4(AE_MOVDA16((WORD32)p_bias[row+N])),48);            \
  temp1_ ##N = SW_SLAA64S_INT64_INT64(temp1_ ##N , bias_shift);\
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N);\
  accu1_ ##N = SW_SLAA64S_INT64_INT64(accu1_ ##N , acc_shift);\
  p_out[(row+N)*out_offset] =AE_MOVINT16_FROMF32X2(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(AE_MOVF64_FROMINT64(accu1_ ##N)),16),-16));

/************************** Multiple of 4 **************************************************************/

#if (UNROLL_S == 1)
#define SETUP_S SETUP_ROW_S(0)
#define KERNEL_S KERNEL_ROW_S_I(0)
#define STORE_S STORE_ROW_S(0)
#define SETUP_S_8   SETUP_ROW_S_8(0)
#define KERNEL_S_8 KERNEL_ROW_S_I_8(0)
#define STORE_S_8   STORE_ROW_S_8(0)
#define SETUP_S_16   SETUP_ROW_S_16(0)
#define KERNEL_S_16 KERNEL_ROW_S_I_16(0)
#define STORE_S_16   STORE_ROW_S_16(0)

#elif (UNROLL_S == 2)
#define SETUP_S  SETUP_ROW_S(0)  SETUP_ROW_S(1)
#define KERNEL_S KERNEL_ROW_S_I(0) KERNEL_ROW_S(1)
#define STORE_S  STORE_ROW_S(0)  STORE_ROW_S(1)
#define SETUP_S_8   SETUP_ROW_S_8(0)  SETUP_ROW_S_8(1)
#define KERNEL_S_8 KERNEL_ROW_S_I_8(0) KERNEL_ROW_S_8(1)
#define STORE_S_8   STORE_ROW_S_8(0)  STORE_ROW_S_8(1)
#define SETUP_S_16   SETUP_ROW_S_16(0)  SETUP_ROW_S_16(1)
#define KERNEL_S_16 KERNEL_ROW_S_I_16(0) KERNEL_ROW_S_16(1)
#define STORE_S_16   STORE_ROW_S_16(0)  STORE_ROW_S_16(1)

#elif (UNROLL_S == 4)
#define SETUP_S  SETUP_ROW_S(0)  SETUP_ROW_S(1)  SETUP_ROW_S(2)  SETUP_ROW_S(3)
#define KERNEL_S KERNEL_ROW_S_I(0) KERNEL_ROW_S(1) KERNEL_ROW_S(2) KERNEL_ROW_S(3)
#define STORE_S  STORE_ROW_S(0)  STORE_ROW_S(1)  STORE_ROW_S(2)  STORE_ROW_S(3)
#define SETUP_S_8  SETUP_ROW_S_8(0)  SETUP_ROW_S_8(1)  SETUP_ROW_S_8(2)  SETUP_ROW_S_8(3)
#define KERNEL_S_8 KERNEL_ROW_S_I_8(0) KERNEL_ROW_S_8(1) KERNEL_ROW_S_8(2) KERNEL_ROW_S_8(3)
#define STORE_S_8  STORE_ROW_S_8(0)  STORE_ROW_S_8(1)  STORE_ROW_S_8(2)  STORE_ROW_S_8(3)
#define SETUP_S_16  SETUP_ROW_S_16(0)  SETUP_ROW_S_16(1)  SETUP_ROW_S_16(2)  SETUP_ROW_S_16(3)
#define KERNEL_S_16 KERNEL_ROW_S_I_16(0) KERNEL_ROW_S_16(1) KERNEL_ROW_S_16(2) KERNEL_ROW_S_16(3)
#define STORE_S_16  STORE_ROW_S_16(0)  STORE_ROW_S_16(1)  STORE_ROW_S_16(2)  STORE_ROW_S_16(3)

#elif (UNROLL_S == 8)
#define SETUP_S   SETUP_ROW_S(0)  SETUP_ROW_S(1)  SETUP_ROW_S(2)  SETUP_ROW_S(3)  SETUP_ROW_S(4)  SETUP_ROW_S(5)  SETUP_ROW_S(6)  SETUP_ROW_S(7)
#define KERNEL_S KERNEL_ROW_S_I(0) KERNEL_ROW_S(1) KERNEL_ROW_S(2) KERNEL_ROW_S(3) KERNEL_ROW_S(4) KERNEL_ROW_S(5) KERNEL_ROW_S(6) KERNEL_ROW_S(7)
#define STORE_S   STORE_ROW_S(0)  STORE_ROW_S(1)  STORE_ROW_S(2)  STORE_ROW_S(3)  STORE_ROW_S(4)  STORE_ROW_S(5)  STORE_ROW_S(6)  STORE_ROW_S(7)
#define SETUP_S_8   SETUP_ROW_S_8(0)  SETUP_ROW_S_8(1)  SETUP_ROW_S_8(2)  SETUP_ROW_S_8(3)  SETUP_ROW_S_8(4)  SETUP_ROW_S_8(5)  SETUP_ROW_S_8(6)  SETUP_ROW_S_8(7)
#define KERNEL_S_8 KERNEL_ROW_S_I_8(0) KERNEL_ROW_S_8(1) KERNEL_ROW_S_8(2) KERNEL_ROW_S_8(3) KERNEL_ROW_S_8(4) KERNEL_ROW_S_8(5) KERNEL_ROW_S_8(6) KERNEL_ROW_S_8(7)
#define STORE_S_8   STORE_ROW_S_8(0)  STORE_ROW_S_8(1)  STORE_ROW_S_8(2)  STORE_ROW_S_8(3)  STORE_ROW_S_8(4)  STORE_ROW_S_8(5)  STORE_ROW_S_8(6)  STORE_ROW_S_8(7)

#define SETUP_S_16   SETUP_ROW_S_16(0)  SETUP_ROW_S_16(1)  SETUP_ROW_S_16(2)  SETUP_ROW_S_16(3)  SETUP_ROW_S_16(4)  SETUP_ROW_S_16(5)  SETUP_ROW_S_16(6)  SETUP_ROW_S_16(7)
#define KERNEL_S_16 KERNEL_ROW_S_I_16(0) KERNEL_ROW_S_16(1) KERNEL_ROW_S_16(2) KERNEL_ROW_S_16(3) KERNEL_ROW_S_16(4) KERNEL_ROW_S_16(5) KERNEL_ROW_S_16(6) KERNEL_ROW_S_16(7)
#define STORE_S_16   STORE_ROW_S_16(0)  STORE_ROW_S_16(1)  STORE_ROW_S_16(2)  STORE_ROW_S_16(3)  STORE_ROW_S_16(4)  STORE_ROW_S_16(5)  STORE_ROW_S_16(6)  STORE_ROW_S_16(7)

#endif

WORD32 xa_nn_matXvec_8x16_16_circ_nb(
  WORD16 * __restrict__ pt_out,
  WORD8  * __restrict__ p_mat,
  WORD16 * __restrict__ p_vec,
  WORD16 * __restrict__ p_bias,
  WORD32 rows,
  WORD32 cols,
  WORD32 out_offset,
  WORD32 bias_shift,
  WORD32 acc_shift)
{
  ae_int16 *p_out = (ae_int16 *)pt_out;
  WORD32 row, col;
  ae_int16x4 temp_src1;
  ae_int16x4 temp_src1_1;
  ae_int16x4 temp_src1_2;
  ae_int16x4 temp_src1_3;
  if ((NULL == p_out) || (NULL == p_mat) || (NULL == p_vec))
  {
    return -1;
  }

  if ((0 >= rows ) || (0 >= cols ) || (cols & 0x3))
  {
    return -2;
  }

  row = 0;
  if(cols%16==0)
  {
      if(rows >= UNROLL_S)
      {
        for (row = 0; row < ( rows & ~(UNROLL_S-1)) ; row+=UNROLL_S)
        {
          ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
          SETUP_S_16;
          for (col = 0; col < (cols>>4); col++) {
            KERNEL_S_16;
            AE_MULA8QW8X16(accu1_0, accu1_1, accu1_2, accu1_3, temp_in1_0,temp_in1_1,temp_in1_2,temp_in1_3,temp_src1,temp_src1_1);
            AE_MULA8QW8X16(accu1_4, accu1_5, accu1_6, accu1_7, temp_in1_4,temp_in1_5,temp_in1_6,temp_in1_7,temp_src1,temp_src1_1);
            AE_MULA8QW8X16(accu1_0, accu1_1, accu1_2, accu1_3, temp_in2_0,temp_in2_1,temp_in2_2,temp_in2_3,temp_src1_2,temp_src1_3);
            AE_MULA8QW8X16(accu1_4, accu1_5, accu1_6, accu1_7, temp_in2_4,temp_in2_5,temp_in2_6,temp_in2_7,temp_src1_2,temp_src1_3);
          }
          STORE_S_16;
        }
      }
      // Handle remaining rows
      for (; row < rows ; row++)
      {
        ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
        SETUP_ROW_S_8(0);
        ae_int64 accu1_1 = ZERO64;
        ae_int64 accu1_2 = ZERO64;
        ae_int64 accu1_3 = ZERO64;
        for (col = 0; col < (cols>>3); col++) {
          KERNEL_ROW_S_I_8(0);
          AE_MULA8QW8X16(accu1_0, accu1_1, accu1_2, accu1_3, temp_in1_0,temp_in1_0,temp_in1_0,temp_in1_0,temp_src1,temp_src1_1);
        }
        STORE_ROW_S_8(0);
      }
  }
  else if(cols%8==0)
  {
      if(rows >=UNROLL_S)
      {
        for (row = 0; row < ( rows & ~(UNROLL_S-1)) ; row+=UNROLL_S)
        {
          ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
          SETUP_S_8;
          for (col = 0; col < (cols>>3); col++) {
            KERNEL_S_8;
            AE_MULA8QW8X16(accu1_0, accu1_1, accu1_2, accu1_3, temp_in1_0,temp_in1_1,temp_in1_2,temp_in1_3,temp_src1,temp_src1_1);
            AE_MULA8QW8X16(accu1_4, accu1_5, accu1_6, accu1_7, temp_in1_4,temp_in1_5,temp_in1_6,temp_in1_7,temp_src1,temp_src1_1);
          }
          STORE_S_8;
        }
      }
      // Handle remaining rows
      for (; row < rows ; row++)
      {
        ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
        SETUP_ROW_S_8(0);
        ae_int64 accu1_1 = ZERO64;
        ae_int64 accu1_2 = ZERO64;
        ae_int64 accu1_3 = ZERO64;
        for (col = 0; col < (cols>>3); col++) {
          KERNEL_ROW_S_I_8(0);
          AE_MULA8QW8X16(accu1_0, accu1_1, accu1_2, accu1_3, temp_in1_0,temp_in1_0,temp_in1_0,temp_in1_0,temp_src1,temp_src1_1);
        }
        STORE_ROW_S_8(0);
      }

  }
  else if(cols%4==0)
  {
      if(rows > UNROLL_S)
      {
        for (row = 0; row < ( rows & ~(UNROLL_S-1)) ; row+=UNROLL_S)
        {
          ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
          SETUP_S;
#pragma ymemory (p_mat1_0)
#pragma ymemory (p_mat1_1)
#pragma ymemory (p_mat1_2)
#pragma ymemory (p_mat1_3)
          for (col = 0; col < (cols>>2); col++) {
            KERNEL_S;
          }
          STORE_S;
        }
      }
      // Handle remaining rows
      for (; row < rows ; row++)
      {
        ae_int16x4 *p_src1 = (ae_int16x4*)p_vec;
        SETUP_ROW_S(0);
        for (col = 0; col < (cols>>2); col++) {
          KERNEL_ROW_S_I(0);
        }
        STORE_ROW_S(0);
      }
  }
  return 0;
}

