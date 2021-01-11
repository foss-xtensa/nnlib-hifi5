/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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

#define ZERO64   AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0))

#if defined(CUST_UNROLL) && (CUST_UNROLL != 0)
#define UNROLL_D CUST_UNROLL
#else
#define UNROLL_D  4 /// Optimal unroll
#endif
#define LIMIT_VARIABLE(_var, _left_limit, _right_limit) \
  _var = _var > _right_limit ? _right_limit : _var < _left_limit ? _left_limit : _var;

/******************  Marcos for multiple of 8 cols *****************/
#define SETUP_ROW_D_8(N) \
  ae_int64 accu1_ ##N, accu2_ ##N;\
  ae_int16x4 *p_mat1_ ##N = (ae_int16x4*) p_mat; \
  AE_ADDCIRC16X4_XC(p_mat1_ ##N, (row+N) * row_offset * sizeof(WORD16)); \
  accu1_ ##N = p_bias[vec];            \
  accu2_ ##N = p_bias[vec+1];            \
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , bias_shift); \
  accu2_ ##N = AE_SLAA64S(accu2_ ##N , bias_shift);

#define KERNEL_ROW_D_8(N) \
  ae_int16x4 temp_in1_ ## N; \
  ae_int16x4 temp_in2_ ## N; \
  AE_L16X4X2_XC(temp_in1_ ## N,temp_in2_ ## N, p_mat1_ ##N, 2*sizeof(ae_int16x4)); \

#define KERNEL_ROW_D_I_8(N) \
  ae_int16x4 temp_in1_ ## N; \
  ae_int16x4 temp_in2_ ## N; \
  AE_L16X4X2_XC(temp_in1_ ## N,temp_in2_ ## N, p_mat1_ ##N, 2*sizeof(ae_int16x4)); \
  AE_L8X8_IP(temp_src1, p_src1, 8); \
  AE_L8X8_IP(temp_src2, p_src2, 8); \

#define STORE_ROW_D_8(N) \
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , acc_shift);\
  accu2_ ##N = AE_SLAA64S(accu2_ ##N , acc_shift);\
  p_dst1[(row+N) * out_row_offset] =AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_ ##N),16),16)); \
  p_dst2[(row+N) * out_row_offset] =AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu2_ ##N),16),16));
/******************  Marcos for multiple of 8 cols *****************/

/******************  Marcos for multiple of 4 cols *****************/

#define SETUP_ROW_D(N) \
  ae_int64 accu1_ ##N, accu2_ ##N;\
  ae_int16x4 *p_mat1_ ##N = (ae_int16x4*) p_mat; \
  AE_ADDCIRC16X4_XC(p_mat1_ ##N, (row+N) * row_offset * sizeof(WORD16)); \
  accu1_ ##N = ZERO64;\
  accu2_ ##N = ZERO64;\

#define KERNEL_ROW_D(N) \
{\
  ae_int16x4 temp_in1; \
  AE_L16X4_XC(temp_in1, p_mat1_ ##N, sizeof(ae_int16x4)); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
  AE_MULAAAAQ16(accu2_ ##N, temp_src2, temp_in1); \
}

#define KERNEL_ROW_D_I(N) \
{\
  ae_int16x4 temp_in1; \
  AE_L16X4_XC(temp_in1, p_mat1_ ##N, sizeof(ae_int16x4)); \
  AE_L8X4F_IP(temp_src1, p_src1, 4); \
  AE_L8X4F_IP(temp_src2, p_src2, 4); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
  AE_MULAAAAQ16(accu2_ ##N, temp_src2, temp_in1); \
}

#define STORE_ROW_D(N) \
  accu1_ ##N = AE_SRAA64(accu1_ ##N , 8);\
  accu2_ ##N = AE_SRAA64(accu2_ ##N , 8);\
  ae_int64 temp1_ ##N = p_bias[vec];            \
  ae_int64 temp2_ ##N = p_bias[vec+1];            \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , bias_shift);\
  temp2_ ##N = AE_SLAA64S(temp2_ ##N , bias_shift);\
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N);\
  accu2_ ##N = AE_ADD64(accu2_ ##N , temp2_ ##N);\
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , acc_shift);\
  accu2_ ##N = AE_SLAA64S(accu2_ ##N , acc_shift);\
  p_dst1[(row+N) * out_row_offset] =AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_ ##N),16),16)); \
  p_dst2[(row+N) * out_row_offset] =AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu2_ ##N),16),16));

/******************  Marcos for multiple of 4 cols *****************/

#if (UNROLL_D == 1)
#define SETUP_D SETUP_ROW_D(0)
#define KERNEL_D KERNEL_ROW_D_I(0)
#define STORE_D STORE_ROW_D(0)
#define SETUP_D_8   SETUP_ROW_D_8(0)
#define KERNEL_D_8 KERNEL_ROW_D_I_8(0)
#define STORE_D_8   STORE_ROW_D_8(0)
#define SETUP_D_16   SETUP_ROW_D_16(0)
#define KERNEL_D_16 KERNEL_ROW_D_I_16(0)
#define STORE_D_16   STORE_ROW_D_16(0)

#elif (UNROLL_D == 2)
#define SETUP_D  SETUP_ROW_D(0)  SETUP_ROW_D(1)
#define KERNEL_D KERNEL_ROW_D_I(0) KERNEL_ROW_D(1)
#define STORE_D  STORE_ROW_D(0)  STORE_ROW_D(1)
#define SETUP_D_16   SETUP_ROW_D_16(0)  SETUP_ROW_D_16(1)
#define KERNEL_D_16 KERNEL_ROW_D_I_16(0) KERNEL_ROW_D_16(1)
#define STORE_D_16   STORE_ROW_D_16(0)  STORE_ROW_D_16(1)

#elif (UNROLL_D == 4)
#define SETUP_D  SETUP_ROW_D(0)  SETUP_ROW_D(1)  SETUP_ROW_D(2)  SETUP_ROW_D(3)
#define KERNEL_D KERNEL_ROW_D_I(0) KERNEL_ROW_D(1) KERNEL_ROW_D(2) KERNEL_ROW_D(3)
#define STORE_D  STORE_ROW_D(0)  STORE_ROW_D(1)  STORE_ROW_D(2)  STORE_ROW_D(3)
#define SETUP_D_8  SETUP_ROW_D_8(0)  SETUP_ROW_D_8(1)  SETUP_ROW_D_8(2)  SETUP_ROW_D_8(3)
#define KERNEL_D_8 KERNEL_ROW_D_I_8(0) KERNEL_ROW_D_8(1) KERNEL_ROW_D_8(2) KERNEL_ROW_D_8(3)
#define STORE_D_8  STORE_ROW_D_8(0)  STORE_ROW_D_8(1)  STORE_ROW_D_8(2)  STORE_ROW_D_8(3)
#define SETUP_D_16  SETUP_ROW_D_16(0)  SETUP_ROW_D_16(1)  SETUP_ROW_D_16(2)  SETUP_ROW_D_16(3)
#define KERNEL_D_16 KERNEL_ROW_D_I_16(0) KERNEL_ROW_D_16(1) KERNEL_ROW_D_16(2) KERNEL_ROW_D_16(3)
#define STORE_D_16  STORE_ROW_D_16(0)  STORE_ROW_D_16(1)  STORE_ROW_D_16(2)  STORE_ROW_D_16(3)

#elif (UNROLL_D == 8)
#define SETUP_D   SETUP_ROW_D(0)  SETUP_ROW_D(1)  SETUP_ROW_D(2)  SETUP_ROW_D(3)  SETUP_ROW_D(4)  SETUP_ROW_D(5)  SETUP_ROW_D(6)  SETUP_ROW_D(7)
#define KERNEL_D KERNEL_ROW_D_I(0) KERNEL_ROW_D(1) KERNEL_ROW_D(2) KERNEL_ROW_D(3) KERNEL_ROW_D(4) KERNEL_ROW_D(5) KERNEL_ROW_D(6) KERNEL_ROW_D(7)
#define STORE_D   STORE_ROW_D(0)  STORE_ROW_D(1)  STORE_ROW_D(2)  STORE_ROW_D(3)  STORE_ROW_D(4)  STORE_ROW_D(5)  STORE_ROW_D(6)  STORE_ROW_D(7)
#define SETUP_D_8   SETUP_ROW_D_8(0)  SETUP_ROW_D_8(1)  SETUP_ROW_D_8(2)  SETUP_ROW_D_8(3)  SETUP_ROW_D_8(4)  SETUP_ROW_D_8(5)  SETUP_ROW_D_8(6)  SETUP_ROW_D_8(7)
#define KERNEL_D_8 KERNEL_ROW_D_I_8(0) KERNEL_ROW_D_8(1) KERNEL_ROW_D_8(2) KERNEL_ROW_D_8(3) KERNEL_ROW_D_8(4) KERNEL_ROW_D_8(5) KERNEL_ROW_D_8(6) KERNEL_ROW_D_8(7)
#define STORE_D_8   STORE_ROW_D_8(0)  STORE_ROW_D_8(1)  STORE_ROW_D_8(2)  STORE_ROW_D_8(3)  STORE_ROW_D_8(4)  STORE_ROW_D_8(5)  STORE_ROW_D_8(6)  STORE_ROW_D_8(7)
#define SETUP_D_16   SETUP_ROW_D_16(0)  SETUP_ROW_D_16(1)  SETUP_ROW_D_16(2)  SETUP_ROW_D_16(3)  SETUP_ROW_D_16(4)  SETUP_ROW_D_16(5)  SETUP_ROW_D_16(6)  SETUP_ROW_D_16(7)
#define KERNEL_D_16 KERNEL_ROW_D_I_16(0) KERNEL_ROW_D_16(1) KERNEL_ROW_D_16(2) KERNEL_ROW_D_16(3) KERNEL_ROW_D_16(4) KERNEL_ROW_D_16(5) KERNEL_ROW_D_16(6) KERNEL_ROW_D_16(7)
#define STORE_D_16   STORE_ROW_D_16(0)  STORE_ROW_D_16(1)  STORE_ROW_D_16(2)  STORE_ROW_D_16(3)  STORE_ROW_D_16(4)  STORE_ROW_D_16(5)  STORE_ROW_D_16(6)  STORE_ROW_D_16(7)

#endif

#if defined(CUST_UNROLL) && (CUST_UNROLL != 0)
#define UNROLL_S CUST_UNROLL
#else
#define UNROLL_S  8 /// Optimal unroll
#endif

#define SETUP_ROW_S(N) \
  ae_int64 accu1_ ##N;\
  ae_int16x4 *p_mat1_ ##N = (ae_int16x4*) p_mat; \
  AE_ADDCIRC16X4_XC(p_mat1_ ##N, (row+N) * row_offset * sizeof(WORD16)); \
  accu1_ ##N = ZERO64;\

#define KERNEL_ROW_S(N) \
{ \
  ae_int16x4 temp_in1; \
  AE_L16X4_XC(temp_in1, p_mat1_ ##N, sizeof(ae_int16x4)); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
}

#define KERNEL_ROW_S_I(N) \
{ \
  ae_int16x4 temp_in1; \
  AE_L16X4_XC(temp_in1, p_mat1_ ##N, sizeof(ae_int16x4)); \
  AE_L8X4F_IP(temp_src1, p_src1, 4); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
}

#define STORE_ROW_S(N) \
  accu1_ ##N = AE_SRAI64(accu1_ ##N , 8);\
  ae_int64 temp1_ ##N = p_bias[vec];            \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , bias_shift);\
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N);\
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , acc_shift);\
  p_dst1[(row+N) * out_row_offset] =AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_ ##N),16),16)); \

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

WORD32 xa_nn_matXvec_8x16_16_circ(
  WORD16 * __restrict__ p_out,
  WORD16 * __restrict__ p_mat,
  WORD8  * __restrict__ p_vec,
  WORD16 * __restrict__ p_bias,
  WORD32 rows,
  WORD32 cols,
  WORD32 row_offset,
  WORD32 vec_count,
  WORD32 vec_offset,
  WORD32 out_col_offset,
  WORD32 out_row_offset,
  WORD32 bias_shift,
  WORD32 acc_shift)
{
  WORD32 row = 0, col = 0, vec=0;
  if ((NULL == p_out) || (NULL == p_mat) || (NULL == p_vec))
  {
    return -1;
  }

  if ((0 >= rows ) || (0 >= cols ) || (cols & 0x3))
  {
    return -2;
  }
  if(0 >= vec_count) return -3;
  if(cols%16==0)
  {
      ae_int8x8 temp_src1, temp_src2;
      ae_int8x8 temp_src3, temp_src4;
      ae_int8x8 temp_src1_1, temp_src2_1;
      ae_int8x8 temp_src3_1, temp_src4_1;
      if(vec_count >= 4)
      {
        // Process two vectors at a time
        for(vec = 0; vec < (vec_count & (~0x3)) ; vec+=4)
        {

          WORD16 *p_dst1 = (WORD16 *)&p_out[vec*out_col_offset];
          WORD16 *p_dst2 = (WORD16 *)&p_out[(vec+1)*out_col_offset];
          WORD16 *p_dst3 = (WORD16 *)&p_out[(vec+2)*out_col_offset];
          WORD16 *p_dst4 = (WORD16 *)&p_out[(vec+3)*out_col_offset];

          row = 0;

          if(rows>=2)
          {
            for (row = 0; row < (rows & (~0x1)) ; row+=2)
            {
              ae_int8x8 *p_src1 = (ae_int8x8 *)&p_vec[vec * vec_offset];
              ae_int8x8 *p_src2 = (ae_int8x8 *)&p_vec[(vec+1) * vec_offset];
              ae_int8x8 *p_src3 = (ae_int8x8 *)&p_vec[(vec+2) * vec_offset];
              ae_int8x8 *p_src4 = (ae_int8x8 *)&p_vec[(vec+3) * vec_offset];

              ae_int64 accu1_0, accu2_0;
              ae_int64 accu3_0, accu4_0;
              ae_int16x8 *p_mat1_0 = (ae_int16x8*) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (row+0) * row_offset * sizeof(WORD16));
              ae_valignx2 p_mat1_0_align;
              AE_LA16X4X2POS_PC(p_mat1_0_align,p_mat1_0);
              accu1_0 = p_bias[vec];
              accu2_0 = p_bias[vec+1];
              accu3_0 = p_bias[vec+2];
              accu4_0 = p_bias[vec+3];
              accu1_0 = AE_SLAA64S(accu1_0 , bias_shift);
              accu2_0 = AE_SLAA64S(accu2_0 , bias_shift);
              accu3_0 = AE_SLAA64S(accu3_0 , bias_shift);
              accu4_0 = AE_SLAA64S(accu4_0 , bias_shift);

              ae_int64 accu1_1, accu2_1;
              ae_int64 accu3_1, accu4_1;
              ae_int16x8 *p_mat1_1 = (ae_int16x8*) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (row+1) * row_offset * sizeof(WORD16));
              ae_valignx2 p_mat1_1_align;
              AE_LA16X4X2POS_PC(p_mat1_1_align,p_mat1_1);

              accu1_1 = p_bias[vec];
              accu2_1 = p_bias[vec+1];
              accu3_1 = p_bias[vec+2];
              accu4_1 = p_bias[vec+3];
              accu1_1 = AE_SLAA64S(accu1_1 , bias_shift);
              accu2_1 = AE_SLAA64S(accu2_1 , bias_shift);
              accu3_1 = AE_SLAA64S(accu3_1 , bias_shift);
              accu4_1 = AE_SLAA64S(accu4_1 , bias_shift);
              ae_int16x4 temp_in1_0;
              ae_int16x4 temp_in2_0;
              ae_int16x4 temp_in3_0;
              ae_int16x4 temp_in4_0;
              ae_int16x4 temp_in1_1;
              ae_int16x4 temp_in2_1;
              ae_int16x4 temp_in3_1;
              ae_int16x4 temp_in4_1;
              for (col = 0; col < cols>>4; col++)
              {
                  AE_LA16X4X2_IC(temp_in1_0, temp_in2_0,p_mat1_0_align, (ae_int16x8 *) p_mat1_0);
                  AE_LA16X4X2_IC(temp_in3_0, temp_in4_0,p_mat1_0_align, (ae_int16x8 *) p_mat1_0);
                  AE_LA16X4X2_IC(temp_in1_1, temp_in2_1,p_mat1_1_align, (ae_int16x8 *) p_mat1_1);
                  AE_LA16X4X2_IC(temp_in3_1, temp_in4_1,p_mat1_1_align, (ae_int16x8 *) p_mat1_1);
                  AE_L8X8X2_IP(temp_src1, temp_src1_1,(ae_int8x16 *) p_src1, 16);
                  AE_L8X8X2_IP(temp_src2, temp_src2_1,(ae_int8x16 *) p_src2, 16);
                  AE_L8X8X2_IP(temp_src3, temp_src3_1,(ae_int8x16 *) p_src3, 16);
                  AE_L8X8X2_IP(temp_src4, temp_src4_1,(ae_int8x16 *) p_src4, 16);
                  AE_MULA8QW8X16(accu1_0, accu2_0, accu3_0, accu4_0, temp_src1,temp_src2,temp_src3,temp_src4,temp_in1_0,temp_in2_0);
                  AE_MULA8QW8X16(accu1_1, accu2_1, accu3_1, accu4_1, temp_src1,temp_src2,temp_src3,temp_src4,temp_in1_1,temp_in2_1);
                  AE_MULA8QW8X16(accu1_0, accu2_0, accu3_0, accu4_0, temp_src1_1,temp_src2_1,temp_src3_1,temp_src4_1,temp_in3_0,temp_in4_0);
                  AE_MULA8QW8X16(accu1_1, accu2_1, accu3_1, accu4_1, temp_src1_1,temp_src2_1,temp_src3_1,temp_src4_1,temp_in3_1,temp_in4_1);
              }
              accu1_0 = AE_SLAA64S(accu1_0 , acc_shift);
              accu2_0 = AE_SLAA64S(accu2_0 , acc_shift);
              accu3_0 = AE_SLAA64S(accu3_0 , acc_shift);
              accu4_0 = AE_SLAA64S(accu4_0 , acc_shift);
              accu1_1 = AE_SLAA64S(accu1_1 , acc_shift);
              accu2_1 = AE_SLAA64S(accu2_1 , acc_shift);
              accu3_1 = AE_SLAA64S(accu3_1 , acc_shift);
              accu4_1 = AE_SLAA64S(accu4_1 , acc_shift);
              p_dst1[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_0),16),16));
              p_dst2[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu2_0),16),16));
              p_dst3[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu3_0),16),16));
              p_dst4[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu4_0),16),16));
              p_dst1[(row+1) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_1),16),16));
              p_dst2[(row+1) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu2_1),16),16));
              p_dst3[(row+1) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu3_1),16),16));
              p_dst4[(row+1) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu4_1),16),16));
            }
          }
          // Handle remaining rows
          for (; row < rows ; row++)
          {
              ae_int8x8 *p_src1 = (ae_int8x8 *)&p_vec[vec * vec_offset];
              ae_int8x8 *p_src2 = (ae_int8x8 *)&p_vec[(vec+1) * vec_offset];
              ae_int8x8 *p_src3 = (ae_int8x8 *)&p_vec[(vec+2) * vec_offset];
              ae_int8x8 *p_src4 = (ae_int8x8 *)&p_vec[(vec+3) * vec_offset];
              ae_int64 accu1_0, accu2_0;
              ae_int64 accu3_0, accu4_0;
              ae_int16x8 *p_mat1_0 = (ae_int16x8*) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (row+0) * row_offset * sizeof(WORD16));
              ae_valignx2 p_mat1_0_align;
              AE_LA16X4X2POS_PC(p_mat1_0_align,p_mat1_0);

              accu1_0 = p_bias[vec];
              accu2_0 = p_bias[vec+1];
              accu3_0 = p_bias[vec+2];
              accu4_0 = p_bias[vec+3];
              accu1_0 = AE_SLAA64S(accu1_0 , bias_shift);
              accu2_0 = AE_SLAA64S(accu2_0 , bias_shift);
              accu3_0 = AE_SLAA64S(accu3_0 , bias_shift);
              accu4_0 = AE_SLAA64S(accu4_0 , bias_shift);
#pragma ymemory (p_mat1_0)
              ae_int16x4 temp_in1_0;
              ae_int16x4 temp_in2_0;
              ae_int16x4 temp_in3_0;
              ae_int16x4 temp_in4_0;
              for (col = 0; col < cols>>4; col++)
              {
                  AE_LA16X4X2_IC(temp_in1_0, temp_in2_0,p_mat1_0_align, (ae_int16x8 *) p_mat1_0);
                  AE_LA16X4X2_IC(temp_in3_0, temp_in4_0,p_mat1_0_align, (ae_int16x8 *) p_mat1_0);
                  AE_L8X8X2_IP(temp_src1, temp_src1_1,(ae_int8x16 *) p_src1, 16);
                  AE_L8X8X2_IP(temp_src2, temp_src2_1,(ae_int8x16 *) p_src2, 16);
                  AE_L8X8X2_IP(temp_src3, temp_src3_1,(ae_int8x16 *) p_src3, 16);
                  AE_L8X8X2_IP(temp_src4, temp_src4_1,(ae_int8x16 *) p_src4, 16);
                  AE_MULA8QW8X16(accu1_0, accu2_0, accu3_0, accu4_0, temp_src1,temp_src2,temp_src3,temp_src4,temp_in1_0,temp_in2_0);
                  AE_MULA8QW8X16(accu1_0, accu2_0, accu3_0, accu4_0, temp_src1_1,temp_src2_1,temp_src3_1,temp_src4_1,temp_in3_0,temp_in4_0);
              }
              accu1_0 = AE_SLAA64S(accu1_0 , acc_shift);
              accu2_0 = AE_SLAA64S(accu2_0 , acc_shift);
              accu3_0 = AE_SLAA64S(accu3_0 , acc_shift);
              accu4_0 = AE_SLAA64S(accu4_0 , acc_shift);
              p_dst1[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_0),16),16));
              p_dst2[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu2_0),16),16));
              p_dst3[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu3_0),16),16));
              p_dst4[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu4_0),16),16));
          }
        }
      }
      if(vec_count %4!=0)
      {
        // Process one vectors at a time
        for(; vec < vec_count; vec++)
        {
            WORD16 *p_dst1 = (WORD16 *)&p_out[vec*out_col_offset];
            row = 0;

            for (row=0; row < rows ; row++)
            {
              ae_int8x8 *p_src1 = (ae_int8x8 *)&p_vec[vec * vec_offset];
              ae_int64 accu1_0 = ZERO64, accu2_0 = ZERO64;
              ae_int64 accu3_0 = ZERO64, accu4_0 = ZERO64;

              ae_int16x8 *p_mat1_0 = (ae_int16x8*) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (row+0) * row_offset * sizeof(WORD16));
              ae_valignx2 p_mat1_0_align;
              AE_LA16X4X2POS_PC(p_mat1_0_align,p_mat1_0);

              accu1_0 = p_bias[vec];
              accu1_0 = AE_SLAA64S(accu1_0 , bias_shift);

              ae_int16x4 temp_in1_0;
              ae_int16x4 temp_in2_0;
              for (col = 0; col < cols>>3; col++)
              {
                AE_LA16X4X2_IC(temp_in1_0, temp_in2_0,p_mat1_0_align, (ae_int16x8 *) p_mat1_0);
                AE_L8X8_IP(temp_src1, p_src1, 8);
                AE_MULA8QW8X16(accu1_0, accu2_0, accu3_0, accu4_0, temp_src1,temp_src1,temp_src1,temp_src1,temp_in1_0,temp_in2_0);
              }
              accu1_0 = AE_SLAA64S(accu1_0 , acc_shift);
              p_dst1[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_0),16),16));
            }
         }
      }

  }
  else if(cols%8==0)
  {
      ae_int8x8 temp_src1, temp_src2;
      ae_int8x8 temp_src3, temp_src4;
      if(vec_count >= 4)
      {
        // Process two vectors at a time
        for(vec = 0; vec < (vec_count & (~0x3)); vec+=4)
        {

          WORD16 *p_dst1 = (WORD16 *)&p_out[vec*out_col_offset];
          WORD16 *p_dst2 = (WORD16 *)&p_out[(vec+1)*out_col_offset];
          WORD16 *p_dst3 = (WORD16 *)&p_out[(vec+2)*out_col_offset];
          WORD16 *p_dst4 = (WORD16 *)&p_out[(vec+3)*out_col_offset];

          row = 0;
          if(rows>=2)
          {
            for (row = 0; row < (rows  & (~0x1)) ; row+=2)
            {
              ae_int8x8 *p_src1 = (ae_int8x8 *)&p_vec[vec * vec_offset];
              ae_int8x8 *p_src2 = (ae_int8x8 *)&p_vec[(vec+1) * vec_offset];
              ae_int8x8 *p_src3 = (ae_int8x8 *)&p_vec[(vec+2) * vec_offset];
              ae_int8x8 *p_src4 = (ae_int8x8 *)&p_vec[(vec+3) * vec_offset];

              ae_int64 accu1_0, accu2_0;
              ae_int64 accu3_0, accu4_0;
              ae_int16x8 *p_mat1_0 = (ae_int16x8*) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (row+0) * row_offset * sizeof(WORD16));
              ae_valignx2 p_mat1_0_align;
              AE_LA16X4X2POS_PC(p_mat1_0_align,p_mat1_0);

              accu1_0 = p_bias[vec];
              accu2_0 = p_bias[vec+1];
              accu3_0 = p_bias[vec+2];
              accu4_0 = p_bias[vec+3];
              accu1_0 = AE_SLAA64S(accu1_0 , bias_shift);
              accu2_0 = AE_SLAA64S(accu2_0 , bias_shift);
              accu3_0 = AE_SLAA64S(accu3_0 , bias_shift);
              accu4_0 = AE_SLAA64S(accu4_0 , bias_shift);

              ae_int64 accu1_1, accu2_1;
              ae_int64 accu3_1, accu4_1;
              ae_int16x8 *p_mat1_1 = (ae_int16x8*) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_1, (row+1) * row_offset * sizeof(WORD16));
              ae_valignx2 p_mat1_1_align;
              AE_LA16X4X2POS_PC(p_mat1_1_align,p_mat1_1);

              accu1_1 = p_bias[vec];
              accu2_1 = p_bias[vec+1];
              accu3_1 = p_bias[vec+2];
              accu4_1 = p_bias[vec+3];
              accu1_1 = AE_SLAA64S(accu1_1 , bias_shift);
              accu2_1 = AE_SLAA64S(accu2_1 , bias_shift);
              accu3_1 = AE_SLAA64S(accu3_1 , bias_shift);
              accu4_1 = AE_SLAA64S(accu4_1 , bias_shift);
              ae_int16x4 temp_in1_0;
              ae_int16x4 temp_in2_0;
              ae_int16x4 temp_in1_1;
              ae_int16x4 temp_in2_1;
#pragma ymemory (p_src1)
#pragma ymemory (p_src2)
#pragma ymemory (p_src3)
#pragma ymemory (p_src4)
              for (col = 0; col < cols>>3; col++)
              {
                  AE_LA16X4X2_IC(temp_in1_0, temp_in2_0,p_mat1_0_align, (ae_int16x8 *) p_mat1_0);
                  AE_LA16X4X2_IC(temp_in1_1, temp_in2_1,p_mat1_1_align, (ae_int16x8 *) p_mat1_1);
                  AE_L8X8_IP(temp_src1, p_src1, 8);
                  AE_L8X8_IP(temp_src2, p_src2, 8);
                  AE_L8X8_IP(temp_src3, p_src3, 8);
                  AE_L8X8_IP(temp_src4, p_src4, 8);
                  AE_MULA8QW8X16(accu1_0, accu2_0, accu3_0, accu4_0, temp_src1,temp_src2,temp_src3,temp_src4,temp_in1_0,temp_in2_0);
                  AE_MULA8QW8X16(accu1_1, accu2_1, accu3_1, accu4_1, temp_src1,temp_src2,temp_src3,temp_src4,temp_in1_1,temp_in2_1);
              }
              accu1_0 = AE_SLAA64S(accu1_0 , acc_shift);
              accu2_0 = AE_SLAA64S(accu2_0 , acc_shift);
              accu3_0 = AE_SLAA64S(accu3_0 , acc_shift);
              accu4_0 = AE_SLAA64S(accu4_0 , acc_shift);
              accu1_1 = AE_SLAA64S(accu1_1 , acc_shift);
              accu2_1 = AE_SLAA64S(accu2_1 , acc_shift);
              accu3_1 = AE_SLAA64S(accu3_1 , acc_shift);
              accu4_1 = AE_SLAA64S(accu4_1 , acc_shift);
              p_dst1[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_0),16),16));
              p_dst2[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu2_0),16),16));
              p_dst3[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu3_0),16),16));
              p_dst4[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu4_0),16),16));
              p_dst1[(row+1) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_1),16),16));
              p_dst2[(row+1) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu2_1),16),16));
              p_dst3[(row+1) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu3_1),16),16));
              p_dst4[(row+1) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu4_1),16),16));
            }
          }
          // Handle remaining rows
          for (; row < rows ; row++)
          {
              ae_int8x8 *p_src1 = (ae_int8x8 *)&p_vec[vec * vec_offset];
              ae_int8x8 *p_src2 = (ae_int8x8 *)&p_vec[(vec+1) * vec_offset];
              ae_int8x8 *p_src3 = (ae_int8x8 *)&p_vec[(vec+2) * vec_offset];
              ae_int8x8 *p_src4 = (ae_int8x8 *)&p_vec[(vec+3) * vec_offset];
              ae_int64 accu1_0, accu2_0;
              ae_int64 accu3_0, accu4_0;
              ae_int16x8 *p_mat1_0 = (ae_int16x8*) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (row+0) * row_offset * sizeof(WORD16));
              ae_valignx2 p_mat1_0_align;
              AE_LA16X4X2POS_PC(p_mat1_0_align,p_mat1_0);

              accu1_0 = p_bias[vec];
              accu2_0 = p_bias[vec+1];
              accu3_0 = p_bias[vec+2];
              accu4_0 = p_bias[vec+3];
              accu1_0 = AE_SLAA64S(accu1_0 , bias_shift);
              accu2_0 = AE_SLAA64S(accu2_0 , bias_shift);
              accu3_0 = AE_SLAA64S(accu3_0 , bias_shift);
              accu4_0 = AE_SLAA64S(accu4_0 , bias_shift);
#pragma ymemory (p_mat1_0)
              ae_int16x4 temp_in1_0;
              ae_int16x4 temp_in2_0;
              for (col = 0; col < cols>>3; col++) {
                  AE_LA16X4X2_IC(temp_in1_0, temp_in2_0,p_mat1_0_align, (ae_int16x8 *) p_mat1_0);
                  //AE_L16X4X2_XC(temp_in1_0, temp_in2_0,(ae_int16x8 *) p_mat1_0, 2*sizeof(ae_int16x4));
                  AE_L8X8_IP(temp_src1, p_src1, 8);
                  AE_L8X8_IP(temp_src2, p_src2, 8);
                  AE_L8X8_IP(temp_src3, p_src3, 8);
                  AE_L8X8_IP(temp_src4, p_src4, 8);
                  AE_MULA8QW8X16(accu1_0, accu2_0, accu3_0, accu4_0, temp_src1,temp_src2,temp_src3,temp_src4,temp_in1_0,temp_in2_0);

              }
              accu1_0 = AE_SLAA64S(accu1_0 , acc_shift);
              accu2_0 = AE_SLAA64S(accu2_0 , acc_shift);
              accu3_0 = AE_SLAA64S(accu3_0 , acc_shift);
              accu4_0 = AE_SLAA64S(accu4_0 , acc_shift);
              p_dst1[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_0),16),16));
              p_dst2[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu2_0),16),16));
              p_dst3[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu3_0),16),16));
              p_dst4[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu4_0),16),16));
          }
        }
      }
      if(vec_count %4!=0)
      {
        // Process one vectors at a time
        for(; vec < vec_count; vec++)
        {
            WORD16 *p_dst1 = (WORD16 *)&p_out[vec*out_col_offset];
            row = 0;
            for (row=0; row < rows ; row++)
            {
              ae_int8x8 *p_src1 = (ae_int8x8 *)&p_vec[vec * vec_offset];
              ae_int64 accu1_0 = ZERO64, accu2_0 = ZERO64;
              ae_int64 accu3_0 = ZERO64, accu4_0 = ZERO64;
              ae_int16x8 *p_mat1_0 = (ae_int16x8*) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, (row+0) * row_offset * sizeof(WORD16));
              ae_valignx2 p_mat1_0_align;
              AE_LA16X4X2POS_PC(p_mat1_0_align,p_mat1_0);
              accu1_0 = p_bias[vec];
              accu1_0 = AE_SLAA64S(accu1_0 , bias_shift);
              ae_int16x4 temp_in1_0;
              ae_int16x4 temp_in2_0;
              for (col = 0; col < cols>>3; col++)
              {
                //AE_L16X4X2_XC(temp_in1_0, temp_in2_0,(ae_int16x8 *) p_mat1_0, 2*sizeof(ae_int16x4));
                AE_LA16X4X2_IC(temp_in1_0, temp_in2_0,p_mat1_0_align, (ae_int16x8 *) p_mat1_0);
                //AE_L16X4_XC(temp_in1_0,p_mat1_0, sizeof(ae_int16x4));
                //AE_L16X4_XC(temp_in2_0,p_mat1_0, sizeof(ae_int16x4));
                AE_L8X8_IP(temp_src1, p_src1, 8);
                AE_MULA8QW8X16(accu1_0, accu2_0, accu3_0, accu4_0, temp_src1,temp_src1,temp_src1,temp_src1,temp_in1_0,temp_in2_0);
              }
              accu1_0 = AE_SLAA64S(accu1_0 , acc_shift);
              p_dst1[(row+0) * out_row_offset] = AE_MOVINT16_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_0),16),16));
            }
         }
      }
  }
  else if(cols%4==0)
  {
      ae_int16x4 temp_src1, temp_src2;
      if(vec_count > 1)
      {
        // Process two vectors at a time
        for(vec = 0; vec < (vec_count & (~0x1)); vec+=2)
        {

          WORD16 *p_dst1 = (WORD16 *)&p_out[vec*out_col_offset];
          WORD16 *p_dst2 = (WORD16 *)&p_out[(vec+1)*out_col_offset];

          row = 0;
          if(rows > UNROLL_D)
          {
            for (row = 0; row < ( rows & ~(UNROLL_D-1)) ; row+=UNROLL_D)
            {
              WORD8 *p_src1 = (WORD8 *)&p_vec[vec * vec_offset];
              WORD8 *p_src2 = (WORD8 *)&p_vec[(vec+1) * vec_offset];
              SETUP_D;
#pragma ymemory (p_mat1_0)
#pragma ymemory (p_mat1_1)
#pragma ymemory (p_mat1_2)
#pragma ymemory (p_mat1_3)
              for (col = 0; col < cols>>2; col++) {
                KERNEL_D ;
              }
              STORE_D;
            }
          }
          // Handle remaining rows
          for (; row < rows ; row++)
          {
            WORD8 *p_src1 = (WORD8 *)&p_vec[vec * vec_offset];
            WORD8 *p_src2 = (WORD8 *)&p_vec[(vec+1) * vec_offset];
            SETUP_ROW_D(0);
            for (col = 0; col < cols>>2; col++) {
              KERNEL_ROW_D_I(0);
            }
            STORE_ROW_D(0);
          }
        }
      }
      if(vec_count & 0x1)
      {
        WORD16 *p_dst1 = (WORD16 *)&p_out[vec*out_col_offset];

        row = 0;
        if(rows > UNROLL_S)
        {
          for (row = 0; row < ( rows & ~(UNROLL_S-1)) ; row+=UNROLL_S)
          {
            WORD8 *p_src1 = (WORD8 *)&p_vec[vec *vec_offset];
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
          WORD8 *p_src1 = (WORD8 *)&p_vec[vec * vec_offset];
          SETUP_ROW_S(0);
          for (col = 0; col < (cols>>2); col++) {
            KERNEL_ROW_S_I(0);
          }
          STORE_ROW_S(0);
        }
      }
  }
  return 0;
}

