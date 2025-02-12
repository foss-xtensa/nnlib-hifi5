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

#define ZERO32   (0)

#define ZERO64   AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0))

#if defined(CUST_UNROLL) && (CUST_UNROLL != 0)
#define UNROLL_S CUST_UNROLL
#else
#define UNROLL_S  8 /// Optimal unroll
#endif

/**************************** Multiple of 4 ***********************************************************/
#define SETUP_ROW_S(N) \
  ae_int64 accu1_ ##N;\
  WORD8 *p_mat1_ ##N = &p_mat[(row+N)*cols]; \
  accu1_ ##N = ZERO64; \

#define KERNEL_ROW_S(N) \
{ \
  ae_int16x4 temp_in1; \
  AE_L8X4F_IP(temp_in1, p_mat1_ ##N, 4); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
}

#define KERNEL_ROW_S_I(N) \
{ \
  ae_int16x4 temp_in1; \
  AE_L8X4F_IP(temp_in1, p_mat1_ ##N, 4); \
  temp_src1 = AE_L8X4F_I(p_src1, 0); \
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_src1, 4); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
}
#define STORE_ROW_WITHOUT_SHIFT(N) \
  ae_int8x8 temp8_ ##N;\
  ae_int64 temp1_ ##N; \
  temp1_ ##N = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(p_bias[row+N]));            \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , 8); \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , -56); \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , bias_shift);\
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N); \
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , acc_shift); \
  temp8_ ##N = AE_MOVINT8X8_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_ ##N),24),-24));\
  p_out[(row+N)*out_offset] = (WORD8)AE_MOVAD8(temp8_ ##N,0);

#define STORE_ROW_S(N) \
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , -16); \
  ae_int8x8 temp8_ ##N;\
  ae_int64 temp1_ ##N; \
  temp1_ ##N = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(p_bias[row+N]));            \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , 8); \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , -56); \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , bias_shift);\
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N); \
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , acc_shift); \
  temp8_ ##N = AE_MOVINT8X8_FROMINT32(AE_SLAA32S(AE_SLAA32S(AE_ROUND32F64SSYM(accu1_ ##N),24),-24));\
  p_out[(row+N)*out_offset] = (WORD8)AE_MOVAD8(temp8_ ##N,0);


/**************************** Multiple of 4 ***********************************************************/

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

WORD32 xa_nn_matXvec_8x8_8_circ_nb(
  WORD8 * __restrict__ p_out,
  WORD8 * __restrict__ p_mat,
  WORD8 * __restrict__ p_vec,
  WORD8 * __restrict__ p_bias,
  WORD32 rows,
  WORD32 cols,
  WORD32 out_offset,
  WORD32 bias_shift,
  WORD32 acc_shift)
{
  WORD32 row, col;
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
      ae_int8x8 temp_src1;
      ae_int8x8 temp_src1_1;
//      ae_int8x8 * output_ptr=(ae_int8x8 *)p_out;
      if(rows >= 8)
      {
        for (row = 0; row < ( rows & ~(8-1)) ; row+=8)
        {
          ae_int8x16 *p_src1 = (ae_int8x16 *)p_vec;
          ae_int32x2 accu1_01;
          ae_int8x8 *p_mat1_0 =(ae_int8x8 *) (&p_mat[(row+0)*cols]);
          ae_int8x8 *p_mat1_1 =(ae_int8x8 *) (&p_mat[(row+1)*cols]);
          ae_int32x2 accu1_23;
          ae_int8x8 *p_mat1_2 =(ae_int8x8 *) (&p_mat[(row+2)*cols]);
          ae_int8x8 *p_mat1_3 =(ae_int8x8 *) (&p_mat[(row+3)*cols]);
          ae_int32x2 accu1_45;
          ae_int8x8 *p_mat1_4 =(ae_int8x8 *) (&p_mat[(row+4)*cols]);
          ae_int8x8 *p_mat1_5 =(ae_int8x8 *) (&p_mat[(row+5)*cols]);
          ae_int32x2 accu1_67;
          ae_int8x8 *p_mat1_6 =(ae_int8x8 *) (&p_mat[(row+6)*cols]);
          ae_int8x8 *p_mat1_7 =(ae_int8x8 *) (&p_mat[(row+7)*cols]);
          accu1_01 = ZERO32;
          accu1_23 = ZERO32;
          accu1_45 = ZERO32;
          accu1_67 = ZERO32;
#pragma ymemory (p_mat1_0)
#pragma ymemory (p_mat1_1)
#pragma ymemory (p_mat1_2)
#pragma ymemory (p_mat1_3)
          ae_int8x8 temp_in0;
          ae_int8x8 temp_in1;
          ae_int8x8 temp_in2;
          ae_int8x8 temp_in3;
          ae_int8x8 temp_in4;
          ae_int8x8 temp_in5;
          ae_int8x8 temp_in6;
          ae_int8x8 temp_in7;
          ae_int8x8 temp_in0_1;
          ae_int8x8 temp_in1_1;
          ae_int8x8 temp_in2_1;
          ae_int8x8 temp_in3_1;
          ae_int8x8 temp_in4_1;
          ae_int8x8 temp_in5_1;
          ae_int8x8 temp_in6_1;
          ae_int8x8 temp_in7_1;
          ae_valignx2 align_p_vec;
          AE_LA8X8X2POS_PC(align_p_vec,p_src1);
          for (col = 0; col < (cols>>4); col++)
          {
              AE_L8X8X2_IP(temp_in0 ,temp_in0_1 , (ae_int8x16 *)p_mat1_0, 16);
              AE_L8X8X2_IP(temp_in1 ,temp_in1_1 , (ae_int8x16 *)p_mat1_1, 16);
              AE_L8X8X2_IP(temp_in2 ,temp_in2_1 , (ae_int8x16 *)p_mat1_2, 16);
              AE_L8X8X2_IP(temp_in3 ,temp_in3_1 , (ae_int8x16 *)p_mat1_3, 16);
              AE_L8X8X2_IP(temp_in4 ,temp_in4_1 , (ae_int8x16 *)p_mat1_4, 16);
              AE_L8X8X2_IP(temp_in5 ,temp_in5_1 , (ae_int8x16 *)p_mat1_5, 16);
              AE_L8X8X2_IP(temp_in6 ,temp_in6_1 , (ae_int8x16 *)p_mat1_6, 16);
              AE_L8X8X2_IP(temp_in7 ,temp_in7_1 , (ae_int8x16 *)p_mat1_7, 16);
              AE_LA8X8X2_IC(temp_src1,temp_src1_1,align_p_vec ,(ae_int8x16 *)  p_src1);
              AE_MULA8Q8X8(accu1_01,accu1_23,temp_in0,temp_in1,temp_in2,temp_in3,temp_src1);
              AE_MULA8Q8X8(accu1_45,accu1_67,temp_in4,temp_in5,temp_in6,temp_in7,temp_src1);
              AE_MULA8Q8X8(accu1_01,accu1_23,temp_in0_1,temp_in1_1,temp_in2_1,temp_in3_1,temp_src1_1);
              AE_MULA8Q8X8(accu1_45,accu1_67,temp_in4_1,temp_in5_1,temp_in6_1,temp_in7_1,temp_src1_1);
          }
          ae_int64 accu1_0;
          ae_int64 accu1_1;
          ae_int64 accu1_2;
          ae_int64 accu1_3;
          ae_int64 accu1_4;
          ae_int64 accu1_5;
          ae_int64 accu1_6;
          ae_int64 accu1_7;
          accu1_0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu1_01,ZERO32));
          accu1_1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_01,ZERO32));
          accu1_2 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu1_23,ZERO32));
          accu1_3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_23,ZERO32));
          accu1_4 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu1_45,ZERO32));
          accu1_5 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_45,ZERO32));
          accu1_6 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu1_67,ZERO32));
          accu1_7 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_67,ZERO32));

          accu1_0 = AE_SLAA64S(accu1_0 , -32);
          accu1_1 = AE_SLAA64S(accu1_1 , -32);
          accu1_2 = AE_SLAA64S(accu1_2 , -32);
          accu1_3 = AE_SLAA64S(accu1_3 , -32);
          accu1_4 = AE_SLAA64S(accu1_4 , -32);
          accu1_5 = AE_SLAA64S(accu1_5 , -32);
          accu1_6 = AE_SLAA64S(accu1_6 , -32);
          accu1_7 = AE_SLAA64S(accu1_7 , -32);

          STORE_ROW_WITHOUT_SHIFT(0);
          STORE_ROW_WITHOUT_SHIFT(1);
          STORE_ROW_WITHOUT_SHIFT(2);
          STORE_ROW_WITHOUT_SHIFT(3);
          STORE_ROW_WITHOUT_SHIFT(4);
          STORE_ROW_WITHOUT_SHIFT(5);
          STORE_ROW_WITHOUT_SHIFT(6);
          STORE_ROW_WITHOUT_SHIFT(7);
        }
      }
      // Handle remaining rows
      for (; row < rows ; row++)
      {
          //output_ptr = (ae_int8x8 *)(&p_out[(row)*out_offset]);
          ae_int8x8 *p_src1 = (ae_int8x8 *)p_vec;
          ae_int32x2 accu1_01;
          ae_int32x2 accu1_23;
          ae_int8x8 *p_mat1_0 =(ae_int8x8 *) (&p_mat[(row+0)*cols]);
          accu1_01 = ZERO32;
          accu1_23 = ZERO32;

          ae_valign align_p_vec;
          AE_LA8X8POS_PC(align_p_vec,p_src1);
          ae_int8x8 temp_in0;
          for (col = 0; col < (cols>>3); col++)
          {
              AE_L8X8_IP(temp_in0 , p_mat1_0, 8);
              AE_LA8X8_IC(temp_src1, align_p_vec , p_src1);
              AE_MULA8Q8X8(accu1_01,accu1_23,temp_in0,temp_in0,temp_in0,temp_in0,temp_src1);
          }
          ae_int64 accu1_0;
          accu1_0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_01,ZERO32));
          accu1_0 = AE_SLAA64S(accu1_0 , -32);
          STORE_ROW_WITHOUT_SHIFT(0);
      }
  }
  else if(cols%8==0)
  {
      ae_int8x8 temp_src1;
//      ae_int8x8 * output_ptr=(ae_int8x8 *)p_out;
      if(rows >= 8)
      {
        for (row = 0; row < ( rows & ~(8-1)) ; row+=8)
        {
          ae_int8x8 *p_src1 = (ae_int8x8 *)p_vec;
          ae_int32x2 accu1_01;
          ae_int8x8 *p_mat1_0 =(ae_int8x8 *) (&p_mat[(row+0)*cols]);
          ae_int8x8 *p_mat1_1 =(ae_int8x8 *) (&p_mat[(row+1)*cols]);
          ae_int32x2 accu1_23;
          ae_int8x8 *p_mat1_2 =(ae_int8x8 *) (&p_mat[(row+2)*cols]);
          ae_int8x8 *p_mat1_3 =(ae_int8x8 *) (&p_mat[(row+3)*cols]);
          ae_int32x2 accu1_45;
          ae_int8x8 *p_mat1_4 =(ae_int8x8 *) (&p_mat[(row+4)*cols]);
          ae_int8x8 *p_mat1_5 =(ae_int8x8 *) (&p_mat[(row+5)*cols]);
          ae_int32x2 accu1_67;
          ae_int8x8 *p_mat1_6 =(ae_int8x8 *) (&p_mat[(row+6)*cols]);
          ae_int8x8 *p_mat1_7 =(ae_int8x8 *) (&p_mat[(row+7)*cols]);
          accu1_01 = ZERO32;
          accu1_23 = ZERO32;
          accu1_45 = ZERO32;
          accu1_67 = ZERO32;
#pragma ymemory (p_mat1_0)
#pragma ymemory (p_mat1_1)
#pragma ymemory (p_mat1_2)
#pragma ymemory (p_mat1_3)
          ae_int8x8 temp_in0;
          ae_int8x8 temp_in1;
          ae_int8x8 temp_in2;
          ae_int8x8 temp_in3;
          ae_int8x8 temp_in4;
          ae_int8x8 temp_in5;
          ae_int8x8 temp_in6;
          ae_int8x8 temp_in7;
          ae_valign align_p_vec;
          AE_LA8X8POS_PC(align_p_vec,p_src1);
          for (col = 0; col < (cols>>3); col++)
          {
              AE_L8X8_IP(temp_in0 , p_mat1_0, 8);
              AE_L8X8_IP(temp_in1 , p_mat1_1, 8);
              AE_L8X8_IP(temp_in2 , p_mat1_2, 8);
              AE_L8X8_IP(temp_in3 , p_mat1_3, 8);
              AE_L8X8_IP(temp_in4 , p_mat1_4, 8);
              AE_L8X8_IP(temp_in5 , p_mat1_5, 8);
              AE_L8X8_IP(temp_in6 , p_mat1_6, 8);
              AE_L8X8_IP(temp_in7 , p_mat1_7, 8);
              AE_LA8X8_IC(temp_src1, align_p_vec , p_src1);
              AE_MULA8Q8X8(accu1_01,accu1_23,temp_in0,temp_in1,temp_in2,temp_in3,temp_src1);
              AE_MULA8Q8X8(accu1_45,accu1_67,temp_in4,temp_in5,temp_in6,temp_in7,temp_src1);
          }
          ae_int64 accu1_0;
          ae_int64 accu1_1;
          ae_int64 accu1_2;
          ae_int64 accu1_3;
          ae_int64 accu1_4;
          ae_int64 accu1_5;
          ae_int64 accu1_6;
          ae_int64 accu1_7;
          accu1_0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu1_01,ZERO32));
          accu1_1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_01,ZERO32));
          accu1_2 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu1_23,ZERO32));
          accu1_3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_23,ZERO32));
          accu1_4 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu1_45,ZERO32));
          accu1_5 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_45,ZERO32));
          accu1_6 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu1_67,ZERO32));
          accu1_7 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_67,ZERO32));

          accu1_0 = AE_SLAA64S(accu1_0 , -32);
          accu1_1 = AE_SLAA64S(accu1_1 , -32);
          accu1_2 = AE_SLAA64S(accu1_2 , -32);
          accu1_3 = AE_SLAA64S(accu1_3 , -32);
          accu1_4 = AE_SLAA64S(accu1_4 , -32);
          accu1_5 = AE_SLAA64S(accu1_5 , -32);
          accu1_6 = AE_SLAA64S(accu1_6 , -32);
          accu1_7 = AE_SLAA64S(accu1_7 , -32);

          STORE_ROW_WITHOUT_SHIFT(0);
          STORE_ROW_WITHOUT_SHIFT(1);
          STORE_ROW_WITHOUT_SHIFT(2);
          STORE_ROW_WITHOUT_SHIFT(3);
          STORE_ROW_WITHOUT_SHIFT(4);
          STORE_ROW_WITHOUT_SHIFT(5);
          STORE_ROW_WITHOUT_SHIFT(6);
          STORE_ROW_WITHOUT_SHIFT(7);
        }
      }
      // Handle remaining rows
      for (; row < rows ; row++)
      {
          //output_ptr = (ae_int8x8 *)(&p_out[(row)*out_offset]);
          ae_int8x8 *p_src1 = (ae_int8x8 *)p_vec;
          ae_int32x2 accu1_01;
          ae_int32x2 accu1_23;
          ae_int8x8 *p_mat1_0 =(ae_int8x8 *) (&p_mat[(row)*cols]);

          accu1_01 = ZERO32;
          accu1_23 = ZERO32;
          ae_valign align_p_vec;
          AE_LA8X8POS_PC(align_p_vec,p_src1);
          ae_int8x8 temp_in0;
          ae_int8x8 zero_temp = AE_MOVINT8X8_FROMINT32X2(0);
          for (col = 0; col < (cols>>3); col++)
          {
              AE_L8X8_IP (temp_in0 , p_mat1_0, 8);
              AE_LA8X8_IC(temp_src1, align_p_vec , p_src1);
              AE_MULA8Q8X8(accu1_01,accu1_23,zero_temp,temp_in0,zero_temp,zero_temp,temp_src1);
          }
          ae_int64 accu1_0;
          accu1_0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_01,ZERO32));
          accu1_0 = AE_SLAA64S(accu1_0 , -32);
          STORE_ROW_WITHOUT_SHIFT(0);
      }
  }
  else if(cols%4==0)
  {
      ae_int16x4 temp_src1;
      if(rows > UNROLL_S)
      {
        for (row = 0; row < ( rows & ~(UNROLL_S-1)) ; row+=UNROLL_S)
        {
          WORD8 *p_src1 = p_vec;
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
        WORD8 *p_src1 = p_vec;
        SETUP_ROW_S(0);
        for (col = 0; col < (cols>>2); col++) {
          KERNEL_ROW_S_I(0);
        }
        STORE_ROW_S(0);
      }
  }

  return 0;
}

