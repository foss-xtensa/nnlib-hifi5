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
#include "xa_nnlib_common_macros_hifi5.h"

#define ZERO32   (0)

#if defined(CUST_UNROLL) && (CUST_UNROLL != 0)
#define UNROLL_D CUST_UNROLL
#else
#define UNROLL_D  4 /// Optimal unroll
#endif
#define LIMIT_VARIABLE(_var, _left_limit, _right_limit) \
  _var = _var > _right_limit ? _right_limit : _var < _left_limit ? _left_limit : _var;

#define SETUP_ROW_D(N) \
  ae_int64 accu1_ ##N, accu2_ ##N;\
  WORD8 *p_mat1_ ##N = p_mat; \
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_ ##N, (row+N) * row_offset * sizeof(WORD8)); \
  accu1_ ##N = ZERO64; \
  accu2_ ##N = ZERO64;

#define KERNEL_ROW_D(N) \
{\
  ae_int16x4 temp_in1; \
  temp_in1 = AE_L8X4F_I(p_mat1_ ##N, 0); \
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_ ##N, 4 * sizeof(WORD8)); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
  AE_MULAAAAQ16(accu2_ ##N, temp_src2, temp_in1); \
}

#define KERNEL_ROW_D_I(N) \
{\
  ae_int16x4 temp_in1; \
  temp_in1 = AE_L8X4F_I(p_mat1_ ##N, 0); \
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_ ##N, 4 * sizeof(WORD8)); \
  AE_L8X4F_IP(temp_src1, p_src1, 4); \
  AE_L8X4F_IP(temp_src2, p_src2, 4); \
  AE_MULAAAAQ16(accu1_ ##N, temp_src1, temp_in1);\
  AE_MULAAAAQ16(accu2_ ##N, temp_src2, temp_in1); \
}
#define STORE_ROW_D_WITHOUT_SHIFT(N) \
  ae_int64 temp1_ ##N, temp2_ ##N; \
  ae_int8x8 temp8_ ##N;\
  ae_int8x8 temp81_ ##N;\
  temp1_ ##N = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(p_bias[vec]));            \
  temp2_ ##N = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(p_bias[vec+1]));            \
  temp1_ ##N = AE_SLAI64S(temp1_ ##N , 8); \
  temp1_ ##N = AE_SRAI64(temp1_ ##N , 56); \
  temp2_ ##N = AE_SLAI64S(temp2_ ##N , 8); \
  temp2_ ##N = AE_SRAI64(temp2_ ##N , 56); \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , bias_shift); \
  temp2_ ##N = AE_SLAA64S(temp2_ ##N , bias_shift); \
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N); \
  accu2_ ##N = AE_ADD64(accu2_ ##N , temp2_ ##N); \
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , acc_shift); \
  accu2_ ##N = AE_SLAA64S(accu2_ ##N , acc_shift); \
  temp8_ ##N = AE_MOVINT8X8_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_ ##N),24),24));\
  p_dst1[(row+N) * out_row_offset] = AE_MOVAD8(temp8_ ##N,0);\
  temp81_ ##N = AE_MOVINT8X8_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu2_ ##N),24),24));\
  p_dst2[(row+N) * out_row_offset] = AE_MOVAD8(temp81_ ##N,0);

#define STORE_ROW_D(N) \
  accu1_ ##N = AE_SRAI64(accu1_ ##N , 16); \
  accu2_ ##N = AE_SRAI64(accu2_ ##N , 16); \
  ae_int64 temp1_ ##N, temp2_ ##N; \
  ae_int8x8 temp8_ ##N;\
  ae_int8x8 temp81_ ##N;\
  temp1_ ##N = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(p_bias[vec]));            \
  temp2_ ##N = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(p_bias[vec+1]));            \
  temp1_ ##N = AE_SLAI64S(temp1_ ##N , 8); \
  temp1_ ##N = AE_SRAI64(temp1_ ##N , 56); \
  temp2_ ##N = AE_SLAI64S(temp2_ ##N , 8); \
  temp2_ ##N = AE_SRAI64(temp2_ ##N , 56); \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , bias_shift); \
  temp2_ ##N = AE_SLAA64S(temp2_ ##N , bias_shift); \
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N); \
  accu2_ ##N = AE_ADD64(accu2_ ##N , temp2_ ##N); \
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , acc_shift); \
  accu2_ ##N = AE_SLAA64S(accu2_ ##N , acc_shift); \
  temp8_ ##N = AE_MOVINT8X8_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_ ##N),24),24));\
  p_dst1[(row+N) * out_row_offset] = AE_MOVAD8(temp8_ ##N,0);\
  temp81_ ##N = AE_MOVINT8X8_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu2_ ##N),24),24));\
  p_dst2[(row+N) * out_row_offset] = AE_MOVAD8(temp81_ ##N,0);


#if (UNROLL_D == 1)
#define SETUP_D SETUP_ROW_D(0)
#define KERNEL_D KERNEL_ROW_D_I(0)
#define STORE_D STORE_ROW_D(0)

#elif (UNROLL_D == 2)
#define SETUP_D  SETUP_ROW_D(0)  SETUP_ROW_D(1)
#define KERNEL_D KERNEL_ROW_D_I(0) KERNEL_ROW_D(1)
#define STORE_D  STORE_ROW_D(0)  STORE_ROW_D(1)

#elif (UNROLL_D == 4)
#define SETUP_D  SETUP_ROW_D(0)  SETUP_ROW_D(1)  SETUP_ROW_D(2)  SETUP_ROW_D(3)
#define KERNEL_D KERNEL_ROW_D_I(0) KERNEL_ROW_D(1) KERNEL_ROW_D(2) KERNEL_ROW_D(3)
#define STORE_D  STORE_ROW_D(0)  STORE_ROW_D(1)  STORE_ROW_D(2)  STORE_ROW_D(3)
#elif (UNROLL_D == 8)
#define SETUP_D   SETUP_ROW_D(0)  SETUP_ROW_D(1)  SETUP_ROW_D(2)  SETUP_ROW_D(3)  SETUP_ROW_D(4)  SETUP_ROW_D(5)  SETUP_ROW_D(6)  SETUP_ROW_D(7)
#define KERNEL_D KERNEL_ROW_D_I(0) KERNEL_ROW_D(1) KERNEL_ROW_D(2) KERNEL_ROW_D(3) KERNEL_ROW_D(4) KERNEL_ROW_D(5) KERNEL_ROW_D(6) KERNEL_ROW_D(7)
#define STORE_D   STORE_ROW_D(0)  STORE_ROW_D(1)  STORE_ROW_D(2)  STORE_ROW_D(3)  STORE_ROW_D(4)  STORE_ROW_D(5)  STORE_ROW_D(6)  STORE_ROW_D(7)

#endif

#if defined(CUST_UNROLL) && (CUST_UNROLL != 0)
#define UNROLL_S CUST_UNROLL
#else
#define UNROLL_S  8 /// Optimal unroll
#endif

#define SETUP_ROW_S(N) \
  ae_int64 accu1_ ##N;\
  WORD8 *p_mat1_ ##N = p_mat; \
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_ ##N, (row+N) * row_offset * sizeof(WORD8)); \
  accu1_ ##N = ZERO64; \

#define KERNEL_ROW_S(N) \
{ \
  ae_int8x8 temp_in8; \
  ae_int32x2 temp_in1, temp_in2; \
  AE_L8_XC(temp_in8, (ae_int8*)p_mat1_ ##N, 1); \
  AE_CVTA32X4F8_L(temp_in1, temp_in2, temp_in8, 0); \
  AE_MULA32_LL(accu1_ ##N, temp_src32_1, temp_in1);\
}

#define KERNEL_ROW_S_I(N) \
{ \
  ae_int8x8 temp_in8, temp_src8; \
  ae_int32x2 temp_in1, temp_in2; \
  AE_L8_IP(temp_src8, (ae_int8*)p_src1, 1); \
  AE_L8_XC(temp_in8, (ae_int8*)p_mat1_ ##N, 1); \
  AE_CVTA32X4F8_L(temp_in1, temp_in2, temp_in8, 0); \
  AE_CVTA32X4F8_L(temp_src32_1, temp_src32_2, temp_src8, 0); \
  AE_MULA32_LL(accu1_ ##N, temp_src32_1, temp_in1);\
}
#define STORE_ROW_S_WITHOUT_SHIFT(N) \
  ae_int64 temp1_ ##N; \
  ae_int8x8 temp8_ ##N;\
  temp1_ ##N = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(p_bias[vec]));            \
  temp1_ ##N = AE_SLAI64S(temp1_ ##N , 8); \
  temp1_ ##N = AE_SRAI64(temp1_ ##N , 56); \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , bias_shift); \
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N); \
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , acc_shift); \
  temp8_ ##N = AE_MOVINT8X8_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_ ##N),24),24));\
  p_dst1[(row+N) * out_row_offset] = AE_MOVAD8(temp8_ ##N,0);

#define STORE_ROW_S(N) \
  ae_int64 temp1_ ##N; \
  ae_int8x8 temp8_ ##N;\
  temp1_ ##N = AE_MOVINT64_FROMINT16X4(AE_MOVDA16(p_bias[vec]));            \
  temp1_ ##N = AE_SLAI64S(temp1_ ##N , 8); \
  temp1_ ##N = AE_SRAI64(temp1_ ##N , 56); \
  temp1_ ##N = AE_SLAA64S(temp1_ ##N , bias_shift); \
  accu1_ ##N = AE_ADD64(accu1_ ##N , temp1_ ##N); \
  accu1_ ##N = AE_SLAA64S(accu1_ ##N , acc_shift); \
  temp8_ ##N = AE_MOVINT8X8_FROMINT32(AE_SRAI32(AE_SLAI32S(AE_ROUND32F64SSYM(accu1_ ##N),24),24));\
  p_dst1[(vec+N) * out_col_offset] = AE_MOVAD8(temp8_ ##N,0);

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

static const long long g_sel_pattern[16] = {
 0xf7e6d5c4L, 0xb3a29180L,
 0xe7d6c5b4L, 0xa3928170L,
 0xd7c6b5a4L, 0x93827160L,
 0xc7b6a594L, 0x83726150L,
 0xb7a69584L, 0x73625140L,
 0xa7968574L, 0x63524130L,
 0x97867564L, 0x53423120L,
 0x87766554L, 0x43322110L
};

static inline void _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_0_1
    ,ae_int32x2* out_0_2
    ,ae_int32x2* out_0_3
    ,ae_int32x2* out_1_0
    ,ae_int32x2* out_1_1
    ,ae_int32x2* out_1_2
    ,ae_int32x2* out_1_3
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset 
    )
{
  int pre_loop_count, loop_count, post_loop_count, pre_loop_shift;
  int c_itr;

  ae_int8x8 vec0_batch_0, vec0_batch_1; 
  ae_int8x8 vec1_batch_0, vec1_batch_1; 
  ae_int8x8 vec2_batch_0, vec2_batch_1; 
  ae_int8x8 vec3_batch_0, vec3_batch_1; 

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

  int align_offset = ((unsigned int)p_vec_0 & 0x7);
  pre_loop_count = 8 - align_offset;
  pre_loop_shift = align_offset * 8;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, -align_offset);
  //TODO: possible out of bound access
  p_vec_0 -= align_offset;

  pre_loop_count += 8; // 16 values loaded in preloop step 
  loop_count = cols - pre_loop_count;
  post_loop_count = loop_count & 0xf;
  loop_count >>= 4;

  int rem_cols_shift_0 = (post_loop_count <= 8) ? (8 - post_loop_count) * 8 : 0;
  int rem_cols_shift_1 = (post_loop_count > 8) ? (16 - post_loop_count) * 8 : 64;

  ae_int8x8* p_mat1_1 = p_mat1_0; //next 8th row 
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1,  1 * row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1; //next 8th row
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2,  1 * row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2; //next 8th row 
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3,  1 * row_offset * sizeof(WORD8));

  ae_int8* p_vec_1 = p_vec_0 + 8 * vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + 8 * vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + 8 * vec_offset;

  ae_valign align_p_mat1_0;
  ae_valign align_p_mat1_1;
  ae_valign align_p_mat1_2;
  ae_valign align_p_mat1_3;

  AE_LA8X8POS_PC(align_p_mat1_0, p_mat1_0);
  AE_LA8X8POS_PC(align_p_mat1_1, p_mat1_1);
  AE_LA8X8POS_PC(align_p_mat1_2, p_mat1_2);
  AE_LA8X8POS_PC(align_p_mat1_3, p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_0_1;
  ae_int32x2 acc_row0_vec2 = *out_0_2;
  ae_int32x2 acc_row0_vec3 = *out_0_3;
                       
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row1_vec1 = *out_1_1;
  ae_int32x2 acc_row1_vec2 = *out_1_2;
  ae_int32x2 acc_row1_vec3 = *out_1_3;

  /* Pre loop computation */
  AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
  AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, p_mat1_0);
  AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
  AE_LA8X8_IC(mat1_row1_1, align_p_mat1_1, p_mat1_1);
  AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
  AE_LA8X8_IC(mat1_row2_1, align_p_mat1_2, p_mat1_2);
  AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);
  AE_LA8X8_IC(mat1_row3_1, align_p_mat1_3, p_mat1_3);

  AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
  AE_L8X8_IP(vec0_batch_1, (ae_int8x8 *)p_vec_0, 8);
  AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
  AE_L8X8_IP(vec1_batch_1, (ae_int8x8 *)p_vec_1, 8);
  AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
  AE_L8X8_IP(vec2_batch_1, (ae_int8x8 *)p_vec_2, 8);
  AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);
  AE_L8X8_IP(vec3_batch_1, (ae_int8x8 *)p_vec_3, 8);

  mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), pre_loop_shift), pre_loop_shift));
  mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), pre_loop_shift), pre_loop_shift));
  mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), pre_loop_shift), pre_loop_shift));
  mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), pre_loop_shift), pre_loop_shift));

  AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
  AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
  AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

  AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
  AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
  AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
  AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);

#pragma no_unroll
  for(c_itr = 0; c_itr < loop_count; c_itr++)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row1_1, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row2_1, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);
    AE_LA8X8_IC(mat1_row3_1, align_p_mat1_3, p_mat1_3);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8*)p_vec_0, 8);
    AE_L8X8_IP(vec0_batch_1, (ae_int8x8*)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8*)p_vec_1, 8);
    AE_L8X8_IP(vec1_batch_1, (ae_int8x8*)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8*)p_vec_2, 8);
    AE_L8X8_IP(vec2_batch_1, (ae_int8x8*)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8*)p_vec_3, 8);
    AE_L8X8_IP(vec3_batch_1, (ae_int8x8*)p_vec_3, 8);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }

  //Remainder loop for cols
  c_itr = 0;
  int rem_shift = rem_cols_shift_0;;
  while(c_itr < post_loop_count)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8*)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8*)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8*)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8*)p_vec_3, 8);

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_shift), rem_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_shift), rem_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_shift), rem_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_shift), rem_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    c_itr += 8;
    rem_shift = rem_cols_shift_1;
  }

  *out_0_0 = acc_row0_vec0;
  *out_0_1 = acc_row0_vec1;
  *out_0_2 = acc_row0_vec2;
  *out_0_3 = acc_row0_vec3;
       
  *out_1_0 = acc_row1_vec0;
  *out_1_1 = acc_row1_vec1;
  *out_1_2 = acc_row1_vec2;
  *out_1_3 = acc_row1_vec3;
}

#if 0
static inline void _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int rem_cols_shift = 64 - (cols & 7) * 8;
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  ae_int8x8 *p_mat1_1 = p_mat1_0; 
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1,  8 * row_offset * sizeof(WORD8));
  ae_int8x8 *p_mat1_2 = p_mat1_1; 
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2,  8 * row_offset * sizeof(WORD8));
  ae_int8x8 *p_mat1_3 = p_mat1_2; 
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3,  8 * row_offset * sizeof(WORD8));

  ae_valign align_p_mat1_0;
  ae_valign align_p_mat1_1;
  ae_valign align_p_mat1_2;
  ae_valign align_p_mat1_3;

  AE_LA8X8POS_PC(align_p_mat1_0, p_mat1_0);
  AE_LA8X8POS_PC(align_p_mat1_1, p_mat1_1);
  AE_LA8X8POS_PC(align_p_mat1_2, p_mat1_2);
  AE_LA8X8POS_PC(align_p_mat1_3, p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);

  int cols_count=cols-(cols&7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}
#endif

static inline void _xa_nn_dot_product_1_rows_4_vecs_offset_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int rem_cols_shift = 64 - (cols & 7) * 8;
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + 8 * row_offset); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + 8 * row_offset); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + 8 * row_offset); 

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  ae_valign align_p_mat1_3 = AE_LA64_PP(p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  /* p_vec needs to be accessed in circular fashion */
  AE_SW_PRIME_CIRC_64(p_vec_0, align_p_vec0);

  int cols_count = cols - (cols & 7);
  for(c_itr = 0; c_itr < cols_count >> 3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IC(vec0_batch_0, align_p_vec0, p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IC(vec0_batch_0, align_p_vec0, p_vec_0);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_4_rows_4_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_0_1
    ,ae_int32x2* out_0_2
    ,ae_int32x2* out_0_3
    ,ae_int32x2* out_1_0
    ,ae_int32x2* out_1_1
    ,ae_int32x2* out_1_2
    ,ae_int32x2* out_1_3
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    ,WORD32      vec_offset 
    )
{
  int rem_cols_shift_0 = ((cols & 15) <= 8)? (8 - (cols & 15)) * 8 : 0;
  int rem_cols_shift_1 = ((cols & 15) > 8)? (16 - (cols & 15)) * 8 : 64;
  int c_itr = 0;

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

  ae_int8x8 vec0_batch_0, vec0_batch_1; 
  ae_int8x8 vec1_batch_0, vec1_batch_1; 
  ae_int8x8 vec2_batch_0, vec2_batch_1; 
  ae_int8x8 vec3_batch_0, vec3_batch_1; 
  ae_int8x8 align_p_vec0, align_p_vec1, align_p_vec2, align_p_vec3; 

  /* p_mat needs to be accessed in circular fashion */
  ae_int8x8* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD8));

  ae_valign align_p_mat1_0;
  ae_valign align_p_mat1_1;
  ae_valign align_p_mat1_2;
  ae_valign align_p_mat1_3;

  AE_LA8X8POS_PC(align_p_mat1_0, p_mat1_0);
  AE_LA8X8POS_PC(align_p_mat1_1, p_mat1_1);
  AE_LA8X8POS_PC(align_p_mat1_2, p_mat1_2);
  AE_LA8X8POS_PC(align_p_mat1_3, p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_0_1;
  ae_int32x2 acc_row0_vec2 = *out_0_2;
  ae_int32x2 acc_row0_vec3 = *out_0_3;

  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row1_vec1 = *out_1_1;
  ae_int32x2 acc_row1_vec2 = *out_1_2;
  ae_int32x2 acc_row1_vec3 = *out_1_3;

  ae_int8* p_vec_1 = p_vec_0 + vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);
  AE_SW_PRIME_64(p_vec_1, align_p_vec1);
  AE_SW_PRIME_64(p_vec_2, align_p_vec2);
  AE_SW_PRIME_64(p_vec_3, align_p_vec3);

  int cols_count = cols -(cols & 15);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row1_1, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row2_1, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);
    AE_LA8X8_IC(mat1_row3_1, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);
    AE_SW_LA8X8_IP(vec0_batch_1, align_p_vec0, p_vec_0);
    AE_SW_LA8X8_IP(vec1_batch_0, align_p_vec1, p_vec_1);
    AE_SW_LA8X8_IP(vec1_batch_1, align_p_vec1, p_vec_1);
    AE_SW_LA8X8_IP(vec2_batch_0, align_p_vec2, p_vec_2);
    AE_SW_LA8X8_IP(vec2_batch_1, align_p_vec2, p_vec_2);
    AE_SW_LA8X8_IP(vec3_batch_0, align_p_vec3, p_vec_3);
    AE_SW_LA8X8_IP(vec3_batch_1, align_p_vec3, p_vec_3);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }

  //Remainder loop for cols
  c_itr <<= 4;
  int rem_shift = rem_cols_shift_0;;
  while(c_itr < cols)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);
    AE_SW_LA8X8_IP(vec1_batch_0, align_p_vec1, p_vec_1);
    AE_SW_LA8X8_IP(vec2_batch_0, align_p_vec2, p_vec_2);
    AE_SW_LA8X8_IP(vec3_batch_0, align_p_vec3, p_vec_3);

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_shift), rem_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_shift), rem_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_shift), rem_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_shift), rem_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);
    c_itr += 8;
    rem_shift = rem_cols_shift_1;
  }

  *out_0_0 = acc_row0_vec0;
  *out_0_1 = acc_row0_vec1;
  *out_0_2 = acc_row0_vec2;
  *out_0_3 = acc_row0_vec3;

  *out_1_0 = acc_row1_vec0;
  *out_1_1 = acc_row1_vec1;
  *out_1_2 = acc_row1_vec2;
  *out_1_3 = acc_row1_vec3;
}

static inline void _xa_nn_dot_product_4_rows_1_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int rem_cols_shift = 64 - (cols & 7)*8;
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;

  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  /* p_mat needs to be accessed in circular fashion */
  ae_int8x8* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD8));

  ae_valign align_p_mat1_0;
  ae_valign align_p_mat1_1;
  ae_valign align_p_mat1_2;
  ae_valign align_p_mat1_3;

  AE_LA8X8POS_PC(align_p_mat1_0, p_mat1_0);
  AE_LA8X8POS_PC(align_p_mat1_1, p_mat1_1);
  AE_LA8X8POS_PC(align_p_mat1_2, p_mat1_2);
  AE_LA8X8POS_PC(align_p_mat1_3, p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;

  ae_int32x2 acc_row1_vec0 = *out_1_0;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);

  int cols_count = cols - (cols & 7);
  for(c_itr = 0; c_itr < cols_count >> 3; c_itr++)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count != cols)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }


  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_4_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int rem_cols_shift = 64 - (cols & 7) * 8;
  int c_itr = 0;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  /* p_vec needs to be accessed in circular fashion */
  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_offset); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_offset); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_offset); 

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  ae_valign align_p_mat1_3 = AE_LA64_PP(p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  AE_SW_PRIME_CIRC_64(p_vec_0, align_p_vec0);

  int cols_count=cols-(cols&7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IC(vec0_batch_0, align_p_vec0, p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IC(vec0_batch_0, align_p_vec0, p_vec_0);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_unaligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    )
{
  int c_itr = 0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0; 
  ae_int8x8 mat1_row0_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  ae_valign align_p_mat1_0;
  AE_LA8X8POS_PC(align_p_mat1_0, p_mat1_0);
  AE_SW_PRIME_64(p_vec_0, align_p_vec0); //TODO: h/w align?

  int rem_cols_shift = 64 - (cols & 7) * 8;
  int cols_count=cols-(cols&7);

#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0, acc_row0_vec1, vec0_batch_0, vec0_batch_0, vec0_batch_0, vec0_batch_0, mat1_row0_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0, acc_row0_vec1, vec0_batch_0, vec0_batch_0, vec0_batch_0, vec0_batch_0, mat1_row0_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

WORD32 xa_nn_matXvec_8x8_8_circ(
  WORD8  * __restrict__ p_out,
  WORD8  * __restrict__ p_mat,
  WORD8  * __restrict__ p_vec,
  WORD8  * __restrict__ p_bias,
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

  //printf("rows , cols, vec_count %d %d %d,\n",rows,cols,vec_count);
  if ((NULL == p_out) || (NULL == p_mat) || (NULL == p_vec))
  {
    return -1;
  }

  if ((0 >= rows ) || (0 >= cols ) )
  {
    return -2;
  }
  if(0 >= vec_count) return -3;
  if(cols%16==0)
  {
      ae_int8x8 temp_src1;
      ae_int8x8 temp_src2;
      ae_int8x8 temp_src1_1;
      ae_int8x8 temp_src2_1;
      if(vec_count >=2)
      {
        for(vec = 0; vec < (vec_count & (~0x1)); vec+=2)
        {
          row=0;
          WORD8 *p_dst1 = (WORD8 *)&p_out[vec*out_col_offset];
          WORD8 *p_dst2 = (WORD8 *)&p_out[(vec+1)*out_col_offset];
          if(rows >= 4)
          {
            for (row = 0; row < ( rows & ~(4-1)) ; row+=4)
            {
              ae_int8x8 *p_src1 = (ae_int8x8 *)&p_vec[vec * vec_offset];
              ae_int8x8 *p_src2 = (ae_int8x8 *)&p_vec[(vec+1) * vec_offset];

              ae_int32x2 accu1_01;
              ae_int32x2 accu2_01;
              ae_int8x16 *p_mat1_0 =(ae_int8x16 *) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, (row+0) * row_offset * sizeof(WORD8));
              ae_valignx2 p_mat1_0_align;
              AE_LA8X8X2POS_PC(p_mat1_0_align,p_mat1_0);

              ae_int8x16 *p_mat1_1 =(ae_int8x16 *) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, (row+1) * row_offset * sizeof(WORD8));
              ae_valignx2 p_mat1_1_align;
              AE_LA8X8X2POS_PC(p_mat1_1_align,p_mat1_1);
              accu1_01 = ZERO32;
              accu2_01 = ZERO32;


              ae_int32x2 accu1_23;
              ae_int32x2 accu2_23;
              ae_int8x16 *p_mat1_2 =(ae_int8x16 *) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, (row+2) * row_offset * sizeof(WORD8));
              ae_valignx2 p_mat1_2_align;
              AE_LA8X8X2POS_PC(p_mat1_2_align,p_mat1_2);

              ae_int8x16 *p_mat1_3 = (ae_int8x16 *) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, (row+3) * row_offset * sizeof(WORD8));
              ae_valignx2 p_mat1_3_align;
              AE_LA8X8X2POS_PC(p_mat1_3_align,p_mat1_3);

              accu1_23 = ZERO32;
              accu2_23 = ZERO32;

              ae_int8x8 temp_in0;
              ae_int8x8 temp_in1;
              ae_int8x8 temp_in2;
              ae_int8x8 temp_in3;

              ae_int8x8 temp_in0_1;
              ae_int8x8 temp_in1_1;
              ae_int8x8 temp_in2_1;
              ae_int8x8 temp_in3_1;
#pragma ymemory (p_mat1_0)
#pragma ymemory (p_mat1_1)
              for (col = 0; col < (cols>>4); col++)
              {
                  AE_LA8X8X2_IC(temp_in0 ,temp_in0_1 , p_mat1_0_align ,(ae_int8x16*) p_mat1_0);
                  AE_LA8X8X2_IC(temp_in1 ,temp_in1_1 , p_mat1_1_align ,(ae_int8x16*) p_mat1_1);
                  AE_LA8X8X2_IC(temp_in2 ,temp_in2_1 , p_mat1_2_align ,(ae_int8x16*) p_mat1_2);
                  AE_LA8X8X2_IC(temp_in3 ,temp_in3_1 , p_mat1_3_align ,(ae_int8x16*) p_mat1_3);
                  AE_L8X8X2_IP(temp_src1,temp_src1_1,(ae_int8x16*)   p_src1, 16);
                  AE_L8X8X2_IP(temp_src2,temp_src2_1,(ae_int8x16*)   p_src2, 16);
                  AE_MULA8Q8X8(accu1_01,accu1_23,temp_in0,temp_in1,temp_in2,temp_in3,temp_src1);
                  AE_MULA8Q8X8(accu2_01,accu2_23,temp_in0,temp_in1,temp_in2,temp_in3,temp_src2);
                  AE_MULA8Q8X8(accu1_01,accu1_23,temp_in0_1,temp_in1_1,temp_in2_1,temp_in3_1,temp_src1_1);
                  AE_MULA8Q8X8(accu2_01,accu2_23,temp_in0_1,temp_in1_1,temp_in2_1,temp_in3_1,temp_src2_1);
              }
              ae_int64 accu1_0;
              ae_int64 accu1_1;
              ae_int64 accu1_2;
              ae_int64 accu1_3;
              ae_int64 accu2_0;
              ae_int64 accu2_1;
              ae_int64 accu2_2;
              ae_int64 accu2_3;
              accu1_0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu1_01,ZERO32));
              accu1_1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_01,ZERO32));
              accu1_2 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu1_23,ZERO32));
              accu1_3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_23,ZERO32));
              accu2_0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu2_01,ZERO32));
              accu2_1 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu2_01,ZERO32));
              accu2_2 = AE_MOVINT64_FROMINT32X2(AE_SEL32_HH(accu2_23,ZERO32));
              accu2_3 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu2_23,ZERO32));
              accu1_0 = AE_SRAI64(accu1_0 , 32);
              accu1_1 = AE_SRAI64(accu1_1 , 32);
              accu1_2 = AE_SRAI64(accu1_2 , 32);
              accu1_3 = AE_SRAI64(accu1_3 , 32);
              accu2_0 = AE_SRAI64(accu2_0 , 32);
              accu2_1 = AE_SRAI64(accu2_1 , 32);
              accu2_2 = AE_SRAI64(accu2_2 , 32);
              accu2_3 = AE_SRAI64(accu2_3 , 32);

              STORE_ROW_D_WITHOUT_SHIFT(0);
              STORE_ROW_D_WITHOUT_SHIFT(1);
              STORE_ROW_D_WITHOUT_SHIFT(2);
              STORE_ROW_D_WITHOUT_SHIFT(3);
            }
          }
          // Handle remaining rows
          for (; row < rows ; row++)
          {
              ae_int8x8 *p_src1 = (ae_int8x8 *)&p_vec[vec * vec_offset];
              ae_int8x8 *p_src2 = (ae_int8x8 *)&p_vec[(vec+1) * vec_offset];

              ae_int32x2 accu1_01;
              ae_int32x2 accu2_01;
              ae_int8x8 *p_mat1_0 =(ae_int8x8 *) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, (row+0) * row_offset * sizeof(WORD8));
              ae_valign p_mat1_0_align;
              AE_LA8X8POS_PC(p_mat1_0_align,p_mat1_0);

              accu1_01 = ZERO32;
              accu2_01 = ZERO32;

              ae_int8x8 temp_in0;
              ae_int8x8 zero_temp = AE_MOVINT8X8_FROMINT32X2(0);
              for (col = 0; col < (cols>>3); col++)
              {
                  AE_LA8X8_IC (temp_in0 ,p_mat1_0_align ,p_mat1_0);
                  AE_L8X8_IP (temp_src1,   p_src1, 8);
                  AE_L8X8_IP (temp_src2,   p_src2, 8);
                  AE_MULA8Q8X8(accu1_01,accu2_01,zero_temp,temp_src1,zero_temp,temp_src2,temp_in0);
              }
              ae_int64 accu1_0;
              ae_int64 accu2_0;
              accu1_0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_01,ZERO32));
              accu2_0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu2_01,ZERO32));
              accu1_0 = AE_SRAI64(accu1_0 , 32);
              accu2_0 = AE_SRAI64(accu2_0 , 32);

              STORE_ROW_D_WITHOUT_SHIFT(0);
          }
        }
      }
      if(vec_count & 0x1)
      {
        for(; vec < vec_count ; vec++)
        {
          WORD8 *p_dst1 = (WORD8 *)&p_out[vec*out_col_offset];
          for (row=0; row < rows ; row++)
          {
              ae_int8x8 *p_src1 = (ae_int8x8 *)&p_vec[vec * vec_offset];

              ae_int32x2 accu1_01;
              ae_int32x2 accu2_01;
              ae_int8x8 *p_mat1_0 =(ae_int8x8 *) p_mat;
              AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, (row+0) * row_offset * sizeof(WORD8));
              ae_valign p_mat1_0_align;
              AE_LA8X8POS_PC(p_mat1_0_align,p_mat1_0);

              accu1_01 = ZERO32;
              accu2_01 = ZERO32;

              ae_int8x8 temp_in0;
              ae_int8x8 zero_temp = AE_MOVINT8X8_FROMINT32X2(0);
              for (col = 0; col < (cols>>3); col++)
              {
                  AE_LA8X8_IC (temp_in0 ,p_mat1_0_align ,p_mat1_0);
                  AE_L8X8_IP (temp_src1,   p_src1, 8);
                  AE_MULA8Q8X8(accu1_01,accu2_01,zero_temp,temp_src1,zero_temp,zero_temp,temp_in0);
              }
              ae_int64 accu1_0;
              accu1_0 = AE_MOVINT64_FROMINT32X2(AE_SEL32_LL(accu1_01,ZERO32));
              accu1_0 = AE_SRAI64(accu1_0 , 32);
              STORE_ROW_S_WITHOUT_SHIFT(0);
          }
        }
      }
  }
  else
  {
    int m_itr = 0, vec_itr = 0;
    ae_int64 bias_array[32];

    int neg_bias = (bias_shift <= 0) ? 1 : 0; 
    int rshift_bias = neg_bias ? XT_MIN(16 , -bias_shift) : -1;
    ae_int32x2 lshift_mul_bias = neg_bias ? AE_MOVDA32(1) : AE_MOVDA32(1 << (bias_shift - 1));

    ae_int16x4 _ae_int16x4_bias; 
    int out_stride = out_row_offset;
    int out_offset = out_col_offset;

    for(vec_itr = 0; vec_itr < (vec_count & ~(32 - 1)); vec_itr += 32)
    {
      int ii, b_itr = 0;
      ae_int16x4 _ae_int16x4_bias1; 
      ae_int8x8 _ae_int8x8_bias; 
      ae_valign align_p_bias_ua;
      WORD8 *p_bias_ua = p_bias + vec_itr;
      align_p_bias_ua = AE_LA64_PP(p_bias_ua);

      for(b_itr = 0; b_itr < 32; b_itr += 8)
      {
        AE_LA8X8_IP(_ae_int8x8_bias, align_p_bias_ua, (ae_int8x8 *)p_bias_ua);
        AE_CVTI16X4X2F8(_ae_int16x4_bias, _ae_int16x4_bias1, _ae_int8x8_bias, 0);
        _ae_int16x4_bias = AE_SRAA16S(_ae_int16x4_bias, rshift_bias);
        _ae_int16x4_bias1 = AE_SRAA16S(_ae_int16x4_bias1, rshift_bias);
        bias_array[b_itr + 0] = AE_MUL32X16_L3(lshift_mul_bias, _ae_int16x4_bias);
        bias_array[b_itr + 1] = AE_MUL32X16_L2(lshift_mul_bias, _ae_int16x4_bias);
        bias_array[b_itr + 2] = AE_MUL32X16_L1(lshift_mul_bias, _ae_int16x4_bias);
        bias_array[b_itr + 3] = AE_MUL32X16_L0(lshift_mul_bias, _ae_int16x4_bias);
        bias_array[b_itr + 4] = AE_MUL32X16_L3(lshift_mul_bias, _ae_int16x4_bias1);
        bias_array[b_itr + 5] = AE_MUL32X16_L2(lshift_mul_bias, _ae_int16x4_bias1);
        bias_array[b_itr + 6] = AE_MUL32X16_L1(lshift_mul_bias, _ae_int16x4_bias1);
        bias_array[b_itr + 7] = AE_MUL32X16_L0(lshift_mul_bias, _ae_int16x4_bias1);
      }
      
      for(ii = 0; ii < 8; ii++)
      {
        WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + ii +  0) * out_offset;
        WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + ii +  8) * out_offset;
        WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + ii + 16) * out_offset;
        WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + ii + 24) * out_offset;

        m_itr = 0;

        for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
        {
          ae_int32x2 acc_row0_vec0 = ZERO32;
          ae_int32x2 acc_row1_vec0 = ZERO32;
          ae_int32x2 acc_row0_vec1 = ZERO32;
          ae_int32x2 acc_row1_vec1 = ZERO32;
          ae_int32x2 acc_row0_vec2 = ZERO32;
          ae_int32x2 acc_row1_vec2 = ZERO32;
          ae_int32x2 acc_row0_vec3 = ZERO32;
          ae_int32x2 acc_row1_vec3 = ZERO32;

          ae_int8* p_vec_0  = (ae_int8 *)(p_vec + (vec_itr + ii) * vec_offset);
          ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat; 
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_offset * sizeof(WORD8));

          _xa_nn_dot_product_4_rows_4_vecs_offset_aligned
            (&acc_row0_vec0
             ,&acc_row0_vec1
             ,&acc_row0_vec2
             ,&acc_row0_vec3
             ,&acc_row1_vec0
             ,&acc_row1_vec1
             ,&acc_row1_vec2
             ,&acc_row1_vec3
             ,p_mat1_0
             ,p_vec_0
             ,cols
             ,row_offset
             ,vec_offset
            );

          ae_int64 acc64_0_0, acc64_0_1, acc64_0_2, acc64_0_3;
          ae_int64 acc64_1_0, acc64_1_1, acc64_1_2, acc64_1_3;
          ae_int64 acc64_2_0, acc64_2_1, acc64_2_2, acc64_2_3;
          ae_int64 acc64_3_0, acc64_3_1, acc64_3_2, acc64_3_3;

          acc64_0_0 = acc64_1_0 = acc64_2_0 = acc64_3_0 = bias_array[ 0 + ii];
          acc64_0_1 = acc64_1_1 = acc64_2_1 = acc64_3_1 = bias_array[ 8 + ii];
          acc64_0_2 = acc64_1_2 = acc64_2_2 = acc64_3_2 = bias_array[16 + ii];
          acc64_0_3 = acc64_1_3 = acc64_2_3 = acc64_3_3 = bias_array[24 + ii];

          AE_ACCW32(acc64_0_0, acc64_1_0, acc_row0_vec0, ZERO32);
          AE_ACCW32(acc64_0_1, acc64_1_1, acc_row0_vec1, ZERO32);
          AE_ACCW32(acc64_0_2, acc64_1_2, acc_row0_vec2, ZERO32);
          AE_ACCW32(acc64_0_3, acc64_1_3, acc_row0_vec3, ZERO32);
          AE_ACCW32(acc64_2_0, acc64_3_0, acc_row1_vec0, ZERO32);
          AE_ACCW32(acc64_2_1, acc64_3_1, acc_row1_vec1, ZERO32);
          AE_ACCW32(acc64_2_2, acc64_3_2, acc_row1_vec2, ZERO32);
          AE_ACCW32(acc64_2_3, acc64_3_3, acc_row1_vec3, ZERO32);

          acc_row0_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_0,acc_shift), AE_SLAA64S(acc64_1_0,acc_shift));
          acc_row1_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_0,acc_shift), AE_SLAA64S(acc64_3_0,acc_shift));
          acc_row0_vec1 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_1,acc_shift), AE_SLAA64S(acc64_1_1,acc_shift));
          acc_row1_vec1 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_1,acc_shift), AE_SLAA64S(acc64_3_1,acc_shift));
          acc_row0_vec2 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_2,acc_shift), AE_SLAA64S(acc64_1_2,acc_shift));
          acc_row1_vec2 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_2,acc_shift), AE_SLAA64S(acc64_3_2,acc_shift));
          acc_row0_vec3 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_3,acc_shift), AE_SLAA64S(acc64_1_3,acc_shift));
          acc_row1_vec3 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_3,acc_shift), AE_SLAA64S(acc64_3_3,acc_shift));

          ae_int8x8 temp_vec0, temp_vec1, temp_vec2, temp_vec3;

          temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row1_vec0);
          temp_vec1 = AE_SAT8X4X32_L(acc_row0_vec1, acc_row1_vec1);
          temp_vec2 = AE_SAT8X4X32_L(acc_row0_vec2, acc_row1_vec2);
          temp_vec3 = AE_SAT8X4X32_L(acc_row0_vec3, acc_row1_vec3);

          AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
          AE_SW_S8_3_XP(temp_vec1, (ae_int8 *) p_dst_1, out_stride);
          AE_SW_S8_3_XP(temp_vec2, (ae_int8 *) p_dst_2, out_stride);
          AE_SW_S8_3_XP(temp_vec3, (ae_int8 *) p_dst_3, out_stride);
          
          AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
          AE_SW_S8_2_XP(temp_vec1, (ae_int8 *) p_dst_1, out_stride);
          AE_SW_S8_2_XP(temp_vec2, (ae_int8 *) p_dst_2, out_stride);
          AE_SW_S8_2_XP(temp_vec3, (ae_int8 *) p_dst_3, out_stride);
          
          AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
          AE_SW_S8_1_XP(temp_vec1, (ae_int8 *) p_dst_1, out_stride);
          AE_SW_S8_1_XP(temp_vec2, (ae_int8 *) p_dst_2, out_stride);
          AE_SW_S8_1_XP(temp_vec3, (ae_int8 *) p_dst_3, out_stride);
          
          AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
          AE_S8_0_XP(temp_vec1, (ae_int8 *) p_dst_1, out_stride);
          AE_S8_0_XP(temp_vec2, (ae_int8 *) p_dst_2, out_stride);
          AE_S8_0_XP(temp_vec3, (ae_int8 *) p_dst_3, out_stride);
        }

        // Remaining rows 
        // TODO: codesize considerations: remove inline for 4 row, 1 vec case ?
        for (; m_itr < rows; m_itr++)
        {
          ae_int32x2 acc_row0_vec0 = ZERO32;
          ae_int32x2 acc_row1_vec0 = ZERO32;

          ae_int8* p_vec_0  = (ae_int8 *)(p_vec + (vec_itr + ii) * vec_offset);
          ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat;
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_offset * sizeof(WORD8));

          _xa_nn_dot_product_1_rows_4_vecs_offset_aligned
            (&acc_row0_vec0
             ,&acc_row1_vec0
             ,(ae_int8x8*)p_vec_0
             ,(ae_int8*)p_mat1_0
             ,cols
             ,vec_offset
            );

          ae_int64 acc64_0_0;
          ae_int64 acc64_1_0;
          ae_int64 acc64_2_0;
          ae_int64 acc64_3_0;

          ae_int8x8 temp_vec0;
          
          acc64_0_0 = bias_array[ 0 + ii];
          acc64_1_0 = bias_array[ 8 + ii];
          acc64_2_0 = bias_array[16 + ii];
          acc64_3_0 = bias_array[24 + ii];

          AE_ACCW32(acc64_0_0, acc64_1_0, acc_row0_vec0, ZERO32);
          AE_ACCW32(acc64_2_0, acc64_3_0, acc_row1_vec0, ZERO32);

          acc_row0_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_0,acc_shift), AE_SLAA64S(acc64_1_0,acc_shift));
          acc_row1_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_0,acc_shift), AE_SLAA64S(acc64_3_0,acc_shift));

          temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row1_vec0);

          AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
          AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);
          AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_2, out_stride);
          AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_3, out_stride);
        }
      }
    }

    // Process loop for 4 rows and 4 vectors 
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;


      //TODO: use 8x4F load
      WORD8 *p_bias_ua = p_bias + vec_itr;
      _ae_int16x4_bias = AE_MOVDA16(*p_bias_ua++);
      _ae_int16x4_bias = AE_SRAA16S(_ae_int16x4_bias, rshift_bias);
      bias_array[0] = AE_MUL32X16_L3(lshift_mul_bias, _ae_int16x4_bias);
      _ae_int16x4_bias = AE_MOVDA16(*p_bias_ua++);
      _ae_int16x4_bias = AE_SRAA16S(_ae_int16x4_bias, rshift_bias);
      bias_array[1] = AE_MUL32X16_L3(lshift_mul_bias, _ae_int16x4_bias);
      _ae_int16x4_bias = AE_MOVDA16(*p_bias_ua++);
      _ae_int16x4_bias = AE_SRAA16S(_ae_int16x4_bias, rshift_bias);
      bias_array[2] = AE_MUL32X16_L3(lshift_mul_bias, _ae_int16x4_bias);
      _ae_int16x4_bias = AE_MOVDA16(*p_bias_ua++);
      _ae_int16x4_bias = AE_SRAA16S(_ae_int16x4_bias, rshift_bias);
      bias_array[3] = AE_MUL32X16_L3(lshift_mul_bias, _ae_int16x4_bias);

      m_itr = 0;

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0 = ZERO32;
        ae_int32x2 acc_row1_vec0 = ZERO32;
        ae_int32x2 acc_row0_vec1 = ZERO32;
        ae_int32x2 acc_row1_vec1 = ZERO32;
        ae_int32x2 acc_row0_vec2 = ZERO32;
        ae_int32x2 acc_row1_vec2 = ZERO32;
        ae_int32x2 acc_row0_vec3 = ZERO32;
        ae_int32x2 acc_row1_vec3 = ZERO32;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_offset * sizeof(WORD8));

        _xa_nn_dot_product_4_rows_4_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,&acc_row0_vec2
           ,&acc_row0_vec3
           ,&acc_row1_vec0
           ,&acc_row1_vec1
           ,&acc_row1_vec2
           ,&acc_row1_vec3
           ,p_mat1_0
           ,p_vec_0
           ,cols
           ,row_offset
           ,vec_offset
          );

        ae_int64 acc64_0_0, acc64_0_1, acc64_0_2, acc64_0_3;
        ae_int64 acc64_1_0, acc64_1_1, acc64_1_2, acc64_1_3;
        ae_int64 acc64_2_0, acc64_2_1, acc64_2_2, acc64_2_3;
        ae_int64 acc64_3_0, acc64_3_1, acc64_3_2, acc64_3_3;

        ae_int8x8 temp_vec0, temp_vec1, temp_vec2, temp_vec3;

        acc64_0_0 = acc64_1_0 = acc64_2_0 = acc64_3_0 = bias_array[0];
        acc64_0_1 = acc64_1_1 = acc64_2_1 = acc64_3_1 = bias_array[1];
        acc64_0_2 = acc64_1_2 = acc64_2_2 = acc64_3_2 = bias_array[2];
        acc64_0_3 = acc64_1_3 = acc64_2_3 = acc64_3_3 = bias_array[3];

        AE_ACCW32(acc64_0_0, acc64_1_0, acc_row0_vec0, ZERO32);
        AE_ACCW32(acc64_0_1, acc64_1_1, acc_row0_vec1, ZERO32);
        AE_ACCW32(acc64_0_2, acc64_1_2, acc_row0_vec2, ZERO32);
        AE_ACCW32(acc64_0_3, acc64_1_3, acc_row0_vec3, ZERO32);
        AE_ACCW32(acc64_2_0, acc64_3_0, acc_row1_vec0, ZERO32);
        AE_ACCW32(acc64_2_1, acc64_3_1, acc_row1_vec1, ZERO32);
        AE_ACCW32(acc64_2_2, acc64_3_2, acc_row1_vec2, ZERO32);
        AE_ACCW32(acc64_2_3, acc64_3_3, acc_row1_vec3, ZERO32);

        acc_row0_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_0,acc_shift), AE_SLAA64S(acc64_1_0,acc_shift));
        acc_row1_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_0,acc_shift), AE_SLAA64S(acc64_3_0,acc_shift));
        acc_row0_vec1 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_1,acc_shift), AE_SLAA64S(acc64_1_1,acc_shift));
        acc_row1_vec1 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_1,acc_shift), AE_SLAA64S(acc64_3_1,acc_shift));
        acc_row0_vec2 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_2,acc_shift), AE_SLAA64S(acc64_1_2,acc_shift));
        acc_row1_vec2 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_2,acc_shift), AE_SLAA64S(acc64_3_2,acc_shift));
        acc_row0_vec3 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_3,acc_shift), AE_SLAA64S(acc64_1_3,acc_shift));
        acc_row1_vec3 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_3,acc_shift), AE_SLAA64S(acc64_3_3,acc_shift));

        temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row1_vec0);
        temp_vec1 = AE_SAT8X4X32_L(acc_row0_vec1, acc_row1_vec1);
        temp_vec2 = AE_SAT8X4X32_L(acc_row0_vec2, acc_row1_vec2);
        temp_vec3 = AE_SAT8X4X32_L(acc_row0_vec3, acc_row1_vec3);

        AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_3_XP(temp_vec1, (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_3_XP(temp_vec2, (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_3_XP(temp_vec3, (ae_int8 *) p_dst_3, out_stride);

        AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_2_XP(temp_vec1, (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(temp_vec2, (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_2_XP(temp_vec3, (ae_int8 *) p_dst_3, out_stride);

        AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_1_XP(temp_vec1, (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_1_XP(temp_vec2, (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_1_XP(temp_vec3, (ae_int8 *) p_dst_3, out_stride);

        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_S8_0_XP(temp_vec1, (ae_int8 *) p_dst_1, out_stride);
        AE_S8_0_XP(temp_vec2, (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(temp_vec3, (ae_int8 *) p_dst_3, out_stride);
      }

      // Remaining vectors
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = ZERO32;
        ae_int32x2 acc_row1_vec0 = ZERO32;

        ae_int8* p_vec_0  = (ae_int8*)(p_vec + vec_itr * vec_offset);
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_offset * sizeof(WORD8));

        _xa_nn_dot_product_1_rows_4_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_vec_0
           ,(ae_int8*)p_mat1_0
           ,cols
           ,vec_offset
          );

        ae_int64 acc64_0_0;
        ae_int64 acc64_1_0;
        ae_int64 acc64_2_0;
        ae_int64 acc64_3_0;

        ae_int8x8 temp_vec0;

        acc64_0_0 = bias_array[0];
        acc64_1_0 = bias_array[1];
        acc64_2_0 = bias_array[2];
        acc64_3_0 = bias_array[3];

        AE_ACCW32(acc64_0_0, acc64_1_0, acc_row0_vec0, ZERO32);
        AE_ACCW32(acc64_2_0, acc64_3_0, acc_row1_vec0, ZERO32);

        acc_row0_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_0,acc_shift), AE_SLAA64S(acc64_1_0,acc_shift));
        acc_row1_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_2_0,acc_shift), AE_SLAA64S(acc64_3_0,acc_shift));

        temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row1_vec0);

        AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst_3, out_stride);
      }
    }

    // remaining vectors 
    for(; vec_itr < vec_count; vec_itr++)
    {
      WORD8 *p_bias_ua = p_bias + vec_itr;
      _ae_int16x4_bias = AE_MOVDA16(*p_bias_ua);
      _ae_int16x4_bias = AE_SRAA16S(_ae_int16x4_bias, rshift_bias);
      bias_array[0] = AE_MUL32X16_L3(lshift_mul_bias, _ae_int16x4_bias);

      WORD8* p_dst = (WORD8*)p_out + (vec_itr + 0) * out_offset;

      m_itr = 0;

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0 = ZERO32;
        ae_int32x2 acc_row0_vec1 = ZERO32;

        ae_int8x8 * p_vec_0  = (ae_int8x8 *)(p_vec + vec_itr * vec_offset);
        WORD8 *p_mat1_0 = (WORD8 *)p_mat;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_offset * sizeof(WORD8));

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec_0
           ,cols
           ,row_offset
          );

        ae_int64 acc64_0_0, acc64_0_1, acc64_0_2, acc64_0_3;
        ae_int8x8 temp_vec0;

        acc64_0_0 = bias_array[0];
        acc64_0_1 = acc64_0_0;
        acc64_0_2 = acc64_0_0;
        acc64_0_3 = acc64_0_0;

        AE_ACCW32(acc64_0_0, acc64_0_1, acc_row0_vec0, ZERO32);
        AE_ACCW32(acc64_0_2, acc64_0_3, acc_row0_vec1, ZERO32);

        acc_row0_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_0,acc_shift), AE_SLAA64S(acc64_0_1,acc_shift));
        acc_row0_vec1 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_2,acc_shift), AE_SLAA64S(acc64_0_3,acc_shift));

        temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec1);

        AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_2_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_1_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
      }

      // Remaining vectors
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = ZERO32;
        ae_int32x2 acc_row0_vec1 = ZERO32;

        ae_int8* p_vec_0  = (ae_int8*)(p_vec + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*)p_mat;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_offset * sizeof(WORD8));

        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_mat1_0
           ,p_vec_0
           ,cols
          );

        ae_int64 acc64_0_0, dummy;
        ae_int8x8 temp_vec0;

        acc64_0_0 = bias_array[0];

        AE_ACCW32(acc64_0_0, dummy, acc_row0_vec0, ZERO32);

        acc_row0_vec0 = AE_ROUND32X2F64SSYM(AE_SLAA64S(acc64_0_0,acc_shift), AE_SLAA64S(acc64_0_0,acc_shift));

        temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);

        //TODO: AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
      }
    }
  }

  return 0;
}

