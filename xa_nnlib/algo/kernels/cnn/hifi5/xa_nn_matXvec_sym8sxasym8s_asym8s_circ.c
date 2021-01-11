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
#include "xa_nn_conv2d_std_state.h"
#include "xa_nnlib_common_macros_hifi5.h"

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2(inp, multiplier, left_shift, right_shift) \
    inp = AE_SLAA32(inp, left_shift); \
    inp = AE_MULFP32X2RAS(inp, AE_MOVDA32(multiplier)); \
    inp = AE_SRAA32SYMS(inp, right_shift);

#define MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out, inp1, inp2, multiplier_23, multiplier_01, l_shift_23, l_shift_01, r_shift_23, r_shift_01, out_off) \
{\
  AE_MUL2P32X4S(inp1, inp2, inp1, inp2, l_shift_01, l_shift_23); \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, multiplier_01, multiplier_23); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, r_shift_01, r_shift_23); \
  out = AE_SAT16X4(inp1, inp2); \
  out = AE_ADD16S(AE_MOVDA16(out_off), out); \
  AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127)); \
}

#define MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_mobnetv2(out, inp1, inp2, multiplier_32, multiplier_10, l_shift_32, l_shift_10, r_shift_32, r_shift_10, out_off) \
{\
  AE_MUL2P32X4S(inp1, inp2, inp1, inp2, l_shift_10, l_shift_32); \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, multiplier_10, multiplier_32); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, r_shift_10, r_shift_32); \
  out = AE_SAT16X4(inp2, inp1); \
  out = AE_ADD16S(AE_MOVDA16(out_off), out); \
}

#define MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_splcase(out, inp1, inp2, multiplier_23, multiplier_01, r_shift_23, r_shift_01, out_off) \
{\
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, AE_NEG32S(multiplier_01), AE_NEG32S(multiplier_23)); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, r_shift_01, r_shift_23); \
  out = AE_SAT16X4(inp1, inp2); \
  out = AE_ADD16S(AE_MOVDA16(out_off), out); \
  AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127)); \
}

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out, inp, inp1, multiplier, l_shift, right_shift, out_off) \
    AE_MUL2P32X4S(inp, inp1, inp, inp1, l_shift, l_shift); \
    AE_MULF2P32X4RAS(inp, inp1, inp, inp1, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
    inp = AE_SRAA32SYMS(inp, AE_MOVAD32_H(right_shift));\
    inp1 = AE_SRAA32SYMS(inp1, AE_MOVAD32_L(right_shift));\
    out = AE_SAT16X4(inp, inp1); \
    out = AE_ADD16S(AE_MOVDA16(out_off), out); \
    AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127)); 

#define PACK_32X2(dst1, src1, src2) \
dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x080a0c0e, 0x00020406)));

extern const long long g_sel_pattern[16];
extern const long long pre_loop_sel_pattern[16];
extern const long long post_loop_sel_pattern[16];

static inline void _xa_nn_dot_product_4_rows_4_vecs_aligned
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

  /* p_mat needs to be accessed in circular fashion */
  ae_int8x8* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD8));

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

  int cols_count = cols -(cols & 15);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_XC(mat1_row0_1, p_mat1_0, 8);
    AE_L8X8_XC(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_XC(mat1_row1_1, p_mat1_1, 8);
    AE_L8X8_XC(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_XC(mat1_row2_1, p_mat1_2, 8);
    AE_L8X8_XC(mat1_row3_0, p_mat1_3, 8);
    AE_L8X8_XC(mat1_row3_1, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec0_batch_1, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec1_batch_1, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec2_batch_1, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);
    AE_L8X8_IP(vec3_batch_1, (ae_int8x8 *)p_vec_3, 8);

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
  int rem_shift = rem_cols_shift_0;
  while(c_itr < cols)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_XC(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_XC(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_XC(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);

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

static inline void _xa_nn_dot_product_4_rows_1_vecs_aligned
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

  /* p_mat needs to be accessed in circular fashion */
  ae_int8x8* p_mat1_1 = p_mat1_0;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2;
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD8));

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  int cols_count = cols - (cols & 7);
  for(c_itr = 0; c_itr < cols_count >> 3; c_itr++)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_XC(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_XC(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_XC(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count != cols)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_XC(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_XC(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_XC(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_4_vecs_aligned
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

  /* p_vec needs to be accessed in circular fashion */
  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_offset); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_offset); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_offset); 

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  int cols_count=cols-(cols&7);
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_XC(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_XC(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    )
{
  int c_itr = 0;
  
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 mat1_row0_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  int rem_cols_shift = 64 - (cols & 7) * 8;
  int cols_count=cols-(cols&7);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);

    AE_MULA8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_L8X8_XC(mat1_row0_0, p_mat1_0, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0, acc_row0_vec1, mat1_row0_0, mat1_row0_0, mat1_row0_0, mat1_row0_0, vec0_batch_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

// All functions imported from conv2d_std_8x8
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
  loop_count = (cols < pre_loop_count)?0:(cols - pre_loop_count);
  post_loop_count = loop_count?(loop_count & 15):((cols + align_offset) & 15);
  loop_count >>= 4;

  int mask_start_end = ((cols + align_offset) < 16)?0:1;
  
  int rem_cols_shift_0 = (post_loop_count <= 8) ? (8 - post_loop_count) * 8 : 0;
  int rem_cols_shift_1 = (post_loop_count > 8) ? (16 - post_loop_count) * 8 : 64;

  ae_int8x8* p_mat1_1 = p_mat1_0; 
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_2 = p_mat1_1;  
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_offset * sizeof(WORD8));
  ae_int8x8* p_mat1_3 = p_mat1_2;  
  AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_offset * sizeof(WORD8));

  // Process every 8th vector
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

  if(align_offset)
  {
    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), pre_loop_shift), pre_loop_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), pre_loop_shift), pre_loop_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), pre_loop_shift), pre_loop_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), pre_loop_shift), pre_loop_shift));
  }

  if(mask_start_end)
  {
    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
  }

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
  if(post_loop_count)
  {
    if(mask_start_end) 
    {
      AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);
      AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, p_mat1_1);
      AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, p_mat1_2);
      AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, p_mat1_3);
      
      AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, p_mat1_0);
      AE_LA8X8_IC(mat1_row1_1, align_p_mat1_1, p_mat1_1);
      AE_LA8X8_IC(mat1_row2_1, align_p_mat1_2, p_mat1_2);
      AE_LA8X8_IC(mat1_row3_1, align_p_mat1_3, p_mat1_3);

      AE_L8X8_IP(vec0_batch_0, (ae_int8x8*)p_vec_0, 8);
      AE_L8X8_IP(vec1_batch_0, (ae_int8x8*)p_vec_1, 8);
      AE_L8X8_IP(vec2_batch_0, (ae_int8x8*)p_vec_2, 8);
      AE_L8X8_IP(vec3_batch_0, (ae_int8x8*)p_vec_3, 8);
      
      AE_L8X8_IP(vec0_batch_1, (ae_int8x8*)p_vec_0, 8);
      AE_L8X8_IP(vec1_batch_1, (ae_int8x8*)p_vec_1, 8);
      AE_L8X8_IP(vec2_batch_1, (ae_int8x8*)p_vec_2, 8);
      AE_L8X8_IP(vec3_batch_1, (ae_int8x8*)p_vec_3, 8);
    }

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift_0), rem_cols_shift_0));

    mat1_row0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_1), rem_cols_shift_1), rem_cols_shift_1));
    mat1_row1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_1), rem_cols_shift_1), rem_cols_shift_1));
    mat1_row2_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_1), rem_cols_shift_1), rem_cols_shift_1));
    mat1_row3_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_1), rem_cols_shift_1), rem_cols_shift_1));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);
    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);
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

  ae_int8x8 *p_mat1_1 = p_mat1_0 + row_offset; //next 8th vector
  ae_int8x8 *p_mat1_2 = p_mat1_1 + row_offset; //next 8th vector
  ae_int8x8 *p_mat1_3 = p_mat1_2 + row_offset; //next 8th vector

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
  ae_valign align_p_vec0; 
  ae_int8x8 mat1_row0_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  ae_valign align_p_mat1_0;
  AE_LA8X8POS_PC(align_p_mat1_0, p_mat1_0);
  align_p_vec0 = AE_LA64_PP(p_vec_0);
  int rem_cols_shift = 64 - (cols & 7) * 8;
  int cols_count=cols-(cols&7);

#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0, acc_row0_vec1, vec0_batch_0, vec0_batch_0, vec0_batch_0, vec0_batch_0, mat1_row0_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec0, (ae_int8x8 *)p_vec_0);

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0, acc_row0_vec1, vec0_batch_0, vec0_batch_0, vec0_batch_0, vec0_batch_0, mat1_row0_0);
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

WORD32 xa_nn_matXvec_sym8sxasym8s_asym8s_circ(
    WORD8 * __restrict__ p_out,
    WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_stride,
    WORD32 out_col_offset,
    WORD32 out_row_offset,
    WORD32 mat1_offset,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  /* Iterators used in for loops */
  int m_itr, vec_itr;
  
  for(vec_itr = 0; vec_itr < vec_count; vec_itr++)
  {
    if((p_out_shift[vec_itr] > 31) || (p_out_shift[vec_itr] < -31))
    {
      return -1;
    }
  }
  if (!p_bias)
  {
    return -1;
  }

  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);
  
  int c_itr = 0;
  int rem_cols_shift = 64 - (cols1 & 7) * 8;
  ae_int8x8 mat_z_b = AE_MOVDA8(-mat1_offset);

  ae_int8x8 vec0_0, vec0_1;
  ae_int8x8 vec1_0, vec1_1;
  ae_int8x8 vec2_0, vec2_1;
  ae_int8x8 vec3_0, vec3_1;

  if(cols1 > 8 && cols1 < 16 && (vec_count & 0x3) == 0 && (out_col_offset == 1))
  {
    ae_int32x2 bias_buffer[2];
    m_itr = 0, vec_itr = 0;

    int out_stride = out_row_offset;

    ae_int32x4 *pt_bias = (ae_int32x4 *)p_bias;
    ae_valignx2 align_p_bias = AE_LA128_PP(pt_bias);
    
    ae_int32x4 *pt_out_mult = (ae_int32x4 *)p_out_multiplier;
    ae_valignx2 align_p_out_mult = AE_LA128_PP(pt_out_mult);
    // Process loop for 4 rows and 4 vectors 
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0);
      m_itr = 0;

      ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
      ae_int8* p_vec_1 = p_vec_0 + vec_stride; 
      ae_int8* p_vec_2 = p_vec_1 + vec_stride;
      ae_int8* p_vec_3 = p_vec_2 + vec_stride;

      ae_int8x8 vec0_batch_0, vec0_batch_1; 
      ae_int8x8 vec1_batch_0, vec1_batch_1; 
      ae_int8x8 vec2_batch_0, vec2_batch_1; 
      ae_int8x8 vec3_batch_0, vec3_batch_1;

      ae_valignx2 align_p_vec0, align_p_vec1, align_p_vec2, align_p_vec3; 

      align_p_vec0 = AE_LA128_PP(p_vec_0);
      align_p_vec1 = AE_LA128_PP(p_vec_1);
      align_p_vec2 = AE_LA128_PP(p_vec_2);
      align_p_vec3 = AE_LA128_PP(p_vec_3);

      AE_LAV8X8X2_XP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0, cols1);
      AE_LAV8X8X2_XP(vec1_batch_0, vec1_batch_1, align_p_vec1, (ae_int8x16 *)p_vec_1, cols1);
      AE_LAV8X8X2_XP(vec2_batch_0, vec2_batch_1, align_p_vec2, (ae_int8x16 *)p_vec_2, cols1);
      AE_LAV8X8X2_XP(vec3_batch_0, vec3_batch_1, align_p_vec3, (ae_int8x16 *)p_vec_3, cols1);
      
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      AE_MULA8Q8X8(acc_row0, acc_row1, vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat_z_b);
      AE_MULA8Q8X8(acc_row0, acc_row1, vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat_z_b);
 
      ae_int32x2 d_bias_0, d_bias_1;
      AE_LA32X2X2_IP(d_bias_0, d_bias_1, align_p_bias, (ae_int32x4 *)pt_bias);
      acc_row0 = AE_SUB32S(d_bias_0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias_1, acc_row1);
      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4 *)bias_buffer, 0);
      
      /* Shifts to match with Tensorflow */
      int p_left_shift[4], p_right_shift[4];
      
      p_left_shift[0] = p_out_shift[vec_itr + 0] < 0 ? 0 : p_out_shift[vec_itr + 0];
      p_left_shift[1] = p_out_shift[vec_itr + 1] < 0 ? 0 : p_out_shift[vec_itr + 1];
      p_left_shift[2] = p_out_shift[vec_itr + 2] < 0 ? 0 : p_out_shift[vec_itr + 2];
      p_left_shift[3] = p_out_shift[vec_itr + 3] < 0 ? 0 : p_out_shift[vec_itr + 3];

      p_right_shift[0] = p_out_shift[vec_itr + 0] > 0 ? 0 : -p_out_shift[vec_itr + 0];
      p_right_shift[1] = p_out_shift[vec_itr + 1] > 0 ? 0 : -p_out_shift[vec_itr + 1];
      p_right_shift[2] = p_out_shift[vec_itr + 2] > 0 ? 0 : -p_out_shift[vec_itr + 2];
      p_right_shift[3] = p_out_shift[vec_itr + 3] > 0 ? 0 : -p_out_shift[vec_itr + 3];

      ae_int32x2 l_mult23 = AE_MOVDA32X2( 1 << p_left_shift[2], 1 << p_left_shift[3]); 
      ae_int32x2 l_mult01 = AE_MOVDA32X2( 1 << p_left_shift[0], 1 << p_left_shift[1]);

      ae_int32x2 r_mult01 = AE_MOVDA32X2( (-1 << (31 - p_right_shift[0])), (-1 << (31 - p_right_shift[1])));
      ae_int32x2 r_mult23 = AE_MOVDA32X2( (-1 << (31 - p_right_shift[2])), (-1 << (31 - p_right_shift[3]))); 

      ae_int32x2 p_out_mult01, p_out_mult23;
      AE_LA32X2X2_IP(p_out_mult01, p_out_mult23, align_p_out_mult, (ae_int32x4 *)pt_out_mult);
      p_out_mult01 = AE_NEG32(p_out_mult01);
      p_out_mult23 = AE_NEG32(p_out_mult23);

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0, acc_row0_vec1, acc_row0_vec2, acc_row0_vec3;
        ae_int32x2 acc_row1_vec0, acc_row1_vec1, acc_row1_vec2, acc_row1_vec3;
        
        /* Initialize accumulators with bias - (ker * inp_zero_bias) */
        AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)bias_buffer, 0);

        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1;
        ae_int8x8 mat1_row1_0, mat1_row1_1;
        ae_int8x8 mat1_row2_0, mat1_row2_1;
        ae_int8x8 mat1_row3_0, mat1_row3_1;

        /* p_mat needs to be accessed in circular fashion */
        ae_int8x8* p_mat1_1 = p_mat1_0;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_stride1 * sizeof(WORD8));
        ae_int8x8* p_mat1_2 = p_mat1_1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_stride1 * sizeof(WORD8));
        ae_int8x8* p_mat1_3 = p_mat1_2;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_stride1 * sizeof(WORD8));

        ae_valignx2 align_p_mat1_0, align_p_mat1_1, align_p_mat1_2,align_p_mat1_3;

        AE_LA8X8X2POS_PC(align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2POS_PC(align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
        AE_LA8X8X2POS_PC(align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
        AE_LA8X8X2POS_PC(align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

        AE_LA8X8X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2_IC(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
        AE_LA8X8X2_IC(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
        AE_LA8X8X2_IC(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0);
        AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row1_0);
        AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row2_0);
        AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row3_0);

        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_1 , vec1_batch_1 , vec2_batch_1 , vec3_batch_1 , mat1_row0_1);
        AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_1 , vec1_batch_1 , vec2_batch_1 , vec3_batch_1 , mat1_row1_1);
        AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_1 , vec1_batch_1 , vec2_batch_1 , vec3_batch_1 , mat1_row2_1);
        AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_1 , vec1_batch_1 , vec2_batch_1 , vec3_batch_1 , mat1_row3_1);

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult23, p_out_mult01, l_mult23, l_mult01, r_mult23, r_mult01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_1, acc_row0_vec1, acc_row1_vec1, p_out_mult23, p_out_mult01, l_mult23, l_mult01, r_mult23, r_mult01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_2, acc_row0_vec2, acc_row1_vec2, p_out_mult23, p_out_mult01, l_mult23, l_mult01, r_mult23, r_mult01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_3, acc_row0_vec3, acc_row1_vec3, p_out_mult23, p_out_mult01, l_mult23, l_mult01, r_mult23, r_mult01, out_zero_bias);
       
        /* Store output */
        ae_int8x8 out32_0, out32_1; 
        PACK_32X2(out32_0, out_0, out_1);
        PACK_32X2(out32_1, out_2, out_3);
        
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0, acc_row1_vec0;
        AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)bias_buffer, 0);
        
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1;
        ae_valignx2 align_p_mat1_0;

        AE_LA8X8X2POS_PC(align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        AE_LA8X8X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0 , vec1_batch_0 , vec2_batch_0 , vec3_batch_0 , mat1_row0_0);
        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_1 , vec1_batch_1 , vec2_batch_1 , vec3_batch_1 , mat1_row0_1);

        /* Apply quantization */
        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult23, p_out_mult01, l_mult23, l_mult01, r_mult23, r_mult01, out_zero_bias); 
        
        /* Store output */
        ae_int8x8 out32_0; 
        PACK_32X2(out32_0, out_0, out_0);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
      }
    }
  }
  else if(cols1 == 27 && (vec_count & 0x3) == 0 && (out_col_offset == 1))
  {
    ae_int32x2 bias_buffer[2];
    m_itr = 0, vec_itr = 0;

    int out_stride = out_row_offset;

    int rem_cols = cols1 - 16;
    // Process loop for 4 rows and 4 vectors 
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0);
      m_itr = 0;

      ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
      ae_int8* p_vec_1 = p_vec_0 + vec_stride; 
      ae_int8* p_vec_2 = p_vec_1 + vec_stride;
      ae_int8* p_vec_3 = p_vec_2 + vec_stride;

      ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
      ae_int8x8 vec1_batch_0, vec1_batch_1, vec1_batch_2, vec1_batch_3; 
      ae_int8x8 vec2_batch_0, vec2_batch_1, vec2_batch_2, vec2_batch_3; 
      ae_int8x8 vec3_batch_0, vec3_batch_1, vec3_batch_2, vec3_batch_3;

      ae_valignx2 align_p_vec0, align_p_vec1, align_p_vec2, align_p_vec3; 

      align_p_vec0 = AE_LA128_PP(p_vec_0);
      align_p_vec1 = AE_LA128_PP(p_vec_1);
      align_p_vec2 = AE_LA128_PP(p_vec_2);
      align_p_vec3 = AE_LA128_PP(p_vec_3);

      AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0);
      AE_LA8X8X2_IP(vec1_batch_0, vec1_batch_1, align_p_vec1, (ae_int8x16 *)p_vec_1);
      AE_LA8X8X2_IP(vec2_batch_0, vec2_batch_1, align_p_vec2, (ae_int8x16 *)p_vec_2);
      AE_LA8X8X2_IP(vec3_batch_0, vec3_batch_1, align_p_vec3, (ae_int8x16 *)p_vec_3);
      
      AE_LAV8X8X2_XP(vec0_batch_2, vec0_batch_3, align_p_vec0, (ae_int8x16 *)p_vec_0, rem_cols);
      AE_LAV8X8X2_XP(vec1_batch_2, vec1_batch_3, align_p_vec1, (ae_int8x16 *)p_vec_1, rem_cols);
      AE_LAV8X8X2_XP(vec2_batch_2, vec2_batch_3, align_p_vec2, (ae_int8x16 *)p_vec_2, rem_cols);
      AE_LAV8X8X2_XP(vec3_batch_2, vec3_batch_3, align_p_vec3, (ae_int8x16 *)p_vec_3, rem_cols);
      
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_0, vec2_batch_0, vec1_batch_0, vec0_batch_0, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_1, vec2_batch_1, vec1_batch_1, vec0_batch_1, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_2, vec2_batch_2, vec1_batch_2, vec0_batch_2, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_3, vec2_batch_3, vec1_batch_3, vec0_batch_3, mat_z_b);
 
      ae_int32x2 d_bias_0, d_bias_1;
      d_bias_1 = AE_MOVDA32X2(p_bias[vec_itr + 3], p_bias[vec_itr + 2]);
      d_bias_0 = AE_MOVDA32X2(p_bias[vec_itr + 1], p_bias[vec_itr + 0]);
      
      acc_row0 = AE_SUB32S(d_bias_0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias_1, acc_row1);
      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4 *)bias_buffer, 0);
      
      /* Shifts to match with Tensorflow */
      int p_left_shift[4], p_right_shift[4];
      
      p_left_shift[3] = p_out_shift[vec_itr + 0] < 0 ? 0 : p_out_shift[vec_itr + 0];
      p_left_shift[2] = p_out_shift[vec_itr + 1] < 0 ? 0 : p_out_shift[vec_itr + 1];
      p_left_shift[1] = p_out_shift[vec_itr + 2] < 0 ? 0 : p_out_shift[vec_itr + 2];
      p_left_shift[0] = p_out_shift[vec_itr + 3] < 0 ? 0 : p_out_shift[vec_itr + 3];

      p_right_shift[3] = p_out_shift[vec_itr + 0] > 0 ? 0 : -p_out_shift[vec_itr + 0];
      p_right_shift[2] = p_out_shift[vec_itr + 1] > 0 ? 0 : -p_out_shift[vec_itr + 1];
      p_right_shift[1] = p_out_shift[vec_itr + 2] > 0 ? 0 : -p_out_shift[vec_itr + 2];
      p_right_shift[0] = p_out_shift[vec_itr + 3] > 0 ? 0 : -p_out_shift[vec_itr + 3];

      ae_int32x2 l_mult32 = AE_MOVDA32X2( 1 << p_left_shift[0], 1 << p_left_shift[1]);
      ae_int32x2 l_mult10 = AE_MOVDA32X2( 1 << p_left_shift[2], 1 << p_left_shift[3]); 

      ae_int32x2 r_mult32 = AE_MOVDA32X2( (-1 << (31 - p_right_shift[0])), (-1 << (31 - p_right_shift[1])));
      ae_int32x2 r_mult10 = AE_MOVDA32X2( (-1 << (31 - p_right_shift[2])), (-1 << (31 - p_right_shift[3]))); 

      ae_int32x2 p_out_mult10, p_out_mult32;
      p_out_mult32 = AE_MOVDA32X2(-p_out_multiplier[vec_itr + 3], -p_out_multiplier[vec_itr + 2]);
      p_out_mult10 = AE_MOVDA32X2(-p_out_multiplier[vec_itr + 1], -p_out_multiplier[vec_itr + 0]);

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0, acc_row0_vec1, acc_row0_vec2, acc_row0_vec3;
        ae_int32x2 acc_row1_vec0, acc_row1_vec1, acc_row1_vec2, acc_row1_vec3;
        
        /* Initialize accumulators with bias - (ker * inp_zero_bias) */
        AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)bias_buffer, 0);

        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
        ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
        ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
        ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;

        /* p_mat needs to be accessed in circular fashion */
        ae_int8x8* p_mat1_1 = p_mat1_0;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_stride1 * sizeof(WORD8));
        ae_int8x8* p_mat1_2 = p_mat1_1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_stride1 * sizeof(WORD8));
        ae_int8x8* p_mat1_3 = p_mat1_2;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_stride1 * sizeof(WORD8));

        ae_valignx2 align_p_mat1_0, align_p_mat1_1, align_p_mat1_2,align_p_mat1_3;

        AE_LA8X8X2POS_PC(align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2POS_PC(align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
        AE_LA8X8X2POS_PC(align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
        AE_LA8X8X2POS_PC(align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

        AE_LA8X8X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2_IC(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
        AE_LA8X8X2_IC(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
        AE_LA8X8X2_IC(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

        AE_LA8X8X2_IC(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2_IC(mat1_row1_2, mat1_row1_3, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
        AE_LA8X8X2_IC(mat1_row2_2, mat1_row2_3, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
        AE_LA8X8X2_IC(mat1_row3_2, mat1_row3_3, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row0_0);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row1_0);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row2_0);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row3_0);

        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row0_1);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row1_1);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row2_1);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row3_1);

        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row0_2);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row1_2);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row2_2);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row3_2);

        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row0_3);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row1_3);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row2_3);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row3_3);

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_mobnetv2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult32, p_out_mult10, l_mult32, l_mult10, r_mult32, r_mult10, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_mobnetv2(out_1, acc_row0_vec1, acc_row1_vec1, p_out_mult32, p_out_mult10, l_mult32, l_mult10, r_mult32, r_mult10, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_mobnetv2(out_2, acc_row0_vec2, acc_row1_vec2, p_out_mult32, p_out_mult10, l_mult32, l_mult10, r_mult32, r_mult10, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_mobnetv2(out_3, acc_row0_vec3, acc_row1_vec3, p_out_mult32, p_out_mult10, l_mult32, l_mult10, r_mult32, r_mult10, out_zero_bias);
        
        /* Store output */
        ae_int8x8 out32_0, out32_1; 
        out32_0 = AE_SAT8X8X16(out_0, out_1);
        out32_1 = AE_SAT8X8X16(out_2, out_3);
        
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0, acc_row1_vec0;
        AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)bias_buffer, 0);
        
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
        ae_valignx2 align_p_mat1_0;

        AE_LA8X8X2POS_PC(align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        AE_LA8X8X2_IC(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2_IC(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);

        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row0_0);
        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row0_1);
        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row0_2);
        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row0_3);

        /* Apply quantization */
        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_mobnetv2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult32, p_out_mult10, l_mult32, l_mult10, r_mult32, r_mult10, out_zero_bias); 
   
        /* Store output */
        ae_int8x8 out32_0; 
        out32_0 = AE_SAT8X8X16(out_0, out_0);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
      }
    }
  }
  else if((cols1 == 40) && (vec_count & 0x3) == 0 && (out_col_offset == 1))
  {
    ae_int32x2 bias_buffer[2];
    m_itr = 0, vec_itr = 0;

    int out_stride = out_row_offset;
    
    // Process loop for 4 rows and 4 vectors 
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0);
      m_itr = 0;

      ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
      ae_int8* p_vec_1 = p_vec_0 + vec_stride; 
      ae_int8* p_vec_2 = p_vec_1 + vec_stride;
      ae_int8* p_vec_3 = p_vec_2 + vec_stride;

      ae_int8x8 vec0_batch_0, vec0_batch_1; 
      ae_int8x8 vec1_batch_0, vec1_batch_1; 
      ae_int8x8 vec2_batch_0, vec2_batch_1; 
      ae_int8x8 vec3_batch_0, vec3_batch_1;
      ae_int8x8 vec0_batch_2, vec0_batch_3; 
      ae_int8x8 vec1_batch_2, vec1_batch_3; 
      ae_int8x8 vec2_batch_2, vec2_batch_3; 
      ae_int8x8 vec3_batch_2, vec3_batch_3;
      ae_int8x8 vec0_batch_4, vec1_batch_4; 
      ae_int8x8 vec2_batch_4, vec3_batch_4;

      ae_valignx2 align_p_vec0, align_p_vec1, align_p_vec2, align_p_vec3; 

      align_p_vec0 = AE_LA128_PP(p_vec_0);
      align_p_vec1 = AE_LA128_PP(p_vec_1);
      align_p_vec2 = AE_LA128_PP(p_vec_2);
      align_p_vec3 = AE_LA128_PP(p_vec_3);

      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      AE_LA8X8X2_IP(vec0_batch_0, vec0_batch_1, align_p_vec0, (ae_int8x16 *)p_vec_0);
      AE_LA8X8X2_IP(vec1_batch_0, vec1_batch_1, align_p_vec1, (ae_int8x16 *)p_vec_1);
      AE_LA8X8X2_IP(vec2_batch_0, vec2_batch_1, align_p_vec2, (ae_int8x16 *)p_vec_2);
      AE_LA8X8X2_IP(vec3_batch_0, vec3_batch_1, align_p_vec3, (ae_int8x16 *)p_vec_3);
      
      AE_LA8X8X2_IP(vec0_batch_2, vec0_batch_3, align_p_vec0, (ae_int8x16 *)p_vec_0);
      AE_LA8X8X2_IP(vec1_batch_2, vec1_batch_3, align_p_vec1, (ae_int8x16 *)p_vec_1);
      AE_LA8X8X2_IP(vec2_batch_2, vec2_batch_3, align_p_vec2, (ae_int8x16 *)p_vec_2);
      AE_LA8X8X2_IP(vec3_batch_2, vec3_batch_3, align_p_vec3, (ae_int8x16 *)p_vec_3);
        
      ae_int8x8 temp0, temp1, temp2, temp3;
      AE_LAV8X8X2_XP(vec0_batch_4, temp0, align_p_vec0, (ae_int8x16 *)p_vec_0, 8);
      AE_LAV8X8X2_XP(vec1_batch_4, temp1, align_p_vec1, (ae_int8x16 *)p_vec_1, 8);
      AE_LAV8X8X2_XP(vec2_batch_4, temp2, align_p_vec2, (ae_int8x16 *)p_vec_2, 8);
      AE_LAV8X8X2_XP(vec3_batch_4, temp3, align_p_vec3, (ae_int8x16 *)p_vec_3, 8);
        
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_0, vec2_batch_0, vec1_batch_0, vec0_batch_0, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_1, vec2_batch_1, vec1_batch_1, vec0_batch_1, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_2, vec2_batch_2, vec1_batch_2, vec0_batch_2, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_3, vec2_batch_3, vec1_batch_3, vec0_batch_3, mat_z_b);
      AE_MULA8Q8X8(acc_row1, acc_row0, vec3_batch_4, vec2_batch_4, vec1_batch_4, vec0_batch_4, mat_z_b);
      
      ae_int32x2 d_bias_0, d_bias_1;
      d_bias_1 = AE_MOVDA32X2(p_bias[vec_itr + 3], p_bias[vec_itr + 2]);
      d_bias_0 = AE_MOVDA32X2(p_bias[vec_itr + 1], p_bias[vec_itr + 0]);
      
      acc_row0 = AE_SUB32S(d_bias_0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias_1, acc_row1);
      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4 *)bias_buffer, 0);
      
      /* Shifts to match with Tensorflow */
      int p_left_shift[4], p_right_shift[4];
      
      p_left_shift[3] = p_out_shift[vec_itr + 0] < 0 ? 0 : p_out_shift[vec_itr + 0];
      p_left_shift[2] = p_out_shift[vec_itr + 1] < 0 ? 0 : p_out_shift[vec_itr + 1];
      p_left_shift[1] = p_out_shift[vec_itr + 2] < 0 ? 0 : p_out_shift[vec_itr + 2];
      p_left_shift[0] = p_out_shift[vec_itr + 3] < 0 ? 0 : p_out_shift[vec_itr + 3];

      p_right_shift[3] = p_out_shift[vec_itr + 0] > 0 ? 0 : -p_out_shift[vec_itr + 0];
      p_right_shift[2] = p_out_shift[vec_itr + 1] > 0 ? 0 : -p_out_shift[vec_itr + 1];
      p_right_shift[1] = p_out_shift[vec_itr + 2] > 0 ? 0 : -p_out_shift[vec_itr + 2];
      p_right_shift[0] = p_out_shift[vec_itr + 3] > 0 ? 0 : -p_out_shift[vec_itr + 3];

      p_left_shift[3] = (1 << p_left_shift[3]);
      p_left_shift[2] = (1 << p_left_shift[2]);
      p_left_shift[1] = (1 << p_left_shift[1]);
      p_left_shift[0] = (1 << p_left_shift[0]);

      p_right_shift[3] = (-1 << (31 - p_right_shift[3]));
      p_right_shift[2] = (-1 << (31 - p_right_shift[2]));
      p_right_shift[1] = (-1 << (31 - p_right_shift[1]));
      p_right_shift[0] = (-1 << (31 - p_right_shift[0]));

      ae_int32x2 p_out_mult10, p_out_mult32;
      p_out_mult32 = AE_MOVDA32X2(-p_out_multiplier[vec_itr + 3], -p_out_multiplier[vec_itr + 2]);
      p_out_mult10 = AE_MOVDA32X2(-p_out_multiplier[vec_itr + 1], -p_out_multiplier[vec_itr + 0]);

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0, acc_row0_vec1, acc_row0_vec2, acc_row0_vec3;
        ae_int32x2 acc_row1_vec0, acc_row1_vec1, acc_row1_vec2, acc_row1_vec3;
        
        /* Initialize accumulators with bias - (ker * inp_zero_bias) */
        AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)bias_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)bias_buffer, 0);

        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3, mat1_row0_4;
        ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3, mat1_row1_4;
        ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3, mat1_row2_4;
        ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3, mat1_row3_4;

        /* p_mat needs to be accessed in circular fashion */
        ae_int8x8* p_mat1_1 = p_mat1_0;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_1, row_stride1 * sizeof(WORD8));
        ae_int8x8* p_mat1_2 = p_mat1_1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_2, row_stride1 * sizeof(WORD8));
        ae_int8x8* p_mat1_3 = p_mat1_2;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_3, row_stride1 * sizeof(WORD8));

        ae_valign align_p_mat1_0, align_p_mat1_1, align_p_mat1_2, align_p_mat1_3;

        AE_LA8X8POS_PC(align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8POS_PC(align_p_mat1_1, (ae_int8x8 *)p_mat1_1);
        AE_LA8X8POS_PC(align_p_mat1_2, (ae_int8x8 *)p_mat1_2);
        AE_LA8X8POS_PC(align_p_mat1_3, (ae_int8x8 *)p_mat1_3);

        AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row1_0, align_p_mat1_1, (ae_int8x8 *)p_mat1_1);
        AE_LA8X8_IC(mat1_row2_0, align_p_mat1_2, (ae_int8x8 *)p_mat1_2);
        AE_LA8X8_IC(mat1_row3_0, align_p_mat1_3, (ae_int8x8 *)p_mat1_3);
        
        AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row1_1, align_p_mat1_1, (ae_int8x8 *)p_mat1_1);
        AE_LA8X8_IC(mat1_row2_1, align_p_mat1_2, (ae_int8x8 *)p_mat1_2);
        AE_LA8X8_IC(mat1_row3_1, align_p_mat1_3, (ae_int8x8 *)p_mat1_3);
        
        AE_LA8X8_IC(mat1_row0_2, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row1_2, align_p_mat1_1, (ae_int8x8 *)p_mat1_1);
        AE_LA8X8_IC(mat1_row2_2, align_p_mat1_2, (ae_int8x8 *)p_mat1_2);
        AE_LA8X8_IC(mat1_row3_2, align_p_mat1_3, (ae_int8x8 *)p_mat1_3);
        
        AE_LA8X8_IC(mat1_row0_3, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row1_3, align_p_mat1_1, (ae_int8x8 *)p_mat1_1);
        AE_LA8X8_IC(mat1_row2_3, align_p_mat1_2, (ae_int8x8 *)p_mat1_2);
        AE_LA8X8_IC(mat1_row3_3, align_p_mat1_3, (ae_int8x8 *)p_mat1_3);
        
        AE_LA8X8_IC(mat1_row0_4, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row1_4, align_p_mat1_1, (ae_int8x8 *)p_mat1_1);
        AE_LA8X8_IC(mat1_row2_4, align_p_mat1_2, (ae_int8x8 *)p_mat1_2);
        AE_LA8X8_IC(mat1_row3_4, align_p_mat1_3, (ae_int8x8 *)p_mat1_3);
        
        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row0_0);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row1_0);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row2_0);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row3_0);
        
        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row0_1);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row1_1);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row2_1);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row3_1);

        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row0_2);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row1_2);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row2_2);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row3_2);
        
        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row0_3);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row1_3);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row2_3);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row3_3);
        
        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_4 , vec2_batch_4 , vec1_batch_4 , vec0_batch_4 , mat1_row0_4);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , vec3_batch_4 , vec2_batch_4 , vec1_batch_4 , vec0_batch_4 , mat1_row1_4);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , vec3_batch_4 , vec2_batch_4 , vec1_batch_4 , vec0_batch_4 , mat1_row2_4);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , vec3_batch_4 , vec2_batch_4 , vec1_batch_4 , vec0_batch_4 , mat1_row3_4);
       
        /* Apply quantization */
        ae_int32x2 l_mult32, l_mult10, r_mult32, r_mult10;
        AE_L32X2X2_I(l_mult32, l_mult10, (ae_int32x4 *)p_left_shift, 0);
        AE_L32X2X2_I(r_mult32, r_mult10, (ae_int32x4 *)p_right_shift, 0);

        ae_int16x4 out_0, out_1, out_2, out_3;
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_mobnetv2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult32, p_out_mult10, l_mult32, l_mult10, r_mult32, r_mult10, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_mobnetv2(out_1, acc_row0_vec1, acc_row1_vec1, p_out_mult32, p_out_mult10, l_mult32, l_mult10, r_mult32, r_mult10, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_mobnetv2(out_2, acc_row0_vec2, acc_row1_vec2, p_out_mult32, p_out_mult10, l_mult32, l_mult10, r_mult32, r_mult10, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_mobnetv2(out_3, acc_row0_vec3, acc_row1_vec3, p_out_mult32, p_out_mult10, l_mult32, l_mult10, r_mult32, r_mult10, out_zero_bias);
       
        /* Store output */
        ae_int8x8 out32_0, out32_1; 
        out32_0 = AE_SAT8X8X16(out_0, out_1);
        out32_1 = AE_SAT8X8X16(out_2, out_3);
        
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_stride);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_stride);
      }

      //remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0, acc_row1_vec0;
        AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)bias_buffer, 0);
        
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        ae_int8x8 mat1_row0_0, mat1_row0_1;
        ae_int8x8 mat1_row0_2, mat1_row0_3;
        ae_int8x8 mat1_row0_4;
        ae_valign align_p_mat1_0;

        AE_LA8X8POS_PC(align_p_mat1_0, (ae_int8x8 *)p_mat1_0);

        AE_LA8X8_IC(mat1_row0_0, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row0_1, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row0_2, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row0_3, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);
        AE_LA8X8_IC(mat1_row0_4, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);

        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_0 , vec2_batch_0 , vec1_batch_0 , vec0_batch_0 , mat1_row0_0);
        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_1 , vec2_batch_1 , vec1_batch_1 , vec0_batch_1 , mat1_row0_1);
        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_2 , vec2_batch_2 , vec1_batch_2 , vec0_batch_2 , mat1_row0_2);
        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_3 , vec2_batch_3 , vec1_batch_3 , vec0_batch_3 , mat1_row0_3);
        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , vec3_batch_4 , vec2_batch_4 , vec1_batch_4 , vec0_batch_4 , mat1_row0_4);

        /* Apply quantization */
        ae_int32x2 l_mult32, l_mult10, r_mult32, r_mult10;
        AE_L32X2X2_I(l_mult32, l_mult10, (ae_int32x4 *)p_left_shift, 0);
        AE_L32X2X2_I(r_mult32, r_mult10, (ae_int32x4 *)p_right_shift, 0);
        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_mobnetv2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult32, p_out_mult10, l_mult32, l_mult10, r_mult32, r_mult10, out_zero_bias); 
        
        /* Store output */
        ae_int8x8 out32_0; 
        out32_0 = AE_SAT8X8X16(out_0, out_0);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_stride);
      }
    }
  }
  else if(((((unsigned)p_mat1) & 7) == 0) && ((((unsigned)p_vec1) & 7) == 0) && ((((unsigned)p_bias) & 3) == 0) &&
     ((row_stride1 & 15) == 0) && (vec_stride & 15) == 0) 
  {
    ae_int32x2 acc_buffer[4];
    m_itr = 0, vec_itr = 0;

    int rem_cols = cols1 & 15;
    int rem_cols_shift0 = ((rem_cols) <= 8)?(8 - (rem_cols)) * 8:0;
    int rem_cols_shift1 = ((rem_cols) > 8)?(16 - (rem_cols)) * 8:64;
    
    int out_stride = out_row_offset;
    int out_offset = out_col_offset;

    // Process loop for 4 rows and 4 vectors 
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      ae_int8x8 *p_vec_0 = (ae_int8x8 *) &p_vec1[(vec_itr) * vec_stride];
      ae_int8x8 *p_vec_1 = (ae_int8x8*)((WORD8 *)p_vec_0 + vec_stride); 
      ae_int8x8 *p_vec_2 = (ae_int8x8*)((WORD8 *)p_vec_1 + vec_stride); 
      ae_int8x8 *p_vec_3 = (ae_int8x8*)((WORD8 *)p_vec_2 + vec_stride); 

      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      int cols_count=cols1-(cols1&15);
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
      {
        AE_L8X8X2_IP(vec0_0, vec0_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec1_0, vec1_1, (ae_int8x16*)p_vec_1, 16);
        AE_L8X8X2_IP(vec2_0, vec2_1, (ae_int8x16*)p_vec_2, 16);
        AE_L8X8X2_IP(vec3_0, vec3_1, (ae_int8x16*)p_vec_3, 16);

        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec1_1 , vec2_1 , vec3_1 , mat_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_L8X8X2_IP(vec0_0, vec0_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec1_0, vec1_1, (ae_int8x16*)p_vec_1, 16);
        AE_L8X8X2_IP(vec2_0, vec2_1, (ae_int8x16*)p_vec_2, 16);
        AE_L8X8X2_IP(vec3_0, vec3_1, (ae_int8x16*)p_vec_3, 16);

        vec0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_0), rem_cols_shift0), rem_cols_shift0));
        vec1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_0), rem_cols_shift0), rem_cols_shift0));
        vec2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_0), rem_cols_shift0), rem_cols_shift0));
        vec3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_0), rem_cols_shift0), rem_cols_shift0));

        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);
          
        if(rem_cols > 8)
        {
          vec0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_1), rem_cols_shift1), rem_cols_shift1));
          vec1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_1), rem_cols_shift1), rem_cols_shift1));
          vec2_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_1), rem_cols_shift1), rem_cols_shift1));
          vec3_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_1), rem_cols_shift1), rem_cols_shift1));
          
          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec1_1 , vec2_1 , vec3_1 , mat_z_b);
        }
      }

      acc_row0 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + 0], p_bias[vec_itr + 1]), acc_row0);
      acc_row1 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + 2], p_bias[vec_itr + 3]), acc_row1);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);
     
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;

      m_itr = 0;

      /* Shifts to match with Tensorflow */
      xtbool2 b0, b1;
      int p_right_shift[4];
      b0 = AE_LT32(AE_MOVDA32X2(p_out_shift[vec_itr], p_out_shift[vec_itr + 1]), ZERO32);
      b1 = AE_LT32(AE_MOVDA32X2(p_out_shift[vec_itr + 2], p_out_shift[vec_itr + 3]), ZERO32);

      ae_int32x2 temp_0 = ZERO32, temp_1 = ZERO32;
      ae_int32x2 temp_2 = ZERO32, temp_3 = ZERO32;
      AE_MOVT32X2(temp_0, AE_MOVDA32X2(-p_out_shift[vec_itr], -p_out_shift[vec_itr + 1]), b0);
      AE_MOVT32X2(temp_1, AE_MOVDA32X2(-p_out_shift[vec_itr + 2], -p_out_shift[vec_itr + 3]), b1);
      AE_MOVF32X2(temp_2, AE_MOVDA32X2(p_out_shift[vec_itr], p_out_shift[vec_itr + 1]), b0);
      AE_MOVF32X2(temp_3, AE_MOVDA32X2(p_out_shift[vec_itr + 2], p_out_shift[vec_itr + 3]), b1);

      ae_int32x2 l_mult0 = AE_MOVDA32(1 << AE_MOVAD32_H(temp_2));
      ae_int32x2 l_mult1 = AE_MOVDA32(1 << AE_MOVAD32_L(temp_2));
      ae_int32x2 l_mult2 = AE_MOVDA32(1 << AE_MOVAD32_H(temp_3));
      ae_int32x2 l_mult3 = AE_MOVDA32(1 << AE_MOVAD32_L(temp_3));

      p_right_shift[0] = AE_MOVAD32_H(temp_0);
      p_right_shift[1] = AE_MOVAD32_L(temp_0);
      p_right_shift[2] = AE_MOVAD32_H(temp_1);
      p_right_shift[3] = AE_MOVAD32_L(temp_1);

      ae_int32x2 l_mult01 = AE_MOVDA32X2((1 << AE_MOVAD32_H(temp_2)),(1 << AE_MOVAD32_L(temp_2)));
      ae_int32x2 l_mult23 = AE_MOVDA32X2((1 << AE_MOVAD32_H(temp_3)),(1 << AE_MOVAD32_L(temp_3)));

      ae_int32x2 r_mult01 = AE_MOVDA32X2( (-1 << (31 - p_right_shift[0])), (-1 << (31 - p_right_shift[1])));
      ae_int32x2 r_mult23 = AE_MOVDA32X2( (-1 << (31 - p_right_shift[2])), (-1 << (31 - p_right_shift[3]))); 

      ae_int32x2 p_out_mult01 = AE_MOVDA32X2(-p_out_multiplier[vec_itr + 0], -p_out_multiplier[vec_itr + 1]);
      ae_int32x2 p_out_mult23 = AE_MOVDA32X2(-p_out_multiplier[vec_itr + 2], -p_out_multiplier[vec_itr + 3]);

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;
        ae_int32x2 acc_row0_vec1;
        ae_int32x2 acc_row1_vec1;
        ae_int32x2 acc_row0_vec2;
        ae_int32x2 acc_row1_vec2;
        ae_int32x2 acc_row0_vec3;
        ae_int32x2 acc_row1_vec3;
        
        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);
        AE_L32X2X2_I(acc_row1_vec0, acc_row1_vec1, (ae_int32x4*)acc_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec2, acc_row0_vec3, (ae_int32x4*)acc_buffer, 16);
        AE_L32X2X2_I(acc_row1_vec2, acc_row1_vec3, (ae_int32x4*)acc_buffer, 16);
        
        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_4_rows_4_vecs_aligned
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
           ,cols1
           ,row_stride1
           ,vec_stride
          );

        ae_int16x4 out_0, out_1, out_2, out_3;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[vec_itr + 0], l_mult0, AE_MOVAD32_H(temp_0), out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[vec_itr + 1], l_mult1, AE_MOVAD32_L(temp_0), out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[vec_itr + 2], l_mult2, AE_MOVAD32_H(temp_1), out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[vec_itr + 3], l_mult3, AE_MOVAD32_L(temp_1), out_zero_bias);
        
        /* Store output */
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row1;

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_1_rows_4_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_vec_0
           ,(ae_int8*)p_mat1_0
           ,cols1
           ,vec_stride
          );

        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult23, p_out_mult01, l_mult23, l_mult01, r_mult23, r_mult01, out_zero_bias); 

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_stride);
      }
    }

    // remaining vectors 
    for(; vec_itr < vec_count; vec_itr++)
    {
      ae_int8x8 *p_vec_0 = (ae_int8x8 *) &p_vec1[(vec_itr) * vec_stride];
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      int cols_count=cols1-(cols1&7);
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)
      {
        AE_L8X8_IP(vec0_0, p_vec_0, 8);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec0_0 , vec0_0 , vec0_0 , mat_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_L8X8_IP(vec0_0, p_vec_0, 8);
        vec0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_0), rem_cols_shift), rem_cols_shift));
        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec0_0 , vec0_0 , vec0_0 , mat_z_b);
      }

      WORD8* p_dst = (WORD8*)p_out + (vec_itr + 0) * out_offset;

      m_itr = 0;

      /* Shifts to match with Tensorflow */
      xtbool2 b0;
      b0 = AE_LT32(AE_MOVDA32(p_out_shift[vec_itr]), ZERO32);

      ae_int32x2 temp_0 = ZERO32, temp_1 = ZERO32;
      AE_MOVT32X2(temp_0, AE_MOVDA32(-p_out_shift[vec_itr]), b0);
      AE_MOVF32X2(temp_1, AE_MOVDA32(p_out_shift[vec_itr]), b0);
      
      ae_int32x2 l_mult0 = AE_MOVDA32(1 << AE_MOVAD32_H(temp_1));

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0 = AE_SUB32S(AE_MOVDA32(p_bias[vec_itr]), AE_MOVDA32(AE_MOVAD32_H(acc_row0)));
        ae_int32x2 acc_row1_vec0 = AE_SUB32S(AE_MOVDA32(p_bias[vec_itr]), AE_MOVDA32(AE_MOVAD32_H(acc_row1)));

        ae_int8x8 * p_vec_0  = (ae_int8x8 *)(p_vec1 + vec_itr * vec_stride);
        WORD8 *p_mat1_0 = (WORD8 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec_0
           ,cols1
           ,row_stride1
          );

        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[vec_itr + 0], l_mult0, temp_0, out_zero_bias);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = AE_SUB32S(AE_MOVDA32(p_bias[vec_itr]), acc_row0);
        ae_int32x2 acc_row1_vec0 = AE_SUB32S(AE_MOVDA32(p_bias[vec_itr]), acc_row1);

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_1_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
          );

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[vec_itr], AE_MOVAD32_H(temp_1), temp_0);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);

        ae_int8x8 temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);

        //TODO: AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
      }
    }
  }
  else if(p_mat1 && p_vec1)
  {
    ae_int32x2 acc_buffer[4];
    m_itr = 0, vec_itr = 0;

    int out_stride = out_row_offset;
    int out_offset = out_col_offset;

    int rem_cols = cols1 & 15;
    
    for(vec_itr = 0; vec_itr < (vec_count & ~(32 - 1)); vec_itr += 32)
    {
      int ii;

      for(ii = 0; ii < 8; ii++)
      {
        ae_int8x8 *p_vec_0 = (ae_int8x8 *) &p_vec1[(vec_itr + ii) * vec_stride];
        ae_int8x8 *p_vec_1 = p_vec_0 + vec_stride; 
        ae_int8x8 *p_vec_2 = p_vec_1 + vec_stride; 
        ae_int8x8 *p_vec_3 = p_vec_2 + vec_stride; 

        ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);
        ae_valignx2 align_p_vec_1 = AE_LA128_PP(p_vec_1);
        ae_valignx2 align_p_vec_2 = AE_LA128_PP(p_vec_2);
        ae_valignx2 align_p_vec_3 = AE_LA128_PP(p_vec_3);

        ae_int32x2 acc_row0 = ZERO32; 
        ae_int32x2 acc_row1 = ZERO32;

        int cols_count=cols1-(cols1&15);
#pragma no_unroll
        for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
        {
          AE_LA8X8X2_IP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0);
          AE_LA8X8X2_IP(vec1_0, vec1_1, align_p_vec_1, (ae_int8x16*)p_vec_1);
          AE_LA8X8X2_IP(vec2_0, vec2_1, align_p_vec_2, (ae_int8x16*)p_vec_2);
          AE_LA8X8X2_IP(vec3_0, vec3_1, align_p_vec_3, (ae_int8x16*)p_vec_3);

          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);
          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec1_1 , vec2_1 , vec3_1 , mat_z_b);
        }

        //Remainder loop for cols1
        if(cols_count!=cols1)
        {
          AE_LAV8X8X2_XP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0, rem_cols);
          AE_LAV8X8X2_XP(vec1_0, vec1_1, align_p_vec_1, (ae_int8x16*)p_vec_1, rem_cols);
          AE_LAV8X8X2_XP(vec2_0, vec2_1, align_p_vec_2, (ae_int8x16*)p_vec_2, rem_cols);
          AE_LAV8X8X2_XP(vec3_0, vec3_1, align_p_vec_3, (ae_int8x16*)p_vec_3, rem_cols);

          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);

          if(rem_cols > 8)
          {
            AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec1_1 , vec2_1 , vec3_1 , mat_z_b);
          }
        }

        acc_row0 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + ii +  0], p_bias[vec_itr + ii +  8]), acc_row0);
        acc_row1 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + ii + 16], p_bias[vec_itr + ii + 24]), acc_row1);
        AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
        AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);
        
        WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + ii +  0) * out_offset;
        WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + ii +  8) * out_offset;
        WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + ii + 16) * out_offset;
        WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + ii + 24) * out_offset;

        m_itr = 0;

        /* Shifts to match with Tensorflow */
        int p_left_shift[4], p_right_shift[4];
        
        p_left_shift[0] = p_out_shift[vec_itr + ii +  0] < 0 ? 0 : p_out_shift[vec_itr + ii +  0];
        p_left_shift[1] = p_out_shift[vec_itr + ii +  8] < 0 ? 0 : p_out_shift[vec_itr + ii +  8];
        p_left_shift[2] = p_out_shift[vec_itr + ii + 16] < 0 ? 0 : p_out_shift[vec_itr + ii + 16];
        p_left_shift[3] = p_out_shift[vec_itr + ii + 24] < 0 ? 0 : p_out_shift[vec_itr + ii + 24];

        p_right_shift[0] = p_out_shift[vec_itr + ii +  0] > 0 ? 0 : -p_out_shift[vec_itr + ii +  0];
        p_right_shift[1] = p_out_shift[vec_itr + ii +  8] > 0 ? 0 : -p_out_shift[vec_itr + ii +  8];
        p_right_shift[2] = p_out_shift[vec_itr + ii + 16] > 0 ? 0 : -p_out_shift[vec_itr + ii + 16];
        p_right_shift[3] = p_out_shift[vec_itr + ii + 24] > 0 ? 0 : -p_out_shift[vec_itr + ii + 24];

        ae_int32x2 l_mult0 = AE_MOVDA32(1 << p_left_shift[0]);
        ae_int32x2 l_mult1 = AE_MOVDA32(1 << p_left_shift[1]);
        ae_int32x2 l_mult2 = AE_MOVDA32(1 << p_left_shift[2]);
        ae_int32x2 l_mult3 = AE_MOVDA32(1 << p_left_shift[3]);

        ae_int32x2 l_mult23 = AE_MOVDA32X2( 1 << p_left_shift[2], 1 << p_left_shift[3]); 
        ae_int32x2 l_mult01 = AE_MOVDA32X2( 1 << p_left_shift[0], 1 << p_left_shift[1]);

        ae_int32x2 r_mult23 = AE_MOVDA32X2( (-1 << (31 - p_right_shift[2])), (-1 << (31 - p_right_shift[3]))); 
        ae_int32x2 r_mult01 = AE_MOVDA32X2( (-1 << (31 - p_right_shift[0])), (-1 << (31 - p_right_shift[1])));

        ae_int32x2 p_out_mult01 = AE_MOVDA32X2(-p_out_multiplier[vec_itr + ii + 0], -p_out_multiplier[vec_itr + ii + 8]);
        ae_int32x2 p_out_mult23 = AE_MOVDA32X2(-p_out_multiplier[vec_itr + ii + 16], -p_out_multiplier[vec_itr + ii + 24]);

        for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
        {
          ae_int32x2 acc_row0_vec0;
          ae_int32x2 acc_row1_vec0;
          ae_int32x2 acc_row0_vec1;
          ae_int32x2 acc_row1_vec1;
          ae_int32x2 acc_row0_vec2;
          ae_int32x2 acc_row1_vec2;
          ae_int32x2 acc_row0_vec3;
          ae_int32x2 acc_row1_vec3;
          
          /* Initialize accumulators */
          AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);
          AE_L32X2X2_I(acc_row1_vec0, acc_row1_vec1, (ae_int32x4*)acc_buffer, 0);
          AE_L32X2X2_I(acc_row0_vec2, acc_row0_vec3, (ae_int32x4*)acc_buffer, 16);
          AE_L32X2X2_I(acc_row1_vec2, acc_row1_vec3, (ae_int32x4*)acc_buffer, 16);
          
          ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + (vec_itr + ii) * vec_stride);
          ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

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
             ,cols1 
             ,row_stride1
             ,vec_stride
            );

            ae_int16x4 out_0, out_1, out_2, out_3;
            MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[vec_itr + ii + 0], l_mult0, p_right_shift[0], out_zero_bias);
            MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[vec_itr + ii + 8], l_mult1, p_right_shift[1], out_zero_bias);
            MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[vec_itr + ii + 16], l_mult2, p_right_shift[2], out_zero_bias);
            MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[vec_itr + ii + 24], l_mult3, p_right_shift[3], out_zero_bias);
          
            /* Store output */
            AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
            AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
            AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
            AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

            AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
            AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
            AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
            AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

            AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
            AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
            AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
            AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

            AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
            AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
            AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
            AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);
        }

        // Remaining rows 
        // TODO: codesize considerations: remove inline for 4 row, 1 vec case ?
        for (; m_itr < rows; m_itr++)
        {
          ae_int32x2 acc_row0_vec0 = acc_row0;
          ae_int32x2 acc_row1_vec0 = acc_row1;

          ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + (vec_itr + ii) * vec_stride);
          ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
          AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

          _xa_nn_dot_product_1_rows_4_vecs_offset_aligned
            (&acc_row0_vec0
             ,&acc_row1_vec0
             ,(ae_int8x8*)p_vec_0
             ,(ae_int8*)p_mat1_0
             ,cols1
             ,vec_stride
            );

          ae_int16x4 out_0;
          MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult23, p_out_mult01, l_mult23, l_mult01, r_mult23, r_mult01, out_zero_bias); 

          AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
          AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_stride);
          AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_stride);
          AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_stride);
        }
      }
    }
    // Process loop for 4 rows and 4 vectors 
    for(; vec_itr < (vec_count & ~(4-1)); vec_itr += 4)
    {
      ae_int8x8 *p_vec_0 = (ae_int8x8 *) &p_vec1[(vec_itr) * vec_stride];
      ae_int8x8 *p_vec_1 = (ae_int8x8*)((WORD8 *)p_vec_0 + vec_stride); 
      ae_int8x8 *p_vec_2 = (ae_int8x8*)((WORD8 *)p_vec_1 + vec_stride); 
      ae_int8x8 *p_vec_3 = (ae_int8x8*)((WORD8 *)p_vec_2 + vec_stride); 

      ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);
      ae_valignx2 align_p_vec_1 = AE_LA128_PP(p_vec_1);
      ae_valignx2 align_p_vec_2 = AE_LA128_PP(p_vec_2);
      ae_valignx2 align_p_vec_3 = AE_LA128_PP(p_vec_3);

      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

        int cols_count=cols1-(cols1&15);
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
      {
          AE_LA8X8X2_IP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0);
          AE_LA8X8X2_IP(vec1_0, vec1_1, align_p_vec_1, (ae_int8x16*)p_vec_1);
          AE_LA8X8X2_IP(vec2_0, vec2_1, align_p_vec_2, (ae_int8x16*)p_vec_2);
          AE_LA8X8X2_IP(vec3_0, vec3_1, align_p_vec_3, (ae_int8x16*)p_vec_3);

          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);
          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec1_1 , vec2_1 , vec3_1 , mat_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_LAV8X8X2_XP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0, rem_cols);
        AE_LAV8X8X2_XP(vec1_0, vec1_1, align_p_vec_1, (ae_int8x16*)p_vec_1, rem_cols);
        AE_LAV8X8X2_XP(vec2_0, vec2_1, align_p_vec_2, (ae_int8x16*)p_vec_2, rem_cols);
        AE_LAV8X8X2_XP(vec3_0, vec3_1, align_p_vec_3, (ae_int8x16*)p_vec_3, rem_cols);

        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec1_0 , vec2_0 , vec3_0 , mat_z_b);

        if(rem_cols > 8)
        {
          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec1_1 , vec2_1 , vec3_1 , mat_z_b);
        }
      }
      
      acc_row0 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + 0], p_bias[vec_itr + 1]), acc_row0);
      acc_row1 = AE_SUB32S(AE_MOVDA32X2(p_bias[vec_itr + 2], p_bias[vec_itr + 3]), acc_row1);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);
        
      WORD8* p_dst_0 = (WORD8*)p_out + (vec_itr + 0) * out_offset;
      WORD8* p_dst_1 = (WORD8*)p_out + (vec_itr + 1) * out_offset;
      WORD8* p_dst_2 = (WORD8*)p_out + (vec_itr + 2) * out_offset;
      WORD8* p_dst_3 = (WORD8*)p_out + (vec_itr + 3) * out_offset;

      m_itr = 0;

      /* Shifts to match with Tensorflow */
      int p_left_shift[4], p_right_shift[4];
      
      p_left_shift[0] = p_out_shift[vec_itr + 0] < 0 ? 0 : p_out_shift[vec_itr + 0];
      p_left_shift[1] = p_out_shift[vec_itr + 1] < 0 ? 0 : p_out_shift[vec_itr + 1];
      p_left_shift[2] = p_out_shift[vec_itr + 2] < 0 ? 0 : p_out_shift[vec_itr + 2];
      p_left_shift[3] = p_out_shift[vec_itr + 3] < 0 ? 0 : p_out_shift[vec_itr + 3];

      p_right_shift[0] = p_out_shift[vec_itr + 0] > 0 ? 0 : -p_out_shift[vec_itr + 0];
      p_right_shift[1] = p_out_shift[vec_itr + 1] > 0 ? 0 : -p_out_shift[vec_itr + 1];
      p_right_shift[2] = p_out_shift[vec_itr + 2] > 0 ? 0 : -p_out_shift[vec_itr + 2];
      p_right_shift[3] = p_out_shift[vec_itr + 3] > 0 ? 0 : -p_out_shift[vec_itr + 3];

      ae_int32x2 l_mult0 = AE_MOVDA32(1 << p_left_shift[0]);
      ae_int32x2 l_mult1 = AE_MOVDA32(1 << p_left_shift[1]);
      ae_int32x2 l_mult2 = AE_MOVDA32(1 << p_left_shift[2]);
      ae_int32x2 l_mult3 = AE_MOVDA32(1 << p_left_shift[3]);

      ae_int32x2 l_mult23 = AE_MOVDA32X2( 1 << p_left_shift[2], 1 << p_left_shift[3]); 
      ae_int32x2 l_mult01 = AE_MOVDA32X2( 1 << p_left_shift[0], 1 << p_left_shift[1]);

      ae_int32x2 r_mult23 = AE_MOVDA32X2( (-1 << (31 - p_right_shift[2])), (-1 << (31 - p_right_shift[3]))); 
      ae_int32x2 r_mult01 = AE_MOVDA32X2( (-1 << (31 - p_right_shift[0])), (-1 << (31 - p_right_shift[1])));

      ae_int32x2 p_out_mult01 = AE_MOVDA32X2(-p_out_multiplier[vec_itr + 0], -p_out_multiplier[vec_itr + 1]);
      ae_int32x2 p_out_mult23 = AE_MOVDA32X2(-p_out_multiplier[vec_itr + 2], -p_out_multiplier[vec_itr + 3]);

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row1_vec0;
        ae_int32x2 acc_row0_vec1;
        ae_int32x2 acc_row1_vec1;
        ae_int32x2 acc_row0_vec2;
        ae_int32x2 acc_row1_vec2;
        ae_int32x2 acc_row0_vec3;
        ae_int32x2 acc_row1_vec3;
        
        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);
        AE_L32X2X2_I(acc_row1_vec0, acc_row1_vec1, (ae_int32x4*)acc_buffer, 0);
        AE_L32X2X2_I(acc_row0_vec2, acc_row0_vec3, (ae_int32x4*)acc_buffer, 16);
        AE_L32X2X2_I(acc_row1_vec2, acc_row1_vec3, (ae_int32x4*)acc_buffer, 16);
        
        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *)p_mat1; 
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

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
           ,cols1
           ,row_stride1
           ,vec_stride
          );

        ae_int16x4 out_0, out_1, out_2, out_3;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[vec_itr + 0], l_mult0, p_right_shift[0], out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[vec_itr + 1], l_mult1, p_right_shift[1], out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[vec_itr + 2], l_mult2, p_right_shift[2], out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[vec_itr + 3], l_mult3, p_right_shift[3], out_zero_bias);
        
        /* Store output */
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);

        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row1;

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8* p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_1_rows_4_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_vec_0
           ,(ae_int8*)p_mat1_0
           ,cols1
           ,vec_stride
          );

        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_mult23, p_out_mult01, l_mult23, l_mult01, r_mult23, r_mult01, out_zero_bias); 

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_stride);
      }
    }

    // remaining vectors 
    for(; vec_itr < vec_count; vec_itr++)
    {
      ae_int8x8 *p_vec_0 = (ae_int8x8 *) &p_vec1[(vec_itr) * vec_stride];
      ae_valignx2 align_p_vec_0 = AE_LA128_PP(p_vec_0);
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      int cols_count=cols1-(cols1&15);
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
      {
        AE_LA8X8X2_IP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec0_0 , vec0_0 , vec0_0 , mat_z_b);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec0_1 , vec0_1 , vec0_1 , mat_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_LAV8X8X2_XP(vec0_0, vec0_1, align_p_vec_0, (ae_int8x16*)p_vec_0, rem_cols);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_0 , vec0_0 , vec0_0 , vec0_0 , mat_z_b);
        
        if(rem_cols > 8)
        {
          AE_MULA8Q8X8(acc_row0 , acc_row1 , vec0_1 , vec0_1 , vec0_1 , vec0_1 , mat_z_b);
        }
      }
      acc_row0 = AE_SUB32S(AE_MOVDA32(p_bias[vec_itr]), acc_row0);

      WORD8* p_dst = (WORD8*)p_out + (vec_itr + 0) * out_offset;

      m_itr = 0;

      /* Shifts to match with Tensorflow */
      xtbool2 b0;
      b0 = AE_LT32(AE_MOVDA32(p_out_shift[vec_itr]), ZERO32);

      ae_int32x2 temp_0 = ZERO32, temp_1 = ZERO32;
      AE_MOVT32X2(temp_0, AE_MOVDA32(-p_out_shift[vec_itr]), b0);
      AE_MOVF32X2(temp_1, AE_MOVDA32(p_out_shift[vec_itr]), b0);
      
      ae_int32x2 l_mult0 = AE_MOVDA32(1 << AE_MOVAD32_H(temp_1));

      for (m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
      {
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row0;

        ae_int8x8 * p_vec_0  = (ae_int8x8 *)(p_vec1 + vec_itr * vec_stride);
        WORD8 *p_mat1_0 = (WORD8 *)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,(ae_int8x8*)p_mat1_0
           ,(ae_int8*)p_vec_0
           ,cols1
           ,row_stride1
          );

        ae_int16x4 out_0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[vec_itr + 0], l_mult0, temp_0, out_zero_bias);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_stride);
      }

      // Remaining rows
      for (; m_itr < rows; m_itr++)
      {
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row0;

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_stride);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*)p_mat1;
        AE_ADDCIRC16X4_XC((ae_int16x4*)p_mat1_0, m_itr * row_stride1 * sizeof(WORD8));

        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
          );

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[vec_itr], AE_MOVAD32_H(temp_1), temp_0);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        AE_MINMAX32(acc_row0_vec0, min_int8, max_int8);

        ae_int8x8 temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);

        //TODO: AE_SW_S8_3_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_stride);
      }
    }
  }
  else
    return -1;
  return 0;
}
