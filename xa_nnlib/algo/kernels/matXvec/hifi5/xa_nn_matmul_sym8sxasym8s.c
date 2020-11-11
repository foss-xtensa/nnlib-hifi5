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
#include "xa_nnlib_common_macros_hifi5.h"

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2(inp, multiplier, left_shift, right_shift) \
  inp = AE_SLAA32(inp, left_shift); \
  inp = AE_MULFP32X2RAS(inp, AE_MOVDA32(multiplier)); \
  inp = AE_SRAA32SYMS(inp, right_shift);

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out, inp1, inp2, multiplier, l_shift, r_shift, out_off) \
  AE_MUL2P32X4S(inp1, inp2, inp1, inp2, l_shift, l_shift); \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
  inp1 = AE_SRAA32SYMS(inp1, r_shift); \
  inp2 = AE_SRAA32SYMS(inp2, r_shift); \
  out = AE_SAT16X4(inp1, inp2); \
  out = AE_ADD16S(AE_MOVDA16(out_off), out); \
  AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127)); 

#define MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out, inp1, inp2, multiplier_23, multiplier_01, l_shift_23, l_shift_01, r_shift_23, r_shift_01, out_off) \
{\
  AE_MUL2P32X4S(inp1, inp2, inp1, inp2, l_shift_01, l_shift_23); \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, multiplier_01, multiplier_23); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, r_shift_01, r_shift_23); \
  out = AE_SAT16X4(inp1, inp2); \
  out = AE_ADD16S(AE_MOVDA16(out_off), out); \
  AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127)); \
}

#define MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_SP_32(out, inp1, inp2, multiplier_32, multiplier_10, l_shift_32, l_shift_10, r_shift_32, r_shift_10, out_off) \
{\
  AE_MUL2P32X4S(inp1, inp2, inp1, inp2, l_shift_10, l_shift_32); \
  AE_MULF2P32X4RAS(inp1, inp2, inp1, inp2, multiplier_10, multiplier_32); \
  AE_MULF2P32X4RS(inp1, inp2, inp1, inp2, r_shift_10, r_shift_32); \
  out = AE_SAT16X4(inp2, inp1); \
  out = AE_ADD16S(AE_MOVDA16(out_off), out); \
}

#define PACK_32X2(dst1, src1, src2) \
  dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x080a0c0e, 0x00020406)));

#define PACK_32X2_per_chan(dst_0, dst_1, src_0, src_1, src_2, src_3) \
{\
  ae_int8x8 dst_temp1, dst_temp2; \
  dst_temp1 = AE_SEL8X8I(AE_MOVINT8X8_FROMINT16X4(src_0), AE_MOVINT8X8_FROMINT16X4(src_1), 25); \
  dst_temp2 = AE_SEL8X8I(AE_MOVINT8X8_FROMINT16X4(src_2), AE_MOVINT8X8_FROMINT16X4(src_3), 25); \
  AE_DSEL8X8(dst_0, dst_1, dst_temp1, dst_temp2, AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x3175b9fd, 0x2064a8ec))); \
}

extern const long long g_sel_pattern[16];
extern const long long pre_loop_sel_pattern[16];
extern const long long post_loop_sel_pattern[16];

static void special_function_for_cols_mul_32
    (WORD8*        p_out_0
    ,const WORD8*  p_mat1
    ,const WORD8*  p_vec1_0
    ,const WORD32* p_bias_0
    ,WORD32        n_rows
    ,WORD32        n_vecs
    ,WORD32        cols
    ,const WORD32* p_out_mul
    ,const WORD32* p_out_shift
    ,WORD32        vec1_z_b
    ,WORD32        out_z_b
    ,WORD32        out_stride
    ,WORD32        row_offset
    ,WORD32        vec_offset
    ,WORD32        out_offset
    )
{
  WORD8 * __restrict__ p_dst_0;
  ae_int8x16 * __restrict__ p_vec_0;
  int c_itr;
  int m_itr = 0, vec_itr = 0;
  ae_int32x2 acc_buffer[4];
  int p_left_mult[4], p_right_mult[4], p_out_mult[4];
  ae_int8x8 vec_z_b = AE_MOVDA8(-vec1_z_b);

  ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
  ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
  ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
  ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;

  ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
  ae_int8x8 vec1_batch_0, vec1_batch_1, vec1_batch_2, vec1_batch_3; 
  ae_int8x8 vec2_batch_0, vec2_batch_1, vec2_batch_2, vec2_batch_3; 
  ae_int8x8 vec3_batch_0, vec3_batch_1, vec3_batch_2, vec3_batch_3;

  for(m_itr = 0; m_itr < (n_rows & ~(4 - 1)); m_itr += 4)
  {
    ae_int8x8* p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_offset];
    ae_int8x8* p_mat1_1 = (ae_int8x8*)((ae_int8*)p_mat1_0 + row_offset); 
    ae_int8x8* p_mat1_2 = (ae_int8x8*)((ae_int8*)p_mat1_1 + row_offset);
    ae_int8x8* p_mat1_3 = (ae_int8x8*)((ae_int8*)p_mat1_2 + row_offset);

    ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
    ae_valignx2 align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
    ae_valignx2 align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
    ae_valignx2 align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

    ae_int32x2 acc_row0 = ZERO32; 
    ae_int32x2 acc_row1 = ZERO32;

#pragma loop_count min=1
#pragma no_unroll
    for(c_itr = 0; c_itr < cols>>5; c_itr++)
    {
      /* Load 4 rows */
      AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
      AE_LA8X8X2_IP(mat1_row1_2, mat1_row1_3, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
      AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
      AE_LA8X8X2_IP(mat1_row2_2, mat1_row2_3, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
      AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16*)p_mat1_3);
      AE_LA8X8X2_IP(mat1_row3_2, mat1_row3_3, align_p_mat1_3, (ae_int8x16*)p_mat1_3);

      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 , vec_z_b);
    }
      
    {
      ae_int32x2 d_bias0, d_bias1;
      ae_valignx2 align_p_bias = AE_LA128_PP((ae_int32x4 *)p_bias_0);
      AE_LA32X2X2_IP(d_bias0, d_bias1, align_p_bias, (ae_int32x4 *)p_bias_0);

      acc_row0 = AE_SUB32S(d_bias0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias1, acc_row1);

      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4*)acc_buffer, 0);
    }

    p_dst_0 = (WORD8*)p_out_0 + (m_itr + 0) * out_stride;

    p_left_mult[0] = p_out_shift[m_itr + 0] < 0 ? 1 : (1 << p_out_shift[m_itr + 0]);
    p_left_mult[1] = p_out_shift[m_itr + 1] < 0 ? 1 : (1 << p_out_shift[m_itr + 1]);
    p_left_mult[2] = p_out_shift[m_itr + 2] < 0 ? 1 : (1 << p_out_shift[m_itr + 2]);
    p_left_mult[3] = p_out_shift[m_itr + 3] < 0 ? 1 : (1 << p_out_shift[m_itr + 3]);

    p_right_mult[0] = p_out_shift[m_itr + 0] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 0]));
    p_right_mult[1] = p_out_shift[m_itr + 1] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 1]));
    p_right_mult[2] = p_out_shift[m_itr + 2] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 2]));
    p_right_mult[3] = p_out_shift[m_itr + 3] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 3]));

    {
      ae_int32x2 d_temp0, d_temp1;
      ae_valignx2 align_p_mult = AE_LA128_PP((ae_int32x4 *)p_out_mul);
      AE_LA32X2X2_IP(d_temp0, d_temp1, align_p_mult, (ae_int32x4 *)p_out_mul);

      d_temp0 = AE_NEG32(d_temp0);
      d_temp1 = AE_NEG32(d_temp1);

      AE_S32X2X2_I(d_temp0, d_temp1, (ae_int32x4*)p_out_mult, 0);
    }

    for (vec_itr = 0; vec_itr < (n_vecs & ~(4 - 1)); vec_itr += 4)
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
      AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buffer, 0);
      AE_L32X2X2_I(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)acc_buffer, 0);
      AE_L32X2X2_I(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)acc_buffer, 0);
      AE_L32X2X2_I(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)acc_buffer, 0);

      p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_offset];
      p_mat1_1 = (ae_int8x8*)((ae_int8*)p_mat1_0 + row_offset); 
      p_mat1_2 = (ae_int8x8*)((ae_int8*)p_mat1_1 + row_offset);
      p_mat1_3 = (ae_int8x8*)((ae_int8*)p_mat1_2 + row_offset);

      align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
      align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
      align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
      align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

      p_vec_0  = (ae_int8x16 *)(p_vec1_0 + vec_itr * vec_offset);

#pragma loop_count min=1
#pragma no_unroll
      for(c_itr = 0; c_itr < cols>>5; c_itr++)
      {
        /* Load 4 rows */
        AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row1_2, mat1_row1_3, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row2_2, mat1_row2_3, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16*)p_mat1_3);
        AE_LA8X8X2_IP(mat1_row3_2, mat1_row3_3, align_p_mat1_3, (ae_int8x16*)p_mat1_3);

        /* Load  4 vectors  */
        AE_L8X8X2_X(vec1_batch_0, vec1_batch_1, p_vec_0, cols);
        AE_L8X8X2_X(vec1_batch_2, vec1_batch_3, p_vec_0, cols+16);
        AE_L8X8X2_X(vec2_batch_0, vec2_batch_1, p_vec_0, 2*cols);
        AE_L8X8X2_X(vec2_batch_2, vec2_batch_3, p_vec_0, 2*cols+16);
        AE_L8X8X2_X(vec3_batch_0, vec3_batch_1, p_vec_0, 3*cols);
        AE_L8X8X2_X(vec3_batch_2, vec3_batch_3, p_vec_0, 3*cols+16);

        AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, p_vec_0, 16);
        AE_L8X8X2_IP(vec0_batch_2, vec0_batch_3, p_vec_0, 16);
        
        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
        AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
        AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
        AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
        AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
        AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
        AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);

        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec0_batch_2);
        AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec1_batch_2);
        AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec2_batch_2);
        AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec3_batch_2);

        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec0_batch_3);
        AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec1_batch_3);
        AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec2_batch_3);
        AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec3_batch_3);
      }   

      /* Apply quantization */
      ae_int16x4 out_0, out_1, out_2, out_3;
      ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;
      ae_int32x2 out_multiplier_01, out_multiplier_23;
      AE_L32X2X2_I(l_mult_01, l_mult_23, (ae_int32x4 *)p_left_mult, 0);
      AE_L32X2X2_I(r_mult_01, r_mult_23, (ae_int32x4 *)p_right_mult, 0);
      AE_L32X2X2_I(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)p_out_mult, 0);

      MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_z_b);
      MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_1, acc_row0_vec1, acc_row1_vec1, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_z_b);
      MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_2, acc_row0_vec2, acc_row1_vec2, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_z_b);
      MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_3, acc_row0_vec3, acc_row1_vec3, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_z_b);

      /* Store output */
      ae_int8x8 out32_0, out32_1; 
      PACK_32X2(out32_0, out_0, out_1);
      PACK_32X2(out32_1, out_2, out_3);

      AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
      AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
      AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);
      AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);
    }
    
    for (; vec_itr < n_vecs; vec_itr++)
    {
      ae_int32x2 acc_row0_vec0; 
      ae_int32x2 acc_row1_vec0;

      /* Initialize accumulators */
      AE_L32X2X2_I(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buffer, 0);

      p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_offset];
      p_mat1_1 = (ae_int8x8*)((ae_int8*)p_mat1_0 + row_offset); 
      p_mat1_2 = (ae_int8x8*)((ae_int8*)p_mat1_1 + row_offset);
      p_mat1_3 = (ae_int8x8*)((ae_int8*)p_mat1_2 + row_offset);

      align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
      align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
      align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
      align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

      p_vec_0  = (ae_int8x16 *)(p_vec1_0 + vec_itr * vec_offset);

#pragma loop_count min=1
      for(c_itr = 0; c_itr < cols>>5; c_itr++)
      {
        /* Load 4 rows */
        AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row1_2, mat1_row1_3, align_p_mat1_1, (ae_int8x16*)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row2_2, mat1_row2_3, align_p_mat1_2, (ae_int8x16*)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16*)p_mat1_3);
        AE_LA8X8X2_IP(mat1_row3_2, mat1_row3_3, align_p_mat1_3, (ae_int8x16*)p_mat1_3);

        /* Load  4 vectors  */
        AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec0_batch_2, vec0_batch_3, (ae_int8x16*)p_vec_0, 16);
        
        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec0_batch_2);
        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_3 , mat1_row1_3 , mat1_row2_3 , mat1_row3_3 ,vec0_batch_3);
      }   

      /* Apply quantization */
      ae_int16x4 out_0;
      ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;
      ae_int32x2 out_multiplier_01, out_multiplier_23;
      AE_L32X2X2_I(l_mult_01, l_mult_23, (ae_int32x4 *)p_left_mult, 0);
      AE_L32X2X2_I(r_mult_01, r_mult_23, (ae_int32x4 *)p_right_mult, 0);
      AE_L32X2X2_I(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)p_out_mult, 0);

      MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_z_b);

      /* Store output */
      ae_int8x8 out32_0; 
      PACK_32X2(out32_0, out_0, out_0);

      AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
    }
  }
}

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
  int c_itr = 0;
  int rem_cols = cols & 15;
  int rem_cols_shift_0 = ((rem_cols)<=8)?(8-(rem_cols))*8:0;
  int rem_cols_shift_1 = ((rem_cols)>8)?(16-(rem_cols))*8:64;
  
  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;

  ae_int8x8 vec0_batch_0, vec0_batch_1; 
  ae_int8x8 vec1_batch_0, vec1_batch_1;
  ae_int8x8 vec2_batch_0, vec2_batch_1;
  ae_int8x8 vec3_batch_0, vec3_batch_1;

  ae_int8x8* p_mat1_1 = (ae_int8x8*)((ae_int8*)p_mat1_0 + row_offset); 
  ae_int8x8* p_mat1_2 = (ae_int8x8*)((ae_int8*)p_mat1_1 + row_offset);
  ae_int8x8* p_mat1_3 = (ae_int8x8*)((ae_int8*)p_mat1_2 + row_offset);

  ae_int8* p_vec_1 = p_vec_0 + vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_0_1;
  ae_int32x2 acc_row0_vec2 = *out_0_2;
  ae_int32x2 acc_row0_vec3 = *out_0_3;
                       
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row1_vec1 = *out_1_1;
  ae_int32x2 acc_row1_vec2 = *out_1_2;
  ae_int32x2 acc_row1_vec3 = *out_1_3;

  int cols_count = cols -(cols & 15);

  for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
  {
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 16);
    AE_L8X8X2_IP(vec1_batch_0, vec1_batch_1, (ae_int8x16 *)p_vec_1, 16);
    AE_L8X8X2_IP(vec2_batch_0, vec2_batch_1, (ae_int8x16 *)p_vec_2, 16);
    AE_L8X8X2_IP(vec3_batch_0, vec3_batch_1, (ae_int8x16 *)p_vec_3, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, (ae_int8x16 *)p_mat1_1, 16);
    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, (ae_int8x16 *)p_mat1_2, 16);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, (ae_int8x16 *)p_mat1_3, 16);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row0_1);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row1_1);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row2_1);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row3_1);
  }

  //Remainder loop for cols
  c_itr <<= 4;
  while(c_itr < cols)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);
    AE_L8X8_IP(vec1_batch_0, (ae_int8x8 *)p_vec_1, 8);
    AE_L8X8_IP(vec2_batch_0, (ae_int8x8 *)p_vec_2, 8);
    AE_L8X8_IP(vec3_batch_0, (ae_int8x8 *)p_vec_3, 8);

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift_0), rem_cols_shift_0));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift_0), rem_cols_shift_0));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);

    c_itr += 8;
    rem_cols_shift_0 = rem_cols_shift_1;
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
  int c_itr = 0;
  int rem_cols = cols & 15;
  int rem_cols_shift_0 = ((rem_cols)<=8)?(8-(rem_cols))*8:0;
  int rem_cols_shift_1 = ((rem_cols)>8)?(16-(rem_cols))*8:64;

  ae_int8x8 mat1_row0_0, mat1_row0_1;
  ae_int8x8 mat1_row1_0, mat1_row1_1;
  ae_int8x8 mat1_row2_0, mat1_row2_1;
  ae_int8x8 mat1_row3_0, mat1_row3_1;
  ae_int8x8 vec0_batch_0, vec0_batch_1; 

  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_offset); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_offset); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_offset); 

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  int cols_count=cols-(cols&15);

  for(c_itr = 0; c_itr < cols_count>>4; c_itr++)
  {
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);
    AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, (ae_int8x16 *)p_mat1_1, 16);
    AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, (ae_int8x16 *)p_mat1_2, 16);
    AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, (ae_int8x16 *)p_mat1_3, 16);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
  }

  //Remainder loop for cols
  c_itr <<= 4;
  while(c_itr < cols)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift_0), rem_cols_shift_0));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
    c_itr += 8;
    rem_cols_shift_0 = rem_cols_shift_1;
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row1_vec0;
}

static inline void _xa_nn_dot_product_1_rows_1_vecs_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols1
    )
{
  int c_itr = 0;
  int rem_cols = cols1 & 15;
  int rem_cols_shift_0 = ((rem_cols)<=8)?(8-(rem_cols))*8:0;
  int rem_cols_shift_1 = ((rem_cols)>8)?(16-(rem_cols))*8:64;
  
  ae_int8x8 vec0_batch_0, vec0_batch_1; 
  ae_int8x8 mat1_row0_0, mat1_row0_1;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  int cols_count = cols1 - (cols1 & 15);

#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count >> 4; c_itr++)
  {
    AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16 *)p_vec_0, 16);

    AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 ,vec0_batch_0);
    AE_MULA8Q8X8(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_1 , mat1_row0_1 , mat1_row0_1 , mat1_row0_1 ,vec0_batch_1);
  }
 
  //Remainder loop for cols1
  c_itr <<= 4;
  while(c_itr < cols1)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(vec0_batch_0, (ae_int8x8 *)p_vec_0, 8);   

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift_0), rem_cols_shift_0));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 ,vec0_batch_0);
    c_itr += 8;
    rem_cols_shift_0 = rem_cols_shift_1;
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

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

  int align_offset = ((unsigned int)p_mat1_0 & 0x7);
  pre_loop_count = 8 - align_offset;
  pre_loop_shift = align_offset * 8;
  p_mat1_0 = (ae_int8x8 *)((ae_int8 *)p_mat1_0 - align_offset);
  //TODO: possible out of bound access
  p_vec_0 -= align_offset;

  pre_loop_count += 8; // 16 values loaded in preloop step
  loop_count = (cols < pre_loop_count)?0:(cols - pre_loop_count);
  post_loop_count = loop_count?(loop_count & 15):((cols + align_offset) & 15);
  loop_count >>= 4;

  int rem_cols_shift_0 = ((post_loop_count)<=8)?(8-(post_loop_count))*8:0;
  int rem_cols_shift_1 = ((post_loop_count)>8)?(16-(post_loop_count))*8:64;

  int mask_start_end = ((cols + align_offset) < 16)?0:1;

  ae_int8x8* p_mat1_1 = p_mat1_0 + row_offset; //next 8th row 
  ae_int8x8* p_mat1_2 = p_mat1_1 + row_offset; //next 8th row
  ae_int8x8* p_mat1_3 = p_mat1_2 + row_offset; //next 8th row 

  ae_int8* p_vec_1 = p_vec_0 + vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  ae_valign align_p_vec_0 = AE_LA64_PP(p_vec_0);
  ae_valign align_p_vec_1 = AE_LA64_PP(p_vec_1);
  ae_valign align_p_vec_2 = AE_LA64_PP(p_vec_2);
  ae_valign align_p_vec_3 = AE_LA64_PP(p_vec_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_0_1;
  ae_int32x2 acc_row0_vec2 = *out_0_2;
  ae_int32x2 acc_row0_vec3 = *out_0_3;
                       
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row1_vec1 = *out_1_1;
  ae_int32x2 acc_row1_vec2 = *out_1_2;
  ae_int32x2 acc_row1_vec3 = *out_1_3;

  /* Pre loop computation */
  AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
  AE_L8X8_IP(mat1_row0_1, p_mat1_0, 8);
  AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
  AE_L8X8_IP(mat1_row1_1, p_mat1_1, 8);
  AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
  AE_L8X8_IP(mat1_row2_1, p_mat1_2, 8);
  AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);
  AE_L8X8_IP(mat1_row3_1, p_mat1_3, 8);

  AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);
  AE_LA8X8_IP(vec0_batch_1, align_p_vec_0, (ae_int8x8 *)p_vec_0);
  AE_LA8X8_IP(vec1_batch_0, align_p_vec_1, (ae_int8x8 *)p_vec_1);
  AE_LA8X8_IP(vec1_batch_1, align_p_vec_1, (ae_int8x8 *)p_vec_1);
  AE_LA8X8_IP(vec2_batch_0, align_p_vec_2, (ae_int8x8 *)p_vec_2);
  AE_LA8X8_IP(vec2_batch_1, align_p_vec_2, (ae_int8x8 *)p_vec_2);
  AE_LA8X8_IP(vec3_batch_0, align_p_vec_3, (ae_int8x8 *)p_vec_3);
  AE_LA8X8_IP(vec3_batch_1, align_p_vec_3, (ae_int8x8 *)p_vec_3);

  if(align_offset)
  {
    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), pre_loop_shift), pre_loop_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), pre_loop_shift), pre_loop_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), pre_loop_shift), pre_loop_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SRLA64(AE_SLAA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), pre_loop_shift), pre_loop_shift));
  }

  if(mask_start_end)
  {
    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row0_1);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row1_1);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row2_1);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row3_1);
  }

#pragma no_unroll
  for(c_itr = 0; c_itr < loop_count; c_itr++)
  {
    AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row0_1, p_mat1_0, 8);
    AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row1_1, p_mat1_1, 8);
    AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row2_1, p_mat1_2, 8);
    AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);
    AE_L8X8_IP(mat1_row3_1, p_mat1_3, 8);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec0_batch_1, align_p_vec_0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec_1, (ae_int8x8 *)p_vec_1);
    AE_LA8X8_IP(vec1_batch_1, align_p_vec_1, (ae_int8x8 *)p_vec_1);
    AE_LA8X8_IP(vec2_batch_0, align_p_vec_2, (ae_int8x8 *)p_vec_2);
    AE_LA8X8_IP(vec2_batch_1, align_p_vec_2, (ae_int8x8 *)p_vec_2);
    AE_LA8X8_IP(vec3_batch_0, align_p_vec_3, (ae_int8x8 *)p_vec_3);
    AE_LA8X8_IP(vec3_batch_1, align_p_vec_3, (ae_int8x8 *)p_vec_3);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row0_1);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row1_1);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row2_1);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_1, vec1_batch_1, vec2_batch_1, vec3_batch_1, mat1_row3_1);
  }

  //Remainder loop for cols
  c_itr = 0;
  int rem_shift = rem_cols_shift_0;

  while(c_itr < post_loop_count)
  {
    if(mask_start_end)
    {
      AE_L8X8_IP(mat1_row0_0, p_mat1_0, 8);
      AE_L8X8_IP(mat1_row1_0, p_mat1_1, 8);
      AE_L8X8_IP(mat1_row2_0, p_mat1_2, 8);
      AE_L8X8_IP(mat1_row3_0, p_mat1_3, 8);

      AE_LA8X8_IP(vec0_batch_0, align_p_vec_0 ,(ae_int8x8*)p_vec_0);
      AE_LA8X8_IP(vec1_batch_0, align_p_vec_1 ,(ae_int8x8*)p_vec_1);
      AE_LA8X8_IP(vec2_batch_0, align_p_vec_2 ,(ae_int8x8*)p_vec_2);
      AE_LA8X8_IP(vec3_batch_0, align_p_vec_3 ,(ae_int8x8*)p_vec_3);
    }

    mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_shift), rem_shift));
    mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_shift), rem_shift));
    mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_shift), rem_shift));
    mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_shift), rem_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);

    c_itr += 8;
    rem_shift = rem_cols_shift_1;
    if(!mask_start_end && (c_itr < post_loop_count))
    {
      mat1_row0_0 = mat1_row0_1;
      mat1_row1_0 = mat1_row1_1;
      mat1_row2_0 = mat1_row2_1;
      mat1_row3_0 = mat1_row3_1;
      vec0_batch_0 = vec0_batch_1;
      vec1_batch_0 = vec1_batch_1;
      vec2_batch_0 = vec2_batch_1;
      vec3_batch_0 = vec3_batch_1;
    }
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

static inline void _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
    (ae_int32x2* out_0_0
    ,ae_int32x2* out_1_0
    ,ae_int8x8*  p_mat1_0
    ,ae_int8*    p_vec_0
    ,WORD32      cols
    ,WORD32      row_offset
    )
{
  int c_itr = 0;
  int rem_cols_shift = 64 - (cols & 7) * 8;

  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  ae_int8x8* p_mat1_1 = p_mat1_0 + row_offset; //next 8th row 
  ae_int8x8* p_mat1_2 = p_mat1_1 + row_offset; //next 8th row
  ae_int8x8* p_mat1_3 = p_mat1_2 + row_offset; //next 8th row 

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  ae_valign align_p_mat1_3 = AE_LA64_PP(p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);

  int cols_count=cols-(cols&7);

  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);
    
    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

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
  int c_itr = 0;

  int rem_cols_shift = 64 - (cols & 7) * 8;
  
  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;

  ae_int8x8 vec0_batch_0; 
  ae_int8x8 vec1_batch_0; 
  ae_int8x8 vec2_batch_0; 
  ae_int8x8 vec3_batch_0; 
  ae_int8x8 align_p_mat1_0, align_p_mat1_1, align_p_mat1_2, align_p_mat1_3; 

  ae_int8x8* p_mat1_1 = (ae_int8x8*)((ae_int8*)p_mat1_0 + row_offset); 
  ae_int8x8* p_mat1_2 = (ae_int8x8*)((ae_int8*)p_mat1_1 + row_offset);
  ae_int8x8* p_mat1_3 = (ae_int8x8*)((ae_int8*)p_mat1_2 + row_offset);

  ae_int8* p_vec_1 = p_vec_0 + vec_offset; 
  ae_int8* p_vec_2 = p_vec_1 + vec_offset;
  ae_int8* p_vec_3 = p_vec_2 + vec_offset;

  ae_valign align_p_vec_0 = AE_LA64_PP(p_vec_0);
  ae_valign align_p_vec_1 = AE_LA64_PP(p_vec_1);
  ae_valign align_p_vec_2 = AE_LA64_PP(p_vec_2);
  ae_valign align_p_vec_3 = AE_LA64_PP(p_vec_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_0_1;
  ae_int32x2 acc_row0_vec2 = *out_0_2;
  ae_int32x2 acc_row0_vec3 = *out_0_3;
                       
  ae_int32x2 acc_row1_vec0 = *out_1_0;
  ae_int32x2 acc_row1_vec1 = *out_1_1;
  ae_int32x2 acc_row1_vec2 = *out_1_2;
  ae_int32x2 acc_row1_vec3 = *out_1_3;

  AE_SW_PRIME_64(p_mat1_0, align_p_mat1_0);
  AE_SW_PRIME_64(p_mat1_1, align_p_mat1_1);
  AE_SW_PRIME_64(p_mat1_2, align_p_mat1_2);
  AE_SW_PRIME_64(p_mat1_3, align_p_mat1_3);

  int cols_count = cols -(cols & 7);
#pragma no_unroll
  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec_1, (ae_int8x8 *)p_vec_1);
    AE_LA8X8_IP(vec2_batch_0, align_p_vec_2, (ae_int8x8 *)p_vec_2);
    AE_LA8X8_IP(vec3_batch_0, align_p_vec_3, (ae_int8x8 *)p_vec_3);

    AE_SW_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_SW_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_SW_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);
  }  

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_SW_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_SW_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_SW_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_SW_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);
    AE_LA8X8_IP(vec1_batch_0, align_p_vec_1, (ae_int8x8 *)p_vec_1);
    AE_LA8X8_IP(vec2_batch_0, align_p_vec_2, (ae_int8x8 *)p_vec_2);
    AE_LA8X8_IP(vec3_batch_0, align_p_vec_3, (ae_int8x8 *)p_vec_3);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));
    vec1_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec1_batch_0), rem_cols_shift), rem_cols_shift));
    vec2_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec2_batch_0), rem_cols_shift), rem_cols_shift));
    vec3_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec3_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row0_0);
    AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row1_0);
    AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row2_0);
    AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , vec0_batch_0, vec1_batch_0, vec2_batch_0, vec3_batch_0, mat1_row3_0);
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
  int c_itr = 0;

  int rem_cols_shift = 64 - (cols & 7) * 8;
  
  ae_int8x8 mat1_row0_0;
  ae_int8x8 mat1_row1_0;
  ae_int8x8 mat1_row2_0;
  ae_int8x8 mat1_row3_0;
  ae_int8x8 vec0_batch_0; 
  ae_int8x8 align_p_vec0;

  ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_offset); 
  ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_offset); 
  ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_offset); 

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_mat1_1 = AE_LA64_PP(p_mat1_1);
  ae_valign align_p_mat1_2 = AE_LA64_PP(p_mat1_2);
  ae_valign align_p_mat1_3 = AE_LA64_PP(p_mat1_3);

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row1_vec0 = *out_1_0;

  AE_SW_PRIME_64(p_vec_0, align_p_vec0);

  int cols_count=cols-(cols&7);

  for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);
    
    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
  }

  //Remainder loop for cols
  if(cols_count!=cols)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
    AE_LA8X8_IP(mat1_row1_0, align_p_mat1_1, p_mat1_1);
    AE_LA8X8_IP(mat1_row2_0, align_p_mat1_2, p_mat1_2);
    AE_LA8X8_IP(mat1_row3_0, align_p_mat1_3, p_mat1_3);

    AE_SW_LA8X8_IP(vec0_batch_0, align_p_vec0, p_vec_0);
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
    ,WORD32      cols1
    )
{
  int c_itr = 0;
  int rem_cols_shift = 64 - (cols1 & 7) * 8;

  ae_int8x8 vec0_batch_0; 
  ae_int8x8 mat1_row0_0;

  ae_int32x2 acc_row0_vec0 = *out_0_0;
  ae_int32x2 acc_row0_vec1 = *out_1_0;

  ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
  ae_valign align_p_vec_0 = AE_LA64_PP(p_vec_0);

  int cols_count = cols1 - (cols1 & 7);

  for(c_itr = 0; c_itr < cols_count >> 3; c_itr++)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 ,vec0_batch_0);
  }

  //Remainder loop for cols1
  if(cols_count!=cols1)
  {
    AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, (ae_int8x8 *)p_mat1_0);

    AE_LA8X8_IP(vec0_batch_0, align_p_vec_0, (ae_int8x8 *)p_vec_0);

    vec0_batch_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(vec0_batch_0), rem_cols_shift), rem_cols_shift));

    AE_MULA8Q8X8(acc_row0_vec0 , acc_row0_vec1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 ,vec0_batch_0);
    
  }

  *out_0_0 = acc_row0_vec0;
  *out_1_0 = acc_row0_vec1;
}

WORD32 xa_nn_matmul_per_chan_sym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_vec1,
    const WORD32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride,                      
    WORD32 vec1_zero_bias,
    const WORD32* __restrict__ p_out_multiplier,
    const WORD32* __restrict__ p_out_shift,
    WORD32 out_zero_bias)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shift, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shift, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
  XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
  XA_NNLIB_ARG_CHK_COND((vec1_zero_bias < -127 || vec1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  int itr = 0;
  for(itr=0; itr<rows; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  ae_int32x2 acc_buffer[4];
  WORD8 * __restrict__ p_dst_0;
  ae_int8* __restrict__ p_vec_0;

  /* Iterators used in for loops */
  int m_itr, vec_itr;
  int ii;

  /* Assign initial value so this value will be used in trailing loop */
  m_itr = 0;
  vec_itr = 0;

  /* Shifts to match with Tensorflow */
  int p_left_shift[4], p_right_shift[4];

  ae_int32x2 min_int8 = AE_MOVDA32(-128);

  int c_itr = 0;
  int rem_cols_shift = 64 - (cols1 & 7) * 8;
  ae_int8x8 vec_z_b = AE_MOVDA8(-vec1_zero_bias);

#undef VEC_UNROLL
#define VEC_UNROLL 4

  /* Special case for cols == 8 */
  if(
      (cols1 == 8) &&
      (row_stride1 == 8) &&
      (vec_offset == 8) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ALIGNED_PTR(p_out, 4) &&
      (out_stride == 1) &&   // NHWC case
      ((out_offset & 0x3) == 0) &&   // NHWC case
      ((rows & 0x3) == 0) &&
      ((vec_count & 0x3) == 0)
    )
  {
    ae_int32x4 *pt_bias = (ae_int32x4 *)p_bias;
    ae_valignx2 align_p_bias = AE_LA128_PP(pt_bias);
    ae_int32x4 *acc_buff = (ae_int32x4 *)acc_buffer;

    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int8x8 mat1_row0_0;
      ae_int8x8 mat1_row1_0;
      ae_int8x8 mat1_row2_0;
      ae_int8x8 mat1_row3_0;
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      /* Load 4 rows */
      AE_LA8X8X2_IP(mat1_row0_0, mat1_row1_0, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row2_0, mat1_row3_0, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);

      ae_int32x2 d_bias0, d_bias1;
      AE_LA32X2X2_IP(d_bias0, d_bias1, align_p_bias, pt_bias);

      acc_row0 = AE_SUB32S(d_bias0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias1, acc_row1);

      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4*)acc_buffer, 0);

      int p_left_mult[4], p_right_mult[4], p_out_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x4 *ptr_right_mult = (ae_int32x4 *)p_right_mult;
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;

      p_left_mult[0] = p_out_shift[m_itr + 0] < 0 ? 1 : (1 << p_out_shift[m_itr + 0]);
      p_left_mult[1] = p_out_shift[m_itr + 1] < 0 ? 1 : (1 << p_out_shift[m_itr + 1]);
      p_left_mult[2] = p_out_shift[m_itr + 2] < 0 ? 1 : (1 << p_out_shift[m_itr + 2]);
      p_left_mult[3] = p_out_shift[m_itr + 3] < 0 ? 1 : (1 << p_out_shift[m_itr + 3]);

      p_right_mult[0] = p_out_shift[m_itr + 0] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 0]));
      p_right_mult[1] = p_out_shift[m_itr + 1] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 1]));
      p_right_mult[2] = p_out_shift[m_itr + 2] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 2]));
      p_right_mult[3] = p_out_shift[m_itr + 3] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 3]));

      p_out_mult[0] = -p_out_multiplier[m_itr + 0];
      p_out_mult[1] = -p_out_multiplier[m_itr + 1];
      p_out_mult[2] = -p_out_multiplier[m_itr + 2];
      p_out_mult[3] = -p_out_multiplier[m_itr + 3];

      p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;
      p_vec_0  = (ae_int8 *)(p_vec1);

#pragma loop_count min=1
      for (vec_itr = 0; vec_itr < (vec_count & ~(4 - 1)); vec_itr += 4)
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
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)acc_buff, 0);

        ae_int8x8 vec0_batch_0; 
        ae_int8x8 vec1_batch_0; 
        ae_int8x8 vec2_batch_0; 
        ae_int8x8 vec3_batch_0; 

        /* Load  4 vectors  */
        AE_L8X8X2_IP(vec0_batch_0, vec1_batch_0, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec2_batch_0, vec3_batch_0, (ae_int8x16*)p_vec_0, 16);

        AE_MULA8Q8X8(acc_row0_vec0, acc_row1_vec0, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0);
        AE_MULA8Q8X8(acc_row0_vec1, acc_row1_vec1, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0);
        AE_MULA8Q8X8(acc_row0_vec2, acc_row1_vec2, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec2_batch_0);
        AE_MULA8Q8X8(acc_row0_vec3, acc_row1_vec3, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec3_batch_0);

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;

        ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;
        ae_int32x2 out_multiplier_01, out_multiplier_23;
        AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);
        AE_L32X2X2_IP(r_mult_01, r_mult_23, (ae_int32x4 *)ptr_right_mult, 0);
        AE_L32X2X2_IP(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);

        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_1, acc_row0_vec1, acc_row1_vec1, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_2, acc_row0_vec2, acc_row1_vec2, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_3, acc_row0_vec3, acc_row1_vec3, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);

        /* Store output */
        ae_int8x8 out32_0, out32_1; 
        PACK_32X2(out32_0, out_0, out_1);
        PACK_32X2(out32_1, out_2, out_3);

        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);
      }
      /*
         for (; vec_itr < vec_count; vec_itr++)
         {
         }
       */
    }
    return 0;
  }

  /* Special case for cols == 16 */
  if(
      (cols1 == 16) &&
      (row_stride1 == 16) &&
      (vec_offset == 16) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ALIGNED_PTR(p_out, 4) &&
      (out_stride == 1) &&   // NHWC case
      ((out_offset & 0x3) == 0) &&   // NHWC case
      ((rows & 0x3) == 0) &&
      ((vec_count & 0x3) == 0)
    )
  {
    ae_int32x4 *pt_bias = (ae_int32x4 *)p_bias;
    ae_valignx2 align_p_bias = AE_LA128_PP(pt_bias);
    ae_int32x4 *acc_buff = (ae_int32x4 *)acc_buffer;

    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int8x8 mat1_row0_0, mat1_row0_1;
      ae_int8x8 mat1_row1_0, mat1_row1_1;
      ae_int8x8 mat1_row2_0, mat1_row2_1;
      ae_int8x8 mat1_row3_0, mat1_row3_1;
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);

      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 , vec_z_b);

      ae_int32x2 d_bias0, d_bias1;
      AE_LA32X2X2_IP(d_bias0, d_bias1, align_p_bias, pt_bias);

      acc_row0 = AE_SUB32S(d_bias0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias1, acc_row1);

      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4*)acc_buffer, 0);

      int p_left_mult[4], p_right_mult[4], p_out_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x4 *ptr_right_mult = (ae_int32x4 *)p_right_mult;
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;

      p_left_mult[0] = p_out_shift[m_itr + 0] < 0 ? 1 : (1 << p_out_shift[m_itr + 0]);
      p_left_mult[1] = p_out_shift[m_itr + 1] < 0 ? 1 : (1 << p_out_shift[m_itr + 1]);
      p_left_mult[2] = p_out_shift[m_itr + 2] < 0 ? 1 : (1 << p_out_shift[m_itr + 2]);
      p_left_mult[3] = p_out_shift[m_itr + 3] < 0 ? 1 : (1 << p_out_shift[m_itr + 3]);

      p_right_mult[0] = p_out_shift[m_itr + 0] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 0]));
      p_right_mult[1] = p_out_shift[m_itr + 1] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 1]));
      p_right_mult[2] = p_out_shift[m_itr + 2] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 2]));
      p_right_mult[3] = p_out_shift[m_itr + 3] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 3]));

      p_out_mult[0] = -p_out_multiplier[m_itr + 0];
      p_out_mult[1] = -p_out_multiplier[m_itr + 1];
      p_out_mult[2] = -p_out_multiplier[m_itr + 2];
      p_out_mult[3] = -p_out_multiplier[m_itr + 3];

      p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;

      p_vec_0  = (ae_int8 *)(p_vec1);

#pragma loop_count min=1
      for (vec_itr = 0; vec_itr < (vec_count & ~(4 - 1)); vec_itr += 4)
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
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)acc_buff, 0);
        ae_int8x8 vec0_batch_0, vec0_batch_1; 
        ae_int8x8 vec1_batch_0, vec1_batch_1; 
        ae_int8x8 vec2_batch_0, vec2_batch_1; 
        ae_int8x8 vec3_batch_0, vec3_batch_1;

        /* Load  4 vectors  */
        AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec1_batch_0, vec1_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec2_batch_0, vec2_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec3_batch_0, vec3_batch_1, (ae_int8x16*)p_vec_0, 16);

        AE_MULA8Q8X8(acc_row0_vec0, acc_row1_vec0, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec0_batch_0);
        AE_MULA8Q8X8(acc_row0_vec1, acc_row1_vec1, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec1_batch_0);
        AE_MULA8Q8X8(acc_row0_vec2, acc_row1_vec2, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec2_batch_0);
        AE_MULA8Q8X8(acc_row0_vec3, acc_row1_vec3, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec3_batch_0);

        AE_MULA8Q8X8(acc_row0_vec0, acc_row1_vec0, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec0_batch_1);
        AE_MULA8Q8X8(acc_row0_vec1, acc_row1_vec1, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec1_batch_1);
        AE_MULA8Q8X8(acc_row0_vec2, acc_row1_vec2, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec2_batch_1);
        AE_MULA8Q8X8(acc_row0_vec3, acc_row1_vec3, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec3_batch_1);

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;

        ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;
        ae_int32x2 out_multiplier_01, out_multiplier_23;
        AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);
        AE_L32X2X2_IP(r_mult_01, r_mult_23, (ae_int32x4 *)ptr_right_mult, 0);
        AE_L32X2X2_IP(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);

        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_1, acc_row0_vec1, acc_row1_vec1, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_2, acc_row0_vec2, acc_row1_vec2, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_3, acc_row0_vec3, acc_row1_vec3, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);

        /* Store output */
        ae_int8x8 out32_0, out32_1; 
        PACK_32X2(out32_0, out_0, out_1);
        PACK_32X2(out32_1, out_2, out_3);

        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);
      }
      /*
      for (; vec_itr < vec_count; vec_itr++)
      {
      }
      */
    }
    return 0;
  }

  /* Special case for cols == 24 */
  if(
      (cols1 == 24) &&
      (row_stride1 == 24) &&
      (vec_offset == 24) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ALIGNED_PTR(p_out, 4) &&
      (out_stride == 1) &&   // NHWC case
      ((out_offset & 0x3) == 0) &&   // NHWC case
      ((rows & 0x3) == 0) &&
      ((vec_count & 0x3) == 0)
    )
  {
    ae_int32x4 *pt_bias = (ae_int32x4 *)p_bias;
    ae_valignx2 align_p_bias = AE_LA128_PP(pt_bias);
    ae_int32x4 *acc_buff = (ae_int32x4 *)acc_buffer;

    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2;
      ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2;
      ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2;
      ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2;
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row0_1, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row0_2, align_p_mat1_0, p_mat1_0);
      AE_LA8X8_IP(mat1_row1_0, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row1_1, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row1_2, align_p_mat1_0, p_mat1_0);
      AE_LA8X8_IP(mat1_row2_0, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row2_1, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row2_2, align_p_mat1_0, p_mat1_0);
      AE_LA8X8_IP(mat1_row3_0, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row3_1, align_p_mat1_0, p_mat1_0); AE_LA8X8_IP(mat1_row3_2, align_p_mat1_0, p_mat1_0);

      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 , vec_z_b);
      AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 , vec_z_b);

      ae_int32x2 d_bias0, d_bias1;
      AE_LA32X2X2_IP(d_bias0, d_bias1, align_p_bias, pt_bias);

      acc_row0 = AE_SUB32S(d_bias0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias1, acc_row1);

      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4*)acc_buffer, 0);

      int p_left_mult[4], p_right_mult[4], p_out_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x4 *ptr_right_mult = (ae_int32x4 *)p_right_mult;
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;

      p_left_mult[0] = p_out_shift[m_itr + 0] < 0 ? 1 : (1 << p_out_shift[m_itr + 0]);
      p_left_mult[1] = p_out_shift[m_itr + 1] < 0 ? 1 : (1 << p_out_shift[m_itr + 1]);
      p_left_mult[2] = p_out_shift[m_itr + 2] < 0 ? 1 : (1 << p_out_shift[m_itr + 2]);
      p_left_mult[3] = p_out_shift[m_itr + 3] < 0 ? 1 : (1 << p_out_shift[m_itr + 3]);

      p_right_mult[0] = p_out_shift[m_itr + 0] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 0]));
      p_right_mult[1] = p_out_shift[m_itr + 1] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 1]));
      p_right_mult[2] = p_out_shift[m_itr + 2] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 2]));
      p_right_mult[3] = p_out_shift[m_itr + 3] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 3]));

      p_out_mult[0] = -p_out_multiplier[m_itr + 0];
      p_out_mult[1] = -p_out_multiplier[m_itr + 1];
      p_out_mult[2] = -p_out_multiplier[m_itr + 2];
      p_out_mult[3] = -p_out_multiplier[m_itr + 3];

      p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;

      p_vec_0  = (ae_int8 *)(p_vec1);

      for (vec_itr = 0; vec_itr < (vec_count & ~(4 - 1)); vec_itr += 4)
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
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)acc_buff, 0);

        ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2; 
        ae_int8x8 vec1_batch_0, vec1_batch_1, vec1_batch_2; 
        ae_int8x8 vec2_batch_0, vec2_batch_1, vec2_batch_2; 
        ae_int8x8 vec3_batch_0, vec3_batch_1, vec3_batch_2; 

        /* Load  4 vectors  */
        AE_L8X8_IP(vec0_batch_0, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec0_batch_1, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec0_batch_2, (ae_int8x8*)p_vec_0, 8);
        AE_L8X8_IP(vec1_batch_0, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec1_batch_1, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec1_batch_2, (ae_int8x8*)p_vec_0, 8);
        AE_L8X8_IP(vec2_batch_0, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec2_batch_1, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec2_batch_2, (ae_int8x8*)p_vec_0, 8);
        AE_L8X8_IP(vec3_batch_0, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec3_batch_1, (ae_int8x8*)p_vec_0, 8); AE_L8X8_IP(vec3_batch_2, (ae_int8x8*)p_vec_0, 8);

        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec0_batch_0);
        AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec1_batch_0);
        AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec2_batch_0);
        AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 ,vec3_batch_0);

        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec0_batch_1);
        AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec1_batch_1);
        AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec2_batch_1);
        AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 ,vec3_batch_1);

        AE_MULA8Q8X8(acc_row0_vec0 , acc_row1_vec0 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec0_batch_2);
        AE_MULA8Q8X8(acc_row0_vec1 , acc_row1_vec1 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec1_batch_2);
        AE_MULA8Q8X8(acc_row0_vec2 , acc_row1_vec2 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec2_batch_2);
        AE_MULA8Q8X8(acc_row0_vec3 , acc_row1_vec3 , mat1_row0_2 , mat1_row1_2 , mat1_row2_2 , mat1_row3_2 ,vec3_batch_2);

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;

        ae_int32x2 l_mult_23, l_mult_01, r_mult_23, r_mult_01;
        ae_int32x2 out_multiplier_01, out_multiplier_23;
        AE_L32X2X2_IP(l_mult_01, l_mult_23, (ae_int32x4 *)ptr_left_mult, 0);
        AE_L32X2X2_IP(r_mult_01, r_mult_23, (ae_int32x4 *)ptr_right_mult, 0);
        AE_L32X2X2_IP(out_multiplier_01, out_multiplier_23, (ae_int32x4 *)ptr_out_mult, 0);

        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_1, acc_row0_vec1, acc_row1_vec1, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_2, acc_row0_vec2, acc_row1_vec2, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_3, acc_row0_vec3, acc_row1_vec3, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);

        /* Store output */
        ae_int8x8 out32_0, out32_1; 
        PACK_32X2(out32_0, out_0, out_1);
        PACK_32X2(out32_1, out_2, out_3);

        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);
      }
      /*
      for (; vec_itr < vec_count; vec_itr++)
      {
      }
      */
    }
    return 0;
  } 

  /* Special case for cols == 32 */
  if(
      (cols1 == 32) &&
      (row_stride1 == 32) &&
      (vec_offset == 32) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ALIGNED_PTR(p_out, 4) &&
      (out_stride == 1) &&   // NHWC case
      ((out_offset & 0x3) == 0) &&   // NHWC case
      ((rows & 0x3) == 0) &&
      ((vec_count & 0x3) == 0)
    )
  {
    ae_int32x4 *acc_buff = (ae_int32x4 *)acc_buffer;

    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int8x8 mat1_row0_0, mat1_row0_1, mat1_row0_2, mat1_row0_3;
      ae_int8x8 mat1_row1_0, mat1_row1_1, mat1_row1_2, mat1_row1_3;
      ae_int8x8 mat1_row2_0, mat1_row2_1, mat1_row2_2, mat1_row2_3;
      ae_int8x8 mat1_row3_0, mat1_row3_1, mat1_row3_2, mat1_row3_3;
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      /* Load 4 rows */
      AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row0_2, mat1_row0_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row1_2, mat1_row1_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row2_2, mat1_row2_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_0, (ae_int8x16*)p_mat1_0);
      AE_LA8X8X2_IP(mat1_row3_2, mat1_row3_3, align_p_mat1_0, (ae_int8x16*)p_mat1_0);

      AE_MULA8Q8X8(acc_row1 , acc_row0 , mat1_row3_0 , mat1_row2_0 , mat1_row1_0 , mat1_row0_0 , vec_z_b);
      AE_MULA8Q8X8(acc_row1 , acc_row0 , mat1_row3_1 , mat1_row2_1 , mat1_row1_1 , mat1_row0_1 , vec_z_b);
      AE_MULA8Q8X8(acc_row1 , acc_row0 , mat1_row3_2 , mat1_row2_2 , mat1_row1_2 , mat1_row0_2 , vec_z_b);
      AE_MULA8Q8X8(acc_row1 , acc_row0 , mat1_row3_3 , mat1_row2_3 , mat1_row1_3 , mat1_row0_3 , vec_z_b);

      ae_int32x2 d_bias0, d_bias1;
      d_bias1 = AE_MOVDA32X2(p_bias[m_itr + 3], p_bias[m_itr + 2]);
      d_bias0 = AE_MOVDA32X2(p_bias[m_itr + 1], p_bias[m_itr + 0]);

      acc_row0 = AE_SUB32S(d_bias0, acc_row0);
      acc_row1 = AE_SUB32S(d_bias1, acc_row1);

      AE_S32X2X2_I(acc_row0, acc_row1, (ae_int32x4*)acc_buffer, 0);

      int p_left_mult[4], p_right_mult[4], p_out_mult[4];
      ae_int32x4 *ptr_left_mult = (ae_int32x4 *)p_left_mult;
      ae_int32x4 *ptr_right_mult = (ae_int32x4 *)p_right_mult;
      ae_int32x4 *ptr_out_mult = (ae_int32x4 *)p_out_mult;

      p_left_mult[3] = p_out_shift[m_itr + 0] < 0 ? 1 : (1 << p_out_shift[m_itr + 0]);
      p_left_mult[2] = p_out_shift[m_itr + 1] < 0 ? 1 : (1 << p_out_shift[m_itr + 1]);
      p_left_mult[1] = p_out_shift[m_itr + 2] < 0 ? 1 : (1 << p_out_shift[m_itr + 2]);
      p_left_mult[0] = p_out_shift[m_itr + 3] < 0 ? 1 : (1 << p_out_shift[m_itr + 3]);

      p_right_mult[3] = p_out_shift[m_itr + 0] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 0]));
      p_right_mult[2] = p_out_shift[m_itr + 1] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 1]));
      p_right_mult[1] = p_out_shift[m_itr + 2] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 2]));
      p_right_mult[0] = p_out_shift[m_itr + 3] > 0 ? (0xFFFFFFFF << 31) : (0xFFFFFFFF << ( 31 + p_out_shift[m_itr + 3]));

      p_out_mult[3] = -p_out_multiplier[m_itr + 0];
      p_out_mult[2] = -p_out_multiplier[m_itr + 1];
      p_out_mult[1] = -p_out_multiplier[m_itr + 2];
      p_out_mult[0] = -p_out_multiplier[m_itr + 3];

      p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;

      ae_int8* p_vec_0  = (ae_int8 *)(p_vec1);

      for (vec_itr = 0; vec_itr < (vec_count & ~(4 - 1)); vec_itr += 4)
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
        AE_L32X2X2_IP(acc_row0_vec0, acc_row1_vec0, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec1, acc_row1_vec1, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec2, acc_row1_vec2, (ae_int32x4*)acc_buff, 0);
        AE_L32X2X2_IP(acc_row0_vec3, acc_row1_vec3, (ae_int32x4*)acc_buff, 0);

        ae_int8x8 vec0_batch_0, vec0_batch_1, vec0_batch_2, vec0_batch_3; 
        ae_int8x8 vec1_batch_0, vec1_batch_1, vec1_batch_2, vec1_batch_3; 
        ae_int8x8 vec2_batch_0, vec2_batch_1, vec2_batch_2, vec2_batch_3; 
        ae_int8x8 vec3_batch_0, vec3_batch_1, vec3_batch_2, vec3_batch_3; 

        /* Load  4 vectors  */
        AE_L8X8X2_IP(vec0_batch_0, vec0_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec0_batch_2, vec0_batch_3, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec1_batch_0, vec1_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec1_batch_2, vec1_batch_3, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec2_batch_0, vec2_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec2_batch_2, vec2_batch_3, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec3_batch_0, vec3_batch_1, (ae_int8x16*)p_vec_0, 16);
        AE_L8X8X2_IP(vec3_batch_2, vec3_batch_3, (ae_int8x16*)p_vec_0, 16);

        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , mat1_row3_0 , mat1_row2_0 , mat1_row1_0 , mat1_row0_0 ,vec0_batch_0);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , mat1_row3_0 , mat1_row2_0 , mat1_row1_0 , mat1_row0_0 ,vec1_batch_0);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , mat1_row3_0 , mat1_row2_0 , mat1_row1_0 , mat1_row0_0 ,vec2_batch_0);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , mat1_row3_0 , mat1_row2_0 , mat1_row1_0 , mat1_row0_0 ,vec3_batch_0);

        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , mat1_row3_1 , mat1_row2_1 , mat1_row1_1 , mat1_row0_1 ,vec0_batch_1);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , mat1_row3_1 , mat1_row2_1 , mat1_row1_1 , mat1_row0_1 ,vec1_batch_1);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , mat1_row3_1 , mat1_row2_1 , mat1_row1_1 , mat1_row0_1 ,vec2_batch_1);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , mat1_row3_1 , mat1_row2_1 , mat1_row1_1 , mat1_row0_1 ,vec3_batch_1);

        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , mat1_row3_2 , mat1_row2_2 , mat1_row1_2 , mat1_row0_2 ,vec0_batch_2);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , mat1_row3_2 , mat1_row2_2 , mat1_row1_2 , mat1_row0_2 ,vec1_batch_2);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , mat1_row3_2 , mat1_row2_2 , mat1_row1_2 , mat1_row0_2 ,vec2_batch_2);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , mat1_row3_2 , mat1_row2_2 , mat1_row1_2 , mat1_row0_2 ,vec3_batch_2);

        AE_MULA8Q8X8(acc_row1_vec0 , acc_row0_vec0 , mat1_row3_3 , mat1_row2_3 , mat1_row1_3 , mat1_row0_3 ,vec0_batch_3);
        AE_MULA8Q8X8(acc_row1_vec1 , acc_row0_vec1 , mat1_row3_3 , mat1_row2_3 , mat1_row1_3 , mat1_row0_3 ,vec1_batch_3);
        AE_MULA8Q8X8(acc_row1_vec2 , acc_row0_vec2 , mat1_row3_3 , mat1_row2_3 , mat1_row1_3 , mat1_row0_3 ,vec2_batch_3);
        AE_MULA8Q8X8(acc_row1_vec3 , acc_row0_vec3 , mat1_row3_3 , mat1_row2_3 , mat1_row1_3 , mat1_row0_3 ,vec3_batch_3);

        /* Apply quantization */
        ae_int16x4 out_0, out_1, out_2, out_3;
        ae_int32x2 l_mult_32, l_mult_10, r_mult_32, r_mult_10;
        ae_int32x2 out_multiplier_10, out_multiplier_32;
        AE_L32X2X2_IP(l_mult_32, l_mult_10, (ae_int32x4 *)ptr_left_mult, 0);
        AE_L32X2X2_IP(r_mult_32, r_mult_10, (ae_int32x4 *)ptr_right_mult, 0);
        AE_L32X2X2_IP(out_multiplier_32, out_multiplier_10, (ae_int32x4 *)ptr_out_mult, 0);

        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_SP_32(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_32, out_multiplier_10, l_mult_32, l_mult_10, r_mult_32, r_mult_10, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_SP_32(out_1, acc_row0_vec1, acc_row1_vec1, out_multiplier_32, out_multiplier_10, l_mult_32, l_mult_10, r_mult_32, r_mult_10, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_SP_32(out_2, acc_row0_vec2, acc_row1_vec2, out_multiplier_32, out_multiplier_10, l_mult_32, l_mult_10, r_mult_32, r_mult_10, out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2_SP_32(out_3, acc_row0_vec3, acc_row1_vec3, out_multiplier_32, out_multiplier_10, l_mult_32, l_mult_10, r_mult_32, r_mult_10, out_zero_bias);

        /* Store output */
        ae_int8x8 out32_0, out32_1; 
        out32_0 = AE_SAT8X8X16(out_0, out_1);
        out32_1 = AE_SAT8X8X16(out_2, out_3);

        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_0), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_H_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);
        AE_S32_L_XP(AE_MOVINT32X2_FROMINT8X8(out32_1), (ae_int32 *)p_dst_0, out_offset);

      }
      /*
         for (; vec_itr < vec_count; vec_itr++)
         {
         }
       */
    }
    return 0;
  } 

  /* Special case for cols == 64, 128, 256 */
  if(((cols1 & 0x1f) == 0) &&
     (cols1 <= 256) &&
     ((row_stride1 & 0x1f) == 0) &&
     ((vec_offset & 0x1f) == 0) &&
     ALIGNED_PTR(p_vec1, 16) &&
     ALIGNED_PTR(p_out, 4) &&
     (out_stride == 1) &&   // NHWC case
     ((out_offset & 0x3) == 0) &&   // NHWC case
     ((rows & 0x3) == 0)
    )
  {
    special_function_for_cols_mul_32
      (p_out,
       p_mat1,
       p_vec1,
       p_bias,
       rows,
       vec_count,
       cols1,
       p_out_multiplier,
       p_out_shift,
       vec1_zero_bias,
       out_zero_bias,
       out_stride,
       row_stride1,
       vec_offset,
       out_offset
      );

    return 0;
  }

  if(
      ALIGNED_PTR(p_mat1, 16) &&
      ALIGNED_PTR(p_vec1, 16) &&
      ((row_stride1 & 15) == 0) &&
      ((vec_offset & 15) == 0)
      )
  {
    ae_int8x8 mat1_row0_0, mat1_row0_1;
    ae_int8x8 mat1_row1_0, mat1_row1_1;
    ae_int8x8 mat1_row2_0, mat1_row2_1;
    ae_int8x8 mat1_row3_0, mat1_row3_1;

    int rem_cols = cols1 & 15;
    int rem_cols_shift_0 = ((rem_cols) <= 8)?(8 - (rem_cols)) * 8:0;
    int rem_cols_shift_1 = ((rem_cols) > 8)?(16 - (rem_cols)) * 8:64;
    for(m_itr = 0; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_stride1); 
      ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_stride1); 
      ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_stride1); 

      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      int cols_count = cols1 - rem_cols;
      for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
      {
        AE_L8X8X2_IP(mat1_row0_0, mat1_row0_1, (ae_int8x16 *)p_mat1_0, 16);
        AE_L8X8X2_IP(mat1_row1_0, mat1_row1_1, (ae_int8x16 *)p_mat1_1, 16);
        AE_L8X8X2_IP(mat1_row2_0, mat1_row2_1, (ae_int8x16 *)p_mat1_2, 16);
        AE_L8X8X2_IP(mat1_row3_0, mat1_row3_1, (ae_int8x16 *)p_mat1_3, 16);

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1 , mat1_row1_1 , mat1_row2_1 , mat1_row3_1 , vec_z_b);
      }

      c_itr <<= 4;
      //Remainder loop for cols1
      while(c_itr < cols1)
      {
        AE_L8X8_IP(mat1_row0_0, (ae_int8x8 *)p_mat1_0, 8);
        AE_L8X8_IP(mat1_row1_0, (ae_int8x8 *)p_mat1_1, 8);
        AE_L8X8_IP(mat1_row2_0, (ae_int8x8 *)p_mat1_2, 8);
        AE_L8X8_IP(mat1_row3_0, (ae_int8x8 *)p_mat1_3, 8);

        mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift_0), rem_cols_shift_0));

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row1_0 , mat1_row2_0 , mat1_row3_0 , vec_z_b);

        c_itr += 8;
        rem_cols_shift_0 = rem_cols_shift_1;
      }

      acc_row0 = AE_SUB32S(AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]), acc_row0);
      acc_row1 = AE_SUB32S(AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]), acc_row1);

      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);

      WORD8* p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;
      WORD8* p_dst_1 = (WORD8*)p_out + (m_itr + 1) * out_stride;
      WORD8* p_dst_2 = (WORD8*)p_out + (m_itr + 2) * out_stride;
      WORD8* p_dst_3 = (WORD8*)p_out + (m_itr + 3) * out_stride;

      p_left_shift[0] = p_out_shift[m_itr + 0] < 0 ? 0 : p_out_shift[m_itr + 0];
      p_left_shift[1] = p_out_shift[m_itr + 1] < 0 ? 0 : p_out_shift[m_itr + 1];
      p_left_shift[2] = p_out_shift[m_itr + 2] < 0 ? 0 : p_out_shift[m_itr + 2];
      p_left_shift[3] = p_out_shift[m_itr + 3] < 0 ? 0 : p_out_shift[m_itr + 3];

      p_right_shift[0] = p_out_shift[m_itr + 0] > 0 ? 0 : -p_out_shift[m_itr + 0];
      p_right_shift[1] = p_out_shift[m_itr + 1] > 0 ? 0 : -p_out_shift[m_itr + 1];
      p_right_shift[2] = p_out_shift[m_itr + 2] > 0 ? 0 : -p_out_shift[m_itr + 2];
      p_right_shift[3] = p_out_shift[m_itr + 3] > 0 ? 0 : -p_out_shift[m_itr + 3];

      ae_int32x2 l_mult_3 = AE_MOVDA32(1 << p_left_shift[3]); 
      ae_int32x2 l_mult_2 = AE_MOVDA32(1 << p_left_shift[2]);
      ae_int32x2 l_mult_1 = AE_MOVDA32(1 << p_left_shift[1]); 
      ae_int32x2 l_mult_0 = AE_MOVDA32(1 << p_left_shift[0]);

      ae_int32x2 l_mult_23 = AE_MOVDA32X2( 1 << p_left_shift[2], 1 << p_left_shift[3]); 
      ae_int32x2 l_mult_01 = AE_MOVDA32X2( 1 << p_left_shift[0], 1 << p_left_shift[1]);

      ae_int32x2 r_mult_23 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[2])), (0xFFFFFFFF << (31 - p_right_shift[3]))); 
      ae_int32x2 r_mult_01 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[0])), (0xFFFFFFFF << (31 - p_right_shift[1])));

      ae_int32x2 out_multiplier_23 = AE_MOVDA32X2(-p_out_multiplier[m_itr + 2], -p_out_multiplier[m_itr + 3]); 
      ae_int32x2 out_multiplier_01 = AE_MOVDA32X2(-p_out_multiplier[m_itr + 0], -p_out_multiplier[m_itr + 1]);

      vec_itr = 0;

      for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
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

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1]; 

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
           ,vec_offset
          );

        ae_int16x4 out_0, out_1, out_2, out_3;

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[m_itr + 0], l_mult_0, p_right_shift[0], out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[m_itr + 1], l_mult_1, p_right_shift[1], out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[m_itr + 2], l_mult_2, p_right_shift[2], out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[m_itr + 3], l_mult_3, p_right_shift[3], out_zero_bias);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
      }

      // Remaining vectors
      for (; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row1;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
          );

        ae_int16x4 out_0;

        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_offset);
      }
    }

    // remaining rows
    for(; m_itr < rows; m_itr++)
    {
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      int cols_count=cols1-(cols1&7);
#pragma no_unroll
      for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)
      {
        AE_L8X8_IP(mat1_row0_0, (ae_int8x8 *)p_mat1_0, 8);

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , vec_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_L8X8_IP(mat1_row0_0, (ae_int8x8 *)p_mat1_0, 8);

        mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , vec_z_b);
      }

      acc_row0 = AE_SUB32S(AE_MOVDA32(p_bias[m_itr + 0]), acc_row0);

      AE_S32X2X2_I(acc_row0, acc_row0, (ae_int32x4*)acc_buffer, 0);

      WORD8* p_dst = (WORD8*)p_out + (m_itr + 0) * out_stride;

      p_left_shift[0] = p_out_shift[m_itr + 0] < 0 ? 0 : p_out_shift[m_itr + 0];

      p_right_shift[0] = p_out_shift[m_itr + 0] > 0 ? 0 : -p_out_shift[m_itr + 0];

      ae_int32x2 l_mult = AE_MOVDA32( 1 << p_left_shift[0]);

      vec_itr = 0;

      for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row0_vec1;

        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);

        ae_int8x8* p_vec_0  = (ae_int8x8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8 *p_mat1_0 = (ae_int8*) &p_mat1[m_itr * row_stride1]; 

        _xa_nn_dot_product_4_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_vec_0
           ,p_mat1_0
           ,cols1
           ,vec_offset
          );

        ae_int16x4 out_0;

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_0, acc_row0_vec0, acc_row0_vec1, p_out_multiplier[m_itr], l_mult, p_right_shift[0], out_zero_bias);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
      }

      // Remaining vectors
      for (; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row0_vec1;

        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*) &p_mat1[m_itr * row_stride1]; 

        _xa_nn_dot_product_1_rows_1_vecs_aligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_mat1_0
           ,p_vec_0
           ,cols1
          );

        ae_int8x8 temp_vec0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[m_itr], p_left_shift[0], p_right_shift[0]);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
        temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_offset);
      }
    }
  }
  else if (p_mat1 && p_vec1)
  {
    ae_int8x8 mat1_row0_0, mat1_row0_1;
    ae_int8x8 mat1_row1_0, mat1_row1_1;
    ae_int8x8 mat1_row2_0, mat1_row2_1;
    ae_int8x8 mat1_row3_0, mat1_row3_1;

    int rem_cols = cols1 & 15;
    int rem_cols_shift_0 = ((rem_cols) <= 8)?(8 - (rem_cols)) * 8:0;
    int rem_cols_shift_1 = ((rem_cols) > 8)?(16 - (rem_cols)) * 8:64;
    for(m_itr = 0; m_itr < (rows & ~(32 - 1)); m_itr += 32)
    {
      for(ii = 0; ii < 8; ii++)
      {
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + ii + 0) * row_stride1]; 
        ae_int8x8* p_mat1_1 = p_mat1_0 + row_stride1; //next 8th row 
        ae_int8x8* p_mat1_2 = p_mat1_1 + row_stride1; //next 8th row
        ae_int8x8* p_mat1_3 = p_mat1_2 + row_stride1; //next 8th row 

        ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
        ae_valignx2 align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
        ae_valignx2 align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
        ae_valignx2 align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

        ae_int32x2 acc_row0 = ZERO32; 
        ae_int32x2 acc_row1 = ZERO32;

        int cols_count=cols1 - rem_cols;
        for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
        {
          AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
          AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
          AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
          AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

          AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec_z_b);
          AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec_z_b);
        }

        //Remainder loop for cols1
        if(cols_count!=cols1)
        {
          AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
          AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
          AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
          AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

          mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift_0), rem_cols_shift_0));
          mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift_0), rem_cols_shift_0));
          mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift_0), rem_cols_shift_0));
          mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift_0), rem_cols_shift_0));

          AE_MULA8Q8X8(acc_row0, acc_row1, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec_z_b);

          if(rem_cols > 8)
          {
            mat1_row0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_1), rem_cols_shift_1), rem_cols_shift_1));
            mat1_row1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_1), rem_cols_shift_1), rem_cols_shift_1));
            mat1_row2_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_1), rem_cols_shift_1), rem_cols_shift_1));
            mat1_row3_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_1), rem_cols_shift_1), rem_cols_shift_1));

            AE_MULA8Q8X8(acc_row0, acc_row1, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec_z_b);
          }
        }

        acc_row0 = AE_SUB32S(AE_MOVDA32X2(p_bias[m_itr + ii +  0], p_bias[m_itr + ii +  8]), acc_row0);
        acc_row1 = AE_SUB32S(AE_MOVDA32X2(p_bias[m_itr + ii + 16], p_bias[m_itr + ii + 24]), acc_row1);

        AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
        AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);

        WORD8* p_dst_0 = (WORD8*)p_out + (m_itr + ii +  0) * out_stride;
        WORD8* p_dst_1 = (WORD8*)p_out + (m_itr + ii +  8) * out_stride;
        WORD8* p_dst_2 = (WORD8*)p_out + (m_itr + ii + 16) * out_stride;
        WORD8* p_dst_3 = (WORD8*)p_out + (m_itr + ii + 24) * out_stride;

        p_left_shift[0] = p_out_shift[m_itr + ii +  0] < 0 ? 0 : p_out_shift[m_itr + ii +  0];
        p_left_shift[1] = p_out_shift[m_itr + ii +  8] < 0 ? 0 : p_out_shift[m_itr + ii +  8];
        p_left_shift[2] = p_out_shift[m_itr + ii + 16] < 0 ? 0 : p_out_shift[m_itr + ii + 16];
        p_left_shift[3] = p_out_shift[m_itr + ii + 24] < 0 ? 0 : p_out_shift[m_itr + ii + 24];

        p_right_shift[0] = p_out_shift[m_itr + ii +  0] > 0 ? 0 : -p_out_shift[m_itr + ii +  0];
        p_right_shift[1] = p_out_shift[m_itr + ii +  8] > 0 ? 0 : -p_out_shift[m_itr + ii +  8];
        p_right_shift[2] = p_out_shift[m_itr + ii + 16] > 0 ? 0 : -p_out_shift[m_itr + ii + 16];
        p_right_shift[3] = p_out_shift[m_itr + ii + 24] > 0 ? 0 : -p_out_shift[m_itr + ii + 24];

        ae_int32x2 l_mult_3 = AE_MOVDA32(1 << p_left_shift[3]); 
        ae_int32x2 l_mult_2 = AE_MOVDA32(1 << p_left_shift[2]);
        ae_int32x2 l_mult_1 = AE_MOVDA32(1 << p_left_shift[1]); 
        ae_int32x2 l_mult_0 = AE_MOVDA32(1 << p_left_shift[0]);

        ae_int32x2 l_mult_23 = AE_MOVDA32X2(1 << p_left_shift[2], 1 << p_left_shift[3]); 
        ae_int32x2 l_mult_01 = AE_MOVDA32X2(1 << p_left_shift[0], 1 << p_left_shift[1]);

        ae_int32x2 r_mult_23 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[2])), (0xFFFFFFFF << (31 - p_right_shift[3]))); 
        ae_int32x2 r_mult_01 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[0])), (0xFFFFFFFF << (31 - p_right_shift[1])));

        ae_int32x2 out_multiplier_23 = AE_MOVDA32X2(-p_out_multiplier[m_itr + ii + 16], -p_out_multiplier[m_itr + ii + 24]); 
        ae_int32x2 out_multiplier_01 = AE_MOVDA32X2(-p_out_multiplier[m_itr + ii + 0], -p_out_multiplier[m_itr + ii + 8]);

        for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
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

          ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
          ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + ii + 0) * row_stride1]; 

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
             ,vec_offset
            );

          ae_int16x4 out_0, out_1, out_2, out_3;

          MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[m_itr + ii +  0], l_mult_0, p_right_shift[0], out_zero_bias);
          MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[m_itr + ii +  8], l_mult_1, p_right_shift[1], out_zero_bias);
          MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[m_itr + ii + 16], l_mult_2, p_right_shift[2], out_zero_bias);
          MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[m_itr + ii + 24], l_mult_3, p_right_shift[3], out_zero_bias);

          AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
          AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
          AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
          AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
          AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
          AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
          AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
          AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);

          AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
          AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
          AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
          AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
          AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
          AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
          AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
          AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        }

        // Remaining vectors
        for (; vec_itr < vec_count; vec_itr++)
        {
          ae_int32x2 acc_row0_vec0 = acc_row0;
          ae_int32x2 acc_row1_vec0 = acc_row1;

          WORD8* p_dst = (WORD8*)p_out + vec_itr * out_offset + (m_itr + ii) * out_stride;
          ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_offset);
          ae_int8x8* p_mat1_0 = (ae_int8x8*) &p_mat1[(m_itr + ii + 0)* row_stride1]; 

          _xa_nn_dot_product_4_rows_1_vecs_offset_aligned
            (&acc_row0_vec0
             ,&acc_row1_vec0
             ,p_mat1_0
             ,p_vec_0
             ,cols1
             ,row_stride1
            );

          ae_int16x4 out_0;

          MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
          AE_SW_S8_6_X(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst,  0 * out_stride);
          AE_SW_S8_4_X(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst,  8 * out_stride);
          AE_SW_S8_2_X(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, 16 * out_stride);
          AE_S8_0_X(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, 24 * out_stride);
        }
      }
    }

    // Process loop for 4 rows and 4 vectors 
    for(; m_itr < (rows & ~(4 - 1)); m_itr += 4)
    {
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_int8x8 *p_mat1_1 = (ae_int8x8*)((WORD8 *)p_mat1_0 + row_stride1); 
      ae_int8x8 *p_mat1_2 = (ae_int8x8*)((WORD8 *)p_mat1_1 + row_stride1); 
      ae_int8x8 *p_mat1_3 = (ae_int8x8*)((WORD8 *)p_mat1_2 + row_stride1); 

      ae_valignx2 align_p_mat1_0 = AE_LA128_PP(p_mat1_0);
      ae_valignx2 align_p_mat1_1 = AE_LA128_PP(p_mat1_1);
      ae_valignx2 align_p_mat1_2 = AE_LA128_PP(p_mat1_2);
      ae_valignx2 align_p_mat1_3 = AE_LA128_PP(p_mat1_3);

      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      int cols_count=cols1 - rem_cols;
      for(c_itr = 0; c_itr < (cols_count >> 4); c_itr++)
      {
        AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec_z_b);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_LA8X8X2_IP(mat1_row0_0, mat1_row0_1, align_p_mat1_0, (ae_int8x16 *)p_mat1_0);
        AE_LA8X8X2_IP(mat1_row1_0, mat1_row1_1, align_p_mat1_1, (ae_int8x16 *)p_mat1_1);
        AE_LA8X8X2_IP(mat1_row2_0, mat1_row2_1, align_p_mat1_2, (ae_int8x16 *)p_mat1_2);
        AE_LA8X8X2_IP(mat1_row3_0, mat1_row3_1, align_p_mat1_3, (ae_int8x16 *)p_mat1_3);

        mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row1_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row2_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_0), rem_cols_shift_0), rem_cols_shift_0));
        mat1_row3_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_0), rem_cols_shift_0), rem_cols_shift_0));

        AE_MULA8Q8X8(acc_row0, acc_row1, mat1_row0_0, mat1_row1_0, mat1_row2_0, mat1_row3_0, vec_z_b);

        if(rem_cols > 8)
        {
          mat1_row0_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_1), rem_cols_shift_1), rem_cols_shift_1));
          mat1_row1_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row1_1), rem_cols_shift_1), rem_cols_shift_1));
          mat1_row2_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row2_1), rem_cols_shift_1), rem_cols_shift_1));
          mat1_row3_1 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row3_1), rem_cols_shift_1), rem_cols_shift_1));

          AE_MULA8Q8X8(acc_row0, acc_row1, mat1_row0_1, mat1_row1_1, mat1_row2_1, mat1_row3_1, vec_z_b);
        }
      }

      acc_row0 = AE_SUB32S(AE_MOVDA32X2(p_bias[m_itr + 0], p_bias[m_itr + 1]), acc_row0);
      acc_row1 = AE_SUB32S(AE_MOVDA32X2(p_bias[m_itr + 2], p_bias[m_itr + 3]), acc_row1);

      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row0)), AE_MOVDA32(AE_MOVAD32_L(acc_row0)), (ae_int32x4*)acc_buffer, 0);
      AE_S32X2X2_I(AE_MOVDA32(AE_MOVAD32_H(acc_row1)), AE_MOVDA32(AE_MOVAD32_L(acc_row1)), (ae_int32x4*)acc_buffer, 16);

      WORD8* p_dst_0 = (WORD8*)p_out + (m_itr + 0) * out_stride;
      WORD8* p_dst_1 = (WORD8*)p_out + (m_itr + 1) * out_stride;
      WORD8* p_dst_2 = (WORD8*)p_out + (m_itr + 2) * out_stride;
      WORD8* p_dst_3 = (WORD8*)p_out + (m_itr + 3) * out_stride;

      p_left_shift[0] = p_out_shift[m_itr + 0] < 0 ? 0 : p_out_shift[m_itr + 0];
      p_left_shift[1] = p_out_shift[m_itr + 1] < 0 ? 0 : p_out_shift[m_itr + 1];
      p_left_shift[2] = p_out_shift[m_itr + 2] < 0 ? 0 : p_out_shift[m_itr + 2];
      p_left_shift[3] = p_out_shift[m_itr + 3] < 0 ? 0 : p_out_shift[m_itr + 3];

      p_right_shift[0] = p_out_shift[m_itr + 0] > 0 ? 0 : -p_out_shift[m_itr + 0];
      p_right_shift[1] = p_out_shift[m_itr + 1] > 0 ? 0 : -p_out_shift[m_itr + 1];
      p_right_shift[2] = p_out_shift[m_itr + 2] > 0 ? 0 : -p_out_shift[m_itr + 2];
      p_right_shift[3] = p_out_shift[m_itr + 3] > 0 ? 0 : -p_out_shift[m_itr + 3];

      ae_int32x2 l_mult_3 = AE_MOVDA32(1 << p_left_shift[3]); 
      ae_int32x2 l_mult_2 = AE_MOVDA32(1 << p_left_shift[2]);
      ae_int32x2 l_mult_1 = AE_MOVDA32(1 << p_left_shift[1]); 
      ae_int32x2 l_mult_0 = AE_MOVDA32(1 << p_left_shift[0]);

      ae_int32x2 l_mult_23 = AE_MOVDA32X2( 1 << p_left_shift[2], 1 << p_left_shift[3]); 
      ae_int32x2 l_mult_01 = AE_MOVDA32X2( 1 << p_left_shift[0], 1 << p_left_shift[1]);

      ae_int32x2 r_mult_23 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[2])), (0xFFFFFFFF << (31 - p_right_shift[3]))); 
      ae_int32x2 r_mult_01 = AE_MOVDA32X2( (0xFFFFFFFF << (31 - p_right_shift[0])), (0xFFFFFFFF << (31 - p_right_shift[1])));

      ae_int32x2 out_multiplier_23 = AE_MOVDA32X2(-p_out_multiplier[m_itr + 2], -p_out_multiplier[m_itr + 3]); 
      ae_int32x2 out_multiplier_01 = AE_MOVDA32X2(-p_out_multiplier[m_itr + 0], -p_out_multiplier[m_itr + 1]);

      vec_itr = 0;

      for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
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

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1]; 

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
           ,vec_offset
          );

        ae_int16x4 out_0, out_1, out_2, out_3;

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, p_out_multiplier[m_itr + 0], l_mult_0, p_right_shift[0], out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_1, acc_row0_vec1, acc_row1_vec1, p_out_multiplier[m_itr + 1], l_mult_1, p_right_shift[1], out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_2, acc_row0_vec2, acc_row1_vec2, p_out_multiplier[m_itr + 2], l_mult_2, p_right_shift[2], out_zero_bias);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_3, acc_row0_vec3, acc_row1_vec3, p_out_multiplier[m_itr + 3], l_mult_3, p_right_shift[3], out_zero_bias);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_1), (ae_int8 *) p_dst_1, out_offset);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_2), (ae_int8 *) p_dst_2, out_offset);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_3), (ae_int8 *) p_dst_3, out_offset);
      }

      // Remaining vectors
      for (; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0 = acc_row0;
        ae_int32x2 acc_row1_vec0 = acc_row1;

        ae_int8* p_vec_0  = (ae_int8 *)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row1_vec0
           ,p_mat1_0
           ,p_vec_0
           ,cols1
           ,row_stride1
          );

        ae_int16x4 out_0;

        MULTIPLYBYQUANTIZEDMULTIPLIER_per_chan_X2_X2(out_0, acc_row0_vec0, acc_row1_vec0, out_multiplier_23, out_multiplier_01, l_mult_23, l_mult_01, r_mult_23, r_mult_01, out_zero_bias);
        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_0, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_1, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_2, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst_3, out_offset);
      }
    }

    // remaining rows
    for(; m_itr < rows; m_itr++)
    {
      ae_int8x8 *p_mat1_0 = (ae_int8x8 *) &p_mat1[(m_itr + 0) * row_stride1];
      ae_valign align_p_mat1_0 = AE_LA64_PP(p_mat1_0);

      ae_int32x2 acc_row0 = ZERO32; 
      ae_int32x2 acc_row1 = ZERO32;

      int cols_count=cols1-(cols1&7);
      for(c_itr = 0; c_itr < (cols_count >> 3); c_itr++)
      {
        AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);
        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , vec_z_b);
      }

      //Remainder loop for cols1
      if(cols_count!=cols1)
      {
        AE_LA8X8_IP(mat1_row0_0, align_p_mat1_0, p_mat1_0);

        mat1_row0_0 = AE_MOVINT8X8_FROMINT64(AE_SLAA64(AE_SRLA64(AE_MOVINT64_FROMINT8X8(mat1_row0_0), rem_cols_shift), rem_cols_shift));

        AE_MULA8Q8X8(acc_row0 , acc_row1 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , mat1_row0_0 , vec_z_b);
      }

      acc_row0 = AE_SUB32S(AE_MOVDA32(p_bias[m_itr + 0]), acc_row0);

      AE_S32X2X2_I(acc_row0, acc_row0, (ae_int32x4*)acc_buffer, 0);

      WORD8* p_dst = (WORD8*)p_out + (m_itr + 0) * out_stride;

      p_left_shift[0] = p_out_shift[m_itr + 0] < 0 ? 0 : p_out_shift[m_itr + 0];

      p_right_shift[0] = p_out_shift[m_itr + 0] > 0 ? 0 : -p_out_shift[m_itr + 0];

      ae_int32x2 l_mult = AE_MOVDA32( 1 << p_left_shift[0]);

      vec_itr = 0;

      for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row0_vec1;

        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);

        ae_int8x8* p_vec_0  = (ae_int8x8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8 *p_mat1_0 = (ae_int8*) &p_mat1[m_itr * row_stride1]; 

        _xa_nn_dot_product_4_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_vec_0
           ,p_mat1_0
           ,cols1
           ,vec_offset
          );

        ae_int16x4 out_0;

        MULTIPLYBYQUANTIZEDMULTIPLIER_X2_X2(out_0, acc_row0_vec0, acc_row0_vec1, p_out_multiplier[m_itr], l_mult, p_right_shift[0], out_zero_bias);

        AE_SW_S8_6_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_SW_S8_4_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_SW_S8_2_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
        AE_S8_0_XP(AE_MOVINT8X8_FROMINT16X4(out_0), (ae_int8 *) p_dst, out_offset);
      }

      // Remaining vectors
      for (; vec_itr < vec_count; vec_itr++)
      {
        ae_int32x2 acc_row0_vec0;
        ae_int32x2 acc_row0_vec1;

        /* Initialize accumulators */
        AE_L32X2X2_I(acc_row0_vec0, acc_row0_vec1, (ae_int32x4*)acc_buffer, 0);

        ae_int8* p_vec_0  = (ae_int8*)(p_vec1 + vec_itr * vec_offset);
        ae_int8x8 *p_mat1_0 = (ae_int8x8*) &p_mat1[m_itr * row_stride1]; 

        _xa_nn_dot_product_1_rows_1_vecs_unaligned
          (&acc_row0_vec0
           ,&acc_row0_vec1
           ,p_mat1_0
           ,p_vec_0
           ,cols1
          );

        ae_int8x8 temp_vec0;
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc_row0_vec0, p_out_multiplier[m_itr], p_left_shift[0], p_right_shift[0]);
        acc_row0_vec0 = AE_ADD32S(acc_row0_vec0, out_zero_bias);
        acc_row0_vec0 = AE_MAX32(acc_row0_vec0, min_int8);
        temp_vec0 = AE_SAT8X4X32_L(acc_row0_vec0, acc_row0_vec0);
        AE_S8_0_XP(temp_vec0, (ae_int8 *) p_dst, out_offset);
      }
    }
  }
  else
  {
    return -1;
  }
    return 0;
}
