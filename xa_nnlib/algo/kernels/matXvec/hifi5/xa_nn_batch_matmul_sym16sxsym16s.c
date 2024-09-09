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
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_quant_macros_hifi5.h"

extern WORD32 xa_nn_matmul_sym16sxsym16s_sym16s(
  WORD16 * __restrict__ p_out,
  const WORD16 * __restrict__ p_mat1,
  const WORD16 * __restrict__ p_vec1,
  const WORD64 * __restrict__ p_bias,
  WORD32 rows,
  WORD32 cols1,
  WORD32 row_stride1,
  WORD32 vec_count,
  WORD32 vec_offset,
  WORD32 out_offset,
  WORD32 out_stride,
  WORD32 mat1_zero_bias,
  WORD32 vec1_zero_bias,
  WORD32 out_multiplier,
  WORD32 out_shift,
  WORD32 out_zero_bias);

WORD32 xa_nn_matmul_mat1_trans_sym16sxsym16s_sym16s(
  WORD16 * __restrict__ p_out,
  const WORD16 * __restrict__ p_mat1,
  const WORD16 * __restrict__ p_vec1,
  WORD32 rows,
  WORD32 cols1,
  WORD32 row_stride1,
  WORD32 vec_count,
  WORD32 vec_offset,
  WORD32 out_offset,
  WORD32 out_stride,
  WORD32 out_multiplier,
  WORD32 out_shift)
{
  int m_itr, vec_itr, c_itr;

  for(vec_itr = 0; vec_itr < vec_count - 1; vec_itr+=2)
  {
    for(m_itr = 0; m_itr < (rows & ~(8-1)); m_itr+=8)
    {
      ae_int64 acc64_00 = AE_ZERO64();
      ae_int64 acc64_10 = AE_ZERO64();
      ae_int64 acc64_20 = AE_ZERO64();
      ae_int64 acc64_30 = AE_ZERO64();
      ae_int64 acc64_40 = AE_ZERO64();
      ae_int64 acc64_50 = AE_ZERO64();
      ae_int64 acc64_60 = AE_ZERO64();
      ae_int64 acc64_70 = AE_ZERO64();

      ae_int64 acc64_01 = AE_ZERO64();
      ae_int64 acc64_11 = AE_ZERO64();
      ae_int64 acc64_21 = AE_ZERO64();
      ae_int64 acc64_31 = AE_ZERO64();
      ae_int64 acc64_41 = AE_ZERO64();
      ae_int64 acc64_51 = AE_ZERO64();
      ae_int64 acc64_61 = AE_ZERO64();
      ae_int64 acc64_71 = AE_ZERO64();

      ae_int32x2 acc_row01_vec0, acc_row23_vec0, acc_row45_vec0, acc_row67_vec0;
      ae_int32x2 acc_row01_vec1, acc_row23_vec1, acc_row45_vec1, acc_row67_vec1;

      ae_int16x4* p_vec_0 = (ae_int16x4 *)(p_vec1 + vec_itr * vec_offset);
      ae_valign vec0_a = AE_LA64_PP(p_vec_0);
      ae_int16x4* p_vec_1 = (ae_int16x4 *)(p_vec1 + (vec_itr + 1) * vec_offset);
      ae_valign vec1_a = AE_LA64_PP(p_vec_1);

      ae_int16x8* p_mat1_0 = (ae_int16x8 *)(p_mat1 + m_itr);
      ae_int16x8* p_mat1_1 = (ae_int16x8 *)(p_mat1 + m_itr + rows);

      ae_valignx2 mat1_0_a, mat1_1_a;
      ae_int16x4 d_mat1_0, d_mat1_4;
      ae_int16x4 d_mat1_1, d_mat1_5;
      ae_int16x4 d_mat1_2, d_mat1_6;
      ae_int16x4 d_mat1_3, d_mat1_7;
      ae_int16x4 d_vec0, d_vec1;

      ae_int16x4 sel_1 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050301,  0x06040200));
      ae_int16x4 sel_2 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050604,  0x03010200));
      WORD32 mat_offset = (rows << 1) - 8;
      for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
      {
        ae_int16x4 d0, d1, d2, d3, d4, d5, d6, d7;

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        mat1_1_a = AE_LA128_PP(p_mat1_1);
        AE_LA16X4X2_IP(d_mat1_0, d_mat1_4, mat1_0_a, p_mat1_0);
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + mat_offset);
        AE_LA16X4X2_IP(d_mat1_1, d_mat1_5, mat1_1_a, p_mat1_1);
        p_mat1_1 = (ae_int16x8*)((ae_int16 *)p_mat1_1 + mat_offset);

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        mat1_1_a = AE_LA128_PP(p_mat1_1);
        AE_LA16X4X2_IP(d_mat1_2, d_mat1_6, mat1_0_a, p_mat1_0);
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + mat_offset);
        AE_LA16X4X2_IP(d_mat1_3, d_mat1_7, mat1_1_a, p_mat1_1);
        p_mat1_1 = (ae_int16x8*)((ae_int16 *)p_mat1_1 + mat_offset);

        AE_DSEL16X4(d0, d1, d_mat1_0, d_mat1_1, sel_1);
        AE_DSEL16X4(d2, d3, d_mat1_2, d_mat1_3, sel_1);

        AE_DSEL16X4(d4, d5, d_mat1_4, d_mat1_5, sel_1);
        AE_DSEL16X4(d6, d7, d_mat1_6, d_mat1_7, sel_1);

        AE_DSEL16X4(d_mat1_0, d_mat1_1, d0, d2, sel_2);
        AE_DSEL16X4(d_mat1_2, d_mat1_3, d1, d3, sel_2);
        AE_DSEL16X4(d_mat1_4, d_mat1_5, d4, d6, sel_2);
        AE_DSEL16X4(d_mat1_6, d_mat1_7, d5, d7, sel_2);

        AE_LA16X4_IP(d_vec0, vec0_a, p_vec_0);
        AE_LA16X4_IP(d_vec1, vec1_a, p_vec_1);

        AE_MULAAAA2Q16(acc64_00, acc64_10, d_mat1_0, d_mat1_1, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_20, acc64_30, d_mat1_2, d_mat1_3, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_40, acc64_50, d_mat1_4, d_mat1_5, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_60, acc64_70, d_mat1_6, d_mat1_7, d_vec0, d_vec0);

        AE_MULAAAA2Q16(acc64_01, acc64_11, d_mat1_0, d_mat1_1, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_21, acc64_31, d_mat1_2, d_mat1_3, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_41, acc64_51, d_mat1_4, d_mat1_5, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_61, acc64_71, d_mat1_6, d_mat1_7, d_vec1, d_vec1);
      }

      if((cols1 & 3) != 0)
      {
        WORD32 rem = (cols1 & 3);
        ae_int16x4 d0, d1, d2, d3, d4, d5, d6, d7;

        d_mat1_1 = d_mat1_5 = AE_ZERO16();
        d_mat1_2 = d_mat1_6 = AE_ZERO16();
        d_mat1_3 = d_mat1_7 = AE_ZERO16();

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        AE_LA16X4X2_IP(d_mat1_0, d_mat1_4, mat1_0_a, p_mat1_0);
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + mat_offset);

        if(rem >= 2)
        {
          mat1_1_a = AE_LA128_PP(p_mat1_1);
          AE_LA16X4X2_IP(d_mat1_1, d_mat1_5, mat1_1_a, p_mat1_1);
        }

        if(rem == 3)
        {
          mat1_0_a = AE_LA128_PP(p_mat1_0);
          AE_LA16X4X2_IP(d_mat1_2, d_mat1_6, mat1_0_a, p_mat1_0);
        }

        AE_DSEL16X4(d0, d1, d_mat1_0, d_mat1_1, sel_1);
        AE_DSEL16X4(d2, d3, d_mat1_2, d_mat1_3, sel_1);

        AE_DSEL16X4(d4, d5, d_mat1_4, d_mat1_5, sel_1);
        AE_DSEL16X4(d6, d7, d_mat1_6, d_mat1_7, sel_1);

        AE_DSEL16X4(d_mat1_0, d_mat1_1, d0, d2, sel_2);
        AE_DSEL16X4(d_mat1_2, d_mat1_3, d1, d3, sel_2);
        AE_DSEL16X4(d_mat1_4, d_mat1_5, d4, d6, sel_2);
        AE_DSEL16X4(d_mat1_6, d_mat1_7, d5, d7, sel_2);

        ae_valignx2 vecx2_a = AE_LA128_PP(p_vec_0);
        AE_LAV16X4X2_XP(d_vec0, d0, vecx2_a, (ae_int16x8 *)p_vec_0, (rem << 1));

        ae_valignx2 vecx2_1_a = AE_LA128_PP(p_vec_1);
        AE_LAV16X4X2_XP(d_vec1, d1, vecx2_1_a, (ae_int16x8 *)p_vec_1, (rem << 1));

        AE_MULAAAA2Q16(acc64_00, acc64_10, d_mat1_0, d_mat1_1, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_20, acc64_30, d_mat1_2, d_mat1_3, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_40, acc64_50, d_mat1_4, d_mat1_5, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_60, acc64_70, d_mat1_6, d_mat1_7, d_vec0, d_vec0);

        AE_MULAAAA2Q16(acc64_01, acc64_11, d_mat1_0, d_mat1_1, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_21, acc64_31, d_mat1_2, d_mat1_3, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_41, acc64_51, d_mat1_4, d_mat1_5, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_61, acc64_71, d_mat1_6, d_mat1_7, d_vec1, d_vec1);
      }

      ae_int16x4 out_0, out_1, out_2, out_3;
#if XCHAL_HAVE_HIFI5S
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row23_vec0, acc64_20, acc64_30, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row45_vec0, acc64_40, acc64_50, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row67_vec0, acc64_60, acc64_70, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec1, acc64_01, acc64_11, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row23_vec1, acc64_21, acc64_31, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row45_vec1, acc64_41, acc64_51, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row67_vec1, acc64_61, acc64_71, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row23_vec0, acc64_20, acc64_30, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row45_vec0, acc64_40, acc64_50, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row67_vec0, acc64_60, acc64_70, out_multiplier, out_shift);

      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec1, acc64_01, acc64_11, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row23_vec1, acc64_21, acc64_31, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row45_vec1, acc64_41, acc64_51, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row67_vec1, acc64_61, acc64_71, out_multiplier, out_shift);
#endif

      out_0 = AE_SAT16X4(acc_row01_vec0, acc_row23_vec0);
      out_1 = AE_SAT16X4(acc_row45_vec0, acc_row67_vec0);

      out_2 = AE_SAT16X4(acc_row01_vec1, acc_row23_vec1);
      out_3 = AE_SAT16X4(acc_row45_vec1, acc_row67_vec1);

      if(out_stride == 1)
      {
        ae_int16x8* p_dst_0 = (ae_int16x8 *)((WORD16*)p_out + vec_itr * out_offset + m_itr * out_stride);

        ae_int16x8* p_dst_1 = (ae_int16x8 *)((WORD16*)p_out + (vec_itr + 1) * out_offset + m_itr * out_stride);

        ae_valignx2 dst_0_align = AE_ZALIGN128();
        ae_valignx2 dst_1_align = AE_ZALIGN128();
        AE_SA16X4X2_IP(out_0, out_1, dst_0_align, p_dst_0);
        AE_SA16X4X2_IP(out_2, out_3, dst_1_align, p_dst_1);
        AE_SA128POS_FP(dst_0_align, p_dst_0);
        AE_SA128POS_FP(dst_1_align, p_dst_1);
      }
      else // out_offset == 1
      {
        ae_int16* p_dst_0 = (ae_int16 *)((WORD16*)p_out + vec_itr * out_offset + m_itr * out_stride);

        ae_int16* p_dst_1 = (ae_int16 *)((WORD16*)p_out + (vec_itr + 1) * out_offset + m_itr * out_stride);

        *p_dst_0 = AE_SEL16_6543(out_0, out_0);     p_dst_0 += out_stride;
        *p_dst_0 = AE_SEL16_5432(out_0, out_0);     p_dst_0 += out_stride;
        *p_dst_0 = AE_SEL16_4321(out_0, out_0);     p_dst_0 += out_stride;
        *p_dst_0 = out_0;                           p_dst_0 += out_stride;

        *p_dst_0 = AE_SEL16_6543(out_1, out_1);     p_dst_0 += out_stride;
        *p_dst_0 = AE_SEL16_5432(out_1, out_1);     p_dst_0 += out_stride;
        *p_dst_0 = AE_SEL16_4321(out_1, out_1);     p_dst_0 += out_stride;
        *p_dst_0 = out_1;                           p_dst_0 += out_stride;

        *p_dst_1 = AE_SEL16_6543(out_2, out_2);     p_dst_1 += out_stride;
        *p_dst_1 = AE_SEL16_5432(out_2, out_2);     p_dst_1 += out_stride;
        *p_dst_1 = AE_SEL16_4321(out_2, out_2);     p_dst_1 += out_stride;
        *p_dst_1 = out_2;                           p_dst_1 += out_stride;

        *p_dst_1 = AE_SEL16_6543(out_3, out_3);     p_dst_1 += out_stride;
        *p_dst_1 = AE_SEL16_5432(out_3, out_3);     p_dst_1 += out_stride;
        *p_dst_1 = AE_SEL16_4321(out_3, out_3);     p_dst_1 += out_stride;
        *p_dst_1 = out_3;                           p_dst_1 += out_stride;
      }
    }

    if(m_itr < rows)
    {
      WORD32 rem_rows = rows - m_itr;

      ae_int64 acc64_00 = AE_ZERO64();
      ae_int64 acc64_10 = AE_ZERO64();
      ae_int64 acc64_20 = AE_ZERO64();
      ae_int64 acc64_30 = AE_ZERO64();
      ae_int64 acc64_40 = AE_ZERO64();
      ae_int64 acc64_50 = AE_ZERO64();
      ae_int64 acc64_60 = AE_ZERO64();
      ae_int64 acc64_70 = AE_ZERO64();

      ae_int64 acc64_01 = AE_ZERO64();
      ae_int64 acc64_11 = AE_ZERO64();
      ae_int64 acc64_21 = AE_ZERO64();
      ae_int64 acc64_31 = AE_ZERO64();
      ae_int64 acc64_41 = AE_ZERO64();
      ae_int64 acc64_51 = AE_ZERO64();
      ae_int64 acc64_61 = AE_ZERO64();
      ae_int64 acc64_71 = AE_ZERO64();

      ae_int32x2 acc_row01_vec0, acc_row23_vec0, acc_row45_vec0, acc_row67_vec0;
      ae_int32x2 acc_row01_vec1, acc_row23_vec1, acc_row45_vec1, acc_row67_vec1;

      ae_int16x4* p_vec_0 = (ae_int16x4 *)(p_vec1 + vec_itr * vec_offset);
      ae_valign vec0_a = AE_LA64_PP(p_vec_0);
      ae_int16x4* p_vec_1 = (ae_int16x4 *)(p_vec1 + (vec_itr + 1) * vec_offset);
      ae_valign vec1_a = AE_LA64_PP(p_vec_1);

      ae_int16x8* p_mat1_0 = (ae_int16x8 *)(p_mat1 + m_itr);
      ae_int16x8* p_mat1_1 = (ae_int16x8 *)(p_mat1 + m_itr + rows);

      ae_valignx2 mat1_0_a, mat1_1_a;
      ae_int16x4 d_mat1_0, d_mat1_4;
      ae_int16x4 d_mat1_1, d_mat1_5;
      ae_int16x4 d_mat1_2, d_mat1_6;
      ae_int16x4 d_mat1_3, d_mat1_7;
      ae_int16x4 d_vec0, d_vec1;

      ae_int16x4 sel_1 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050301,  0x06040200));
      ae_int16x4 sel_2 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050604,  0x03010200));
      WORD32 mat_offset = (rows << 1) - rem_rows;
      for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
      {
        ae_int16x4 d0, d1, d2, d3, d4, d5, d6, d7;

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        mat1_1_a = AE_LA128_PP(p_mat1_1);
        AE_LAV16X4X2_XP(d_mat1_0, d_mat1_4, mat1_0_a, p_mat1_0, (rem_rows << 1));
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + mat_offset);
        AE_LAV16X4X2_XP(d_mat1_1, d_mat1_5, mat1_1_a, p_mat1_1, (rem_rows << 1));
        p_mat1_1 = (ae_int16x8*)((ae_int16 *)p_mat1_1 + mat_offset);

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        mat1_1_a = AE_LA128_PP(p_mat1_1);
        AE_LAV16X4X2_XP(d_mat1_2, d_mat1_6, mat1_0_a, p_mat1_0, (rem_rows << 1));
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + mat_offset);
        AE_LAV16X4X2_XP(d_mat1_3, d_mat1_7, mat1_1_a, p_mat1_1, (rem_rows << 1));
        p_mat1_1 = (ae_int16x8*)((ae_int16 *)p_mat1_1 + mat_offset);

        AE_DSEL16X4(d0, d1, d_mat1_0, d_mat1_1, sel_1);
        AE_DSEL16X4(d2, d3, d_mat1_2, d_mat1_3, sel_1);

        AE_DSEL16X4(d4, d5, d_mat1_4, d_mat1_5, sel_1);
        AE_DSEL16X4(d6, d7, d_mat1_6, d_mat1_7, sel_1);

        AE_DSEL16X4(d_mat1_0, d_mat1_1, d0, d2, sel_2);
        AE_DSEL16X4(d_mat1_2, d_mat1_3, d1, d3, sel_2);
        AE_DSEL16X4(d_mat1_4, d_mat1_5, d4, d6, sel_2);
        AE_DSEL16X4(d_mat1_6, d_mat1_7, d5, d7, sel_2);

        AE_LA16X4_IP(d_vec0, vec0_a, p_vec_0);
        AE_LA16X4_IP(d_vec1, vec1_a, p_vec_1);

        AE_MULAAAA2Q16(acc64_00, acc64_10, d_mat1_0, d_mat1_1, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_20, acc64_30, d_mat1_2, d_mat1_3, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_40, acc64_50, d_mat1_4, d_mat1_5, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_60, acc64_70, d_mat1_6, d_mat1_7, d_vec0, d_vec0);

        AE_MULAAAA2Q16(acc64_01, acc64_11, d_mat1_0, d_mat1_1, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_21, acc64_31, d_mat1_2, d_mat1_3, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_41, acc64_51, d_mat1_4, d_mat1_5, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_61, acc64_71, d_mat1_6, d_mat1_7, d_vec1, d_vec1);
      }

      if((cols1 & 3) != 0)
      {
        WORD32 rem = (cols1 & 3);
        ae_int16x4 d0, d1, d2, d3, d4, d5, d6, d7;

        d_mat1_1 = d_mat1_5 = AE_ZERO16();
        d_mat1_2 = d_mat1_6 = AE_ZERO16();
        d_mat1_3 = d_mat1_7 = AE_ZERO16();

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        AE_LAV16X4X2_XP(d_mat1_0, d_mat1_4, mat1_0_a, p_mat1_0, (rem_rows << 1));
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + (rows << 1) - rem_rows);

        if(rem >= 2)
        {
          mat1_1_a = AE_LA128_PP(p_mat1_1);
          AE_LAV16X4X2_XP(d_mat1_1, d_mat1_5, mat1_1_a, p_mat1_1, (rem_rows << 1));
        }

        if(rem == 3)
        {
          mat1_0_a = AE_LA128_PP(p_mat1_0);
          AE_LAV16X4X2_XP(d_mat1_2, d_mat1_6, mat1_0_a, p_mat1_0, (rem_rows << 1));
        }

        AE_DSEL16X4(d0, d1, d_mat1_0, d_mat1_1, sel_1);
        AE_DSEL16X4(d2, d3, d_mat1_2, d_mat1_3, sel_1);

        AE_DSEL16X4(d4, d5, d_mat1_4, d_mat1_5, sel_1);
        AE_DSEL16X4(d6, d7, d_mat1_6, d_mat1_7, sel_1);

        AE_DSEL16X4(d_mat1_0, d_mat1_1, d0, d2, sel_2);
        AE_DSEL16X4(d_mat1_2, d_mat1_3, d1, d3, sel_2);
        AE_DSEL16X4(d_mat1_4, d_mat1_5, d4, d6, sel_2);
        AE_DSEL16X4(d_mat1_6, d_mat1_7, d5, d7, sel_2);

        ae_valignx2 vecx2_a = AE_LA128_PP(p_vec_0);
        AE_LAV16X4X2_XP(d_vec0, d0, vecx2_a, (ae_int16x8 *)p_vec_0, (rem << 1));

        ae_valignx2 vecx2_1_a = AE_LA128_PP(p_vec_1);
        AE_LAV16X4X2_XP(d_vec1, d1, vecx2_1_a, (ae_int16x8 *)p_vec_1, (rem << 1));

        AE_MULAAAA2Q16(acc64_00, acc64_10, d_mat1_0, d_mat1_1, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_20, acc64_30, d_mat1_2, d_mat1_3, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_40, acc64_50, d_mat1_4, d_mat1_5, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_60, acc64_70, d_mat1_6, d_mat1_7, d_vec0, d_vec0);

        AE_MULAAAA2Q16(acc64_01, acc64_11, d_mat1_0, d_mat1_1, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_21, acc64_31, d_mat1_2, d_mat1_3, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_41, acc64_51, d_mat1_4, d_mat1_5, d_vec1, d_vec1);
        AE_MULAAAA2Q16(acc64_61, acc64_71, d_mat1_6, d_mat1_7, d_vec1, d_vec1);
      }

      ae_int16x4 out_0, out_1, out_2, out_3;
#if XCHAL_HAVE_HIFI5S
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row23_vec0, acc64_20, acc64_30, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row45_vec0, acc64_40, acc64_50, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row67_vec0, acc64_60, acc64_70, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec1, acc64_01, acc64_11, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row23_vec1, acc64_21, acc64_31, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row45_vec1, acc64_41, acc64_51, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row67_vec1, acc64_61, acc64_71, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row23_vec0, acc64_20, acc64_30, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row45_vec0, acc64_40, acc64_50, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row67_vec0, acc64_60, acc64_70, out_multiplier, out_shift);

      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec1, acc64_01, acc64_11, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row23_vec1, acc64_21, acc64_31, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row45_vec1, acc64_41, acc64_51, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row67_vec1, acc64_61, acc64_71, out_multiplier, out_shift);
#endif

      out_0 = AE_SAT16X4(acc_row01_vec0, acc_row23_vec0);
      out_1 = AE_SAT16X4(acc_row45_vec0, acc_row67_vec0);

      out_2 = AE_SAT16X4(acc_row01_vec1, acc_row23_vec1);
      out_3 = AE_SAT16X4(acc_row45_vec1, acc_row67_vec1);

      if(out_stride == 1)
      {
        ae_int16x8* p_dst_0 = (ae_int16x8 *)((WORD16*)p_out + vec_itr * out_offset + m_itr * out_stride);
        ae_int16x8* p_dst_1 = (ae_int16x8 *)((WORD16*)p_out + (vec_itr + 1) * out_offset + m_itr * out_stride);

        ae_valignx2 dst_0_align = AE_ZALIGN128();
        ae_valignx2 dst_1_align = AE_ZALIGN128();
        AE_SAV16X4X2_XP(out_0, out_1, dst_0_align, p_dst_0, (rem_rows << 1));
        AE_SAV16X4X2_XP(out_2, out_3, dst_1_align, p_dst_1, (rem_rows << 1));
        AE_SA128POS_FP(dst_0_align, p_dst_0);
        AE_SA128POS_FP(dst_1_align, p_dst_1);
      }
      else
      {
        ae_int16* p_dst_0 = (ae_int16 *)((WORD16*)p_out + vec_itr * out_offset + m_itr * out_stride);
        ae_int16* p_dst_1 = (ae_int16 *)((WORD16*)p_out + (vec_itr + 1) * out_offset + m_itr * out_stride);

        out_0 = AE_SHORTSWAP(out_0);
        out_1 = AE_SHORTSWAP(out_1);
        out_2 = AE_SHORTSWAP(out_2);
        out_3 = AE_SHORTSWAP(out_3);

        WORD32 i;
        ae_int16x4 sel_shift = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x00040703,  0x06020501));
        for(i = 0; i < rem_rows; i++)
        {
          *p_dst_0 = out_0;           p_dst_0 += out_stride;
          *p_dst_1 = out_2;           p_dst_1 += out_stride;
          AE_DSEL16X4(out_1, out_0, out_1, out_0, sel_shift);
          AE_DSEL16X4(out_3, out_2, out_3, out_2, sel_shift);
        }
      }
    }
  }

  if((vec_count & 1) != 0)
  {
    for(m_itr = 0; m_itr < (rows & ~(8-1)); m_itr+=8)
    {
      ae_int64 acc64_00 = AE_ZERO64();
      ae_int64 acc64_10 = AE_ZERO64();
      ae_int64 acc64_20 = AE_ZERO64();
      ae_int64 acc64_30 = AE_ZERO64();
      ae_int64 acc64_40 = AE_ZERO64();
      ae_int64 acc64_50 = AE_ZERO64();
      ae_int64 acc64_60 = AE_ZERO64();
      ae_int64 acc64_70 = AE_ZERO64();

      ae_int32x2 acc_row01_vec0, acc_row23_vec0, acc_row45_vec0, acc_row67_vec0;

      ae_int16x4* p_vec_0 = (ae_int16x4 *)(p_vec1 + (vec_count - 1) * vec_offset);
      ae_valign vec_a = AE_LA64_PP(p_vec_0);

      ae_int16x8* p_mat1_0 = (ae_int16x8 *)(p_mat1 + m_itr);
      ae_int16x8* p_mat1_1 = (ae_int16x8 *)(p_mat1 + m_itr + rows);

      ae_valignx2 mat1_0_a, mat1_1_a;
      ae_int16x4 d_mat1_0, d_mat1_4;
      ae_int16x4 d_mat1_1, d_mat1_5;
      ae_int16x4 d_mat1_2, d_mat1_6;
      ae_int16x4 d_mat1_3, d_mat1_7;
      ae_int16x4 d_vec0;

      ae_int16x4 sel_1 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050301,  0x06040200));
      ae_int16x4 sel_2 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050604,  0x03010200));
      WORD32 mat_offset = (rows << 1) - 8;
      for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
      {
        ae_int16x4 d0, d1, d2, d3, d4, d5, d6, d7;

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        mat1_1_a = AE_LA128_PP(p_mat1_1);
        AE_LA16X4X2_IP(d_mat1_0, d_mat1_4, mat1_0_a, p_mat1_0);
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + mat_offset);
        AE_LA16X4X2_IP(d_mat1_1, d_mat1_5, mat1_1_a, p_mat1_1);
        p_mat1_1 = (ae_int16x8*)((ae_int16 *)p_mat1_1 + mat_offset);

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        mat1_1_a = AE_LA128_PP(p_mat1_1);
        AE_LA16X4X2_IP(d_mat1_2, d_mat1_6, mat1_0_a, p_mat1_0);
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + mat_offset);
        AE_LA16X4X2_IP(d_mat1_3, d_mat1_7, mat1_1_a, p_mat1_1);
        p_mat1_1 = (ae_int16x8*)((ae_int16 *)p_mat1_1 + mat_offset);

        AE_DSEL16X4(d0, d1, d_mat1_0, d_mat1_1, sel_1);
        AE_DSEL16X4(d2, d3, d_mat1_2, d_mat1_3, sel_1);

        AE_DSEL16X4(d4, d5, d_mat1_4, d_mat1_5, sel_1);
        AE_DSEL16X4(d6, d7, d_mat1_6, d_mat1_7, sel_1);

        AE_DSEL16X4(d_mat1_0, d_mat1_1, d0, d2, sel_2);
        AE_DSEL16X4(d_mat1_2, d_mat1_3, d1, d3, sel_2);
        AE_DSEL16X4(d_mat1_4, d_mat1_5, d4, d6, sel_2);
        AE_DSEL16X4(d_mat1_6, d_mat1_7, d5, d7, sel_2);

        AE_LA16X4_IP(d_vec0, vec_a, p_vec_0);

        AE_MULAAAA2Q16(acc64_00, acc64_10, d_mat1_0, d_mat1_1, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_20, acc64_30, d_mat1_2, d_mat1_3, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_40, acc64_50, d_mat1_4, d_mat1_5, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_60, acc64_70, d_mat1_6, d_mat1_7, d_vec0, d_vec0);
      }

      if((cols1 & 3) != 0)
      {
        WORD32 rem = (cols1 & 3);
        ae_int16x4 d0, d1, d2, d3, d4, d5, d6, d7;

        d_mat1_1 = d_mat1_5 = AE_ZERO16();
        d_mat1_2 = d_mat1_6 = AE_ZERO16();
        d_mat1_3 = d_mat1_7 = AE_ZERO16();

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        AE_LA16X4X2_IP(d_mat1_0, d_mat1_4, mat1_0_a, p_mat1_0);
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + mat_offset);

        if(rem >= 2)
        {
          mat1_1_a = AE_LA128_PP(p_mat1_1);
          AE_LA16X4X2_IP(d_mat1_1, d_mat1_5, mat1_1_a, p_mat1_1);
        }

        if(rem == 3)
        {
          mat1_0_a = AE_LA128_PP(p_mat1_0);
          AE_LA16X4X2_IP(d_mat1_2, d_mat1_6, mat1_0_a, p_mat1_0);
        }

        AE_DSEL16X4(d0, d1, d_mat1_0, d_mat1_1, sel_1);
        AE_DSEL16X4(d2, d3, d_mat1_2, d_mat1_3, sel_1);

        AE_DSEL16X4(d4, d5, d_mat1_4, d_mat1_5, sel_1);
        AE_DSEL16X4(d6, d7, d_mat1_6, d_mat1_7, sel_1);

        AE_DSEL16X4(d_mat1_0, d_mat1_1, d0, d2, sel_2);
        AE_DSEL16X4(d_mat1_2, d_mat1_3, d1, d3, sel_2);
        AE_DSEL16X4(d_mat1_4, d_mat1_5, d4, d6, sel_2);
        AE_DSEL16X4(d_mat1_6, d_mat1_7, d5, d7, sel_2);

        ae_valignx2 vecx2_a = AE_LA128_PP(p_vec_0);
        AE_LAV16X4X2_XP(d_vec0, d0, vecx2_a, (ae_int16x8 *)p_vec_0, (rem << 1));

        AE_MULAAAA2Q16(acc64_00, acc64_10, d_mat1_0, d_mat1_1, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_20, acc64_30, d_mat1_2, d_mat1_3, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_40, acc64_50, d_mat1_4, d_mat1_5, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_60, acc64_70, d_mat1_6, d_mat1_7, d_vec0, d_vec0);
      }

      ae_int16x4 out_0, out_1;
#if XCHAL_HAVE_HIFI5S
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row23_vec0, acc64_20, acc64_30, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row45_vec0, acc64_40, acc64_50, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row67_vec0, acc64_60, acc64_70, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row23_vec0, acc64_20, acc64_30, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row45_vec0, acc64_40, acc64_50, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row67_vec0, acc64_60, acc64_70, out_multiplier, out_shift);
#endif

      out_0 = AE_SAT16X4(acc_row01_vec0, acc_row23_vec0);
      out_1 = AE_SAT16X4(acc_row45_vec0, acc_row67_vec0);

      if(out_stride == 1)
      {
        ae_int16x8* p_dst_0 = (ae_int16x8 *)((WORD16*)p_out + (vec_count - 1) * out_offset + m_itr * out_stride);


        ae_valignx2 dst_0_align = AE_ZALIGN128();
        AE_SA16X4X2_IP(out_0, out_1, dst_0_align, p_dst_0);
        AE_SA128POS_FP(dst_0_align, p_dst_0);
      }
      else // out_offset == 1
      {
        ae_int16* p_dst_0 = (ae_int16 *)((WORD16*)p_out + (vec_count - 1) * out_offset + m_itr * out_stride);

        *p_dst_0 = AE_SEL16_6543(out_0, out_0);     p_dst_0 += out_stride;
        *p_dst_0 = AE_SEL16_5432(out_0, out_0);     p_dst_0 += out_stride;
        *p_dst_0 = AE_SEL16_4321(out_0, out_0);     p_dst_0 += out_stride;
        *p_dst_0 = out_0;                           p_dst_0 += out_stride;

        *p_dst_0 = AE_SEL16_6543(out_1, out_1);     p_dst_0 += out_stride;
        *p_dst_0 = AE_SEL16_5432(out_1, out_1);     p_dst_0 += out_stride;
        *p_dst_0 = AE_SEL16_4321(out_1, out_1);     p_dst_0 += out_stride;
        *p_dst_0 = out_1;                           p_dst_0 += out_stride;
      }
    }

    if(m_itr < rows)
    {
      WORD32 rem_rows = rows - m_itr;
      ae_int64 acc64_00 = AE_ZERO64();
      ae_int64 acc64_10 = AE_ZERO64();
      ae_int64 acc64_20 = AE_ZERO64();
      ae_int64 acc64_30 = AE_ZERO64();
      ae_int64 acc64_40 = AE_ZERO64();
      ae_int64 acc64_50 = AE_ZERO64();
      ae_int64 acc64_60 = AE_ZERO64();
      ae_int64 acc64_70 = AE_ZERO64();

      ae_int32x2 acc_row01_vec0, acc_row23_vec0, acc_row45_vec0, acc_row67_vec0;

      ae_int16x4* p_vec_0 = (ae_int16x4 *)(p_vec1 + (vec_count - 1) * vec_offset);
      ae_valign vec_a = AE_LA64_PP(p_vec_0);

      ae_int16x8* p_mat1_0 = (ae_int16x8 *)(p_mat1 + m_itr);
      ae_int16x8* p_mat1_1 = (ae_int16x8 *)(p_mat1 + m_itr + rows);

      ae_valignx2 mat1_0_a, mat1_1_a;
      ae_int16x4 d_mat1_0, d_mat1_4;
      ae_int16x4 d_mat1_1, d_mat1_5;
      ae_int16x4 d_mat1_2, d_mat1_6;
      ae_int16x4 d_mat1_3, d_mat1_7;
      ae_int16x4 d_vec0;

      ae_int16x4 sel_1 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050301,  0x06040200));
      ae_int16x4 sel_2 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050604,  0x03010200));
      for(c_itr = 0; c_itr < (cols1 >> 2); c_itr++)
      {
        ae_int16x4 d0, d1, d2, d3, d4, d5, d6, d7;

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        mat1_1_a = AE_LA128_PP(p_mat1_1);
        AE_LAV16X4X2_XP(d_mat1_0, d_mat1_4, mat1_0_a, p_mat1_0, (rem_rows << 1));
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + (rows << 1) - rem_rows);
        AE_LAV16X4X2_XP(d_mat1_1, d_mat1_5, mat1_1_a, p_mat1_1, (rem_rows << 1));
        p_mat1_1 = (ae_int16x8*)((ae_int16 *)p_mat1_1 + (rows << 1) - rem_rows);

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        mat1_1_a = AE_LA128_PP(p_mat1_1);
        AE_LAV16X4X2_XP(d_mat1_2, d_mat1_6, mat1_0_a, p_mat1_0, (rem_rows << 1));
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + (rows << 1) - rem_rows);
        AE_LAV16X4X2_XP(d_mat1_3, d_mat1_7, mat1_1_a, p_mat1_1, (rem_rows << 1));
        p_mat1_1 = (ae_int16x8*)((ae_int16 *)p_mat1_1 + (rows << 1) - rem_rows);

        AE_DSEL16X4(d0, d1, d_mat1_0, d_mat1_1, sel_1);
        AE_DSEL16X4(d2, d3, d_mat1_2, d_mat1_3, sel_1);

        AE_DSEL16X4(d4, d5, d_mat1_4, d_mat1_5, sel_1);
        AE_DSEL16X4(d6, d7, d_mat1_6, d_mat1_7, sel_1);

        AE_DSEL16X4(d_mat1_0, d_mat1_1, d0, d2, sel_2);
        AE_DSEL16X4(d_mat1_2, d_mat1_3, d1, d3, sel_2);
        AE_DSEL16X4(d_mat1_4, d_mat1_5, d4, d6, sel_2);
        AE_DSEL16X4(d_mat1_6, d_mat1_7, d5, d7, sel_2);

        AE_LA16X4_IP(d_vec0, vec_a, p_vec_0);

        AE_MULAAAA2Q16(acc64_00, acc64_10, d_mat1_0, d_mat1_1, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_20, acc64_30, d_mat1_2, d_mat1_3, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_40, acc64_50, d_mat1_4, d_mat1_5, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_60, acc64_70, d_mat1_6, d_mat1_7, d_vec0, d_vec0);
      }

      if((cols1 & 3) != 0)
      {
        WORD32 rem = (cols1 & 3);
        ae_int16x4 d0, d1, d2, d3, d4, d5, d6, d7;

        d_mat1_1 = d_mat1_5 = AE_ZERO16();
        d_mat1_2 = d_mat1_6 = AE_ZERO16();
        d_mat1_3 = d_mat1_7 = AE_ZERO16();

        mat1_0_a = AE_LA128_PP(p_mat1_0);
        AE_LAV16X4X2_XP(d_mat1_0, d_mat1_4, mat1_0_a, p_mat1_0, (rem_rows << 1));
        p_mat1_0 = (ae_int16x8*)((ae_int16 *)p_mat1_0 + (rows << 1) - rem_rows);

        if(rem >= 2)
        {
          mat1_1_a = AE_LA128_PP(p_mat1_1);
          AE_LAV16X4X2_XP(d_mat1_1, d_mat1_5, mat1_1_a, p_mat1_1, (rem_rows << 1));
        }

        if(rem == 3)
        {
          mat1_0_a = AE_LA128_PP(p_mat1_0);
          AE_LAV16X4X2_XP(d_mat1_2, d_mat1_6, mat1_0_a, p_mat1_0, (rem_rows << 1));
        }

        AE_DSEL16X4(d0, d1, d_mat1_0, d_mat1_1, sel_1);
        AE_DSEL16X4(d2, d3, d_mat1_2, d_mat1_3, sel_1);

        AE_DSEL16X4(d4, d5, d_mat1_4, d_mat1_5, sel_1);
        AE_DSEL16X4(d6, d7, d_mat1_6, d_mat1_7, sel_1);

        AE_DSEL16X4(d_mat1_0, d_mat1_1, d0, d2, sel_2);
        AE_DSEL16X4(d_mat1_2, d_mat1_3, d1, d3, sel_2);
        AE_DSEL16X4(d_mat1_4, d_mat1_5, d4, d6, sel_2);
        AE_DSEL16X4(d_mat1_6, d_mat1_7, d5, d7, sel_2);

        ae_valignx2 vecx2_a = AE_LA128_PP(p_vec_0);
        AE_LAV16X4X2_XP(d_vec0, d0, vecx2_a, (ae_int16x8 *)p_vec_0, (rem << 1));

        AE_MULAAAA2Q16(acc64_00, acc64_10, d_mat1_0, d_mat1_1, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_20, acc64_30, d_mat1_2, d_mat1_3, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_40, acc64_50, d_mat1_4, d_mat1_5, d_vec0, d_vec0);
        AE_MULAAAA2Q16(acc64_60, acc64_70, d_mat1_6, d_mat1_7, d_vec0, d_vec0);
      }

      ae_int16x4 out_0, out_1;
#if XCHAL_HAVE_HIFI5S
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row23_vec0, acc64_20, acc64_30, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row45_vec0, acc64_40, acc64_50, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32_HIFI5S(acc_row67_vec0, acc64_60, acc64_70, out_multiplier, ((15 - out_shift) << 16) | (15 - out_shift));
#else
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row01_vec0, acc64_00, acc64_10, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row23_vec0, acc64_20, acc64_30, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row45_vec0, acc64_40, acc64_50, out_multiplier, out_shift);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(acc_row67_vec0, acc64_60, acc64_70, out_multiplier, out_shift);
#endif

      out_0 = AE_SAT16X4(acc_row01_vec0, acc_row23_vec0);
      out_1 = AE_SAT16X4(acc_row45_vec0, acc_row67_vec0);

      if(out_stride == 1)
      {
        ae_int16x8* p_dst_0 = (ae_int16x8 *)((WORD16*)p_out + (vec_count - 1) * out_offset + m_itr * out_stride);

        ae_valignx2 dst_0_align = AE_ZALIGN128();
        AE_SAV16X4X2_XP(out_0, out_1, dst_0_align, p_dst_0, (rem_rows << 1));
        AE_SA128POS_FP(dst_0_align, p_dst_0);
      }
      else
      {
        ae_int16* p_dst_0 = (ae_int16 *)((WORD16*)p_out + (vec_count - 1) * out_offset + m_itr * out_stride);

        out_0 = AE_SHORTSWAP(out_0);
        out_1 = AE_SHORTSWAP(out_1);

        WORD32 i;
        ae_int16x4 sel_shift = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x00040703,  0x06020501));
        for(i = 0; i < rem_rows; i++)
        {
          *p_dst_0 = out_0;           p_dst_0 += out_stride;
          AE_DSEL16X4(out_1, out_0, out_1, out_0, sel_shift);
        }
      }
    }
  }

  return 0;
}

WORD32 xa_nn_batch_matmul_sym16sxsym16s_sym16s(
    WORD16 * __restrict__ p_out,
    const WORD32 *const p_out_shape,
    const WORD16 * __restrict__ p_mat1,
    const WORD32 *const p_mat1_shape,
    const WORD16 * __restrict__ p_mat2,
    const WORD32 *const p_mat2_shape,
    WORD32 mat1_transpose,
    WORD32 mat2_transpose,
    WORD32 mat1_zero_bias,
    WORD32 mat2_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias,
    VOID   *p_scratch)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat2_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat2, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((mat1_transpose != 0 && mat1_transpose != 1), -1);
  XA_NNLIB_ARG_CHK_COND((mat2_transpose != 0 && mat2_transpose != 1), -1);
  XA_NNLIB_ARG_CHK_COND((mat1_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((mat2_zero_bias != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias != 0), -1);

  WORD32 itr;
  WORD32 p_mat1_final_shape[5], p_mat2_final_shape[5];
  for(itr = 0; itr < 5; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_mat1_shape[itr] <= 0 || p_mat2_shape[itr] <= 0 || p_out_shape[itr] <= 0), -1);
    p_mat1_final_shape[itr] = p_mat1_shape[itr];
    p_mat2_final_shape[itr] = p_mat2_shape[itr];
  }

  const WORD16 *p_mat1_final, *p_mat2_final;
  p_mat1_final = p_mat1;
  p_mat2_final = p_mat2;

  if(mat1_transpose == 1 && !(mat2_transpose == 0 && p_mat2_final_shape[3] <= 14))
  {
    WORD32 mat1_size, ret;
    WORD32 permute_vec[5] = {0, 1, 2, 4, 3};
    p_mat1_final_shape[3] = p_mat1_shape[4];
    p_mat1_final_shape[4] = p_mat1_shape[3];
    ret = xa_nn_transpose_16_16((WORD16 *)p_scratch,
                                p_mat1_final_shape,
                                p_mat1,
                                p_mat1_shape,
                                permute_vec,
                                5,
                                5);
    if(ret != 0)
      return -1;
    p_mat1_final = (const WORD16 *)p_scratch;
    mat1_size = p_mat1_shape[0] * p_mat1_shape[1] * p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
    p_scratch = (VOID *)(p_mat1_final + mat1_size);
  }

  if(mat2_transpose == 1 && !(mat1_transpose == 0 && p_mat1_final_shape[3] <= 14))
  {
    WORD32 ret;
    WORD32 permute_vec[5] = {0, 1, 2, 4, 3};
    p_mat2_final_shape[3] = p_mat2_shape[4];
    p_mat2_final_shape[4] = p_mat2_shape[3];
    ret = xa_nn_transpose_16_16((WORD16 *)p_scratch,
                                p_mat2_final_shape,
                                p_mat2,
                                p_mat2_shape,
                                permute_vec,
                                5,
                                5);
    if(ret != 0)
      return -1;
    p_mat2_final = (const WORD16 *)p_scratch;
  }

  WORD32 accum_depth, mat1_rows, mat2_cols;
  accum_depth = p_mat1_final_shape[4];
  mat1_rows = p_mat1_final_shape[3];
  mat2_cols = p_mat2_final_shape[3];

  WORD32 mat1_ext0, mat1_ext1, mat1_ext2;
  mat1_ext0 = p_mat1_shape[0] == 1 ? 0 : p_mat1_shape[1] * p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
  mat1_ext1 = p_mat1_shape[1] == 1 ? 0 : p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
  mat1_ext2 = p_mat1_shape[2] == 1 ? 0 : p_mat1_shape[3] * p_mat1_shape[4];

  WORD32 mat2_ext0, mat2_ext1, mat2_ext2;
  mat2_ext0 = p_mat2_shape[0] == 1 ? 0 : p_mat2_shape[1] * p_mat2_shape[2] * p_mat2_shape[3] * p_mat2_shape[4];
  mat2_ext1 = p_mat2_shape[1] == 1 ? 0 : p_mat2_shape[2] * p_mat2_shape[3] * p_mat2_shape[4];
  mat2_ext2 = p_mat2_shape[2] == 1 ? 0 : p_mat2_shape[3] * p_mat2_shape[4];

  WORD32 b0, b1, b2;
  for(b0 = 0; b0 < p_out_shape[0]; b0++)
  {
    const WORD16 *ptr0_mat1 = p_mat1_final + b0 * mat1_ext0;
    const WORD16 *ptr0_mat2 = p_mat2_final + b0 * mat2_ext0;
    for(b1 = 0; b1 < p_out_shape[1]; b1++)
    {
      const WORD16 *ptr1_mat1 = ptr0_mat1 + b1 * mat1_ext1;
      const WORD16 *ptr1_mat2 = ptr0_mat2 + b1 * mat2_ext1;
      for(b2 = 0; b2 < p_out_shape[2]; b2++)
      {
        WORD32 ret = 0;
        const WORD16 *ptr2_mat1 = ptr1_mat1 + b2 * mat1_ext2;
        const WORD16 *ptr2_mat2 = ptr1_mat2 + b2 * mat2_ext2;
        if(mat1_transpose == 1 && mat2_transpose == 0 && p_mat2_final_shape[3] <= 14)
        {
          WORD16 *ptr_out = p_out + ((b0 * p_out_shape[1] + b1) * p_out_shape[2] + b2) * p_mat2_final_shape[3] * p_mat1_final_shape[4];
          ret = xa_nn_matmul_mat1_trans_sym16sxsym16s_sym16s(ptr_out,
                                                             ptr2_mat1,
                                                             ptr2_mat2,
                                                             p_mat1_final_shape[4],
                                                             p_mat2_final_shape[4],
                                                             p_mat2_final_shape[4],
                                                             p_mat2_final_shape[3],
                                                             p_mat2_final_shape[4],
                                                             p_mat1_final_shape[4],
                                                             1,
                                                             out_multiplier,
                                                             out_shift);
        }
        else if(mat2_transpose == 1 && mat1_transpose == 0 && p_mat1_final_shape[3] <= 14)
        {
          WORD16 *ptr_out = p_out + ((b0 * p_out_shape[1] + b1) * p_out_shape[2] + b2) * p_mat2_final_shape[4] * p_mat1_final_shape[3];
          ret = xa_nn_matmul_mat1_trans_sym16sxsym16s_sym16s(ptr_out,
                                                             ptr2_mat2,
                                                             ptr2_mat1,
                                                             p_mat2_final_shape[4],
                                                             p_mat1_final_shape[4],
                                                             p_mat1_final_shape[4],
                                                             p_mat1_final_shape[3],
                                                             p_mat1_final_shape[4],
                                                             1,
                                                             p_mat1_final_shape[3],
                                                             out_multiplier,
                                                             out_shift);
        }
        else
        {
          WORD16 *ptr_out = p_out + ((b0 * p_out_shape[1] + b1) * p_out_shape[2] + b2) * mat1_rows * mat2_cols;
          ret = xa_nn_matmul_sym16sxsym16s_sym16s(ptr_out,
                                                  ptr2_mat1,
                                                  ptr2_mat2,
                                                  NULL,
                                                  mat1_rows,
                                                  accum_depth,
                                                  accum_depth,
                                                  mat2_cols,
                                                  accum_depth,
                                                  mat1_rows,
                                                  1,
                                                  mat1_zero_bias,
                                                  mat2_zero_bias,
                                                  out_multiplier,
                                                  out_shift,
                                                  out_zero_bias);
        }
        if(ret != 0)
          return -1;
      }
    }
  }
  return 0;
}
