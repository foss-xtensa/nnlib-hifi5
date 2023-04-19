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

/* Helper functions for 16x16 matrix-vector multiplication */

/* Multiply four rows and one vector, aligned access case, continuous row accesses in four streams */
static inline void multiply_four_row_one_vec_16x16_aligned_cache
        (ae_int64 * p_accu1
        ,ae_int64 * p_accu2
        ,ae_int64 * p_accu3
        ,ae_int64 * p_accu4
        ,WORD16 ** pp_row1
        ,WORD16 ** pp_row2
        ,WORD16 ** pp_row3
        ,WORD16 ** pp_row4
        ,int row_offset
        ,WORD16 * p_vec
        ,WORD32   cols
        )
{
    int col;

    ae_int16x8 * p_row1 = (ae_int16x8 *)(*pp_row1);
    ae_int16x8 * p_row2 = (ae_int16x8 *)(*pp_row2);
    ae_int16x8 * p_row3 = (ae_int16x8 *)(*pp_row3);
    ae_int16x8 * p_row4 = (ae_int16x8 *)(*pp_row4);

    ae_int16x8 *p_src1 = (ae_int16x8 *)p_vec;

    ae_int64 accu1 = *p_accu1;
    ae_int64 accu2 = *p_accu2;
    ae_int64 accu3 = *p_accu3;
    ae_int64 accu4 = *p_accu4;

    for (col = 0; col < (cols >> 4); col++)
    {
        ae_int16x4 in11, in12;
        ae_int16x4 in21, in22;
        ae_int16x4 in31, in32;
        ae_int16x4 in41, in42;

        ae_int16x4 src11, src12;

        AE_L16X4X2_I(in11 ,in12, p_row1, 16);
        AE_L16X4X2_I(in21 ,in22, p_row2, 16);
        AE_L16X4X2_I(in31 ,in32, p_row3, 16);
        AE_L16X4X2_I(in41 ,in42, p_row4, 16);

        AE_L16X4X2_I(src11, src12, p_src1, 16);

        AE_MULAAAA2Q16(accu1, accu2 , src11, src11, in11, in21);
        AE_MULAAAA2Q16(accu1, accu2 , src12, src12, in12, in22);
        AE_MULAAAA2Q16(accu3, accu4 , src11, src11, in31, in41);
        AE_MULAAAA2Q16(accu3, accu4 , src12, src12, in32, in42);

        AE_L16X4X2_IP(in11 ,in12, p_row1, 32);
        AE_L16X4X2_IP(in21 ,in22, p_row2, 32);
        AE_L16X4X2_IP(in31 ,in32, p_row3, 32);
        AE_L16X4X2_IP(in41 ,in42, p_row4, 32);

        AE_L16X4X2_IP(src11, src12, p_src1, 32);

        AE_MULAAAA2Q16(accu1, accu2 , src11, src11, in11, in21);
        AE_MULAAAA2Q16(accu1, accu2 , src12, src12, in12, in22);
        AE_MULAAAA2Q16(accu3, accu4 , src11, src11, in31, in41);
        AE_MULAAAA2Q16(accu3, accu4 , src12, src12, in32, in42);
    }

#if 0 /* supporting multiple of 16 cols */
    if((cols > 0) && (cols & 15))
    {
        ae_int16x4 in11, in12;
        ae_int16x4 in21, in22;
        ae_int16x4 in31, in32;
        ae_int16x4 in41, in42;

        ae_int16x4 src11, src12;

        AE_L16X4X2_IP(in11 ,in12, p_row1, 16);
        AE_L16X4X2_IP(in21 ,in22, p_row2, 16);
        AE_L16X4X2_IP(in31 ,in32, p_row3, 16);
        AE_L16X4X2_IP(in41 ,in42, p_row4, 16);

        AE_L16X4X2_IP(src11, src12, p_src1, 16);

        AE_MULAAAA2Q16(accu1, accu2 , src11, src11, in11, in21);
        AE_MULAAAA2Q16(accu1, accu2 , src12, src12, in12, in22);
        AE_MULAAAA2Q16(accu3, accu4 , src11, src11, in31, in41);
        AE_MULAAAA2Q16(accu3, accu4 , src12, src12, in32, in42);
    }

    if((cols > 0) && (cols & 7))
    {
        ae_int16x8 *p_src1 = (ae_int16x8 *)(p_vec +  (cols & ~7));

        ae_int16x8 *p_row1 = (ae_int16x8 *)(p_mat + (row + 0)* row_offset + (cols & ~7));
        ae_int16x8 *p_row2 = (ae_int16x8 *)(p_mat + (row + 1)* row_offset + (cols & ~7));
        ae_int16x8 *p_row3 = (ae_int16x8 *)(p_mat + (row + 2)* row_offset + (cols & ~7));
        ae_int16x8 *p_row4 = (ae_int16x8 *)(p_mat + (row + 3)* row_offset + (cols & ~7));

        for (col = 0; col < (cols & 7) ; col++)
        {
            ae_int16x4 in11;
            ae_int16x4 in21;
            ae_int16x4 in31;
            ae_int16x4 in41;
            ae_int16x4 src11;

            AE_L16_IP(in11, (ae_int16 *)p_row1, 2);
            AE_L16_IP(in21, (ae_int16 *)p_row2, 2);
            AE_L16_IP(in31, (ae_int16 *)p_row3, 2);
            AE_L16_IP(in41, (ae_int16 *)p_row4, 2);

            AE_L16_IP(src11, (ae_int16 *)p_src1, 2);

            AE_MULA16_00(accu1, in11, src11);
            AE_MULA16_00(accu2, in21, src11);
            AE_MULA16_00(accu3, in31, src11);
            AE_MULA16_00(accu4, in41, src11);
        }
    }
#endif
    *pp_row1 = (WORD16 *)p_row1 + (row_offset - cols);
    *pp_row2 = (WORD16 *)p_row2 + (row_offset - cols);
    *pp_row3 = (WORD16 *)p_row3 + (row_offset - cols);
    *pp_row4 = (WORD16 *)p_row4 + (row_offset - cols);

    *p_accu1 = accu1;
    *p_accu2 = accu2;
    *p_accu3 = accu3;
    *p_accu4 = accu4;
}

/* Multiply three rows and one vector, unaligned access case */
static inline void multiply_three_row_one_vec_16x16
        (ae_int64 * p_accu1
        ,ae_int64 * p_accu2
        ,ae_int64 * p_accu3
        ,WORD16 * p_mat
        ,WORD32   row_offset
        ,WORD32   row
        ,WORD16 * p_vec
        ,WORD32   cols
        )
{
    int col;
    ae_int16x8 * p_row1 = (ae_int16x8 *)(p_mat + (row + 0) * row_offset);
    ae_int16x8 * p_row2 = (ae_int16x8 *)((ae_int16 *)p_row1 + row_offset);
    ae_int16x8 * p_row3 = (ae_int16x8 *)((ae_int16 *)p_row2 + row_offset);

    ae_int16x8 *p_src1 = (ae_int16x8 *)p_vec;

    ae_valignx2 align_mat1 = AE_LA128_PP(p_row1);
    ae_valignx2 align_mat2 = AE_LA128_PP(p_row2);
    ae_valignx2 align_mat3 = AE_LA128_PP(p_row3);

    ae_valignx2 align_src1 = AE_LA128_PP(p_src1);

    ae_int64 accu1 = *p_accu1;
    ae_int64 accu2 = *p_accu2;
    ae_int64 accu3 = *p_accu3;

    ae_int64 accu4 = AE_ZERO64();

    for (col = 0; col < (cols >> 3); col++)
    {
        ae_int16x4 in11, in12;
        ae_int16x4 in21, in22;
        ae_int16x4 in31, in32;

        ae_int16x4 src11, src12;

        AE_LA16X4X2_IP(in11 ,in12, align_mat1 ,p_row1);
        AE_LA16X4X2_IP(in21 ,in22, align_mat2 ,p_row2);
        AE_LA16X4X2_IP(in31 ,in32, align_mat3 ,p_row3);

        AE_LA16X4X2_IP(src11, src12, align_src1, p_src1);

        AE_MULAAAA2Q16(accu1, accu4 , src11, src12, in11, in12);
        AE_MULAAAA2Q16(accu2, accu3 , src11, src11, in21, in31);
        AE_MULAAAA2Q16(accu2, accu3 , src12, src12, in22, in32);
    }

    if((cols > 0) && (cols & 7))
    {
        ae_int16x8 *p_src1 = (ae_int16x8 *)(p_vec +  (cols & ~7));

        ae_int16x8 *p_row1 = (ae_int16x8 *)(p_mat + (row + 0)* row_offset + (cols & ~7));
        ae_int16x8 *p_row2 = (ae_int16x8 *)(p_mat + (row + 1)* row_offset + (cols & ~7));
        ae_int16x8 *p_row3 = (ae_int16x8 *)(p_mat + (row + 2)* row_offset + (cols & ~7));

        for (col = 0; col < (cols & 7) ; col++)
        {
            ae_int16x4 in11;
            ae_int16x4 in21;
            ae_int16x4 in31;
            ae_int16x4 src11;

            AE_L16_IP(in11, (ae_int16 *)p_row1, 2);
            AE_L16_IP(in21, (ae_int16 *)p_row2, 2);
            AE_L16_IP(in31, (ae_int16 *)p_row3, 2);

            AE_L16_IP(src11, (ae_int16 *)p_src1, 2);

            AE_MULA16_00(accu1, in11, src11);
            AE_MULA16_00(accu2, in21, src11);
            AE_MULA16_00(accu3, in31, src11);
        }
    }

    *p_accu1 = AE_ADD64S(accu1, accu4);
    *p_accu2 = accu2;
    *p_accu3 = accu3;
}


/* Multiply one rows and one vector, aligned access cases */
static inline void multiply_one_row_one_vec_16x16_aligned
        (ae_int64 * p_accu
        ,WORD16 * p_mat
        ,WORD32   row_offset
        ,WORD32   row
        ,WORD16 * p_vec
        ,WORD32   cols
        )
{
    int col;
    ae_int16x8 *p_src1 = (ae_int16x8 *)p_vec;
    ae_int16x8 *p_row1 = (ae_int16x8 *)(p_mat + row * row_offset);

    ae_int64 accu1 = *p_accu;
    ae_int64 accu2 = AE_ZERO64();

    for (col = 0; col < (cols >> 4); col++)
    {
        ae_int16x4 in11, in12;
        ae_int16x4 src11, src12;

        AE_L16X4X2_I(in11 ,in12, p_row1, 16);
        AE_L16X4X2_I(src11, src12, p_src1, 16);
        AE_MULAAAA2Q16(accu1, accu2 , src11, src12, in11, in12);

        AE_L16X4X2_IP(in11 ,in12, p_row1, 32);
        AE_L16X4X2_IP(src11, src12, p_src1, 32);
        AE_MULAAAA2Q16(accu1, accu2 , src11, src12, in11, in12);
    }

    if((cols > 0) && (cols & 15))
    {
        ae_int16x4 in11, in12;
        ae_int16x4 src11, src12;

        AE_L16X4X2_IP(in11 ,in12, p_row1, 16);
        AE_L16X4X2_IP(src11, src12, p_src1, 16);
        AE_MULAAAA2Q16(accu1, accu2 , src11, src12, in11, in12);
    }

#if 0
    if((cols > 0) && (cols & 7))
    {
        ae_int16x8 *p_src1 = (ae_int16x8 *)(p_vec +  (cols & ~7));
        ae_int16x8 *p_row1 = (ae_int16x8 *)(p_mat + row * row_offset + (cols & ~7));

        for (col = 0; col < (cols & 7) ; col++)
        {
            ae_int16x4 in11;
            ae_int16x4 src11;

            AE_L16_IP(in11, (ae_int16 *)p_row1, 2);
            AE_L16_IP(src11, (ae_int16 *)p_src1, 2);
            AE_MULA16_00(accu1, in11, src11);
        }
    }
#endif

    *p_accu = AE_ADD64S(accu1, accu2);
}

/* Multiply one row and one vector, unaligned access cases */
static inline void multiply_one_row_one_vec_16x16
        (ae_int64 * p_accu
        ,WORD16 * p_mat
        ,WORD32   row_offset
        ,WORD32   row
        ,WORD16 * p_vec
        ,WORD32   cols
        )
{
    int col;
    ae_int16x8 *p_src1 = (ae_int16x8 *)p_vec;
    ae_int16x8 *p_row1 = (ae_int16x8 *)(p_mat + row * row_offset); ;

    ae_valignx2 align_mat1 = AE_LA128_PP(p_row1);
    ae_valignx2 align_src1 = AE_LA128_PP(p_src1);

    ae_int64 accu1 = *p_accu;
    ae_int64 accu2 = AE_ZERO64();

    for (col = 0; col < (cols >> 3); col++)
    {
        ae_int16x4 in11, in12;
        ae_int16x4 src11, src12;

        AE_LA16X4X2_IP(in11 ,in12, align_mat1 ,p_row1);
        AE_LA16X4X2_IP(src11, src12, align_src1, p_src1);
        AE_MULAAAA2Q16(accu1, accu2 , src11, src12, in11, in12);
    }

    if((cols > 0) && (cols & 7))
    {
        ae_int16x8 *p_src1 = (ae_int16x8 *)(p_vec +  (cols & ~7));
        ae_int16x8 *p_row1 = (ae_int16x8 *)(p_mat + row * row_offset + (cols & ~7));

        for (col = 0; col < (cols & 7) ; col++)
        {
            ae_int16x4 in11;
            ae_int16x4 src11;

            AE_L16_IP(in11, (ae_int16 *)p_row1, 2);
            AE_L16_IP(src11, (ae_int16 *)p_src1, 2);
            AE_MULA16_00(accu1, in11, src11);
        }
    }

    *p_accu = AE_ADD64S(accu1, accu2);
}


WORD32 xa_nn_matXvec_16x16_16(
         WORD16 * __restrict__ p_out,           /* output */
         WORD16 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD16 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         WORD16 * __restrict__ p_bias,          /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 row_stride2,                    /* row stride for matrix2 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift)                       /* bias shift amount */

{

  if (!p_bias)
  {
    return -1;
  }

  acc_shift = acc_shift +32;
  LIMIT_ACC_LSH

  if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && !((unsigned int)p_mat2 & 15) && !((unsigned int)p_vec2 & 15) && (cols1% 16 ==0) && (cols2% 16 ==0) && row_stride1 % 8 ==0 && row_stride2 %8==0)
  {
    int row = 0;
    int rows_by_four = rows >> 2;

    WAE_SAR(bias_shift);

    WORD16 * p_m1_row1 = (p_mat1 + row * row_stride1);
    WORD16 * p_m1_row2 = (p_m1_row1 + row_stride1 * rows_by_four);
    WORD16 * p_m1_row3 = (p_m1_row2 + row_stride1 * rows_by_four);
    WORD16 * p_m1_row4 = (p_m1_row3 + row_stride1 * rows_by_four);

    WORD16 * p_m2_row1 = (p_mat2 + row * row_stride2);
    WORD16 * p_m2_row2 = (p_m2_row1 + row_stride2 * rows_by_four);
    WORD16 * p_m2_row3 = (p_m2_row2 + row_stride2 * rows_by_four);
    WORD16 * p_m2_row4 = (p_m2_row3 + row_stride2 * rows_by_four);

    for (row = 0; row < (rows_by_four) ; row += 1)
    {
        ae_int32x2 out1_32;
        ae_int32x2 out2_32;
        ae_int16x4 out_16;

        ae_int64 accu1 = AE_SLAS64S(p_bias[row + 0 * rows_by_four]);
        ae_int64 accu2 = AE_SLAS64S(p_bias[row + 1 * rows_by_four]);
        ae_int64 accu3 = AE_SLAS64S(p_bias[row + 2 * rows_by_four]);
        ae_int64 accu4 = AE_SLAS64S(p_bias[row + 3 * rows_by_four]);

        multiply_four_row_one_vec_16x16_aligned_cache
            (&accu1
            ,&accu2
            ,&accu3
            ,&accu4
            ,&p_m1_row1
            ,&p_m1_row2
            ,&p_m1_row3
            ,&p_m1_row4
            ,row_stride1
            ,p_vec1
            ,cols1
            );

        multiply_four_row_one_vec_16x16_aligned_cache
            (&accu1
            ,&accu2
            ,&accu3
            ,&accu4
            ,&p_m2_row1
            ,&p_m2_row2
            ,&p_m2_row3
            ,&p_m2_row4
            ,row_stride2
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);
        accu4 = AE_SLAA64S(accu4, acc_shift);

        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
        out2_32 = AE_ROUND32X2F64SSYM(accu3, accu4);
        out_16 = AE_SAT16X4(out1_32, out2_32);

        p_out[(row + 0)]  = (WORD16)AE_MOVAD16_3(out_16);
        p_out[(row + 1 * rows_by_four)]  = (WORD16)AE_MOVAD16_2(out_16);
        p_out[(row + 2 * rows_by_four)]  = (WORD16)AE_MOVAD16_1(out_16);
        p_out[(row + 3 * rows_by_four)]  = (WORD16)AE_MOVAD16_0(out_16);
    }

    row = rows_by_four * 4;

    for (; row < rows ; row++)
    {
        ae_int32x2 out1_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16_aligned
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        multiply_one_row_one_vec_16x16_aligned
            (&accu1
            ,p_mat2
            ,row_stride2
            ,row
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
        p_out[(row+0)]  = AE_SAT16X4(out1_32, out1_32);
    }
  }
  else if (p_mat1 && p_vec1 && p_mat2 && p_vec2)
  {
    int row = 0;

    WAE_SAR(bias_shift);

    for (row = 0; row < (rows - 2) ; row += 3)
    {
        ae_int32x2 out1_32;
        ae_int32x2 out2_32;
        ae_int16x4 out_16;

        ae_int64 accu1 = AE_SLAS64S(p_bias[row + 0]);
        ae_int64 accu2 = AE_SLAS64S(p_bias[row + 1]);
        ae_int64 accu3 = AE_SLAS64S(p_bias[row + 2]);

        multiply_three_row_one_vec_16x16
            (&accu1
            ,&accu2
            ,&accu3
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        multiply_three_row_one_vec_16x16
            (&accu1
            ,&accu2
            ,&accu3
            ,p_mat2
            ,row_stride2
            ,row
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);

        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
        out2_32 = AE_ROUND32X2F64SSYM(accu3, accu3);
        out_16 = AE_SAT16X4(out1_32, out2_32);

        p_out[(row+0)]  = (WORD16)AE_MOVAD16_3(out_16);
        p_out[(row+1)]  = (WORD16)AE_MOVAD16_2(out_16);
        p_out[(row+2)]  = (WORD16)AE_MOVAD16_1(out_16);
    }

    for (; row < rows ; row++)
    {
        ae_int32x2 out1_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        multiply_one_row_one_vec_16x16
            (&accu1
            ,p_mat2
            ,row_stride2
            ,row
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
        p_out[(row+0)]  = AE_SAT16X4(out1_32, out1_32);
    }
  }
  else if (p_mat1 && p_vec1 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && (cols1% 16 ==0) && (row_stride1 % 8 == 0))
  {
    int row = 0;
    int rows_by_four = rows >> 2;

    WAE_SAR(bias_shift);

    for (row = 0; row < (rows_by_four) ; row += 1)
    {
        ae_int32x2 out1_32;
        ae_int32x2 out2_32;
        ae_int16x4 out_16;

        ae_int64 accu1 = AE_SLAS64S(p_bias[row + 0 * rows_by_four]);
        ae_int64 accu2 = AE_SLAS64S(p_bias[row + 1 * rows_by_four]);
        ae_int64 accu3 = AE_SLAS64S(p_bias[row + 2 * rows_by_four]);
        ae_int64 accu4 = AE_SLAS64S(p_bias[row + 3 * rows_by_four]);

        WORD16 * p_m1_row1 = (p_mat1 + row * row_stride1);
        WORD16 * p_m1_row2 = (p_m1_row1 + row_stride1 * rows_by_four);
        WORD16 * p_m1_row3 = (p_m1_row2 + row_stride1 * rows_by_four);
        WORD16 * p_m1_row4 = (p_m1_row3 + row_stride1 * rows_by_four);

        multiply_four_row_one_vec_16x16_aligned_cache
            (&accu1
            ,&accu2
            ,&accu3
            ,&accu4
            ,&p_m1_row1
            ,&p_m1_row2
            ,&p_m1_row3
            ,&p_m1_row4
            ,row_stride1
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);
        accu4 = AE_SLAA64S(accu4, acc_shift);

        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
        out2_32 = AE_ROUND32X2F64SSYM(accu3, accu4);
        out_16 = AE_SAT16X4(out1_32, out2_32);

        p_out[(row+0 * rows_by_four)]  = (WORD16)AE_MOVAD16_3(out_16);
        p_out[(row+1 * rows_by_four)]  = (WORD16)AE_MOVAD16_2(out_16);
        p_out[(row+2 * rows_by_four)]  = (WORD16)AE_MOVAD16_1(out_16);
        p_out[(row+3 * rows_by_four)]  = (WORD16)AE_MOVAD16_0(out_16);
    }

    row = rows_by_four * 4;

    for (; row < rows ; row++)
    {
        ae_int32x2 out1_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16_aligned
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
        p_out[(row+0)]  = AE_SAT16X4(out1_32, out1_32);
    }
  }
  else if (p_mat1 && p_vec1 )
  {
    int row = 0;

    for (row = 0; row < (rows - 2) ; row += 3)
    {
        ae_int32x2 out1_32;
        ae_int32x2 out2_32;
        ae_int16x4 out_16;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row + 0], bias_shift);
        ae_int64 accu2 = AE_SLAA64S(p_bias[row + 1], bias_shift);
        ae_int64 accu3 = AE_SLAA64S(p_bias[row + 2], bias_shift);

        multiply_three_row_one_vec_16x16
            (&accu1
            ,&accu2
            ,&accu3
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);

        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
        out2_32 = AE_ROUND32X2F64SSYM(accu3, accu3);
        out_16 = AE_SAT16X4(out1_32, out2_32);

        p_out[(row+0)]  = (WORD16)AE_MOVAD16_3(out_16);
        p_out[(row+1)]  = (WORD16)AE_MOVAD16_2(out_16);
        p_out[(row+2)]  = (WORD16)AE_MOVAD16_1(out_16);
    }

    for (; row < rows ; row++)
    {
        ae_int32x2 out1_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
        p_out[(row+0)]  = AE_SAT16X4(out1_32, out1_32);
    }
  }
  else
  {
    return -1;
  }

  return 0;
}

WORD32 xa_nn_matXvec_16x16_32(
         WORD32 * __restrict__ p_out,           /* output */
         WORD16 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD16 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         WORD16 * __restrict__ p_bias,          /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 row_stride2,                    /* row stride for matrix2 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift)                       /* bias shift amount */
{

  if (!p_bias)
  {
    return -1;
  }

  acc_shift += 32;;

  if ( p_mat1 && p_vec1 && p_mat2 && p_vec2 &&  !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && !((unsigned int)p_mat2 & 15) && !((unsigned int)p_vec2 & 15) && (cols1% 16 ==0) && (cols2% 16 ==0) && row_stride1 % 8 ==0 && row_stride2 %8==0)
  {
    int row = 0;
    int rows_by_four = rows >> 2;

    for (row = 0; row < (rows_by_four) ; row += 1)
    {
        ae_int32x2 out1_32;
        ae_int32x2 out2_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row + 0 * rows_by_four], bias_shift);
        ae_int64 accu2 = AE_SLAA64S(p_bias[row + 1 * rows_by_four], bias_shift);
        ae_int64 accu3 = AE_SLAA64S(p_bias[row + 2 * rows_by_four], bias_shift);
        ae_int64 accu4 = AE_SLAA64S(p_bias[row + 3 * rows_by_four], bias_shift);

        WORD16 * p_m1_row1 = (p_mat1 + row * row_stride1);
        WORD16 * p_m1_row2 = (p_m1_row1 + row_stride1 * rows_by_four);
        WORD16 * p_m1_row3 = (p_m1_row2 + row_stride1 * rows_by_four);
        WORD16 * p_m1_row4 = (p_m1_row3 + row_stride1 * rows_by_four);

        WORD16 * p_m2_row1 = (p_mat2 + row * row_stride2);
        WORD16 * p_m2_row2 = (p_m2_row1 + row_stride2 * rows_by_four);
        WORD16 * p_m2_row3 = (p_m2_row2 + row_stride2 * rows_by_four);
        WORD16 * p_m2_row4 = (p_m2_row3 + row_stride2 * rows_by_four);

        multiply_four_row_one_vec_16x16_aligned_cache
            (&accu1
            ,&accu2
            ,&accu3
            ,&accu4
            ,&p_m1_row1
            ,&p_m1_row2
            ,&p_m1_row3
            ,&p_m1_row4
            ,row_stride1
            ,p_vec1
            ,cols1
            );

        multiply_four_row_one_vec_16x16_aligned_cache
            (&accu1
            ,&accu2
            ,&accu3
            ,&accu4
            ,&p_m2_row1
            ,&p_m2_row2
            ,&p_m2_row3
            ,&p_m2_row4
            ,row_stride2
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);
        accu4 = AE_SLAA64S(accu4, acc_shift);

        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
        out2_32 = AE_ROUND32X2F64SSYM(accu3, accu4);

        p_out[(row+0 * rows_by_four)]  = AE_MOVAD32_H(out1_32);
        p_out[(row+1 * rows_by_four)]  = AE_MOVAD32_L(out1_32);
        p_out[(row+2 * rows_by_four)]  = AE_MOVAD32_H(out2_32);
        p_out[(row+3 * rows_by_four)]  = AE_MOVAD32_L(out2_32);
    }

    row = rows_by_four * 4;

    for (; row < rows ; row++)
    {
        ae_int32x2 out1_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16_aligned
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        multiply_one_row_one_vec_16x16_aligned
            (&accu1
            ,p_mat2
            ,row_stride2
            ,row
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
        p_out[(row+0)]  = out1_32;
    }
  }
  else if (p_mat1 && p_vec1 && p_mat2 && p_vec2)
  {
    int row = 0;

    for (row = 0; row < (rows - 2) ; row += 3)
    {
        ae_int32x2 out1_32;
        ae_int32x2 out2_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row + 0], bias_shift);
        ae_int64 accu2 = AE_SLAA64S(p_bias[row + 1], bias_shift);
        ae_int64 accu3 = AE_SLAA64S(p_bias[row + 2], bias_shift);

        multiply_three_row_one_vec_16x16
            (&accu1
            ,&accu2
            ,&accu3
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        multiply_three_row_one_vec_16x16
            (&accu1
            ,&accu2
            ,&accu3
            ,p_mat2
            ,row_stride2
            ,row
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);

        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
        out2_32 = AE_ROUND32X2F64SSYM(accu3, accu3);

        p_out[(row+0)]  = AE_MOVAD32_H(out1_32);
        p_out[(row+1)]  = AE_MOVAD32_L(out1_32);
        p_out[(row+2)]  = AE_MOVAD32_H(out2_32);
    }

    for (; row < rows ; row++)
    {
        ae_int32x2 out1_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        multiply_one_row_one_vec_16x16
            (&accu1
            ,p_mat2
            ,row_stride2
            ,row
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
        p_out[(row+0)]  = out1_32;
    }
  }
  else if (p_mat1 && p_vec1 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && cols1%16==0 && row_stride1%8== 0)
  {
    int row = 0;
    int rows_by_four = rows >> 2;

    for (row = 0; row < (rows_by_four) ; row += 1)
    {
        ae_int32x2 out1_32;
        ae_int32x2 out2_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row + 0 * rows_by_four], bias_shift);
        ae_int64 accu2 = AE_SLAA64S(p_bias[row + 1 * rows_by_four], bias_shift);
        ae_int64 accu3 = AE_SLAA64S(p_bias[row + 2 * rows_by_four], bias_shift);
        ae_int64 accu4 = AE_SLAA64S(p_bias[row + 3 * rows_by_four], bias_shift);

        WORD16 * p_m1_row1 = (p_mat1 + row * row_stride1);
        WORD16 * p_m1_row2 = (p_m1_row1 + row_stride1 * rows_by_four);
        WORD16 * p_m1_row3 = (p_m1_row2 + row_stride1 * rows_by_four);
        WORD16 * p_m1_row4 = (p_m1_row3 + row_stride1 * rows_by_four);

        multiply_four_row_one_vec_16x16_aligned_cache
            (&accu1
            ,&accu2
            ,&accu3
            ,&accu4
            ,&p_m1_row1
            ,&p_m1_row2
            ,&p_m1_row3
            ,&p_m1_row4
            ,row_stride1
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);
        accu4 = AE_SLAA64S(accu4, acc_shift);

        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
        out2_32 = AE_ROUND32X2F64SSYM(accu3, accu4);

        p_out[(row+0 * rows_by_four)]  = AE_MOVAD32_H(out1_32);
        p_out[(row+1 * rows_by_four)]  = AE_MOVAD32_L(out1_32);
        p_out[(row+2 * rows_by_four)]  = AE_MOVAD32_H(out2_32);
        p_out[(row+3 * rows_by_four)]  = AE_MOVAD32_L(out2_32);
    }

    row = rows_by_four * 4;

    for (; row < rows ; row++)
    {
        ae_int32x2 out1_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16_aligned
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
        p_out[(row+0)]  = out1_32;
    }
  }
  else if (p_mat1 && p_vec1 )
  {
    int row = 0;

    for (row = 0; row < (rows - 2) ; row += 3)
    {
        ae_int32x2 out1_32;
        ae_int32x2 out2_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row + 0], bias_shift);
        ae_int64 accu2 = AE_SLAA64S(p_bias[row + 1], bias_shift);
        ae_int64 accu3 = AE_SLAA64S(p_bias[row + 2], bias_shift);

        multiply_three_row_one_vec_16x16
            (&accu1
            ,&accu2
            ,&accu3
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);

        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
        out2_32 = AE_ROUND32X2F64SSYM(accu3, accu3);

        p_out[(row+0)]  = AE_MOVAD32_H(out1_32);
        p_out[(row+1)]  = AE_MOVAD32_L(out1_32);
        p_out[(row+2)]  = AE_MOVAD32_H(out2_32);
    }

    for (; row < rows ; row++)
    {
        ae_int32x2 out1_32;

        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
        p_out[(row+0)]  = out1_32;
    }

  }
  else
  {
    return -1;
  }

  return 0;
}

WORD32 xa_nn_matXvec_16x16_64(
         WORD64 * __restrict__ p_out,           /* output */
         WORD16 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD16 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         WORD16 * __restrict__ p_bias,          /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 row_stride2,                    /* row stride for matrix2 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift)                       /* bias shift amount */
{

  if (!p_bias)
  {
    return -1;
  }

  ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(WORD16, WORD16, WORD64);

  if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && !((unsigned int)p_mat2 & 15) && !((unsigned int)p_vec2 & 15) && (cols1% 16 ==0) && (cols2% 16 ==0) && row_stride1 % 8 ==0 && row_stride2 %8==0)
  {
    int row = 0;
    int rows_by_four = rows >> 2;

    for (row = 0; row < (rows_by_four) ; row += 1)
    {
        ae_int64 accu1 = AE_SLAA64S(p_bias[row + 0 * rows_by_four], bias_shift);
        ae_int64 accu2 = AE_SLAA64S(p_bias[row + 1 * rows_by_four], bias_shift);
        ae_int64 accu3 = AE_SLAA64S(p_bias[row + 2 * rows_by_four], bias_shift);
        ae_int64 accu4 = AE_SLAA64S(p_bias[row + 3 * rows_by_four], bias_shift);

        WORD16 * p_m1_row1 = (p_mat1 + row * row_stride1);
        WORD16 * p_m1_row2 = (p_m1_row1 + row_stride1 * rows_by_four);
        WORD16 * p_m1_row3 = (p_m1_row2 + row_stride1 * rows_by_four);
        WORD16 * p_m1_row4 = (p_m1_row3 + row_stride1 * rows_by_four);

        WORD16 * p_m2_row1 = (p_mat2 + row * row_stride2);
        WORD16 * p_m2_row2 = (p_m2_row1 + row_stride2 * rows_by_four);
        WORD16 * p_m2_row3 = (p_m2_row2 + row_stride2 * rows_by_four);
        WORD16 * p_m2_row4 = (p_m2_row3 + row_stride2 * rows_by_four);

        multiply_four_row_one_vec_16x16_aligned_cache
            (&accu1
            ,&accu2
            ,&accu3
            ,&accu4
            ,&p_m1_row1
            ,&p_m1_row2
            ,&p_m1_row3
            ,&p_m1_row4
            ,row_stride1
            ,p_vec1
            ,cols1
            );

        multiply_four_row_one_vec_16x16_aligned_cache
            (&accu1
            ,&accu2
            ,&accu3
            ,&accu4
            ,&p_m2_row1
            ,&p_m2_row2
            ,&p_m2_row3
            ,&p_m2_row4
            ,row_stride2
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);
        accu4 = AE_SLAA64S(accu4, acc_shift);

        p_out[(row+0 * rows_by_four)]  = accu1;
        p_out[(row+1 * rows_by_four)]  = accu2;
        p_out[(row+2 * rows_by_four)]  = accu3;
        p_out[(row+3 * rows_by_four)]  = accu4;
    }

    row = rows_by_four * 4;

    for (; row < rows ; row++)
    {
        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16_aligned
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        multiply_one_row_one_vec_16x16_aligned
            (&accu1
            ,p_mat2
            ,row_stride2
            ,row
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        p_out[(row+0)]  = accu1;
    }
  }
  else if (p_mat1 && p_vec1 && p_mat2 && p_vec2)
  {
    int row = 0;

    for (row = 0; row < (rows - 2) ; row += 3)
    {
        ae_int64 accu1 = AE_SLAA64S(p_bias[row + 0], bias_shift);
        ae_int64 accu2 = AE_SLAA64S(p_bias[row + 1], bias_shift);
        ae_int64 accu3 = AE_SLAA64S(p_bias[row + 2], bias_shift);

        multiply_three_row_one_vec_16x16
            (&accu1
            ,&accu2
            ,&accu3
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        multiply_three_row_one_vec_16x16
            (&accu1
            ,&accu2
            ,&accu3
            ,p_mat2
            ,row_stride2
            ,row
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);

        p_out[(row+0)]  = accu1;
        p_out[(row+1)]  = accu2;
        p_out[(row+2)]  = accu3;
    }

    for (; row < rows ; row++)
    {
        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        multiply_one_row_one_vec_16x16
            (&accu1
            ,p_mat2
            ,row_stride2
            ,row
            ,p_vec2
            ,cols2
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        p_out[(row+0)]  = accu1;
    }
  }
  else if (p_mat1 && p_vec1 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && cols1%16==0 && row_stride1%8== 0)
  {
    int row = 0;
    int rows_by_four = rows >> 2;

    for (row = 0; row < (rows_by_four) ; row += 1)
    {
        ae_int64 accu1 = AE_SLAA64S(p_bias[row + 0 * rows_by_four], bias_shift);
        ae_int64 accu2 = AE_SLAA64S(p_bias[row + 1 * rows_by_four], bias_shift);
        ae_int64 accu3 = AE_SLAA64S(p_bias[row + 2 * rows_by_four], bias_shift);
        ae_int64 accu4 = AE_SLAA64S(p_bias[row + 3 * rows_by_four], bias_shift);

        WORD16 * p_m1_row1 = (p_mat1 + row * row_stride1);
        WORD16 * p_m1_row2 = (p_m1_row1 + row_stride1 * rows_by_four);
        WORD16 * p_m1_row3 = (p_m1_row2 + row_stride1 * rows_by_four);
        WORD16 * p_m1_row4 = (p_m1_row3 + row_stride1 * rows_by_four);

        multiply_four_row_one_vec_16x16_aligned_cache
            (&accu1
            ,&accu2
            ,&accu3
            ,&accu4
            ,&p_m1_row1
            ,&p_m1_row2
            ,&p_m1_row3
            ,&p_m1_row4
            ,row_stride1
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);
        accu4 = AE_SLAA64S(accu4, acc_shift);

        p_out[(row+0 * rows_by_four)]  = accu1;
        p_out[(row+1 * rows_by_four)]  = accu2;
        p_out[(row+2 * rows_by_four)]  = accu3;
        p_out[(row+3 * rows_by_four)]  = accu4;
    }

    row = rows_by_four * 4;

    for (; row < rows ; row++)
    {
        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16_aligned
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        p_out[(row+0)]  = accu1;
    }
  }
  else if (p_mat1 && p_vec1 )
  {
    int row = 0;

    for (row = 0; row < (rows - 2) ; row += 3)
    {
        ae_int64 accu1 = AE_SLAA64S(p_bias[row + 0], bias_shift);
        ae_int64 accu2 = AE_SLAA64S(p_bias[row + 1], bias_shift);
        ae_int64 accu3 = AE_SLAA64S(p_bias[row + 2], bias_shift);

        multiply_three_row_one_vec_16x16
            (&accu1
            ,&accu2
            ,&accu3
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        accu2 = AE_SLAA64S(accu2, acc_shift);
        accu3 = AE_SLAA64S(accu3, acc_shift);

        p_out[(row+0)]  = accu1;
        p_out[(row+1)]  = accu2;
        p_out[(row+2)]  = accu3;
    }

    for (; row < rows ; row++)
    {
        ae_int64 accu1 = AE_SLAA64S(p_bias[row], bias_shift);

        multiply_one_row_one_vec_16x16
            (&accu1
            ,p_mat1
            ,row_stride1
            ,row
            ,p_vec1
            ,cols1
            );

        accu1 = AE_SLAA64S(accu1, acc_shift);
        p_out[(row+0)]  = accu1;
    }
  }
  else
  {
    return -1;
  }

  return 0;
}

WORD32 xa_nn_matXvec_16x16_16_tanh(
         WORD16 * __restrict__ p_out,           /* output */
         WORD16 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD16 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         VOID   * __restrict__ p_bias,          /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 row_stride2,                    /* row stride for matrix2 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift,                       /* bias shift amount */
         WORD32 bias_precision,                 /* 16 or 64 */
         VOID   * __restrict__ p_scratch)       /* Scratch pointer arg, only if required */
{

  if (!p_bias)
  {
    return -1;
  }

  int err=0;

  switch(bias_precision)
  {
    default:
    case 16:
      {
        err = xa_nn_matXvec_16x16_32(
        ((WORD32 *)p_scratch),       /* output stored in scratch*/
        p_mat1,          /* matrix1: rows x cols1 */
        p_mat2,          /* matrix2: rows x cols2 */
        p_vec1,          /* vec1: cols1 x 1 */
        p_vec2,          /* vec2: cols2 x 1 */
        ((WORD16 *)p_bias),          /* bias */
        rows,
        cols1,
        cols2,
        row_stride1,                  /* row stride for matrix1 */
        row_stride2,                  /* row stride for matrix2 */
        acc_shift,                    /* out accumulator shift amount */
        bias_shift);                   /* bias shift amount */

        if(err){
            return -1;
        }
        break;
      }
    case 64:
      {
          acc_shift += 32;;
          if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && !((unsigned int)p_mat2 & 15) && !((unsigned int)p_vec2 & 15) && (cols1% 16 ==0) && (cols2% 16 ==0) && row_stride1 % 8 ==0 && row_stride2 %8==0)
          {
              int row = 0;
              int rows_by_four = rows >> 2;
              WORD32 * p_out_scratch = (WORD32 *)p_scratch;

              for (row = 0; row < (rows_by_four) ; row += 1)
              {
                  ae_int32x2 out1_32;
                  ae_int32x2 out2_32;

                  ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 0 * rows_by_four], bias_shift);
                  ae_int64 accu2 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 1 * rows_by_four], bias_shift);
                  ae_int64 accu3 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 2 * rows_by_four], bias_shift);
                  ae_int64 accu4 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 3 * rows_by_four], bias_shift);

                  WORD16 * p_m1_row1 = (p_mat1 + row * row_stride1);
                  WORD16 * p_m1_row2 = (p_m1_row1 + row_stride1 * rows_by_four);
                  WORD16 * p_m1_row3 = (p_m1_row2 + row_stride1 * rows_by_four);
                  WORD16 * p_m1_row4 = (p_m1_row3 + row_stride1 * rows_by_four);

                  WORD16 * p_m2_row1 = (p_mat2 + row * row_stride2);
                  WORD16 * p_m2_row2 = (p_m2_row1 + row_stride2 * rows_by_four);
                  WORD16 * p_m2_row3 = (p_m2_row2 + row_stride2 * rows_by_four);
                  WORD16 * p_m2_row4 = (p_m2_row3 + row_stride2 * rows_by_four);

                  multiply_four_row_one_vec_16x16_aligned_cache
                      (&accu1
                       ,&accu2
                       ,&accu3
                       ,&accu4
                       ,&p_m1_row1
                       ,&p_m1_row2
                       ,&p_m1_row3
                       ,&p_m1_row4
                       ,row_stride1
                       ,p_vec1
                       ,cols1
                      );

                  multiply_four_row_one_vec_16x16_aligned_cache
                      (&accu1
                       ,&accu2
                       ,&accu3
                       ,&accu4
                       ,&p_m2_row1
                       ,&p_m2_row2
                       ,&p_m2_row3
                       ,&p_m2_row4
                       ,row_stride2
                       ,p_vec2
                       ,cols2
                      );

                  accu1 = AE_SLAA64S(accu1, acc_shift);
                  accu2 = AE_SLAA64S(accu2, acc_shift);
                  accu3 = AE_SLAA64S(accu3, acc_shift);
                  accu4 = AE_SLAA64S(accu4, acc_shift);

                  out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
                  out2_32 = AE_ROUND32X2F64SSYM(accu3, accu4);

                  p_out_scratch[(row+0 * rows_by_four)]  = AE_MOVAD32_H(out1_32);
                  p_out_scratch[(row+1 * rows_by_four)]  = AE_MOVAD32_L(out1_32);
                  p_out_scratch[(row+2 * rows_by_four)]  = AE_MOVAD32_H(out2_32);
                  p_out_scratch[(row+3 * rows_by_four)]  = AE_MOVAD32_L(out2_32);
              }

              row = rows_by_four * 4;

              for (; row < rows ; row++)
              {
                  ae_int32x2 out1_32;

                  ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row], bias_shift);

                  multiply_one_row_one_vec_16x16_aligned
                      (&accu1
                       ,p_mat1
                       ,row_stride1
                       ,row
                       ,p_vec1
                       ,cols1
                      );

                  multiply_one_row_one_vec_16x16_aligned
                      (&accu1
                       ,p_mat2
                       ,row_stride2
                       ,row
                       ,p_vec2
                       ,cols2
                      );

                  accu1 = AE_SLAA64S(accu1, acc_shift);
                  out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
                  p_out_scratch[(row+0)]  = AE_MOVAD32_H(out1_32);
              }
          }
          else if (p_mat1 && p_vec1 && p_mat2 && p_vec2)
          {
              int row = 0;
              WORD32 * p_out_scratch = (WORD32 *)p_scratch;

              for (row = 0; row < (rows - 2) ; row += 3)
              {
                  ae_int32x2 out1_32;
                  ae_int32x2 out2_32;

                  ae_int64 accu1 = AE_SLAA64S(((ae_int64*)p_bias)[row + 0], bias_shift);
                  ae_int64 accu2 = AE_SLAA64S(((ae_int64*)p_bias)[row + 1], bias_shift);
                  ae_int64 accu3 = AE_SLAA64S(((ae_int64*)p_bias)[row + 2], bias_shift);

                  multiply_three_row_one_vec_16x16
                      (&accu1
                       ,&accu2
                       ,&accu3
                       ,p_mat1
                       ,row_stride1
                       ,row
                       ,p_vec1
                       ,cols1
                      );

                  multiply_three_row_one_vec_16x16
                      (&accu1
                       ,&accu2
                       ,&accu3
                       ,p_mat2
                       ,row_stride2
                       ,row
                       ,p_vec2
                       ,cols2
                      );

                  accu1 = AE_SLAA64S(accu1, acc_shift);
                  accu2 = AE_SLAA64S(accu2, acc_shift);
                  accu3 = AE_SLAA64S(accu3, acc_shift);

                  out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
                  out2_32 = AE_ROUND32X2F64SSYM(accu3, accu3);

                  p_out_scratch[(row+0)]  = AE_MOVAD32_H(out1_32);
                  p_out_scratch[(row+1)]  = AE_MOVAD32_L(out1_32);
                  p_out_scratch[(row+2)]  = AE_MOVAD32_H(out2_32);
              }

              for (; row < rows ; row++)
              {
                  ae_int32x2 out1_32;

                  ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row], bias_shift);

                  multiply_one_row_one_vec_16x16
                      (&accu1
                       ,p_mat1
                       ,row_stride1
                       ,row
                       ,p_vec1
                       ,cols1
                      );

                  multiply_one_row_one_vec_16x16
                      (&accu1
                       ,p_mat2
                       ,row_stride2
                       ,row
                       ,p_vec2
                       ,cols2
                      );

                  accu1 = AE_SLAA64S(accu1, acc_shift);
                  out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
                  p_out_scratch[(row+0)]  = out1_32;
              }
          }
        else if (p_mat1 && p_vec1 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && cols1%16==0 && row_stride1 % 8 ==0)
        {
            int row = 0;
            int rows_by_four = rows >> 2;

            WORD32 * p_out_scratch = (WORD32 *)p_scratch;

            for (row = 0; row < (rows_by_four) ; row += 1)
            {
                ae_int32x2 out1_32;
                ae_int32x2 out2_32;

                ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 0 * rows_by_four], bias_shift);
                ae_int64 accu2 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 1 * rows_by_four], bias_shift);
                ae_int64 accu3 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 2 * rows_by_four], bias_shift);
                ae_int64 accu4 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 3 * rows_by_four], bias_shift);

                WORD16 * p_m1_row1 = (p_mat1 + row * row_stride1);
                WORD16 * p_m1_row2 = (p_m1_row1 + row_stride1 * rows_by_four);
                WORD16 * p_m1_row3 = (p_m1_row2 + row_stride1 * rows_by_four);
                WORD16 * p_m1_row4 = (p_m1_row3 + row_stride1 * rows_by_four);

                multiply_four_row_one_vec_16x16_aligned_cache
                    (&accu1
                     ,&accu2
                     ,&accu3
                     ,&accu4
                     ,&p_m1_row1
                     ,&p_m1_row2
                     ,&p_m1_row3
                     ,&p_m1_row4
                     ,row_stride1
                     ,p_vec1
                     ,cols1
                    );

                accu1 = AE_SLAA64S(accu1, acc_shift);
                accu2 = AE_SLAA64S(accu2, acc_shift);
                accu3 = AE_SLAA64S(accu3, acc_shift);
                accu4 = AE_SLAA64S(accu4, acc_shift);

                out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
                out2_32 = AE_ROUND32X2F64SSYM(accu3, accu4);

                p_out_scratch[(row+0 * rows_by_four)]  = AE_MOVAD32_H(out1_32);
                p_out_scratch[(row+1 * rows_by_four)]  = AE_MOVAD32_L(out1_32);
                p_out_scratch[(row+2 * rows_by_four)]  = AE_MOVAD32_H(out2_32);
                p_out_scratch[(row+3 * rows_by_four)]  = AE_MOVAD32_L(out2_32);
            }

            row = rows_by_four * 4;

            for (; row < rows ; row++)
            {
                ae_int32x2 out1_32;

                ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row], bias_shift);

                multiply_one_row_one_vec_16x16_aligned
                    (&accu1
                     ,p_mat1
                     ,row_stride1
                     ,row
                     ,p_vec1
                     ,cols1
                    );

                accu1 = AE_SLAA64S(accu1, acc_shift);
                out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
                p_out_scratch[(row+0)]  = AE_MOVAD32_H(out1_32);
            }
        }
          else if (p_mat1 && p_vec1 )
          {
              int row = 0;
              WORD32 * p_out_scratch = (WORD32 *)p_scratch;

              for (row = 0; row < (rows - 2) ; row += 3)
              {
                  ae_int32x2 out1_32;
                  ae_int32x2 out2_32;

                  ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 0], bias_shift);
                  ae_int64 accu2 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 1], bias_shift);
                  ae_int64 accu3 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 2], bias_shift);

                  multiply_three_row_one_vec_16x16
                      (&accu1
                       ,&accu2
                       ,&accu3
                       ,p_mat1
                       ,row_stride1
                       ,row
                       ,p_vec1
                       ,cols1
                      );

                  accu1 = AE_SLAA64S(accu1, acc_shift);
                  accu2 = AE_SLAA64S(accu2, acc_shift);
                  accu3 = AE_SLAA64S(accu3, acc_shift);

                  out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
                  out2_32 = AE_ROUND32X2F64SSYM(accu3, accu3);

                  p_out_scratch[(row+0)]  = AE_MOVAD32_H(out1_32);
                  p_out_scratch[(row+1)]  = AE_MOVAD32_L(out1_32);
                  p_out_scratch[(row+2)]  = AE_MOVAD32_H(out2_32);
              }

              for (; row < rows ; row++)
              {
                  ae_int32x2 out1_32;

                  ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row], bias_shift);

                  multiply_one_row_one_vec_16x16
                      (&accu1
                       ,p_mat1
                       ,row_stride1
                       ,row
                       ,p_vec1
                       ,cols1
                      );

                  accu1 = AE_SLAA64S(accu1, acc_shift);
                  out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
                  p_out_scratch[(row+0)]  = out1_32;
              }
          }
        else
        {
          return -1;
        }

        break;
      }
  }

  xa_nn_vec_tanh_32_16((pWORD16) p_out, (pWORD32) p_scratch, rows);

  return 0;
}

WORD32 xa_nn_matXvec_16x16_16_sigmoid(
         WORD16 * __restrict__ p_out,           /* output */
         WORD16 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD16 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         VOID   * __restrict__ p_bias,          /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 row_stride2,                    /* row stride for matrix2 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift,                       /* bias shift amount */
         WORD32 bias_precision,                 /* 16 or 64 */
         VOID   * __restrict__ p_scratch)       /* Scratch pointer arg, only if required */
{

  if (!p_bias)
  {
    return -1;
  }

  int err=0;

  switch(bias_precision)
  {
      default:
      case 16:
          {
              err = xa_nn_matXvec_16x16_32(
                      ((WORD32 *)p_scratch),       /* output stored in scratch*/
                      p_mat1,          /* matrix1: rows x cols1 */
                      p_mat2,          /* matrix2: rows x cols2 */
                      p_vec1,          /* vec1: cols1 x 1 */
                      p_vec2,          /* vec2: cols2 x 1 */
                      ((WORD16 *)p_bias),          /* bias */
                      rows,
                      cols1,
                      cols2,
                      row_stride1,                  /* row stride for matrix1 */
                      row_stride2,                  /* row stride for matrix2 */
                      acc_shift,                    /* out accumulator shift amount */
                      bias_shift);                   /* bias shift amount */

              if(err){
                  return -1;
              }
              break;
          }
      case 64:
          {
              ADJUST_ACC_LSH_AND_BIAS_LSH_AxB_C(WORD16, WORD16, WORD32);
              if (p_mat1 && p_vec1 && p_mat2 && p_vec2 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && !((unsigned int)p_mat2 & 15) && !((unsigned int)p_vec2 & 15) && (cols1% 16 ==0) && (cols2% 16 ==0) && row_stride1 % 8 ==0 && row_stride2 %8==0)
              {
                  int row = 0;
                  int rows_by_four = rows >> 2;

                  WORD32 * p_out_scratch = (WORD32 *)p_scratch;

                  for (row = 0; row < (rows_by_four) ; row += 1)
                  {
                      ae_int32x2 out1_32;
                      ae_int32x2 out2_32;

                      ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 0 * rows_by_four], bias_shift);
                      ae_int64 accu2 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 1 * rows_by_four], bias_shift);
                      ae_int64 accu3 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 2 * rows_by_four], bias_shift);
                      ae_int64 accu4 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 3 * rows_by_four], bias_shift);

                      WORD16 * p_m1_row1 = (p_mat1 + row * row_stride1);
                      WORD16 * p_m1_row2 = (p_m1_row1 + row_stride1 * rows_by_four);
                      WORD16 * p_m1_row3 = (p_m1_row2 + row_stride1 * rows_by_four);
                      WORD16 * p_m1_row4 = (p_m1_row3 + row_stride1 * rows_by_four);

                      WORD16 * p_m2_row1 = (p_mat2 + row * row_stride2);
                      WORD16 * p_m2_row2 = (p_m2_row1 + row_stride2 * rows_by_four);
                      WORD16 * p_m2_row3 = (p_m2_row2 + row_stride2 * rows_by_four);
                      WORD16 * p_m2_row4 = (p_m2_row3 + row_stride2 * rows_by_four);

                      multiply_four_row_one_vec_16x16_aligned_cache
                          (&accu1
                           ,&accu2
                           ,&accu3
                           ,&accu4
                           ,&p_m1_row1
                           ,&p_m1_row2
                           ,&p_m1_row3
                           ,&p_m1_row4
                           ,row_stride1
                           ,p_vec1
                           ,cols1
                          );

                      multiply_four_row_one_vec_16x16_aligned_cache
                          (&accu1
                           ,&accu2
                           ,&accu3
                           ,&accu4
                           ,&p_m2_row1
                           ,&p_m2_row2
                           ,&p_m2_row3
                           ,&p_m2_row4
                           ,row_stride2
                           ,p_vec2
                           ,cols2
                          );

                      accu1 = AE_SLAA64S(accu1, acc_shift);
                      accu2 = AE_SLAA64S(accu2, acc_shift);
                      accu3 = AE_SLAA64S(accu3, acc_shift);
                      accu4 = AE_SLAA64S(accu4, acc_shift);

                      out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
                      out2_32 = AE_ROUND32X2F64SSYM(accu3, accu4);

                      p_out_scratch[(row+0 * rows_by_four)]  = AE_MOVAD32_H(out1_32);
                      p_out_scratch[(row+1 * rows_by_four)]  = AE_MOVAD32_L(out1_32);
                      p_out_scratch[(row+2 * rows_by_four)]  = AE_MOVAD32_H(out2_32);
                      p_out_scratch[(row+3 * rows_by_four)]  = AE_MOVAD32_L(out2_32);
                  }

                  row = rows_by_four * 4;

                  for (; row < rows ; row++)
                  {
                      ae_int32x2 out1_32;

                      ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row], bias_shift);

                      multiply_one_row_one_vec_16x16_aligned
                          (&accu1
                           ,p_mat1
                           ,row_stride1
                           ,row
                           ,p_vec1
                           ,cols1
                          );

                      multiply_one_row_one_vec_16x16_aligned
                          (&accu1
                           ,p_mat2
                           ,row_stride2
                           ,row
                           ,p_vec2
                           ,cols2
                          );

                      accu1 = AE_SLAA64S(accu1, acc_shift);
                      out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
                      p_out_scratch[(row+0)]  = AE_MOVAD32_H(out1_32);
                  }
              }
              else if (p_mat1 && p_vec1 && p_mat2 && p_vec2)
              {
                  int row = 0;
                  WORD32 * p_out_scratch = (WORD32 *)p_scratch;

                  for (row = 0; row < (rows - 2) ; row += 3)
                  {
                      ae_int32x2 out1_32;
                      ae_int32x2 out2_32;

                      ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 0], bias_shift);
                      ae_int64 accu2 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 1], bias_shift);
                      ae_int64 accu3 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 2], bias_shift);

                      multiply_three_row_one_vec_16x16
                          (&accu1
                           ,&accu2
                           ,&accu3
                           ,p_mat1
                           ,row_stride1
                           ,row
                           ,p_vec1
                           ,cols1
                          );

                      multiply_three_row_one_vec_16x16
                          (&accu1
                           ,&accu2
                           ,&accu3
                           ,p_mat2
                           ,row_stride2
                           ,row
                           ,p_vec2
                           ,cols2
                          );

                      accu1 = AE_SLAA64S(accu1, acc_shift);
                      accu2 = AE_SLAA64S(accu2, acc_shift);
                      accu3 = AE_SLAA64S(accu3, acc_shift);

                      out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
                      out2_32 = AE_ROUND32X2F64SSYM(accu3, accu3);

                      p_out_scratch[(row+0)]  = AE_MOVAD32_H(out1_32);
                      p_out_scratch[(row+1)]  = AE_MOVAD32_L(out1_32);
                      p_out_scratch[(row+2)]  = AE_MOVAD32_H(out2_32);
                  }

                  for (; row < rows ; row++)
                  {
                      ae_int32x2 out1_32;

                      ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row], bias_shift);

                      multiply_one_row_one_vec_16x16
                          (&accu1
                           ,p_mat1
                           ,row_stride1
                           ,row
                           ,p_vec1
                           ,cols1
                          );

                      multiply_one_row_one_vec_16x16
                          (&accu1
                           ,p_mat2
                           ,row_stride2
                           ,row
                           ,p_vec2
                           ,cols2
                          );

                      accu1 = AE_SLAA64S(accu1, acc_shift);
                      out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
                      p_out_scratch[(row+0)]  = out1_32;
                  }

              }
              else if (p_mat1 && p_vec1 && !((unsigned int)p_mat1 & 15) && !((unsigned int)p_vec1 & 15) && cols1%16==0 && row_stride1%8 ==0)
              {
                  int row = 0;
                  int rows_by_four = rows >> 2;

                  WORD32 * p_out_scratch = (WORD32 *)p_scratch;

                  for (row = 0; row < (rows_by_four) ; row += 1)
                  {
                      ae_int32x2 out1_32;
                      ae_int32x2 out2_32;

                      ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 0 * rows_by_four], bias_shift);
                      ae_int64 accu2 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 1 * rows_by_four], bias_shift);
                      ae_int64 accu3 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 2 * rows_by_four], bias_shift);
                      ae_int64 accu4 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 3 * rows_by_four], bias_shift);

                      WORD16 * p_m1_row1 = (p_mat1 + row * row_stride1);
                      WORD16 * p_m1_row2 = (p_m1_row1 + row_stride1 * rows_by_four);
                      WORD16 * p_m1_row3 = (p_m1_row2 + row_stride1 * rows_by_four);
                      WORD16 * p_m1_row4 = (p_m1_row3 + row_stride1 * rows_by_four);

                      multiply_four_row_one_vec_16x16_aligned_cache
                          (&accu1
                           ,&accu2
                           ,&accu3
                           ,&accu4
                           ,&p_m1_row1
                           ,&p_m1_row2
                           ,&p_m1_row3
                           ,&p_m1_row4
                           ,row_stride1
                           ,p_vec1
                           ,cols1
                          );

                      accu1 = AE_SLAA64S(accu1, acc_shift);
                      accu2 = AE_SLAA64S(accu2, acc_shift);
                      accu3 = AE_SLAA64S(accu3, acc_shift);
                      accu4 = AE_SLAA64S(accu4, acc_shift);

                      out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
                      out2_32 = AE_ROUND32X2F64SSYM(accu3, accu4);

                      p_out_scratch[(row+0 * rows_by_four)]  = AE_MOVAD32_H(out1_32);
                      p_out_scratch[(row+1 * rows_by_four)]  = AE_MOVAD32_L(out1_32);
                      p_out_scratch[(row+2 * rows_by_four)]  = AE_MOVAD32_H(out2_32);
                      p_out_scratch[(row+3 * rows_by_four)]  = AE_MOVAD32_L(out2_32);
                  }

                  row = rows_by_four * 4;

                  for (; row < rows ; row++)
                  {
                      ae_int32x2 out1_32;

                      ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row], bias_shift);

                      multiply_one_row_one_vec_16x16_aligned
                          (&accu1
                           ,p_mat1
                           ,row_stride1
                           ,row
                           ,p_vec1
                           ,cols1
                          );

                      accu1 = AE_SLAA64S(accu1, acc_shift);
                      out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
                      p_out_scratch[(row+0)]  = AE_MOVAD32_H(out1_32);
                  }
              }
              else if (p_mat1 && p_vec1 )
              {
                  int row = 0;
                  WORD32 * p_out_scratch = (WORD32 *)p_scratch;

                  for (row = 0; row < (rows - 2) ; row += 3)
                  {
                      ae_int32x2 out1_32;
                      ae_int32x2 out2_32;

                      ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 0], bias_shift);
                      ae_int64 accu2 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 1], bias_shift);
                      ae_int64 accu3 = AE_SLAA64S(((ae_int64 *)p_bias)[row + 2], bias_shift);

                      multiply_three_row_one_vec_16x16
                          (&accu1
                           ,&accu2
                           ,&accu3
                           ,p_mat1
                           ,row_stride1
                           ,row
                           ,p_vec1
                           ,cols1
                          );

                      accu1 = AE_SLAA64S(accu1, acc_shift);
                      accu2 = AE_SLAA64S(accu2, acc_shift);
                      accu3 = AE_SLAA64S(accu3, acc_shift);

                      out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
                      out2_32 = AE_ROUND32X2F64SSYM(accu3, accu3);

                      p_out_scratch[(row+0)]  = AE_MOVAD32_H(out1_32);
                      p_out_scratch[(row+1)]  = AE_MOVAD32_L(out1_32);
                      p_out_scratch[(row+2)]  = AE_MOVAD32_H(out2_32);
                  }

                  for (; row < rows ; row++)
                  {
                      ae_int32x2 out1_32;

                      ae_int64 accu1 = AE_SLAA64S(((ae_int64 *)p_bias)[row], bias_shift);

                      multiply_one_row_one_vec_16x16
                          (&accu1
                           ,p_mat1
                           ,row_stride1
                           ,row
                           ,p_vec1
                           ,cols1
                          );

                      accu1 = AE_SLAA64S(accu1, acc_shift);
                      out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
                      p_out_scratch[(row+0)]  = out1_32;
                  }
              }
              else
              {
                  return -1;
              }

              break;
          }
  }

  xa_nn_vec_sigmoid_32_16((pWORD16) p_out, (pWORD32) p_scratch, rows);

  return 0;
}
