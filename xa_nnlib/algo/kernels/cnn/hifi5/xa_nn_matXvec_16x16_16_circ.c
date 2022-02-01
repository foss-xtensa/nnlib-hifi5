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
#include "xa_nnlib_common.h"
#define UNROLL_D  2

WORD32 xa_nn_matXvec_16x16_16_circ(
  WORD16 * __restrict__ p_out,
  WORD16 * __restrict__ p_mat,
  WORD16 * __restrict__ p_vec,
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

    /* Process two vectors at a time */
    for(vec = 0; vec < (vec_count & (~0x1)); vec+=2)
    {
        WORD16 *p_dst1 = (WORD16 *)&p_out[vec*out_col_offset];
        WORD16 *p_dst2 = (WORD16 *)&p_out[(vec+1)*out_col_offset];

        row = 0;

        /* Process 2 rows and 2 vectors */
        for (row = 0; row < ( rows & ~(UNROLL_D-1)) ; row+=UNROLL_D)
        {
            ae_valignx2 align_mat1, align_mat2;
            ae_valignx2 align_src1, align_src2;
            ae_int32x2 out1_32, out2_32;
            ae_int16x4 out;

            ae_int64 accu1 = AE_SLAA64S(p_bias[vec], bias_shift);
            ae_int64 accu2 = AE_SLAA64S(p_bias[vec], bias_shift);
            ae_int64 accu3 = AE_SLAA64S(p_bias[vec + 1], bias_shift);
            ae_int64 accu4 = AE_SLAA64S(p_bias[vec + 1], bias_shift);

            ae_int16x8 *p_src1 = (ae_int16x8 *)&p_vec[vec * vec_offset];
            ae_int16x8 *p_src2 = (ae_int16x8 *)&p_vec[(vec+1) * vec_offset];

            ae_int16x8 *p_mat1 = (ae_int16x8 *)p_mat;
            ae_int16x8 *p_mat2 = (ae_int16x8 *)p_mat;

            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1, (row + 0) * row_offset * sizeof(WORD16));
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat2, (row + 1) * row_offset * sizeof(WORD16));

            AE_LA16X4X2POS_PC(align_mat1, p_mat1);
            AE_LA16X4X2POS_PC(align_mat2, p_mat2);

            align_src1 = AE_LA128_PP(p_src1);
            align_src2 = AE_LA128_PP(p_src2);

#pragma ymemory (p_mat1)
#pragma ymemory (p_mat2)
            for (col = 0; col < (cols >> 3); col++)
            {
                ae_int16x4 in11, in12;
                ae_int16x4 in21, in22;

                ae_int16x4 src11, src12;
                ae_int16x4 src21, src22;

                AE_LA16X4X2_IC(in11 ,in12, align_mat1 ,p_mat1);
                AE_LA16X4X2_IC(in21, in22, align_mat2 ,p_mat2);

                AE_LA16X4X2_IP(src11, src12, align_src1, p_src1);
                AE_LA16X4X2_IP(src21, src22, align_src2, p_src2);

                AE_MULAAAA2Q16(accu1, accu2 , src11, src11, in11, in21);
                AE_MULAAAA2Q16(accu1, accu2 , src12, src12, in12, in22);
                AE_MULAAAA2Q16(accu3, accu4 , src21, src21, in11, in21);
                AE_MULAAAA2Q16(accu3, accu4 , src22, src22, in12, in22);
            }

            if(cols & 7)
            {
                ae_int16x8 *p_src1 = (ae_int16x8 *)&p_vec[vec * vec_offset +  (cols & ~7)];
                ae_int16x8 *p_src2 = (ae_int16x8 *)&p_vec[(vec+1) * vec_offset + (cols & ~7)];

                ae_int16x8 *p_mat1 = (ae_int16x8 *)p_mat;
                ae_int16x8 *p_mat2 = (ae_int16x8 *)p_mat;

                AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1, ((row + 0) * row_offset + (cols & ~7)) * sizeof(WORD16));
                AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat2, ((row + 1) * row_offset + (cols & ~7)) * sizeof(WORD16));

                for (col = 0; col < (cols & 7) ; col++)
                {
                    ae_int16x4 in11;
                    ae_int16x4 in21;

                    ae_int16x4 src11;
                    ae_int16x4 src21;

                    AE_L16_XC(in11, (ae_int16 *)p_mat1, 2);
                    AE_L16_XC(in21, (ae_int16 *)p_mat2, 2);

                    AE_L16_IP(src11, (ae_int16 *)p_src1, 2);
                    AE_L16_IP(src21, (ae_int16 *)p_src2, 2);

                    AE_MULA16_00(accu1, in11, src11);
                    AE_MULA16_00(accu2, in21, src11);
                    AE_MULA16_00(accu3, in11, src21);
                    AE_MULA16_00(accu4, in21, src21);
                }
            }

            accu1 = AE_SLAA64S(accu1, acc_shift);
            accu2 = AE_SLAA64S(accu2, acc_shift);
            accu3 = AE_SLAA64S(accu3, acc_shift);
            accu4 = AE_SLAA64S(accu4, acc_shift);

            out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
            out2_32 = AE_ROUND32X2F64SSYM(accu3, accu4);
            out = AE_SAT16X4(out1_32, out2_32);

            p_dst1[(row+0) * out_row_offset] = AE_MOVAD16_3(out);
            p_dst1[(row+1) * out_row_offset] = AE_MOVAD16_2(out);
            p_dst2[(row+0) * out_row_offset] = AE_MOVAD16_1(out);
            p_dst2[(row+1) * out_row_offset] = AE_MOVAD16_0(out);
        }

        /* Process 1 row and 2 vectors for remaining rows */
        for (; row < rows ; row++)
        {
            ae_valignx2 align_mat1;
            ae_valignx2 align_src1, align_src2;
            ae_int32x2 out1_32;
            ae_int16x4 out;

            ae_int16x8 *p_src1 = (ae_int16x8 *)&p_vec[vec * vec_offset];
            ae_int16x8 *p_src2 = (ae_int16x8 *)&p_vec[(vec+1) * vec_offset];

            ae_int64 accu1 = AE_SLAA64S(p_bias[vec], bias_shift);
            ae_int64 accu2 = AE_SLAA64S(p_bias[vec + 1], bias_shift);

            ae_int16x8 *p_mat1 = (ae_int16x8 *)p_mat;
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1, (row) * row_offset * sizeof(WORD16));

            AE_LA16X4X2POS_PC(align_mat1, p_mat1);

            align_src1 = AE_LA128_PP(p_src1);
            align_src2 = AE_LA128_PP(p_src2);

#pragma ymemory (p_mat1)
            for (col = 0; col < (cols >> 3); col++)
            {
                ae_int16x4 in11, in12;
                ae_int16x4 src11, src12;
                ae_int16x4 src21, src22;

                AE_LA16X4X2_IC(in11 ,in12, align_mat1 ,p_mat1);

                AE_LA16X4X2_IP(src11, src12, align_src1, p_src1);
                AE_LA16X4X2_IP(src21, src22, align_src2, p_src2);

                AE_MULAAAA2Q16(accu1, accu2 , src11, src21, in11, in11);
                AE_MULAAAA2Q16(accu1, accu2 , src12, src22, in12, in12);
            }

            if(cols & 7)
            {
                ae_int16x8 *p_src1 = (ae_int16x8 *)&p_vec[vec * vec_offset +  (cols & ~7)];
                ae_int16x8 *p_src2 = (ae_int16x8 *)&p_vec[(vec+1) * vec_offset + (cols & ~7)];

                ae_int16x8 *p_mat1 = (ae_int16x8 *)p_mat;

                AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1, ((row + 0) * row_offset + (cols & ~7)) * sizeof(WORD16));

                for (col = 0; col < (cols & 7) ; col++)
                {
                    ae_int16x4 in11;
                    ae_int16x4 src11;
                    ae_int16x4 src21;

                    AE_L16_XC(in11, (ae_int16 *)p_mat1, 2);

                    AE_L16_IP(src11, (ae_int16 *)p_src1, 2);
                    AE_L16_IP(src21, (ae_int16 *)p_src2, 2);

                    AE_MULA16_00(accu1, in11, src11);
                    AE_MULA16_00(accu2, in11, src21);
                }
            }

            accu1 = AE_SLAA64S(accu1, acc_shift);
            accu2 = AE_SLAA64S(accu2, acc_shift);
            out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
            out = AE_SAT16X4(out1_32, out1_32);

            p_dst1[(row+0) * out_row_offset] = AE_MOVAD16_3(out);
            p_dst2[(row+0) * out_row_offset] = AE_MOVAD16_2(out);
        }
    }

    if(vec_count & 0x1)
    {
        WORD16 *p_dst1 = (WORD16 *)&p_out[vec*out_col_offset];
        row = 0;

        /* Process 3 rows and 1 vector at a time, utilising the four aligning registers */
        for (row = 0; row < (rows - 2) ; row += 3)
        {
            ae_valignx2 align_mat1, align_mat2, align_mat3;
            ae_valignx2 align_src1;
            ae_int32x2 out1_32, out2_32;
            ae_int16x4 out;


            ae_int16x8 *p_src1 = (ae_int16x8 *)&p_vec[vec *vec_offset];

            ae_int64 accu1 = AE_SLAA64S(p_bias[vec], bias_shift);
            ae_int64 accu2 = AE_SLAA64S(p_bias[vec], bias_shift);
            ae_int64 accu3 = AE_SLAA64S(p_bias[vec], bias_shift);
            ae_int64 accu4 = AE_ZERO64();

            ae_int16x8 *p_mat1 = (ae_int16x8 *)p_mat;
            ae_int16x8 *p_mat2 = (ae_int16x8 *)p_mat;
            ae_int16x8 *p_mat3 = (ae_int16x8 *)p_mat;

            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1, (row) * row_offset * sizeof(WORD16));
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat2, (row + 1) * row_offset * sizeof(WORD16));
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat3, (row + 2) * row_offset * sizeof(WORD16));

            AE_LA16X4X2POS_PC(align_mat1, p_mat1);
            AE_LA16X4X2POS_PC(align_mat2, p_mat2);
            AE_LA16X4X2POS_PC(align_mat3, p_mat3);

            align_src1 = AE_LA128_PP(p_src1);

#pragma ymemory (p_mat1)
#pragma ymemory (p_mat2)
#pragma ymemory (p_mat3)
            for (col = 0; col < (cols>>3); col++)
            {
                ae_int16x4 in11, in12;
                ae_int16x4 in21, in22;
                ae_int16x4 in31, in32;
                ae_int16x4 src11, src12;

                AE_LA16X4X2_IC(in11 ,in12, align_mat1 ,p_mat1);
                AE_LA16X4X2_IC(in21, in22, align_mat2 ,p_mat2);
                AE_LA16X4X2_IC(in31, in32, align_mat3 ,p_mat3);

                AE_LA16X4X2_IP(src11, src12, align_src1, p_src1);

                AE_MULAAAA2Q16(accu1, accu2 , src11, src11, in11, in21);
                AE_MULAAAA2Q16(accu1, accu2 , src12, src12, in12, in22);
                AE_MULAAAA2Q16(accu3, accu4 , src11, src12, in31, in32);
            }

            if(cols & 7)
            {
                ae_int16x8 *p_src1 = (ae_int16x8 *)&p_vec[vec * vec_offset +  (cols & ~7)];

                ae_int16x8 *p_mat1 = (ae_int16x8 *)p_mat;
                ae_int16x8 *p_mat2 = (ae_int16x8 *)p_mat;
                ae_int16x8 *p_mat3 = (ae_int16x8 *)p_mat;

                AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1, ((row + 0) * row_offset + (cols & ~7)) * sizeof(WORD16));
                AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat2, ((row + 1) * row_offset + (cols & ~7)) * sizeof(WORD16));
                AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat3, ((row + 2) * row_offset + (cols & ~7)) * sizeof(WORD16));

                for (col = 0; col < (cols & 7) ; col++)
                {
                    ae_int16x4 in11;
                    ae_int16x4 in21;
                    ae_int16x4 in31;
                    ae_int16x4 src11;

                    AE_L16_XC(in11, (ae_int16 *)p_mat1, 2);
                    AE_L16_XC(in21, (ae_int16 *)p_mat2, 2);
                    AE_L16_XC(in31, (ae_int16 *)p_mat3, 2);

                    AE_L16_IP(src11, (ae_int16 *)p_src1, 2);

                    AE_MULA16_00(accu1, in11, src11);
                    AE_MULA16_00(accu2, in21, src11);
                    AE_MULA16_00(accu3, in31, src11);
                }
            }

            accu3 = AE_ADD64S(accu3, accu4);
            accu1 = AE_SLAA64S(accu1, acc_shift);
            accu2 = AE_SLAA64S(accu2, acc_shift);
            accu3 = AE_SLAA64S(accu3, acc_shift);

            out1_32 = AE_ROUND32X2F64SSYM(accu1, accu2);
            out2_32 = AE_ROUND32X2F64SSYM(accu3, accu3);
            out = AE_SAT16X4(out1_32, out2_32);

            p_dst1[(row+0) * out_row_offset] = AE_MOVAD16_3(out);
            p_dst1[(row+1) * out_row_offset] = AE_MOVAD16_2(out);
            p_dst1[(row+2) * out_row_offset] = AE_MOVAD16_1(out);
        }

        /* Processing 1 row and 1 vec */
        for (; row < rows ; row++)
        {
            ae_valignx2 align_mat1;
            ae_valignx2 align_src1;
            ae_int32x2 out1_32;
            ae_int16x4 out;


            ae_int16x8 *p_src1 = (ae_int16x8 *)&p_vec[vec * vec_offset];

            ae_int16x8 *p_mat1 = (ae_int16x8 *)p_mat;
            AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1, (row) * row_offset * sizeof(WORD16));

            ae_int64 accu1 = AE_SLAA64S(p_bias[vec], bias_shift);
            ae_int64 accu2 = AE_ZERO64();

            AE_LA16X4X2POS_PC(align_mat1, p_mat1);

            align_src1 = AE_LA128_PP(p_src1);

            for (col = 0; col < (cols>>3); col++)
            {
                ae_int16x4 in11, in12;
                ae_int16x4 src11, src12;

                AE_LA16X4X2_IC(in11 ,in12, align_mat1 ,p_mat1);
                AE_LA16X4X2_IP(src11, src12, align_src1, p_src1);
                AE_MULAAAA2Q16(accu1, accu2 , src11, src12, in11, in12);
            }

            if(cols & 7)
            {
                ae_int16x8 *p_src1 = (ae_int16x8 *)&p_vec[vec * vec_offset +  (cols & ~7)];

                ae_int16x8 *p_mat1 = (ae_int16x8 *)p_mat;

                AE_ADDCIRC16X4_XC((ae_int16x4 *)p_mat1, ((row + 0) * row_offset + (cols & ~7)) * sizeof(WORD16));

                for (col = 0; col < (cols & 7) ; col++)
                {
                    ae_int16x4 in11;
                    ae_int16x4 src11;

                    AE_L16_XC(in11, (ae_int16 *)p_mat1, 2);
                    AE_L16_IP(src11, (ae_int16 *)p_src1, 2);
                    AE_MULA16_00(accu1, in11, src11);
                }
            }

            accu1 = AE_ADD64S(accu1, accu2);
            accu1 = AE_SLAA64S(accu1, acc_shift);
            out1_32 = AE_ROUND32X2F64SSYM(accu1, accu1);
            out = AE_SAT16X4(out1_32, out1_32);
            p_dst1[(row+0) * out_row_offset] = AE_MOVAD16_0(out);
        }
    }

    return 0;
}

