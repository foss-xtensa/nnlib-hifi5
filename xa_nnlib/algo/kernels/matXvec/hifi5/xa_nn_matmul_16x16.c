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
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common.h"

WORD32 xa_nn_matmul_16x16_16(
         WORD16 * __restrict__ p_out,          
         const WORD16 * __restrict__ p_mat1,   
         const WORD16 * __restrict__ p_vec1,   
         const WORD16 * __restrict__ p_bias,   
         WORD32 rows,
         WORD32 cols1,
         WORD32 row_stride1,                   
         WORD32 acc_shift,                     
         WORD32 bias_shift,                    
         WORD32 vec_count,
         WORD32 vec_offset,
         WORD32 out_offset,
         WORD32 out_stride)                      
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
    XA_NNLIB_ARG_CHK_COND((acc_shift < -31 || acc_shift > 31), -1);
    XA_NNLIB_ARG_CHK_COND((bias_shift < -31 || bias_shift > 31), -1);
    XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
  
    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    vec_itr = 0;

    acc_shift = acc_shift + 32;
    acc_shift = acc_shift > 63 ? 63 : acc_shift < -63 ? -63 : acc_shift;
    bias_shift = bias_shift > 63 ? 63 : bias_shift < -63 ? -63 : bias_shift;
    if(vec_count > 2)
    {
        for (vec_itr = 0; vec_itr < (vec_count & ~(2-1)); vec_itr += 2)
        {
            ae_int16 bias = (0);
            ae_int64 sat_bias = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
            ae_int16 *pbias = (ae_int16 *) p_bias;
            for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
            {
                ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_0_1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_1_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_1_1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));

                ae_int16x4 vec_batch_0_0  = AE_MOVDA16(0);
                ae_int16x4 vec_batch_0_1  = AE_MOVDA16(0);
                ae_int16x4 vec_batch_1_0  = AE_MOVDA16(0);
                ae_int16x4 vec_batch_1_1  = AE_MOVDA16(0);
                ae_int16x4 mat1_0_0 = AE_MOVDA16(0);
                ae_int16x4 mat1_0_1 = AE_MOVDA16(0);
                ae_int16x4 mat1_1_0 = AE_MOVDA16(0);
                ae_int16x4 mat1_1_1 = AE_MOVDA16(0);

                ae_int16x8 *p_vec_batch_0  = (ae_int16x8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int16x8 *p_vec_batch_1  = (ae_int16x8 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                ae_int16x8 *p_mat1_0 = (ae_int16x8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_int16x8 *p_mat1_1 = (ae_int16x8 *) &p_mat1[(m_itr+1)*row_stride1];

                ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
                ae_valignx2 align_vec1 = AE_LA128_PP(p_vec_batch_1);
                ae_valignx2 align_mat_0 = AE_LA128_PP(p_mat1_0);
                ae_valignx2 align_mat_1 = AE_LA128_PP(p_mat1_1);

                int cols1_count = cols1- cols1%8;
                for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
                {
                    AE_LA16X4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
                    AE_LA16X4X2_IP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1);
                    AE_LA16X4X2_IP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
                    AE_LA16X4X2_IP(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);
                    AE_MULAAAA2Q16(acc_0_0, acc_1_0, vec_batch_0_0, vec_batch_0_0, mat1_0_0, mat1_1_0);
                    AE_MULAAAA2Q16(acc_0_0, acc_1_0, vec_batch_0_1, vec_batch_0_1, mat1_0_1, mat1_1_1);
                    AE_MULAAAA2Q16(acc_0_1, acc_1_1, vec_batch_1_0, vec_batch_1_0, mat1_0_0, mat1_1_0);
                    AE_MULAAAA2Q16(acc_0_1, acc_1_1, vec_batch_1_1, vec_batch_1_1, mat1_0_1, mat1_1_1);
                }

                if(cols1%8 !=0)
                {
                    AE_LAV16X4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols1%8)*2);
                    AE_LAV16X4X2_XP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1, (cols1%8)*2);
                    AE_LAV16X4X2_XP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0, (cols1%8)*2);
                    AE_LAV16X4X2_XP(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1, (cols1%8)*2);
                    AE_MULAAAA2Q16(acc_0_0, acc_1_0, vec_batch_0_0, vec_batch_0_0, mat1_0_0, mat1_1_0);
                    AE_MULAAAA2Q16(acc_0_0, acc_1_0, vec_batch_0_1, vec_batch_0_1, mat1_0_1, mat1_1_1);
                    AE_MULAAAA2Q16(acc_0_1, acc_1_1, vec_batch_1_0, vec_batch_1_0, mat1_0_0, mat1_1_0);
                    AE_MULAAAA2Q16(acc_0_1, acc_1_1, vec_batch_1_1, vec_batch_1_1, mat1_0_1, mat1_1_1);
                }
                if(p_bias!=NULL)
                {
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_0_0 = AE_ADD64S(acc_0_0, sat_bias);
                    acc_0_1 = AE_ADD64S(acc_0_1, sat_bias);
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_1_0 = AE_ADD64S(acc_1_0, sat_bias);
                    acc_1_1 = AE_ADD64S(acc_1_1, sat_bias);
                }
                ae_int32 tmp_var_0_0_0;
                ae_f32x2 tmp_var_0_0_1 =
                AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_0, acc_shift)), 16);
                tmp_var_0_0_0 = AE_SRAI32(tmp_var_0_0_1, 16);
                (*((WORD16 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD16)(*((UWORD32 *)&tmp_var_0_0_0));
                ae_int32 tmp_var_1_0_0;
                ae_f32x2 tmp_var_1_0_1 =
                AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(acc_1_0, acc_shift)), 16);
                tmp_var_1_0_0 = AE_SRAI32(tmp_var_1_0_1, 16);
                (*((WORD16 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride)) = (WORD16)(*((UWORD32 *)&tmp_var_1_0_0));
                ae_int32 tmp_var_0_1_0;
                ae_f32x2 tmp_var_0_1_1 =
                AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_1, acc_shift)), 16);
                tmp_var_0_1_0 = AE_SRAI32(tmp_var_0_1_1, 16);
                (*((WORD16 *) p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride)) = (WORD16)(*((UWORD32 *)&tmp_var_0_1_0));
                ae_int32 tmp_var_1_1_0;
                ae_f32x2 tmp_var_1_1_1 =
                AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(acc_1_1, acc_shift)), 16);
                tmp_var_1_1_0 = AE_SRAI32(tmp_var_1_1_1, 16);
                (*((WORD16 *) p_out + (vec_itr + 1)*out_offset + (m_itr + 1)*out_stride)) = (WORD16)(*((UWORD32 *)&tmp_var_1_1_0));
            }
            //Remaining row
            for(; m_itr < rows; m_itr++)
            {
                ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_0_1 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int16x4 vec_batch_0_0  = AE_MOVDA16(0);
                ae_int16x4 vec_batch_0_1  = AE_MOVDA16(0);
                ae_int16x8 *p_vec_batch_0  = (ae_int16x8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
                ae_int16x4 vec_batch_1_0  = AE_MOVDA16(0);
                ae_int16x4 vec_batch_1_1  = AE_MOVDA16(0);
                ae_int16x8 *p_vec_batch_1  = (ae_int16x8 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                ae_valignx2 align_vec1 = AE_LA128_PP(p_vec_batch_1);
                ae_int16x4 mat1_0_0 = AE_MOVDA16(0);
                ae_int16x4 mat1_0_1 = AE_MOVDA16(0);
                ae_int16x8 *p_mat1_0 = (ae_int16x8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_valignx2 align_mat_0 = AE_LA128_PP(p_mat1_0);
                int cols1_count = cols1- cols1%8;

                for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
                {
                    AE_LA16X4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
                    AE_LA16X4X2_IP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1);
                    AE_LA16X4X2_IP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
                    AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_0, mat1_0_0, vec_batch_0_0, vec_batch_1_0);
                    AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_1, mat1_0_1, vec_batch_0_1, vec_batch_1_1);
                }
                if(cols1%8 != 0)
                {
                    AE_LAV16X4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols1%8 *2));
                    AE_LAV16X4X2_XP(vec_batch_1_0, vec_batch_1_1, align_vec1, p_vec_batch_1, (cols1%8 *2));
                    AE_LAV16X4X2_XP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0, (cols1%8) *2);
                    AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_0, mat1_0_0, vec_batch_0_0, vec_batch_1_0);
                    AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_1, mat1_0_1, vec_batch_0_1, vec_batch_1_1);
                }

                ae_int16_loadip(bias, pbias, 2);
                sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                if(p_bias!=NULL)
                {
                  acc_0_0 = AE_ADD64S(acc_0_0, sat_bias);
                  acc_0_1 = AE_ADD64S(acc_0_1, sat_bias);
                }
                ae_int32 tmp_var_0_0_0;
                ae_f32x2 tmp_var_0_0_1 =
                AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_0, acc_shift)), 16);
                tmp_var_0_0_0 = AE_SRAI32(tmp_var_0_0_1, 16);
                (*((WORD16 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD16)(*((UWORD32 *)&tmp_var_0_0_0));
                ae_int32 tmp_var_0_1_0;
                ae_f32x2 tmp_var_0_1_1 =
                AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_1, acc_shift)), 16);
                tmp_var_0_1_0 = AE_SRAI32(tmp_var_0_1_1, 16);
                (*((WORD16 *) p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride)) = (WORD16)(*((UWORD32 *)&tmp_var_0_1_0));
            }

        }
    }
    {
        /* Tail loop for vec unroll */
        for(; vec_itr < vec_count; vec_itr++)
        {
            ae_int16 bias = (0);
            ae_int64 sat_bias = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
            ae_int16 *pbias = (ae_int16 *) p_bias;
            for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
            {
                ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 acc_1_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int16x4 vec_batch_0_0  = AE_MOVDA16(0);
                ae_int16x4 vec_batch_0_1  = AE_MOVDA16(0);
                ae_int16x8 *p_vec_batch_0  = (ae_int16x8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
                ae_int16x4 mat1_0_0 = AE_MOVDA16(0);
                ae_int16x4 mat1_0_1 = AE_MOVDA16(0);
                ae_int16x8 *p_mat1_0 = (ae_int16x8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_valignx2 align_mat_0 = AE_LA128_PP(p_mat1_0);
                ae_int16x4 mat1_1_0 = AE_MOVDA16(0);
                ae_int16x4 mat1_1_1 = AE_MOVDA16(0);
                ae_int16x8 *p_mat1_1 = (ae_int16x8 *) &p_mat1[(m_itr+1)*row_stride1];
                ae_valignx2 align_mat_1 = AE_LA128_PP(p_mat1_1);
                int cols1_count = cols1 - cols1%8;

                for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
                {
                    AE_LA16X4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
                    AE_LA16X4X2_IP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
                    AE_LA16X4X2_IP(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1);
                    AE_MULAAAA2Q16(acc_0_0, acc_1_0, vec_batch_0_0, vec_batch_0_0, mat1_0_0, mat1_1_0);
                    AE_MULAAAA2Q16(acc_0_0, acc_1_0, vec_batch_0_1, vec_batch_0_1, mat1_0_1, mat1_1_1);
                }
                if(cols1%8 != 0)
                {
                    AE_LAV16X4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols1%8) * 2);
                    AE_LAV16X4X2_XP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0, (cols1%8) * 2);
                    AE_LAV16X4X2_XP(mat1_1_0, mat1_1_1, align_mat_1, p_mat1_1, (cols1%8) * 2);
                    AE_MULAAAA2Q16(acc_0_0, acc_1_0, vec_batch_0_0, vec_batch_0_0, mat1_0_0, mat1_1_0);
                    AE_MULAAAA2Q16(acc_0_0, acc_1_0, vec_batch_0_1, vec_batch_0_1, mat1_0_1, mat1_1_1);
                }
                if(p_bias!=(void *)0)
                {
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_0_0 = AE_ADD64S(acc_0_0, sat_bias);
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_1_0 = AE_ADD64S(acc_1_0, sat_bias);
                }
                ae_int32 tmp_var_0_0_0;
                ae_f32x2 tmp_var_0_0_1 =
                AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_0, acc_shift)), 16);
                tmp_var_0_0_0 = AE_SRAI32(tmp_var_0_0_1, 16);
                (*((WORD16 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD16)(*((UWORD32 *)&tmp_var_0_0_0));
                ae_int32 tmp_var_1_0_0;
                ae_f32x2 tmp_var_1_0_1 =
                AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(acc_1_0, acc_shift)), 16);
                tmp_var_1_0_0 = AE_SRAI32(tmp_var_1_0_1, 16);
                (*((WORD16 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride)) = (WORD16)(*((UWORD32 *)&tmp_var_1_0_0));
            }

            for(; m_itr < rows; m_itr++)
            {
                ae_int64 acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int64 dummy_acc_0_0 = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
                ae_int16x4 vec_batch_0_0  = AE_MOVDA16(0);
                ae_int16x4 vec_batch_0_1  = AE_MOVDA16(0);
                ae_int16x8 *p_vec_batch_0  = (ae_int16x8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_valignx2 align_vec0 = AE_LA128_PP(p_vec_batch_0);
                ae_int16x4 mat1_0_0 = AE_MOVDA16(0);
                ae_int16x4 mat1_0_1 = AE_MOVDA16(0);
                ae_int16x8 *p_mat1_0 = (ae_int16x8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_valignx2 align_mat_0 = AE_LA128_PP(p_mat1_0);
                int cols1_count = cols1 - cols1%8;

                for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
                {
                    AE_LA16X4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0);
                    AE_LA16X4X2_IP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0);
                    AE_MULAAAA2Q16(acc_0_0, dummy_acc_0_0, vec_batch_0_0, vec_batch_0_0, mat1_0_0, mat1_0_0);
                    AE_MULAAAA2Q16(acc_0_0, dummy_acc_0_0, vec_batch_0_1, vec_batch_0_1, mat1_0_1, mat1_0_1);
                }
                if(cols1%8 != 0)
                {
                    AE_LAV16X4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec_batch_0, (cols1%8 * 2));
                    AE_LAV16X4X2_XP(mat1_0_0, mat1_0_1, align_mat_0, p_mat1_0, (cols1%8 * 2));
                    AE_MULAAAA2Q16(acc_0_0, dummy_acc_0_0, vec_batch_0_0, vec_batch_0_0, mat1_0_0, mat1_0_0);
                    AE_MULAAAA2Q16(acc_0_0, dummy_acc_0_0, vec_batch_0_1, vec_batch_0_1, mat1_0_1, mat1_0_1);
                }
                if(p_bias!=(void *)0)
                {
                  ae_int16_loadip(bias, pbias, 2);
                  sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                  acc_0_0 = AE_ADD64S(acc_0_0, sat_bias);
                }
                ae_int32 tmp_var_0_0_0;
                ae_f32x2 tmp_var_0_0_1 =
                AE_SLAI32S(AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_0, acc_shift)), 16);
                tmp_var_0_0_0 = AE_SRAI32(tmp_var_0_0_1, 16);
                (*((WORD16 *) p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride)) = (WORD16)(*((UWORD32 *)&tmp_var_0_0_0));
            }
        }
    }
  return 0;
}
