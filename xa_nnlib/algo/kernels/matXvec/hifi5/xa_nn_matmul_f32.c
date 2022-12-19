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
#include "common_fpu.h"
#include "xa_nnlib_common.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_matmul_f32xf32_f32,(
    FLOAT32 * __restrict__ p_out,        
    const FLOAT32 * __restrict__ p_mat1, 
    const FLOAT32 * __restrict__ p_vec1, 
    const FLOAT32 * __restrict__ p_bias, 
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,                   
    WORD32 vec_count,                     
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride))                      

#else
WORD32 xa_nn_matmul_f32xf32_f32(
    FLOAT32 * __restrict__ p_out,          
    const FLOAT32 * __restrict__ p_mat1,   
    const FLOAT32 * __restrict__ p_vec1,   
    const FLOAT32 * __restrict__ p_bias,   
    WORD32 rows,
    WORD32 cols1,
    WORD32 row_stride1,                    
    WORD32 vec_count,                      
    WORD32 vec_offset,
    WORD32 out_offset,
    WORD32 out_stride)                      
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
    XA_NNLIB_ARG_CHK_COND((vec_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_offset == 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_stride == 0), -1);
  
    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
    xtfloat* p_out_tmp;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    vec_itr = 0;

        if(vec_count > 2)
        {
            for (vec_itr = 0; vec_itr < (vec_count & ~(2-1)); vec_itr += 2)
            {
                xtfloat bias = 0.0f;
                xtfloat *pbias = (xtfloat *) p_bias;
                for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
                {
                    xtfloatx2 acc_0_0_0 = 0.0f, acc_0_1_0 = 0.0f, acc_1_0_0 = 0.0f, acc_1_1_0 = 0.0f;
                    xtfloatx2 vec_batch_1_0 = 0.0f, vec_batch_0_0 = 0.0f;
                    xtfloatx2 mat1_0_0 = 0.0f, mat1_1_0 = 0.0f;
                    xtfloat acc_0_0_1 = 0.0f, acc1_0_0 = 0.0f, acc_0_1_1 = 0.0f, acc1_0_1 = 0.0f, acc_1_0_1 = 0.0f;
                    xtfloat acc1_1_0 = 0.0f, acc_1_1_1 = 0.0f, acc1_1_1 = 0.0f;
                    xtfloat mat1_0_1 = 0.0f, mat1_1_1 = 0.0f;
                    xtfloat vec_batch_0_1 = 0.0f, vec_batch_1_1 = 0.0;
                    xtfloat *p_vec_batch_0_1, *p_vec_batch_1_1;
                    xtfloat *p_mat1_0_1, *p_mat1_1_1;
                    xtfloatx2 *p_vec_batch_0_0  = (xtfloatx2 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                    xtfloatx2 *p_vec_batch_1_0  = (xtfloatx2 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                    xtfloatx2 *p_mat1_0_0 = (xtfloatx2 *) &p_mat1[(m_itr+0)*row_stride1];
                    xtfloatx2 *p_mat1_1_0 = (xtfloatx2 *) &p_mat1[(m_itr+1)*row_stride1];
                    ae_valign align_vec_batch_0_0 = AE_LA64_PP(p_vec_batch_0_0);
                    ae_valign align_vec_batch_1_0 = AE_LA64_PP(p_vec_batch_1_0);
                    ae_valign align_mat1_0_0 = AE_LA64_PP(p_mat1_0_0);
                    ae_valign align_mat1_1_0 = AE_LA64_PP(p_mat1_1_0);

                    int cols1_count = cols1- cols1%2;
                    for(c_itr = 0; c_itr < (cols1_count >> 1); c_itr++)
                    {
                        XT_LASX2IP(vec_batch_0_0, align_vec_batch_0_0, p_vec_batch_0_0);
                        XT_LASX2IP(vec_batch_1_0, align_vec_batch_1_0, p_vec_batch_1_0);
                        XT_LASX2IP(mat1_0_0, align_mat1_0_0, p_mat1_0_0);
                        XT_LASX2IP(mat1_1_0, align_mat1_1_0, p_mat1_1_0);
                        XT_MADD_SX2(acc_0_0_0, vec_batch_0_0, mat1_0_0);
                        XT_MADD_SX2(acc_1_0_0, vec_batch_0_0, mat1_1_0);
                        XT_MADD_SX2(acc_0_1_0, vec_batch_1_0, mat1_0_0);
                        XT_MADD_SX2(acc_1_1_0, vec_batch_1_0, mat1_1_0);
                    }
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        p_vec_batch_0_1 = (xtfloat *)p_vec_batch_0_0;
                        XT_LSIP(vec_batch_0_1, p_vec_batch_0_1, 4);
                        p_vec_batch_1_1 = (xtfloat *)p_vec_batch_1_0;
                        XT_LSIP(vec_batch_1_1, p_vec_batch_1_1, 4);
                        p_mat1_0_1 = (xtfloat *)p_mat1_0_0;
                        XT_LSIP(mat1_0_1, p_mat1_0_1, 4);
                        p_mat1_1_1 = (xtfloat *)p_mat1_1_0;
                        XT_LSIP(mat1_1_1, p_mat1_1_1, 4);
                        XT_MADD_S(acc_0_0_1, vec_batch_0_1, mat1_0_1);
                        XT_MADD_S(acc_1_0_1, vec_batch_0_1, mat1_1_1);
                        XT_MADD_S(acc_0_1_1, vec_batch_1_1, mat1_0_1);
                        XT_MADD_S(acc_1_1_1, vec_batch_1_1, mat1_1_1);
                    }
                    XT_LSIP(bias, pbias, 4);
                    acc1_0_0 = XT_RADD_SX2(acc_0_0_0);
                    acc_0_0_1 = XT_ADD_S(acc1_0_0, acc_0_0_1);
                    if(p_bias!=NULL)
                    {
                      acc_0_0_1 = XT_ADD_S(acc_0_0_1, bias);
                    }
                    acc1_0_1 = XT_RADD_SX2(acc_0_1_0);
                    acc_0_1_1 = XT_ADD_S(acc1_0_1, acc_0_1_1);
                    if(p_bias!=NULL)
                    {
                      acc_0_1_1 = XT_ADD_S(acc_0_1_1, bias);
                    }
                    XT_LSIP(bias, pbias, 4);
                    acc1_1_0 = XT_RADD_SX2(acc_1_0_0);
                    acc_1_0_1 = XT_ADD_S(acc1_1_0, acc_1_0_1);
                    if(p_bias!=NULL)
                    {
                      acc_1_0_1 = XT_ADD_S(acc_1_0_1, bias);
                    }
                     acc1_1_1 = XT_RADD_SX2(acc_1_1_0);
                     acc_1_1_1 = XT_ADD_S(acc1_1_1, acc_1_1_1);
                     if(p_bias!=NULL)
                     {
                       acc_1_1_1 = XT_ADD_S(acc_1_1_1, bias);
                     }
                    p_out_tmp = p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride;
                    XT_SSIP(acc_0_0_1,p_out_tmp,0);
                    p_out_tmp = p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride;
                    XT_SSIP(acc_1_0_1,p_out_tmp,0);
                    p_out_tmp = p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride;
                    XT_SSIP(acc_0_1_1,p_out_tmp,0);
                    p_out_tmp = p_out + (vec_itr + 1)*out_offset + (m_itr + 1)*out_stride;
                    XT_SSIP(acc_1_1_1,p_out_tmp,0);
                }
                //Remaining row
                for(; m_itr < rows; m_itr++)
                {
                    xtfloatx2 acc_0_0_0 = 0.0f, acc_0_1_0 = 0.0f;
                    xtfloatx2 vec_batch_0_0 = 0.0f, vec_batch_1_0 = 0.0f;
                    xtfloatx2 mat1_0_0 = 0.0f;
                    xtfloat acc_0_0_1 = 0.0f, acc1_0_0 = 0.0f, acc_0_1_1 = 0.0f, acc1_0_1 = 0.0f;
                    xtfloat vec_batch_0_1 = 0.0f, vec_batch_1_1 = 0.0f;
                    xtfloat mat1_0_1 = 0.0f;
                    xtfloat *p_vec_batch_0_1, *p_vec_batch_1_1;
                    xtfloat *p_mat1_0_1;
                    xtfloatx2 *p_vec_batch_0_0  = (xtfloatx2 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                    xtfloatx2 *p_vec_batch_1_0  = (xtfloatx2 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                    xtfloatx2 *p_mat1_0_0 = (xtfloatx2 *) &p_mat1[(m_itr+0)*row_stride1];
                    ae_valign align_vec_batch_0_0 = AE_LA64_PP(p_vec_batch_0_0);
                    ae_valign align_vec_batch_1_0 = AE_LA64_PP(p_vec_batch_1_0);
                    ae_valign align_mat1_0_0 = AE_LA64_PP(p_mat1_0_0);

                    int cols1_count = cols1- cols1%2;

                    for(c_itr = 0; c_itr < (cols1_count >> 1); c_itr++)
                    {
                        XT_LASX2IP(vec_batch_0_0, align_vec_batch_0_0, p_vec_batch_0_0);
                        XT_LASX2IP(vec_batch_1_0, align_vec_batch_1_0, p_vec_batch_1_0);
                        XT_LASX2IP(mat1_0_0, align_mat1_0_0, p_mat1_0_0);
                        XT_MADD_SX2(acc_0_0_0, vec_batch_0_0, mat1_0_0);
                        XT_MADD_SX2(acc_0_1_0, vec_batch_1_0, mat1_0_0);
                    }
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        p_vec_batch_0_1 = (xtfloat *)p_vec_batch_0_0;
                        XT_LSIP(vec_batch_0_1, p_vec_batch_0_1, 4);
                        p_vec_batch_1_1 = (xtfloat *)p_vec_batch_1_0;
                        XT_LSIP(vec_batch_1_1, p_vec_batch_1_1, 4);
                        p_mat1_0_1 = (xtfloat *)p_mat1_0_0;
                        XT_LSIP(mat1_0_1, p_mat1_0_1, 4);
                        XT_MADD_S(acc_0_0_1, vec_batch_0_1, mat1_0_1);
                        XT_MADD_S(acc_0_1_1, vec_batch_1_1, mat1_0_1);
                    }
                    XT_LSIP(bias, pbias, 4);
                    acc1_0_0 = XT_RADD_SX2(acc_0_0_0);
                    acc_0_0_1 = XT_ADD_S(acc1_0_0, acc_0_0_1);
                    if(p_bias!=NULL)
                    {
                      acc_0_0_1 = XT_ADD_S(acc_0_0_1, bias);
                    }
                    acc1_0_1 = XT_RADD_SX2(acc_0_1_0);
                    acc_0_1_1 = XT_ADD_S(acc1_0_1, acc_0_1_1);
                    if(p_bias!=NULL)
                    {
                      acc_0_1_1 = XT_ADD_S(acc_0_1_1, bias);
                    }
                    p_out_tmp = p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride;
                    XT_SSIP(acc_0_0_1,p_out_tmp,0);
                    p_out_tmp = p_out + (vec_itr + 1)*out_offset + (m_itr + 0)*out_stride;
                    XT_SSIP(acc_0_1_1,p_out_tmp,0);

                }

            }
        }
        {
            /* Tail loop for vec unroll */
            for(; vec_itr < vec_count; vec_itr++)
            {
                xtfloat bias = 0.0f;
                xtfloat *pbias = (xtfloat *) p_bias;
                for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
                {
                    xtfloatx2 acc_0_0_0 = 0.0f, acc_1_0_0 = 0.0f;
                    xtfloatx2 mat1_0_0 = 0.0f, mat1_1_0 = 0.0f;
                    xtfloatx2 vec_batch_0_0 = 0.0f;
                    xtfloat acc_0_0_1 = 0.0f, acc1_0_0 = 0.0f, acc_1_0_1 = 0.0f, acc1_1_0 = 0.0f;
                    xtfloat vec_batch_0_1 = 0.0f;
                    xtfloat mat1_0_1 = 0.0f, mat1_1_1 = 0.0f;
                    xtfloat *p_vec_batch_0_1;
                    xtfloat *p_mat1_0_1, *p_mat1_1_1;
                    xtfloatx2 *p_vec_batch_0_0  = (xtfloatx2 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                    xtfloatx2 *p_mat1_0_0 = (xtfloatx2 *) &p_mat1[(m_itr+0)*row_stride1];
                    xtfloatx2 *p_mat1_1_0 = (xtfloatx2 *) &p_mat1[(m_itr+1)*row_stride1];
                    ae_valign align_vec_batch_0_0 = AE_LA64_PP(p_vec_batch_0_0);
                    ae_valign align_mat1_0_0 = AE_LA64_PP(p_mat1_0_0);
                    ae_valign align_mat1_1_0 = AE_LA64_PP(p_mat1_1_0);

                    int cols1_count = cols1 - cols1%2;

                    for(c_itr = 0; c_itr < (cols1_count >> 1); c_itr++)
                    {
                        XT_LASX2IP(vec_batch_0_0, align_vec_batch_0_0, p_vec_batch_0_0);
                        XT_LASX2IP(mat1_0_0, align_mat1_0_0, p_mat1_0_0);
                        XT_LASX2IP(mat1_1_0, align_mat1_1_0, p_mat1_1_0);
                        XT_MADD_SX2(acc_0_0_0, vec_batch_0_0, mat1_0_0);
                        XT_MADD_SX2(acc_1_0_0, vec_batch_0_0, mat1_1_0);
                    }
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        p_vec_batch_0_1 = (xtfloat *)p_vec_batch_0_0;
                        XT_LSIP(vec_batch_0_1, p_vec_batch_0_1, 4);
                        p_mat1_0_1 = (xtfloat *)p_mat1_0_0;
                        XT_LSIP(mat1_0_1, p_mat1_0_1, 4);
                        p_mat1_1_1 = (xtfloat *)p_mat1_1_0;
                        XT_LSIP(mat1_1_1, p_mat1_1_1, 4);
                        XT_MADD_S(acc_0_0_1, vec_batch_0_1, mat1_0_1);
                        XT_MADD_S(acc_1_0_1, vec_batch_0_1, mat1_1_1);
                    }  
                    if(p_bias!=(void *)0)
                    {
                        XT_LSIP(bias, pbias, 4);
                    }
                    acc1_0_0 = XT_RADD_SX2(acc_0_0_0);
                    acc_0_0_1 = XT_ADD_S(acc1_0_0, acc_0_0_1);
                    if(p_bias!=(void *)0)
                    {
                      acc_0_0_1 = XT_ADD_S(acc_0_0_1, bias);
                    }
                    if(p_bias!=(void *)0)
                    {
                       XT_LSIP(bias, pbias, 4);
                    }
                    acc1_1_0 = XT_RADD_SX2(acc_1_0_0);
                    acc_1_0_1 = XT_ADD_S(acc1_1_0, acc_1_0_1);
                    if(p_bias!=(void *)0)
                    {
                      acc_1_0_1 = XT_ADD_S(acc_1_0_1, bias);
                    }
                    p_out_tmp = p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride;
                    XT_SSIP(acc_0_0_1,p_out_tmp,0);
                    p_out_tmp = p_out + (vec_itr + 0)*out_offset + (m_itr + 1)*out_stride;
                    XT_SSIP(acc_1_0_1,p_out_tmp,0);
                }

                for(; m_itr < rows; m_itr++)
                {
                    xtfloatx2 acc_0_0_0 = 0.0f, vec_batch_0_0  = 0.0f;
                    xtfloatx2 mat1_0_0 = 0.0f;
                    xtfloat acc_0_0_1 = 0.0f, acc1_0_0 = 0.0f, mat1_0_1 = 0.0f;
                    xtfloat vec_batch_0_1 = 0.0f;
                    xtfloat *p_vec_batch_0_1;
                    xtfloat *p_mat1_0_1;
                    xtfloatx2 *p_vec_batch_0_0  = (xtfloatx2 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                    xtfloatx2 *p_mat1_0_0 = (xtfloatx2 *) &p_mat1[(m_itr+0)*row_stride1];
                    ae_valign align_vec_batch_0_0 = AE_LA64_PP(p_vec_batch_0_0);
                    ae_valign align_mat1_0_0 = AE_LA64_PP(p_mat1_0_0);
                    int cols1_count = cols1 - cols1%2;

                    for(c_itr = 0; c_itr < (cols1_count >> 1); c_itr++)
                    {
                        XT_LASX2IP(vec_batch_0_0, align_vec_batch_0_0, p_vec_batch_0_0);
                        XT_LASX2IP(mat1_0_0, align_mat1_0_0, p_mat1_0_0);
                        XT_MADD_SX2(acc_0_0_0, vec_batch_0_0, mat1_0_0);
                    }
                    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
                    {
                        p_vec_batch_0_1 = (xtfloat *)p_vec_batch_0_0;
                        XT_LSIP(vec_batch_0_1, p_vec_batch_0_1, 4);
                        p_mat1_0_1 = (xtfloat *)p_mat1_0_0;
                        XT_LSIP(mat1_0_1, p_mat1_0_1, 4);
                        XT_MADD_S(acc_0_0_1, vec_batch_0_1, mat1_0_1);
                    }
                    if(p_bias!=(void *)0)
                    {
                       XT_LSIP(bias, pbias, 4);
                    }
                    acc1_0_0 = XT_RADD_SX2(acc_0_0_0);
                    acc_0_0_1 = XT_ADD_S(acc1_0_0, acc_0_0_1);
                    if(p_bias!=(void *)0)
                    {
                      acc_0_0_1 = XT_ADD_S(acc_0_0_1, bias);
                    }
                    p_out_tmp = p_out + (vec_itr + 0)*out_offset + (m_itr + 0)*out_stride;
                    XT_SSIP(acc_0_0_1,p_out_tmp,0);
                }
            }
        }

    return 0;
}
#endif
