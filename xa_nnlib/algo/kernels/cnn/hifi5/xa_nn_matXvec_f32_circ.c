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
#include "common_fpu.h"
#include "xa_nnlib_common.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_matXvec_f32_circ,(
    FLOAT32 *__restrict__ p_out,
    FLOAT32 * __restrict__ p_mat,
    FLOAT32 * __restrict__ p_vec,
    FLOAT32 * __restrict__ p_bias,
    WORD32 rows,
    WORD32 cols,
    WORD32 row_offset,
    WORD32 vec_count,
    WORD32 vec_offset,
    WORD32 bias_row_offset,
    WORD32 out_col_offset,
    WORD32 out_row_offset))
#else /* #if !HAVE_VFPU */
#ifdef ROW_UNROLL
#undef ROW_UNROLL
#endif
#define ROW_UNROLL 4

#define INCREMENT_IN_BYTES_FOR_FLOAT32      sizeof(FLOAT32)
#define INCREMENT_IN_BYTES_FOR_FLOAT32x2    2*sizeof(FLOAT32)

#define SETUP_ACC_BATCH_VEC_UNROLL(idx_row)     UNROLL_SETUP_ACC_BATCH(idx_row,0)   UNROLL_SETUP_ACC_BATCH(idx_row,1)
#define KERNEL_MAT_VEC_BATCH_VEC_UNROLL(idx_row)   UNROLL_KERNEL_MAT_VEC_BATCH(idx_row,0)     UNROLL_KERNEL_MAT_VEC_BATCH(idx_row,1)
#define ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row)      UNROLL_ADD_BIAS_ACC_BATCH(idx_row,0)     UNROLL_ADD_BIAS_ACC_BATCH(idx_row,1)
#define STORE_ACC_BATCH_VEC_UNROLL(idx_row)     UNROLL_STORE_ACC_BATCH(idx_row,0)     UNROLL_STORE_ACC_BATCH(idx_row,1)



#define SETUP_ACC_BATCH_ROW_FOR_f32(idx_row)\
    SETUP_ACC_BATCH_VEC_UNROLL(idx_row);

#define SETUP_ACC_BATCH_FOR_f32(idx_row,idx_vec) \
    xtfloatx2 _xtfloatx2_acc_ ##idx_row ##_ ##idx_vec = (xtfloatx2)0.0f; \
    xtfloatx2 _xtfloatx2_acc_1_ ##idx_row ##_ ##idx_vec = (xtfloatx2)0.0f; \
    xtfloat _xtfloat_acc_ ##idx_row ##_ ##idx_vec = (xtfloat) 0.0f;

#define SETUP_MAT_f32(idx) \
    xtfloatx2 _xtfloatx2_mat_ ## idx = (xtfloatx2)0.0f; \
    xtfloatx2 _xtfloatx2_mat_1_ ## idx = (xtfloatx2)0.0f; \
    xtfloatx4 *_xtfloatx4_p_mat_ ## idx = (xtfloatx4 *) p_mat; \
    ae_valignx2 _xtfloatx4_p_mat_valign_ ## idx;\
    AE_ADDCIRC16X4_XC((ae_int16x4 *)_xtfloatx4_p_mat_ ##idx, (m_itr+idx)*row_offset*sizeof(FLOAT32));\
    AE_LASX2X2POS_PC(_xtfloatx4_p_mat_valign_ ## idx,_xtfloatx4_p_mat_ ## idx);


#define SETUP_VEC_OFFSET_BATCH_f32(idx_vec)\
    xtfloatx2 _xtfloatx2_vec_batch_ ##idx_vec  = (xtfloatx2)0.0f ; \
    xtfloatx2 _xtfloatx2_vec_batch_1_ ##idx_vec  = (xtfloatx2)0.0f ; \
    xtfloatx4 *_xtfloatx4_p_vec_batch_ ##idx_vec  = (xtfloatx4 *)(&p_vec[(vec_itr + idx_vec)*vec_offset]);\

#define SETUP_BIAS_BATCH_ROW_FOR_f32(idx_row)\
    SETUP_BIAS_BATCH_VEC_UNROLL(idx_row);

#define SETUP_BIAS_BATCH_FOR_f32(idx_row,idx_vec) \
    xtfloat _xtfloat_bias_ ##idx_row ##_ ##idx_vec = p_bias[(vec_itr + idx_vec)]; \

#define LOAD_VEC_BATCH_f32(idx_vec) \
    AE_LSX2X2_IP(_xtfloatx2_vec_batch_ ##idx_vec,_xtfloatx2_vec_batch_1_ ##idx_vec,_xtfloatx4_p_vec_batch_ ##idx_vec,16);

#define LOAD_ROW_MAT_f32(idx) \
    AE_LASX2X2_IC(_xtfloatx2_mat_ ## idx, _xtfloatx2_mat_1_ ## idx,_xtfloatx4_p_mat_valign_ ## idx,(xtfloatx4 *)_xtfloatx4_p_mat_ ## idx);

#define KERNEL_MAT_VEC_BATCH_ROW_f32(idx_row)\
    KERNEL_MAT_VEC_BATCH_VEC_UNROLL(idx_row);\

#define KERNEL_MAT_VEC_BATCH_f32(idx_row,idx_vec) \
    MADD_SX2X2(_xtfloatx2_acc_ ##idx_row ##_ ##idx_vec,_xtfloatx2_acc_1_ ##idx_row ##_ ##idx_vec, _xtfloatx2_vec_batch_ ##idx_vec, _xtfloatx2_vec_batch_1_ ##idx_vec, _xtfloatx2_mat_ ##idx_row,_xtfloatx2_mat_1_ ##idx_row);\

#define ADD_BIAS_BATCH_ROW_ACC_FOR_f32(idx_row)\
    ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row);

#define ADD_BIAS_BATCH_ACC_FOR_f32(idx_row,idx_vec)\
    _xtfloatx2_acc_ ##idx_row ##_ ##idx_vec = _xtfloatx2_acc_ ##idx_row ##_ ##idx_vec + _xtfloatx2_acc_1_ ##idx_row ##_ ##idx_vec;\
    _xtfloat_acc_ ##idx_row ##_ ##idx_vec = RADD_SX2(_xtfloatx2_acc_ ##idx_row ##_ ##idx_vec);\
    _xtfloat_acc_ ##idx_row ##_ ##idx_vec = ADD_S(_xtfloat_acc_ ##idx_row ##_ ##idx_vec, _xtfloat_bias_ ##idx_row ##_ ##idx_vec);

#define STORE_ACC_BATCH_ROW_AT_OUT_f32(idx_row)\
    STORE_ACC_BATCH_VEC_UNROLL(idx_row);

#define STORE_ACC_BATCH_AT_OUT_OFFSET_f32(idx_row,idx_vec) \
    /*p_out value stored in a tmp pointer to make it inout for ISA */\
    p_out_tmp = &(p_out[(vec_itr + idx_vec)*out_col_offset + (m_itr + idx_row)*out_row_offset]);\
    AE_SSIP(_xtfloat_acc_ ##idx_row ##_ ##idx_vec,p_out_tmp,0);

#define VEC_UNROLL 2
#define UNROLL_ROW_SETUP_ACC_BATCH          SETUP_ACC_BATCH_ROW_FOR_f32
#define UNROLL_SETUP_ACC_BATCH              SETUP_ACC_BATCH_FOR_f32
#define UNROLL_SETUP_MAT                    SETUP_MAT_f32
#define UNROLL_SETUP_VEC_BATCH              SETUP_VEC_OFFSET_BATCH_f32
#define UNROLL_ROW_SETUP_BIAS_BATCH         SETUP_BIAS_BATCH_ROW_FOR_f32
#define UNROLL_SETUP_BIAS_BATCH             SETUP_BIAS_BATCH_FOR_f32
#define UNROLL_LOAD_VEC_BATCH               LOAD_VEC_BATCH_f32
#define UNROLL_LOAD_ROW_MAT                 LOAD_ROW_MAT_f32
#define UNROLL_ROW_KERNEL_MAT_VEC_BATCH     KERNEL_MAT_VEC_BATCH_ROW_f32
#define UNROLL_KERNEL_MAT_VEC_BATCH         KERNEL_MAT_VEC_BATCH_f32
#define UNROLL_ROW_ADD_BIAS_ACC             ADD_BIAS_BATCH_ROW_ACC_FOR_f32
#define UNROLL_ADD_BIAS_ACC_BATCH           ADD_BIAS_BATCH_ACC_FOR_f32
#define UNROLL_ROW_STORE_ACC                STORE_ACC_BATCH_ROW_AT_OUT_f32
#define UNROLL_STORE_ACC_BATCH              STORE_ACC_BATCH_AT_OUT_OFFSET_f32

/* ==================================================================================================== */
#undef SETUP_MAT
#undef STORE_ACC
#if (ROW_UNROLL == 1)
#define SETUP_MAT           UNROLL_SETUP_MAT(0)
#define STORE_ACC           UNROLL_STORE_ACC(0)
#elif (ROW_UNROLL == 2)
#define SETUP_MAT           UNROLL_SETUP_MAT(0)           UNROLL_SETUP_MAT(1)
#define STORE_ACC           UNROLL_STORE_ACC(0)           UNROLL_STORE_ACC(1)
#elif (ROW_UNROLL == 4)
#define SETUP_MAT           UNROLL_SETUP_MAT(0)           UNROLL_SETUP_MAT(1)           UNROLL_SETUP_MAT(2)           UNROLL_SETUP_MAT(3)
#define STORE_ACC           UNROLL_STORE_ACC(0)           UNROLL_STORE_ACC(1)           UNROLL_STORE_ACC(2)           UNROLL_STORE_ACC(3)
#elif (ROW_UNROLL == 8)
#define SETUP_MAT           UNROLL_SETUP_MAT(0)           UNROLL_SETUP_MAT(1)           UNROLL_SETUP_MAT(2)           UNROLL_SETUP_MAT(3)           UNROLL_SETUP_MAT(4)           UNROLL_SETUP_MAT(5)           UNROLL_SETUP_MAT(6)           UNROLL_SETUP_MAT(7)
#define STORE_ACC           UNROLL_STORE_ACC(0)           UNROLL_STORE_ACC(1)           UNROLL_STORE_ACC(2)           UNROLL_STORE_ACC(3)           UNROLL_STORE_ACC(4)           UNROLL_STORE_ACC(5)           UNROLL_STORE_ACC(6)           UNROLL_STORE_ACC(7)
#endif /* (ROW_UNROLL == 1) */

#if (ROW_UNROLL == 4 && VEC_UNROLL == 2)

#define SETUP_VEC_BATCH     UNROLL_SETUP_VEC_BATCH(0)   UNROLL_SETUP_VEC_BATCH(1)

#define SETUP_BIAS_BATCH     UNROLL_ROW_SETUP_BIAS_BATCH(0)     UNROLL_ROW_SETUP_BIAS_BATCH(1)     UNROLL_ROW_SETUP_BIAS_BATCH(2)     UNROLL_ROW_SETUP_BIAS_BATCH(3)
#define SETUP_BIAS_BATCH_VEC_UNROLL(idx_row)     UNROLL_SETUP_BIAS_BATCH(idx_row,0)   UNROLL_SETUP_BIAS_BATCH(idx_row,1)
#define SETUP_BIAS_BATCH_TAIL    UNROLL_SETUP_BIAS_BATCH(0,0)     UNROLL_SETUP_BIAS_BATCH(1,0)     UNROLL_SETUP_BIAS_BATCH(2,0)     UNROLL_SETUP_BIAS_BATCH(3,0)

#define SETUP_ACC_BATCH     UNROLL_ROW_SETUP_ACC_BATCH(0)     UNROLL_ROW_SETUP_ACC_BATCH(1)     UNROLL_ROW_SETUP_ACC_BATCH(2)     UNROLL_ROW_SETUP_ACC_BATCH(3)
#define SETUP_ACC_BATCH_VEC_UNROLL(idx_row)     UNROLL_SETUP_ACC_BATCH(idx_row,0)   UNROLL_SETUP_ACC_BATCH(idx_row,1)
#define SETUP_ACC_BATCH_TAIL    UNROLL_SETUP_ACC_BATCH(0,0)     UNROLL_SETUP_ACC_BATCH(1,0)     UNROLL_SETUP_ACC_BATCH(2,0)     UNROLL_SETUP_ACC_BATCH(3,0)

#define LOAD_VEC_BATCH      UNROLL_LOAD_VEC_BATCH(0)    UNROLL_LOAD_VEC_BATCH(1)
#define LOAD_MAT            UNROLL_LOAD_ROW_MAT(0)     UNROLL_LOAD_ROW_MAT(1)     UNROLL_LOAD_ROW_MAT(2)     UNROLL_LOAD_ROW_MAT(3)

#define KERNEL_MAT_VEC_BATCH       UNROLL_ROW_KERNEL_MAT_VEC_BATCH(0)     UNROLL_ROW_KERNEL_MAT_VEC_BATCH(1)     UNROLL_ROW_KERNEL_MAT_VEC_BATCH(2)     UNROLL_ROW_KERNEL_MAT_VEC_BATCH(3)
#define KERNEL_MAT_VEC_BATCH_VEC_UNROLL(idx_row)   UNROLL_KERNEL_MAT_VEC_BATCH(idx_row,0)     UNROLL_KERNEL_MAT_VEC_BATCH(idx_row,1)
#define KERNEL_MAT_VEC_BATCH_TAIL  UNROLL_KERNEL_MAT_VEC_BATCH(0,0)   UNROLL_KERNEL_MAT_VEC_BATCH(1,0)   UNROLL_KERNEL_MAT_VEC_BATCH(2,0)   UNROLL_KERNEL_MAT_VEC_BATCH(3,0)

#define ADD_BIAS_ACC_BATCH      UNROLL_ROW_ADD_BIAS_ACC(0)      UNROLL_ROW_ADD_BIAS_ACC(1)      UNROLL_ROW_ADD_BIAS_ACC(2)      UNROLL_ROW_ADD_BIAS_ACC(3)
#define ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row)      UNROLL_ADD_BIAS_ACC_BATCH(idx_row,0)     UNROLL_ADD_BIAS_ACC_BATCH(idx_row,1)
#define ADD_BIAS_ACC_BATCH_TAIL   UNROLL_ADD_BIAS_ACC_BATCH(0,0)     UNROLL_ADD_BIAS_ACC_BATCH(1,0)     UNROLL_ADD_BIAS_ACC_BATCH(2,0)     UNROLL_ADD_BIAS_ACC_BATCH(3,0)

#define STORE_ACC_BATCH     UNROLL_ROW_STORE_ACC(0)     UNROLL_ROW_STORE_ACC(1)     UNROLL_ROW_STORE_ACC(2)     UNROLL_ROW_STORE_ACC(3)
#define STORE_ACC_BATCH_VEC_UNROLL(idx_row)     UNROLL_STORE_ACC_BATCH(idx_row,0)     UNROLL_STORE_ACC_BATCH(idx_row,1)
#define STORE_ACC_BATCH_TAIL    UNROLL_STORE_ACC_BATCH(0,0)     UNROLL_STORE_ACC_BATCH(1,0)     UNROLL_STORE_ACC_BATCH(2,0)     UNROLL_STORE_ACC_BATCH(3,0)

#endif /* (ROW_UNROLL == 4 && VEC_UNROLL == 2)*/

WORD32 xa_nn_matXvec_f32_circ(
    FLOAT32 *__restrict__ p_out,            /* output pointer */
    FLOAT32 *__restrict__ p_mat,            /* matrix: rows x cols */
    FLOAT32 *__restrict__ p_vec,            /* vec: cols x 1 */
    FLOAT32 *__restrict__ p_bias,           /* bias TBD: Need array? */
    WORD32 rows,                            /* Number of rows in matrix */
    WORD32 cols,                            /* Number of columns in matrix */
    WORD32 row_offset,                      /* row stride for matrix */
    WORD32 vec_count,                       /* number of vectors: 2, 4, 2n */
    WORD32 vec_offset,                      /* offset from current to next vector */
    WORD32 out_col_offset,
    WORD32 out_row_offset)
{
    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
    xtfloat* p_out_tmp;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    vec_itr = 0;
    if(cols%4==0)//&& row_offset %8==0 && ((UWORD32)p_mat)%16==0)
    {
        if(rows >= ROW_UNROLL)
        {
            if(vec_count >= VEC_UNROLL)
            {
                for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
                {
                    for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
                    {
                        SETUP_BIAS_BATCH;
                        SETUP_ACC_BATCH;
                        SETUP_VEC_BATCH;
                        SETUP_MAT;

                        for(c_itr = 0; c_itr < (cols >> 2); c_itr++)
                        {
                            LOAD_VEC_BATCH;
                            LOAD_MAT;
                            KERNEL_MAT_VEC_BATCH;
                        }

                        ADD_BIAS_ACC_BATCH;
                        STORE_ACC_BATCH;
                    }
                    for(; m_itr < rows; m_itr++)
                    {
                        UNROLL_ROW_SETUP_BIAS_BATCH(0);
                        UNROLL_ROW_SETUP_ACC_BATCH(0);
                        SETUP_VEC_BATCH;
                        UNROLL_SETUP_MAT(0);

                        for(c_itr = 0; c_itr < (cols >> 2); c_itr++)
                        {
                            LOAD_VEC_BATCH;
                            UNROLL_LOAD_ROW_MAT(0);
                            UNROLL_ROW_KERNEL_MAT_VEC_BATCH(0);
                        }

                        UNROLL_ROW_ADD_BIAS_ACC(0);
                        UNROLL_ROW_STORE_ACC(0);
                    }
                }
            }
            {
                /* Tail loop for vec unroll */
                for(; vec_itr < vec_count; vec_itr++)
                {
                    for(m_itr = 0; m_itr < (rows & ~(ROW_UNROLL-1)); m_itr += ROW_UNROLL)
                    {
                        SETUP_BIAS_BATCH_TAIL;
                        SETUP_ACC_BATCH_TAIL;
                        UNROLL_SETUP_VEC_BATCH(0);
                        SETUP_MAT;

                        for(c_itr = 0; c_itr < (cols >> 2); c_itr++)
                        {
                            UNROLL_LOAD_VEC_BATCH(0);
                            LOAD_MAT;
                            KERNEL_MAT_VEC_BATCH_TAIL;
                        }

                        ADD_BIAS_ACC_BATCH_TAIL;
                        STORE_ACC_BATCH_TAIL;
                    }

                    for(; m_itr < rows; m_itr++)
                    {
                        UNROLL_SETUP_BIAS_BATCH(0,0);
                        UNROLL_SETUP_ACC_BATCH(0,0);
                        UNROLL_SETUP_VEC_BATCH(0);
                        UNROLL_SETUP_MAT(0);

                        for(c_itr = 0; c_itr < (cols >> 2); c_itr++)
                        {
                            UNROLL_LOAD_VEC_BATCH(0);
                            UNROLL_LOAD_ROW_MAT(0);
                            UNROLL_KERNEL_MAT_VEC_BATCH(0,0);
                        }

                        UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                        UNROLL_STORE_ACC_BATCH(0,0);
                    }
                }
            }
        }
        else
        {
            if(vec_count >= VEC_UNROLL)
            {
                for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
                {
                    for(m_itr = 0; m_itr < rows; m_itr++)
                    {
                        UNROLL_ROW_SETUP_BIAS_BATCH(0);
                        UNROLL_ROW_SETUP_ACC_BATCH(0);
                        SETUP_VEC_BATCH;
                        UNROLL_SETUP_MAT(0);

                        for(c_itr = 0; c_itr < (cols >> 2); c_itr++)
                        {
                            LOAD_VEC_BATCH;
                            UNROLL_LOAD_ROW_MAT(0);
                            UNROLL_ROW_KERNEL_MAT_VEC_BATCH(0);
                        }

                        UNROLL_ROW_ADD_BIAS_ACC(0);
                        UNROLL_ROW_STORE_ACC(0);
                    }
                }
            }
            { /* Tail loop for vec unroll */
                for(; vec_itr < vec_count; vec_itr++)
                {
                    for(m_itr = 0; m_itr < rows; m_itr++)
                    {
                        UNROLL_SETUP_BIAS_BATCH(0,0);
                        UNROLL_SETUP_ACC_BATCH(0,0);
                        UNROLL_SETUP_VEC_BATCH(0);
                        UNROLL_SETUP_MAT(0);

                        for(c_itr = 0; c_itr < (cols >> 2); c_itr++)
                        {
                            UNROLL_LOAD_VEC_BATCH(0);
                            UNROLL_LOAD_ROW_MAT(0);
                            UNROLL_KERNEL_MAT_VEC_BATCH(0,0);
                        }

                        UNROLL_ADD_BIAS_ACC_BATCH(0,0);
                        UNROLL_STORE_ACC_BATCH(0,0);
                    }
                }
            }
        }
    }
    else
    {
            // Support of non-multiples of 4 cols
            for (vec_itr = 0; vec_itr < (vec_count & (~0x1)); vec_itr+=2)
            {
                for(m_itr = 0; m_itr < (rows & (~0x1)); m_itr+=2)
                {
                    ae_valignx2 align_mat0, align_mat1;
                    ae_valignx2 align_vec0, align_vec1;
                    xtfloatx2 _xtfloat_bias_0 = p_bias[vec_itr];
                    xtfloatx2 _xtfloat_bias_1 = p_bias[vec_itr+1];

                    xtfloatx2 _xtfloatx2_acc_0_0 = (xtfloatx2)0.0f;
                    xtfloatx2 _xtfloatx2_acc_0_1 = (xtfloatx2)0.0f;
                    xtfloatx2 _xtfloatx2_acc_1_0 = (xtfloatx2)0.0f;
                    xtfloatx2 _xtfloatx2_acc_1_1 = (xtfloatx2)0.0f;

                    xtfloatx2 _xtfloatx2_acc_0_0_1 = (xtfloatx2)0.0f;
                    xtfloatx2 _xtfloatx2_acc_0_1_1 = (xtfloatx2)0.0f;
                    xtfloatx2 _xtfloatx2_acc_1_0_1 = (xtfloatx2)0.0f;
                    xtfloatx2 _xtfloatx2_acc_1_1_1 = (xtfloatx2)0.0f;

                    xtfloat _xtfloat_acc_0_0 = (xtfloat)0.0f;
                    xtfloat _xtfloat_acc_0_1 = (xtfloat)0.0f;
                    xtfloat _xtfloat_acc_1_0 = (xtfloat)0.0f;
                    xtfloat _xtfloat_acc_1_1 = (xtfloat)0.0f;

                    xtfloatx2 _xtfloatx2_vec_batch_0  = (xtfloatx2)0.0f ;
                    xtfloatx2 _xtfloatx2_vec_batch_1  = (xtfloatx2)0.0f ;
                    xtfloatx2 _xtfloatx2_vec_batch_00  = (xtfloatx2)0.0f ;
                    xtfloatx2 _xtfloatx2_vec_batch_11  = (xtfloatx2)0.0f ;
                    xtfloat _xtfloat_vec_batch_0  = (xtfloat)0.0f ;
                    xtfloat _xtfloat_vec_batch_1  = (xtfloat)0.0f ;

                    xtfloatx4 *_xtfloatx4_p_vec_batch_0  = (xtfloatx4 *)(&p_vec[(vec_itr)*vec_offset]);
                    xtfloatx4 *_xtfloatx4_p_vec_batch_1  = (xtfloatx4 *)(&p_vec[(vec_itr+1)*vec_offset]);

                    xtfloatx2 _xtfloatx2_mat_0 = (xtfloatx2)0.0f;
                    xtfloatx2 _xtfloatx2_mat_1 = (xtfloatx2)0.0f;
                    xtfloatx2 _xtfloatx2_mat_00 = (xtfloatx2)0.0f;
                    xtfloatx2 _xtfloatx2_mat_11 = (xtfloatx2)0.0f;
                    xtfloat _xtfloat_mat_0 = (xtfloat)0.0f;
                    xtfloat _xtfloat_mat_1 = (xtfloat)0.0f;

                    xtfloatx4 *_xtfloatx4_p_mat_0 = (xtfloatx4 *) p_mat;
                    xtfloatx4 *_xtfloatx4_p_mat_1 = (xtfloatx4 *) p_mat;

                    AE_ADDCIRC16X4_XC((ae_int16x4 *)_xtfloatx4_p_mat_0, (m_itr)*row_offset*sizeof(FLOAT32));
                    AE_ADDCIRC16X4_XC((ae_int16x4 *)_xtfloatx4_p_mat_1, (m_itr+1)*row_offset*sizeof(FLOAT32));

                    align_vec0 = AE_LA128_PP(_xtfloatx4_p_vec_batch_0);
                    align_vec1 = AE_LA128_PP(_xtfloatx4_p_vec_batch_1);

                    AE_LASX2X2POS_PC(align_mat0, _xtfloatx4_p_mat_0);
                    AE_LASX2X2POS_PC(align_mat1, _xtfloatx4_p_mat_1);

                    for(c_itr = 0; c_itr < (cols>>2); c_itr++)
                    {
                        AE_LASX2X2_IP(_xtfloatx2_vec_batch_0, _xtfloatx2_vec_batch_00, align_vec0, _xtfloatx4_p_vec_batch_0);
                        AE_LASX2X2_IP(_xtfloatx2_vec_batch_1, _xtfloatx2_vec_batch_11, align_vec1, _xtfloatx4_p_vec_batch_1);

                        AE_LASX2X2_IC(_xtfloatx2_mat_0, _xtfloatx2_mat_00, align_mat0, _xtfloatx4_p_mat_0);
                        AE_LASX2X2_IC(_xtfloatx2_mat_1, _xtfloatx2_mat_11, align_mat1, _xtfloatx4_p_mat_1);

                        MADD_SX2X2(_xtfloatx2_acc_0_0,_xtfloatx2_acc_0_1,_xtfloatx2_vec_batch_0,_xtfloatx2_vec_batch_1,_xtfloatx2_mat_0,_xtfloatx2_mat_0);
                        MADD_SX2X2(_xtfloatx2_acc_1_0,_xtfloatx2_acc_1_1,_xtfloatx2_vec_batch_0,_xtfloatx2_vec_batch_1,_xtfloatx2_mat_1,_xtfloatx2_mat_1);

                        MADD_SX2X2(_xtfloatx2_acc_0_0_1,_xtfloatx2_acc_0_1_1,_xtfloatx2_vec_batch_00,_xtfloatx2_vec_batch_11,_xtfloatx2_mat_00,_xtfloatx2_mat_00);
                        MADD_SX2X2(_xtfloatx2_acc_1_0_1,_xtfloatx2_acc_1_1_1,_xtfloatx2_vec_batch_00,_xtfloatx2_vec_batch_11,_xtfloatx2_mat_11,_xtfloatx2_mat_11);
                    }
                    _xtfloatx4_p_mat_0 = (xtfloatx4 *) p_mat;
                    _xtfloatx4_p_mat_1 = (xtfloatx4 *) p_mat;
                    AE_ADDCIRC16X4_XC((ae_int16x4 *)_xtfloatx4_p_mat_0, ((m_itr  )*row_offset + (cols>>2)*4 )*sizeof(FLOAT32));
                    AE_ADDCIRC16X4_XC((ae_int16x4 *)_xtfloatx4_p_mat_1, ((m_itr+1)*row_offset + (cols>>2)*4 )*sizeof(FLOAT32));
                    for(c_itr = ((cols>>2)*4) ; c_itr < cols; c_itr++)
                    {
                        AE_LSIP(_xtfloat_vec_batch_0,(xtfloat *) _xtfloatx4_p_vec_batch_0, INCREMENT_IN_BYTES_FOR_FLOAT32);
                        AE_LSIP(_xtfloat_vec_batch_1,(xtfloat *) _xtfloatx4_p_vec_batch_1, INCREMENT_IN_BYTES_FOR_FLOAT32);

                        AE_LSXC(_xtfloat_mat_0,(xtfloat *) _xtfloatx4_p_mat_0, INCREMENT_IN_BYTES_FOR_FLOAT32);
                        AE_LSXC(_xtfloat_mat_1,(xtfloat *) _xtfloatx4_p_mat_1, INCREMENT_IN_BYTES_FOR_FLOAT32);

                        MADD_S(_xtfloat_acc_0_0, _xtfloat_vec_batch_0, _xtfloat_mat_0);
                        MADD_S(_xtfloat_acc_0_1, _xtfloat_vec_batch_1, _xtfloat_mat_0);
                        MADD_S(_xtfloat_acc_1_0, _xtfloat_vec_batch_0, _xtfloat_mat_1);
                        MADD_S(_xtfloat_acc_1_1, _xtfloat_vec_batch_1, _xtfloat_mat_1);
                    }

                   _xtfloatx2_acc_0_0 = _xtfloatx2_acc_0_0 + _xtfloatx2_acc_0_0_1;
                   _xtfloatx2_acc_0_1 = _xtfloatx2_acc_0_1 + _xtfloatx2_acc_0_1_1;
                   _xtfloatx2_acc_1_0 = _xtfloatx2_acc_1_0 + _xtfloatx2_acc_1_0_1;
                   _xtfloatx2_acc_1_1 = _xtfloatx2_acc_1_1 + _xtfloatx2_acc_1_1_1;

                   _xtfloat_acc_0_0= _xtfloat_acc_0_0 + RADD_SX2(_xtfloatx2_acc_0_0);
                   _xtfloat_acc_0_1= _xtfloat_acc_0_1 + RADD_SX2(_xtfloatx2_acc_0_1);
                   _xtfloat_acc_1_0= _xtfloat_acc_1_0 + RADD_SX2(_xtfloatx2_acc_1_0);
                   _xtfloat_acc_1_1= _xtfloat_acc_1_1 + RADD_SX2(_xtfloatx2_acc_1_1);

                    _xtfloat_acc_0_0=ADD_S(_xtfloat_acc_0_0,_xtfloat_bias_0);
                    _xtfloat_acc_0_1=ADD_S(_xtfloat_acc_0_1,_xtfloat_bias_1);
                    _xtfloat_acc_1_0=ADD_S(_xtfloat_acc_1_0,_xtfloat_bias_0);
                    _xtfloat_acc_1_1=ADD_S(_xtfloat_acc_1_1,_xtfloat_bias_1);

                    p_out_tmp = &(p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]);
                    AE_SSIP(_xtfloat_acc_0_0,p_out_tmp,0);
                    p_out_tmp = &(p_out[(vec_itr)*out_col_offset + (m_itr+1)*out_row_offset]);
                    AE_SSIP(_xtfloat_acc_1_0,p_out_tmp,0);
                    p_out_tmp = &(p_out[(vec_itr+1)*out_col_offset + (m_itr)*out_row_offset]);
                    AE_SSIP(_xtfloat_acc_0_1,p_out_tmp,0);
                    p_out_tmp = &(p_out[(vec_itr+1)*out_col_offset + (m_itr+1)*out_row_offset]);
                    AE_SSIP(_xtfloat_acc_1_1,p_out_tmp,0);
                }
                for (; m_itr < rows ; m_itr++)
                {
                    xtfloat _xtfloat_bias_0 = p_bias[vec_itr];
                    xtfloat _xtfloat_bias_1 = p_bias[vec_itr+1];
                    xtfloat _xtfloatx2_acc_0_0 = (xtfloat)0.0f;
                    xtfloat _xtfloatx2_acc_0_1 = (xtfloat)0.0f;

                    xtfloat _xtfloatx2_vec_batch_0  = (xtfloat)0.0f ;
                    xtfloat _xtfloatx2_vec_batch_1  = (xtfloat)0.0f ;
                    xtfloat *_xtfloatx2_p_vec_batch_0  = (xtfloat *)(&p_vec[(vec_itr)*vec_offset]);
                    xtfloat *_xtfloatx2_p_vec_batch_1  = (xtfloat *)(&p_vec[(vec_itr+1)*vec_offset]);

                    xtfloat _xtfloatx2_mat_0 = (xtfloat)0.0f;
                    xtfloat *_xtfloatx2_p_mat_0 = (xtfloat *) p_mat;

                    AE_ADDCIRC16X4_XC((ae_int16x4 *)_xtfloatx2_p_mat_0, (m_itr)*row_offset*sizeof(FLOAT32));

                    for(c_itr = 0; c_itr < cols; c_itr++)
                    {
                        AE_LSIP(_xtfloatx2_vec_batch_0, _xtfloatx2_p_vec_batch_0, INCREMENT_IN_BYTES_FOR_FLOAT32);
                        AE_LSIP(_xtfloatx2_vec_batch_1, _xtfloatx2_p_vec_batch_1, INCREMENT_IN_BYTES_FOR_FLOAT32);
                        AE_LSXC(_xtfloatx2_mat_0, _xtfloatx2_p_mat_0, INCREMENT_IN_BYTES_FOR_FLOAT32);
                        MADD_S(_xtfloatx2_acc_0_0, _xtfloatx2_vec_batch_0, _xtfloatx2_mat_0);
                        MADD_S(_xtfloatx2_acc_0_1, _xtfloatx2_vec_batch_1, _xtfloatx2_mat_0);
                    }
                    _xtfloatx2_acc_0_0=ADD_S(_xtfloatx2_acc_0_0,_xtfloat_bias_0);
                    _xtfloatx2_acc_0_1=ADD_S(_xtfloatx2_acc_0_1,_xtfloat_bias_1);
                    p_out_tmp = &(p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]);
                    AE_SSIP(_xtfloatx2_acc_0_0,p_out_tmp,0);
                    p_out_tmp = &(p_out[(vec_itr+1)*out_col_offset + (m_itr)*out_row_offset]);
                    AE_SSIP(_xtfloatx2_acc_0_1,p_out_tmp,0);
                }

            }
            if(vec_count & 0x1)
            {
                vec_itr = vec_count -1;
                for(m_itr = 0; m_itr < (rows); m_itr ++)
                {
                    xtfloat _xtfloat_bias = p_bias[vec_itr];
                    xtfloat _xtfloatx2_acc_0_0 = (xtfloat)0.0f;
                    xtfloat _xtfloatx2_vec_batch_0  = (xtfloat)0.0f ;
                    xtfloat *_xtfloatx2_p_vec_batch_0  = (xtfloat *)(&p_vec[(vec_itr)*vec_offset]);
                    xtfloat _xtfloatx2_mat_0 = (xtfloat)0.0f;
                    xtfloat *_xtfloatx2_p_mat_0 = (xtfloat *) p_mat;
                    AE_ADDCIRC16X4_XC((ae_int16x4 *)_xtfloatx2_p_mat_0, (m_itr)*row_offset*sizeof(FLOAT32));
                    for(c_itr = 0; c_itr < cols; c_itr++)
                    {
                        AE_LSIP(_xtfloatx2_vec_batch_0, _xtfloatx2_p_vec_batch_0, INCREMENT_IN_BYTES_FOR_FLOAT32);
                        AE_LSXC(_xtfloatx2_mat_0, _xtfloatx2_p_mat_0, INCREMENT_IN_BYTES_FOR_FLOAT32);
                        MADD_S(_xtfloatx2_acc_0_0, _xtfloatx2_vec_batch_0, _xtfloatx2_mat_0);
                    }
                    _xtfloatx2_acc_0_0=ADD_S(_xtfloatx2_acc_0_0,_xtfloat_bias);
                    p_out_tmp = &(p_out[(vec_itr)*out_col_offset + (m_itr)*out_row_offset]);
                    AE_SSIP(_xtfloatx2_acc_0_0,p_out_tmp,0);
                }
            }

    }
    return 0;
}
#endif /* #if !HAVE_VFPU */
