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
#define ROW_UNROLL 8

extern WORD32 xa_nn_matXvec_8x16_64(
         WORD64 * __restrict__ p_out,           /* output */
         WORD8 * __restrict__ p_mat1,          /* matrix1: rows x cols1 */
         WORD8 * __restrict__ p_mat2,          /* matrix2: rows x cols2 */
         WORD16 * __restrict__ p_vec1,          /* vec1: cols1 x 1 */
         WORD16 * __restrict__ p_vec2,          /* vec2: cols2 x 1 */
         WORD16 * __restrict__ p_bias,          /* bias */
         WORD32 rows,
         WORD32 cols1,
         WORD32 cols2,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 row_stride2,                    /* row stride for matrix2 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift) ;                     /* bias shift amount */
/*----------------------------Main function---------------------------------*/

WORD32 xa_nn_matXvec_batch_8x16_64(

         WORD64 ** __restrict__ p_out,          /* array of output pointers */
         WORD8 *  __restrict__ p_mat1,         /* matrix1: rows x cols1 */
         WORD16 ** __restrict__ p_vec1,         /* vec1: cols1 x 1 */
         WORD16 *  __restrict__ p_bias,         /* bias TBD: Need array? */
         WORD32 rows,
         WORD32 cols1,
         WORD32 row_stride1,                    /* row stride for matrix1 */
         WORD32 acc_shift,                        /* out accumulator shift amount */
         WORD32 bias_shift,                       /* bias shift amount */
         WORD32 vec_count)                      /* number of vectors: 2, 4, 2n */
{
    int i;
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    for(i = 0; i < vec_count; i++)
    {
      XA_NNLIB_ARG_CHK_PTR(p_out[i], -1);
      XA_NNLIB_ARG_CHK_PTR(p_vec1[i], -1);
    }
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD64 *), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD16 *), -1);
    int isAlignedvec = 1;
    for(i = 0; i < vec_count; i++)
    {
      XA_NNLIB_ARG_CHK_ALIGN(p_out[i], sizeof(WORD64), -1);
      XA_NNLIB_ARG_CHK_ALIGN(p_vec1[i],sizeof(WORD16), -1);
      isAlignedvec &= (((unsigned int)p_vec1[i] & 15) == 0);
    }
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);
    XA_NNLIB_ARG_CHK_COND((vec_count <= 0), -1);
    
    /* Iterators used in for loops */
    int m_itr, c_itr, vec_itr;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    vec_itr = 0;

    #define VEC_UNROLL 2
    #define SETUP_BIAS                          SETUP_BIAS_16b

    LIMIT_ACC_LSH

    if (((unsigned int)p_mat1 & 15) == 0 && isAlignedvec && ((cols1&15) == 0) && ((row_stride1&15) == 0))
    {
        if(rows > ROW_UNROLL)
        {
            if(vec_count > VEC_UNROLL)
            {
                for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
                {
                    ae_int16 _ae_int16_bias = ZERO16;
                    ae_int64 _ae_int64_sat_bias = ZERO64;
                    ae_int16 *_ae_int16_p_bias = (ae_int16 *) p_bias;
                    for(m_itr = 0; m_itr < (rows & ~(4-1)); m_itr += 4)
                    {
                        ae_int64 * output_ptr=(ae_int64*)(p_out[vec_itr]+m_itr);
                        ae_int64 * output_ptr1=(ae_int64*)(p_out[vec_itr+1]+m_itr);
                        ae_int16x8 * _ae_int16x8_p_vec1 = (ae_int16x8 *) (p_vec1[vec_itr]);
                        ae_int16x8 * _ae_int16x8_p_vec2 = (ae_int16x8 *) (p_vec1[vec_itr+1]);
                        ae_int64 _ae_int64_acc_0=ZERO64;
                        ae_int64 _ae_int64_acc_1=ZERO64;
                        ae_int64 _ae_int64_acc_2=ZERO64;
                        ae_int64 _ae_int64_acc_3=ZERO64;
                        ae_int64 _ae_int64_acc_8=ZERO64;
                        ae_int64 _ae_int64_acc_9=ZERO64;
                        ae_int64 _ae_int64_acc_10=ZERO64;
                        ae_int64 _ae_int64_acc_11=ZERO64;
                        SETUP_MAT1_8b_8x16(0);
                        SETUP_MAT1_8b_8x16(1);
                        SETUP_MAT1_8b_8x16(2);
                        SETUP_MAT1_8b_8x16(3);
                        ae_int8x16 * _ae_int8x16_p_mat1_0  = (ae_int8x16 *) &p_mat1[(m_itr+0)*row_stride1];
                        ae_int8x16 * _ae_int8x16_p_mat1_1  = (ae_int8x16 *) &p_mat1[(m_itr+1)*row_stride1];
                        ae_int8x16 * _ae_int8x16_p_mat1_2  = (ae_int8x16 *) &p_mat1[(m_itr+2)*row_stride1];
                        ae_int8x16 * _ae_int8x16_p_mat1_3  = (ae_int8x16 *) &p_mat1[(m_itr+3)*row_stride1];
                        ae_int16x4 _ae_int16x4_vec1,_ae_int16x4_vec1_1,_ae_int16x4_vec1_2,_ae_int16x4_vec1_3;
                        ae_int16x4 _ae_int16x4_vec2,_ae_int16x4_vec2_1,_ae_int16x4_vec2_2,_ae_int16x4_vec2_3;


                        #pragma ymemory (_ae_int16x8_p_vec1)
                        #pragma ymemory (_ae_int16x8_p_vec2)

                        for(c_itr = 0; c_itr < (cols1 >> 4); c_itr++)
                        {
                              AE_L16X4X2_IP(_ae_int16x4_vec1  ,_ae_int16x4_vec1_1, _ae_int16x8_p_vec1, 16);
                              AE_L16X4X2_IP(_ae_int16x4_vec1_2,_ae_int16x4_vec1_3, _ae_int16x8_p_vec1, 16);
                              AE_L16X4X2_IP(_ae_int16x4_vec2  ,_ae_int16x4_vec2_1, _ae_int16x8_p_vec2, 16);
                              AE_L16X4X2_IP(_ae_int16x4_vec2_2,_ae_int16x4_vec2_3, _ae_int16x8_p_vec2, 16);

                              AE_L8X8X2_IP( _ae_int8x8_mat1_0, _ae_int8x8_mat1_1_0,_ae_int8x16_p_mat1_0, 16);
                              AE_L8X8X2_IP( _ae_int8x8_mat1_1, _ae_int8x8_mat1_1_1,_ae_int8x16_p_mat1_1, 16);
                              AE_L8X8X2_IP( _ae_int8x8_mat1_2, _ae_int8x8_mat1_1_2,_ae_int8x16_p_mat1_2, 16);
                              AE_L8X8X2_IP( _ae_int8x8_mat1_3, _ae_int8x8_mat1_1_3,_ae_int8x16_p_mat1_3, 16);

                            AE_MULA8QW8X16(_ae_int64_acc_0 ,_ae_int64_acc_1 ,_ae_int64_acc_2 ,_ae_int64_acc_3 ,_ae_int8x8_mat1_0  ,_ae_int8x8_mat1_1  ,_ae_int8x8_mat1_2  ,_ae_int8x8_mat1_3  ,_ae_int16x4_vec1  ,_ae_int16x4_vec1_1);
                            AE_MULA8QW8X16(_ae_int64_acc_0 ,_ae_int64_acc_1 ,_ae_int64_acc_2 ,_ae_int64_acc_3 ,_ae_int8x8_mat1_1_0,_ae_int8x8_mat1_1_1,_ae_int8x8_mat1_1_2,_ae_int8x8_mat1_1_3,_ae_int16x4_vec1_2,_ae_int16x4_vec1_3);
                            AE_MULA8QW8X16(_ae_int64_acc_8 ,_ae_int64_acc_9 ,_ae_int64_acc_10,_ae_int64_acc_11,_ae_int8x8_mat1_0  , _ae_int8x8_mat1_1  ,_ae_int8x8_mat1_2  ,_ae_int8x8_mat1_3  ,_ae_int16x4_vec2  ,_ae_int16x4_vec2_1);
                            AE_MULA8QW8X16(_ae_int64_acc_8 ,_ae_int64_acc_9 ,_ae_int64_acc_10,_ae_int64_acc_11,_ae_int8x8_mat1_1_0, _ae_int8x8_mat1_1_1,_ae_int8x8_mat1_1_2,_ae_int8x8_mat1_1_3,_ae_int16x4_vec2_2,_ae_int16x4_vec2_3);
                        }


                        ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16);
                        _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift);
                        _ae_int64_acc_0 = AE_ADD64S(_ae_int64_acc_0, _ae_int64_sat_bias);
                        _ae_int64_acc_8 = AE_ADD64S(_ae_int64_acc_8, _ae_int64_sat_bias);

                        ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16);
                        _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift);
                        _ae_int64_acc_1 = AE_ADD64S(_ae_int64_acc_1, _ae_int64_sat_bias);
                        _ae_int64_acc_9 = AE_ADD64S(_ae_int64_acc_9, _ae_int64_sat_bias);

                        ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16);
                        _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift);
                        _ae_int64_acc_2 = AE_ADD64S(_ae_int64_acc_2, _ae_int64_sat_bias);
                        _ae_int64_acc_10 = AE_ADD64S(_ae_int64_acc_10, _ae_int64_sat_bias);

                        ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16);
                        _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift);
                        _ae_int64_acc_3 = AE_ADD64S(_ae_int64_acc_3, _ae_int64_sat_bias);
                        _ae_int64_acc_11 = AE_ADD64S(_ae_int64_acc_11, _ae_int64_sat_bias);

                        _ae_int64_acc_0 =  AE_SLAA64S(_ae_int64_acc_0,acc_shift);
                        _ae_int64_acc_1 =  AE_SLAA64S(_ae_int64_acc_1,acc_shift);
                        _ae_int64_acc_2 =  AE_SLAA64S(_ae_int64_acc_2,acc_shift);
                        _ae_int64_acc_3 =  AE_SLAA64S(_ae_int64_acc_3,acc_shift);
                        _ae_int64_acc_8  = AE_SLAA64S(_ae_int64_acc_8 ,acc_shift);
                        _ae_int64_acc_9  = AE_SLAA64S(_ae_int64_acc_9 ,acc_shift);
                        _ae_int64_acc_10 = AE_SLAA64S(_ae_int64_acc_10,acc_shift);
                        _ae_int64_acc_11 = AE_SLAA64S(_ae_int64_acc_11,acc_shift);
                        AE_S64_IP(_ae_int64_acc_0 ,output_ptr ,sizeof(ae_int64));
                        AE_S64_IP(_ae_int64_acc_1 ,output_ptr ,sizeof(ae_int64));
                        AE_S64_IP(_ae_int64_acc_2 ,output_ptr ,sizeof(ae_int64));
                        AE_S64_IP(_ae_int64_acc_3 ,output_ptr ,sizeof(ae_int64));
                        AE_S64_IP(_ae_int64_acc_8 ,output_ptr1,sizeof(ae_int64));
                        AE_S64_IP(_ae_int64_acc_9 ,output_ptr1,sizeof(ae_int64));
                        AE_S64_IP(_ae_int64_acc_10,output_ptr1,sizeof(ae_int64));
                        AE_S64_IP(_ae_int64_acc_11,output_ptr1,sizeof(ae_int64));
                    }
                    for(; m_itr < rows; m_itr++)
                    {
                        ae_int64 _ae_int64_acc_0=ZERO64;
                        ae_int64 _ae_int64_acc_1=ZERO64;
                        ae_int64 _ae_int64_acc_2=ZERO64;
                        ae_int64 _ae_int64_acc_3=ZERO64;
                        ae_int64 * output_ptr=(ae_int64 *)(p_out[vec_itr]+m_itr);
                        ae_int64 * output_ptr1=(ae_int64 *)(p_out[vec_itr+1]+m_itr);
                        ae_int16x8 * _ae_int16x8_p_vec1 = (ae_int16x8 *) (p_vec1[vec_itr]);
                        ae_int16x8 * _ae_int16x8_p_vec2 = (ae_int16x8 *) (p_vec1[vec_itr+1]);
                        ae_int8x8 * _ae_int8x8_p_mat1_0  = (ae_int8x8 *) &p_mat1[(m_itr+0)*row_stride1];
                        ZER0_8x8_Temp_Variable;
                        ae_int8x8 _ae_int8x8_mat1_0;
                        ae_int16x4 _ae_int16x4_vec1,_ae_int16x4_vec1_1;
                        ae_int16x4 _ae_int16x4_vec2,_ae_int16x4_vec2_1;
                        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
                        {
                              AE_L16X4X2_IP(_ae_int16x4_vec1,_ae_int16x4_vec1_1, _ae_int16x8_p_vec1, 2*INCREMENT_IN_BYTES_FOR_INT16X4);
                              AE_L16X4X2_IP(_ae_int16x4_vec2,_ae_int16x4_vec2_1, _ae_int16x8_p_vec2, 2*INCREMENT_IN_BYTES_FOR_INT16X4);
                              AE_L8X8_IP(_ae_int8x8_mat1_0,_ae_int8x8_p_mat1_0, 8);

                              AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,
                                             _ae_int8x8_mat1_0, zero_temp, zero_temp,zero_temp,
                                            _ae_int16x4_vec1,_ae_int16x4_vec1_1);

                              AE_MULA8QW8X16(_ae_int64_acc_1  , _ae_int64_acc_0  , _ae_int64_acc_2  ,_ae_int64_acc_3,
                                             _ae_int8x8_mat1_0, zero_temp, zero_temp,zero_temp,
                                            _ae_int16x4_vec2,_ae_int16x4_vec2_1);


                        }
                        ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16);
                        _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift);
                        _ae_int64_acc_0 = AE_ADD64S(_ae_int64_acc_0, _ae_int64_sat_bias);
                        _ae_int64_acc_1 = AE_ADD64S(_ae_int64_acc_1, _ae_int64_sat_bias);
                        _ae_int64_acc_0 =  AE_SLAA64S(_ae_int64_acc_0,acc_shift);
                        _ae_int64_acc_1 =  AE_SLAA64S(_ae_int64_acc_1,acc_shift);
                        AE_S64_IP(_ae_int64_acc_0,output_ptr,sizeof(ae_int64));
                        AE_S64_IP(_ae_int64_acc_1,output_ptr1,sizeof(ae_int64));

                    }
                }
            }
            {
                for(; vec_itr < vec_count; vec_itr++)
                {
                    xa_nn_matXvec_8x16_64(p_out[vec_itr],p_mat1,NULL,p_vec1[vec_itr],NULL,p_bias,rows,cols1,0,row_stride1,0,acc_shift,bias_shift);
                }
            }
        }
        else
        {
            if(vec_count > VEC_UNROLL)
            {
                for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
                {
                    ae_int16 _ae_int16_bias = ZERO16;
                    ae_int64 _ae_int64_sat_bias = ZERO64;
                    ae_int16 *_ae_int16_p_bias = (ae_int16 *) p_bias;
                    for(m_itr = 0; m_itr < rows; m_itr++)
                    {
                        ae_int64 _ae_int64_acc_0=ZERO64;
                        ae_int64 _ae_int64_acc_1=ZERO64;
                        ae_int64 _ae_int64_acc_2=ZERO64;
                        ae_int64 _ae_int64_acc_3=ZERO64;
                        ae_int64 * output_ptr=(ae_int64 *)(p_out[vec_itr]+m_itr);
                        ae_int64 * output_ptr1=(ae_int64 *)(p_out[vec_itr+1]+m_itr);
                        ae_int16x8 * _ae_int16x8_p_vec1 = (ae_int16x8 *) (p_vec1[vec_itr]);
                        ae_int16x8 * _ae_int16x8_p_vec2 = (ae_int16x8 *) (p_vec1[vec_itr+1]);
                        ae_int8x8 * _ae_int8x8_p_mat1_0  = (ae_int8x8 *) &p_mat1[(m_itr+0)*row_stride1];
                        ZER0_8x8_Temp_Variable;
                        ae_int8x8 _ae_int8x8_mat1_0;
                        ae_int16x4 _ae_int16x4_vec1,_ae_int16x4_vec1_1;
                        ae_int16x4 _ae_int16x4_vec2,_ae_int16x4_vec2_1;
                        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
                        {
                              AE_L16X4X2_IP(_ae_int16x4_vec1,_ae_int16x4_vec1_1, _ae_int16x8_p_vec1, 2*INCREMENT_IN_BYTES_FOR_INT16X4);
                              AE_L16X4X2_IP(_ae_int16x4_vec2,_ae_int16x4_vec2_1, _ae_int16x8_p_vec2, 2*INCREMENT_IN_BYTES_FOR_INT16X4);
                              AE_L8X8_IP(_ae_int8x8_mat1_0,_ae_int8x8_p_mat1_0, 8);

                              AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,
                                             _ae_int8x8_mat1_0, zero_temp, zero_temp,zero_temp,
                                            _ae_int16x4_vec1,_ae_int16x4_vec1_1);

                              AE_MULA8QW8X16(_ae_int64_acc_1  , _ae_int64_acc_0  , _ae_int64_acc_2  ,_ae_int64_acc_3,
                                             _ae_int8x8_mat1_0, zero_temp, zero_temp,zero_temp,
                                            _ae_int16x4_vec2,_ae_int16x4_vec2_1);


                        }
                        ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16);
                        _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift);
                        _ae_int64_acc_0 = AE_ADD64S(_ae_int64_acc_0, _ae_int64_sat_bias);
                        _ae_int64_acc_1 = AE_ADD64S(_ae_int64_acc_1, _ae_int64_sat_bias);
                        _ae_int64_acc_0 =  AE_SLAA64S(_ae_int64_acc_0,acc_shift);
                        _ae_int64_acc_1 =  AE_SLAA64S(_ae_int64_acc_1,acc_shift);
                        AE_S64_IP(_ae_int64_acc_0,output_ptr,sizeof(ae_int64));
                        AE_S64_IP(_ae_int64_acc_1,output_ptr1,sizeof(ae_int64));
                    }
                }
            }
            {
                for(; vec_itr < vec_count; vec_itr++)
                {
                    ae_int16 _ae_int16_bias = ZERO16;
                    ae_int64 _ae_int64_sat_bias = ZERO64;
                    ae_int16 *_ae_int16_p_bias = (ae_int16 *) p_bias;
                    for(m_itr = 0; m_itr < rows; m_itr++)
                    {
                        ae_int64 _ae_int64_acc_0=ZERO64;
                        ae_int64 _ae_int64_acc_1=ZERO64;
                        ae_int64 _ae_int64_acc_2=ZERO64;
                        ae_int64 _ae_int64_acc_3=ZERO64;
                        ae_int64 * output_ptr=(ae_int64 *)(p_out[vec_itr]+m_itr);
                        ae_int16x8 * _ae_int16x8_p_vec1 = (ae_int16x8 *) (p_vec1[vec_itr]);
                        ae_int8x8 * _ae_int8x8_p_mat1_0  = (ae_int8x8 *) &p_mat1[(m_itr+0)*row_stride1];
                        ZER0_8x8_Temp_Variable;
                        ae_int8x8 _ae_int8x8_mat1_0;
                        ae_int16x4 _ae_int16x4_vec1,_ae_int16x4_vec1_1;
                        for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
                        {
                              AE_L16X4X2_IP(_ae_int16x4_vec1,_ae_int16x4_vec1_1, _ae_int16x8_p_vec1, 2*INCREMENT_IN_BYTES_FOR_INT16X4);
                              AE_L8X8_IP(_ae_int8x8_mat1_0,_ae_int8x8_p_mat1_0, 8);

                              AE_MULA8QW8X16(_ae_int64_acc_0  , _ae_int64_acc_1  , _ae_int64_acc_2  ,_ae_int64_acc_3,
                                             _ae_int8x8_mat1_0, zero_temp, zero_temp,zero_temp,
                                            _ae_int16x4_vec1,_ae_int16x4_vec1_1);

                        }
                        ae_int16_loadip(_ae_int16_bias, _ae_int16_p_bias, INCREMENT_IN_BYTES_FOR_INT16);
                        _ae_int64_sat_bias = AE_SLAA64S(((ae_int64) _ae_int16_bias), bias_shift);
                        _ae_int64_acc_0 = AE_ADD64S(_ae_int64_acc_0, _ae_int64_sat_bias);
                        _ae_int64_acc_0 =  AE_SLAA64S(_ae_int64_acc_0,acc_shift);

                        AE_S64_IP(_ae_int64_acc_0,output_ptr,sizeof(ae_int64));
                    }
                }
            }
        }
    }
    else if (p_mat1 && p_vec1)
    {
        ZER0_8x8_Temp_Variable;
        if(rows > 2 && vec_count > VEC_UNROLL)
        {
            for (vec_itr = 0; vec_itr < (vec_count & ~(VEC_UNROLL-1)); vec_itr += VEC_UNROLL)
            {
              SETUP_BIAS;
              m_itr = 0;
              for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
              {
                SETUP_ACC_64b_8x16(0);
                SETUP_ACC_64b_8x16(1);
                SETUP_ACC_64b_8x16(2);
                SETUP_ACC_64b_8x16(3);
                SETUP_MAT1_8b_UNALIGNED(0);
                SETUP_MAT1_8b_UNALIGNED(1);
                SETUP_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                SETUP_VEC1_16b_UNALIGNED_8x16_BATCH(1);
                int cols_count=cols1-cols1%8;
                for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
                {
                    LOAD_ROW_MAT1_8b_UNALIGNED(0);
                    LOAD_ROW_MAT1_8b_UNALIGNED(1);
                    LOAD_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                    LOAD_VEC1_16b_UNALIGNED_8x16_BATCH(1);
                    KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_BATCH;
                }
                for(c_itr = cols_count; c_itr < cols1; c_itr++)
                {
                    LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);

                    LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(1);
                    LOAD_VEC1_16b_UNALIGNED_8x16_SINGLE_ELEMENT_BATCH(0);
                    LOAD_VEC1_16b_UNALIGNED_8x16_SINGLE_ELEMENT_BATCH(1);
                    KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_BATCH;
                }
                ADD_BIAS_16b_ACC_FOR_8bx16b_BATCH(0,2);
                ADD_BIAS_16b_ACC_FOR_8bx16b_BATCH(1,3);
                STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(0,0,0);
                STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(1,0,1);
                STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(0,1,2);
                STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(1,1,3);
              }
              // remaining 1 row
              {
                  for(; m_itr < rows; m_itr++)
                  {
                    SETUP_ACC_64b_8x16(0);
                    SETUP_ACC_64b_8x16(1);
                    SETUP_ACC_64b_8x16(2);
                    SETUP_ACC_64b_8x16(3);
                    SETUP_MAT1_8b_UNALIGNED(0);
                    SETUP_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                    SETUP_VEC1_16b_UNALIGNED_8x16_BATCH(1);
                    int cols_count=cols1-cols1%8;
                    for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
                    {
                        LOAD_ROW_MAT1_8b_UNALIGNED(0);
                        LOAD_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                        LOAD_VEC1_16b_UNALIGNED_8x16_BATCH(1);
                        KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_BATCH_SINGLE_ROW;
                    }
                    for(c_itr = cols_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);
                        LOAD_VEC1_16b_UNALIGNED_8x16_SINGLE_ELEMENT_BATCH(0);
                        LOAD_VEC1_16b_UNALIGNED_8x16_SINGLE_ELEMENT_BATCH(1);
                        KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_BATCH_SINGLE_ROW;
                    }
                    ADD_BIAS_16b_ACC_FOR_8bx16b_BATCH(0,2);
                    STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(0,0,0);
                    STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(0,1,2);
                  }
               }
            }
            // Remaining one vector
            for (; vec_itr < vec_count ; vec_itr++)
            {
                SETUP_BIAS;
                m_itr = 0;
                if(rows > 2)
                {
                  for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
                  {
                    SETUP_ACC_64b_8x16(0);
                    SETUP_ACC_64b_8x16(1);
                    SETUP_ACC_64b_8x16(2);
                    SETUP_ACC_64b_8x16(3);
                    SETUP_MAT1_8b_UNALIGNED(0);
                    SETUP_MAT1_8b_UNALIGNED(1);
                    SETUP_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                    int cols_count=cols1-cols1%8;
                    for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
                    {
                        LOAD_ROW_MAT1_8b_UNALIGNED(0);
                        LOAD_ROW_MAT1_8b_UNALIGNED(1);
                        LOAD_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                        KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_VECTOR;
                    }
                    for(c_itr = cols_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);
                        LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(1);
                        LOAD_VEC1_16b_UNALIGNED_8x16_SINGLE_ELEMENT_BATCH(0);
                        KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_VECTOR;
                    }
                    ADD_BIAS_16b_ACC_FOR_8bx16b(0);
                    ADD_BIAS_16b_ACC_FOR_8bx16b(1);
                    STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(0,0,0);
                    STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(1,0,1);
                  }
                }
                {
                  for(; m_itr < rows; m_itr++)
                  {
                    SETUP_ACC_64b_8x16(0);
                    SETUP_ACC_64b_8x16(1);
                    SETUP_ACC_64b_8x16(2);
                    SETUP_ACC_64b_8x16(3);
                    SETUP_MAT1_8b_UNALIGNED(0);
                    SETUP_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                    int cols_count=cols1-cols1%8;
                    for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
                    {
                        LOAD_ROW_MAT1_8b_UNALIGNED(0);
                        LOAD_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                        KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ROW_8x16_BATCH;
                    }
                    for(c_itr = cols_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);
                        LOAD_VEC1_16b_UNALIGNED_8x16_SINGLE_ELEMENT_BATCH(0);
                        KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW_8x16_batch;
                    }
                    ADD_BIAS_16b_ACC_FOR_8bx16b(0);
                    STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(0,0,0);
                  }
                }
            }

        }
        else
        {
            for (vec_itr = 0; vec_itr < vec_count ; vec_itr++)
            {
                SETUP_BIAS;
                m_itr = 0;
                if(rows > 2)
                {
                  for(m_itr = 0; m_itr < (rows & ~(2-1)); m_itr += 2)
                  {
                    SETUP_ACC_64b_8x16(0);
                    SETUP_ACC_64b_8x16(1);
                    SETUP_ACC_64b_8x16(2);
                    SETUP_ACC_64b_8x16(3);
                    SETUP_MAT1_8b_UNALIGNED(0);
                    SETUP_MAT1_8b_UNALIGNED(1);
                    SETUP_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                    int cols_count=cols1-cols1%8;
                    for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
                    {
                        LOAD_ROW_MAT1_8b_UNALIGNED(0);
                        LOAD_ROW_MAT1_8b_UNALIGNED(1);
                        LOAD_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                        KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_VECTOR;
                    }
                    for(c_itr = cols_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);
                        LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(1);
                        LOAD_VEC1_16b_UNALIGNED_8x16_SINGLE_ELEMENT_BATCH(0);
                        KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_VECTOR;
                    }
                    ADD_BIAS_16b_ACC_FOR_8bx16b(0);
                    ADD_BIAS_16b_ACC_FOR_8bx16b(1);
                    STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(0,0,0);
                    STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(1,0,1);
                  }
                }
                {
                  for(; m_itr < rows; m_itr++)
                  {
                    SETUP_ACC_64b_8x16(0);
                    SETUP_ACC_64b_8x16(1);
                    SETUP_ACC_64b_8x16(2);
                    SETUP_ACC_64b_8x16(3);
                    SETUP_MAT1_8b_UNALIGNED(0);
                    SETUP_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                    int cols_count=cols1-cols1%8;
                    for(c_itr = 0; c_itr < cols_count>>3; c_itr++)
                    {
                        LOAD_ROW_MAT1_8b_UNALIGNED(0);
                        LOAD_VEC1_16b_UNALIGNED_8x16_BATCH(0);
                        KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ROW_8x16_BATCH;
                    }
                    for(c_itr = cols_count; c_itr < cols1; c_itr++)
                    {
                        LOAD_ROW_MAT1_8b_UNALIGNED_SINGLE_ELEMENT(0);
                        LOAD_VEC1_16b_UNALIGNED_8x16_SINGLE_ELEMENT_BATCH(0);
                        KERNEL_MAT1_VEC1_8b_16b_UNALIGNED_SINGLE_ELEMENT_SINGLE_ROW_8x16_batch;
                    }
                    ADD_BIAS_16b_ACC_FOR_8bx16b(0);
                    STORE_ACC_BATCH_16bx16b_AT_OUT_64b_UNALIGNED_SUPPORT(0,0,0);
                  }
                }

            }
        }
    }
    else
    {
      return -1;
    }

    #undef UNROLL_ROW_SETUP_ACC_BATCH
    #undef UNROLL_SETUP_ACC_BATCH
    #undef UNROLL_SETUP_MAT1
    #undef UNROLL_SETUP_VEC_BATCH
    #undef SETUP_BIAS
    #undef UNROLL_LOAD_VEC_BATCH
    #undef UNROLL_LOAD_ROW_MAT1
    #undef LOAD_BIAS
    #undef UNROLL_ROW_KERNEL_MAT1_VEC_BATCH
    #undef UNROLL_KERNEL_MAT1_VEC_BATCH
    #undef UNROLL_ROW_ADD_BIAS_ACC
    #undef UNROLL_ADD_BIAS_ACC_BATCH
    #undef UNROLL_ROW_STORE_ACC
    #undef UNROLL_STORE_ACC_BATCH
    #undef VEC_UNROLL
    return 0;
}

    #undef ROW_UNROLL
