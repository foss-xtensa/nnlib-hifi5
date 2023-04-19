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
#include "xa_nnlib_common_macros_hifi5.h"

extern const long long g_sel_pattern[16];

static inline void _xa_nn_dot_product_4_rows_4_vecs_unaligned
(ae_int64* out_0,
 ae_int64* out_1,
 ae_int64* out_2,
 ae_int64* out_3,
 ae_int64* out_4,
 ae_int64* out_5,
 ae_int64* out_6,
 ae_int64* out_7,
 ae_int64* out_8,
 ae_int64* out_9,
 ae_int64* out_10,
 ae_int64* out_11,
 ae_int64* out_12,
 ae_int64* out_13,
 ae_int64* out_14, 
 ae_int64* out_15,
 ae_int8*    p_mat1,
 ae_int8*    p_mat2,
 ae_int8*    p_mat3,
 ae_int8*    p_mat4,
 ae_int16x8*   p_vec1,    
 ae_int16x8*   p_vec2,
 ae_int16x8*   p_vec3,
 ae_int16x8*   p_vec4,
 WORD32    cols1)
{
    ae_int64 acc_0 = *out_0; 
    ae_int64 acc_1 = *out_1;
    ae_int64 acc_2 = *out_2;
    ae_int64 acc_3 = *out_3;
    ae_int64 acc_4 = *out_4;
    ae_int64 acc_5 = *out_5;
    ae_int64 acc_6 = *out_6;
    ae_int64 acc_7 = *out_7;
    ae_int64 acc_8 = *out_8;
    ae_int64 acc_9 = *out_9;
    ae_int64 acc_10 = *out_10;
    ae_int64 acc_11 = *out_11;
    ae_int64 acc_12 = *out_12;
    ae_int64 acc_13 = *out_13;
    ae_int64 acc_14 = *out_14;
    ae_int64 acc_15 = *out_15;   

    ae_int16x4 vec_batch_0_0_0 = 0, vec_batch_0_0_1 = 0, vec_batch_0_1_0 = 0, vec_batch_0_1_1 = 0;
    ae_int16x4 vec_batch_1_0_0 = 0, vec_batch_1_0_1 = 0, vec_batch_1_1_0 = 0, vec_batch_1_1_1 = 0;

    ae_valignx2 align_vec_batch_0 = AE_LA128_PP(p_vec1);                
    ae_valignx2 align_vec_batch_1 = AE_LA128_PP(p_vec2);  
    ae_valignx2 align_vec_batch_2 = AE_LA128_PP(p_vec3);  
    ae_valignx2 align_vec_batch_3 = AE_LA128_PP(p_vec4);  

    ae_int8x8 mat1_0_0 = 0, mat1_0_1 = 0, mat1_1_0 = 0, mat1_1_1 = 0;
    ae_int8x8 align_mat1_0_0 = 0, align_mat1_0_1 = 0, align_mat1_1_0 = 0, align_mat1_1_1 = 0;

    AE_SW_PRIME_64(p_mat1, align_mat1_0_0);
    AE_SW_PRIME_64(p_mat2, align_mat1_0_1);
    AE_SW_PRIME_64(p_mat3, align_mat1_1_0);
    AE_SW_PRIME_64(p_mat4, align_mat1_1_1);

    int c_itr;
    for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
    {
        AE_LA16X4X2_IP(vec_batch_0_0_0, vec_batch_0_0_1, align_vec_batch_0, p_vec1);
        AE_LA16X4X2_IP(vec_batch_0_1_0, vec_batch_0_1_1, align_vec_batch_1, p_vec2);
        AE_LA16X4X2_IP(vec_batch_1_0_0, vec_batch_1_0_1, align_vec_batch_2, p_vec3);
        AE_LA16X4X2_IP(vec_batch_1_1_0, vec_batch_1_1_1, align_vec_batch_3, p_vec4);

        AE_SW_LA8X8_IP(mat1_0_0, align_mat1_0_0, p_mat1);
        AE_SW_LA8X8_IP(mat1_0_1, align_mat1_0_1, p_mat2);
        AE_SW_LA8X8_IP(mat1_1_0, align_mat1_1_0, p_mat3);
        AE_SW_LA8X8_IP(mat1_1_1, align_mat1_1_1, p_mat4);

        AE_MULA8QW8X16(acc_0, acc_1, acc_2, acc_3, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_0_0_0, vec_batch_0_0_1);
        AE_MULA8QW8X16(acc_4, acc_5, acc_6, acc_7, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_0_1_0, vec_batch_0_1_1);
        AE_MULA8QW8X16(acc_8, acc_9, acc_10, acc_11, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_1_0_0, vec_batch_1_0_1);
        AE_MULA8QW8X16(acc_12, acc_13, acc_14, acc_15, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_1_1_0, vec_batch_1_1_1);
    }
    if ((cols1 & 7) != 0)
    {
        AE_LAV16X4X2_XP(vec_batch_0_0_0, vec_batch_0_0_1, align_vec_batch_0, p_vec1, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_0_1_0, vec_batch_0_1_1, align_vec_batch_1, p_vec2, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_1_0_0, vec_batch_1_0_1, align_vec_batch_2, p_vec3, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_1_1_0, vec_batch_1_1_1, align_vec_batch_3, p_vec4, (cols1 & 7) * 2);

        AE_SW_LA8X8_IP(mat1_0_0, align_mat1_0_0, p_mat1);
        AE_SW_LA8X8_IP(mat1_0_1, align_mat1_0_1, p_mat2);
        AE_SW_LA8X8_IP(mat1_1_0, align_mat1_1_0, p_mat3);
        AE_SW_LA8X8_IP(mat1_1_1, align_mat1_1_1, p_mat4);

        AE_MULA8QW8X16(acc_0, acc_1, acc_2, acc_3, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_0_0_0, vec_batch_0_0_1);
        AE_MULA8QW8X16(acc_4, acc_5, acc_6, acc_7, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_0_1_0, vec_batch_0_1_1);
        AE_MULA8QW8X16(acc_8, acc_9, acc_10, acc_11, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_1_0_0, vec_batch_1_0_1);
        AE_MULA8QW8X16(acc_12, acc_13, acc_14, acc_15, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_1_1_0, vec_batch_1_1_1);
    }  

    *out_0 =  acc_0;
    *out_1 =  acc_1;
    *out_2 =  acc_2;
    *out_3 =  acc_3;
    *out_4 =  acc_4;
    *out_5 =  acc_5;
    *out_6 =  acc_6;
    *out_7 =  acc_7;
    *out_8 =  acc_8;
    *out_9 =  acc_9;
    *out_10 = acc_10;
    *out_11 = acc_11;
    *out_12 = acc_12;
    *out_13 = acc_13;
    *out_14 = acc_14;
    *out_15 = acc_15;
}

static inline void _xa_nn_dot_product_1_row_4_vecs_unaligned
(ae_int64* out_0,
 ae_int64* out_1,
 ae_int64* out_2,
 ae_int64* out_3,
 ae_int8*    p_mat1,
 ae_int16x8* p_vec1,    
 ae_int16x8* p_vec2,
 ae_int16x8* p_vec3,
 ae_int16x8* p_vec4,
 WORD32    cols1)
{
    ae_int64 acc_0_0 = *out_0;
    ae_int64 acc_0_1 = *out_1;
    ae_int64 acc_1_0 = *out_2;
    ae_int64 acc_1_1 = *out_3;

    ae_int16x4 vec_batch_0_0_0 = 0, vec_batch_0_0_1  = 0, vec_batch_0_1_0 = 0, vec_batch_0_1_1 = 0, vec_batch_1_0_0 = 0, vec_batch_1_0_1 = 0, vec_batch_1_1_0 = 0, vec_batch_1_1_1 = 0;
    ae_int16x4 mat1_0_0 = 0,  mat1_0_1 = 0;
    ae_valignx2 align_vec_batch_0 = AE_LA128_PP(p_vec1);                
    ae_valignx2 align_vec_batch_1 = AE_LA128_PP(p_vec2);  
    ae_valignx2 align_vec_batch_2 = AE_LA128_PP(p_vec3);  
    ae_valignx2 align_vec_batch_3 = AE_LA128_PP(p_vec4);  

    ae_valign align_mat_0 = AE_LA64_PP(p_mat1);

    int cols1_count = cols1- cols1%8;
    int c_itr;
    for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
    {
        AE_LA16X4X2_IP(vec_batch_0_0_0, vec_batch_0_0_1, align_vec_batch_0, p_vec1);
        AE_LA16X4X2_IP(vec_batch_0_1_0, vec_batch_0_1_1, align_vec_batch_1, p_vec2);
        AE_LA16X4X2_IP(vec_batch_1_0_0, vec_batch_1_0_1, align_vec_batch_2, p_vec3);
        AE_LA16X4X2_IP(vec_batch_1_1_0, vec_batch_1_1_1, align_vec_batch_3, p_vec4);

        AE_LA8X4S_IP(mat1_0_0, align_mat_0, (WORD8 *)p_mat1);
        AE_LA8X4S_IP(mat1_0_1, align_mat_0, (WORD8 *)p_mat1);

        AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_0, mat1_0_0, vec_batch_0_0_0, vec_batch_0_1_0);
        AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_1, mat1_0_1, vec_batch_0_0_1, vec_batch_0_1_1);
        AE_MULAAAA2Q16(acc_1_0, acc_1_1, mat1_0_0, mat1_0_0, vec_batch_1_0_0, vec_batch_1_1_0);
        AE_MULAAAA2Q16(acc_1_0, acc_1_1, mat1_0_1, mat1_0_1, vec_batch_1_0_1, vec_batch_1_1_1);
    }
    if((cols1 & 7) != 0)
    {
        AE_LAV16X4X2_XP(vec_batch_0_0_0, vec_batch_0_0_1, align_vec_batch_0, p_vec1, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_0_1_0, vec_batch_0_1_1, align_vec_batch_1, p_vec2, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_1_0_0, vec_batch_1_0_1, align_vec_batch_2, p_vec3, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_1_1_0, vec_batch_1_1_1, align_vec_batch_3, p_vec4, (cols1 & 7) * 2);

        AE_LA8X4S_IP(mat1_0_0, align_mat_0, (WORD8 *)p_mat1);
        AE_LA8X4S_IP(mat1_0_1, align_mat_0, (WORD8 *)p_mat1);

        AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_0, mat1_0_0, vec_batch_0_0_0, vec_batch_0_1_0);
        AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_1, mat1_0_1, vec_batch_0_0_1, vec_batch_0_1_1);
        AE_MULAAAA2Q16(acc_1_0, acc_1_1, mat1_0_0, mat1_0_0, vec_batch_1_0_0, vec_batch_1_1_0);
        AE_MULAAAA2Q16(acc_1_0, acc_1_1, mat1_0_1, mat1_0_1, vec_batch_1_0_1, vec_batch_1_1_1);                  
    }    

    *out_0 =  acc_0_0;
    *out_1 =  acc_0_1;
    *out_2 =  acc_1_0;
    *out_3 =  acc_1_1;

}

static inline void _xa_nn_dot_product_4_rows_1_vec_unaligned
(ae_int64* out_0,
 ae_int64* out_1,
 ae_int64* out_2,
 ae_int64* out_3,
 ae_int8*    p_mat1,
 ae_int8*    p_mat2,
 ae_int8*    p_mat3,
 ae_int8*    p_mat4,
 ae_int16x8* p_vec1,    
 WORD32    cols1)
{
    ae_int64 acc_0 = *out_0;
    ae_int64 acc_1 = *out_1;
    ae_int64 acc_2 = *out_2;
    ae_int64 acc_3 = *out_3;

    ae_int16x4 vec_batch_0_0  = 0, vec_batch_0_1 = 0;
    ae_valignx2 align_vec0 = AE_LA128_PP(p_vec1);
    ae_int8x8 mat0_1 = 0,  mat0_2 = 0, mat0_3 = 0,  mat0_4 = 0;
    ae_valign align_mat0_1 = AE_LA64_PP(p_mat1), align_mat0_2 = AE_LA64_PP(p_mat2), align_mat0_3 = AE_LA64_PP(p_mat3), align_mat0_4 = AE_LA64_PP(p_mat4);

    int cols1_count = cols1 - cols1%8;
    int c_itr;
    for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
    {
        AE_LA16X4X2_IP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec1);
        AE_LA8X8_IP(mat0_1, align_mat0_1, (ae_int8x8 *)p_mat1);
        AE_LA8X8_IP(mat0_2, align_mat0_2, (ae_int8x8 *)p_mat2);
        AE_LA8X8_IP(mat0_3, align_mat0_3, (ae_int8x8 *)p_mat3);
        AE_LA8X8_IP(mat0_4, align_mat0_4, (ae_int8x8 *)p_mat4);
        AE_MULA8QW8X16(acc_0, acc_1, acc_2, acc_3, mat0_1, mat0_2, mat0_3, mat0_4, vec_batch_0_0, vec_batch_0_1);
    }
    if((cols1&7) !=0)
    {
        AE_LAV16X4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec1, (cols1&7) * 2 );
        AE_LA8X8_IP(mat0_1, align_mat0_1, (ae_int8x8 *)p_mat1);
        AE_LA8X8_IP(mat0_2, align_mat0_2, (ae_int8x8 *)p_mat2);
        AE_LA8X8_IP(mat0_3, align_mat0_3, (ae_int8x8 *)p_mat3);
        AE_LA8X8_IP(mat0_4, align_mat0_4, (ae_int8x8 *)p_mat4);
        AE_MULA8QW8X16(acc_0, acc_1, acc_2, acc_3, mat0_1, mat0_2, mat0_3, mat0_4, vec_batch_0_0, vec_batch_0_1);
    }
    *out_0 =  acc_0;
    *out_1 =  acc_1;
    *out_2 =  acc_2;
    *out_3 =  acc_3;    
}

static inline void _xa_nn_dot_product_1_row_1_vec_unaligned
(ae_int64* out_0,
 ae_int8*    p_mat1,
 ae_int16x4* p_vec1,    
 WORD32    cols1)
{
    ae_int64 acc_0_0 = *out_0;
    ae_int16x4 vec_batch_0  = 0;
    ae_int16x4 mat1_0 = 0;
    ae_valign align_vec0 = AE_LA64_PP(p_vec1);
    ae_valign align_mat_0 = AE_LA64_PP(p_mat1);
    int cols1_count = cols1 - cols1%4;
    int c_itr;
    for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
    {
        AE_LA16X4_IP(vec_batch_0, align_vec0, p_vec1);
        AE_LA8X4S_IP(mat1_0, align_mat_0, (WORD8 *)p_mat1);
        AE_MULAAAAQ16(acc_0_0, vec_batch_0, mat1_0);
    }
    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
    {
        AE_L16_IP(vec_batch_0, (ae_int16 *)p_vec1, 2);
        AE_L16_IP(mat1_0, (ae_int16 *)p_mat1, 2);
        mat1_0 = AE_MOVDA16(((WORD16)*((WORD8 *)(p_mat1))));
        p_mat1++;
        AE_MULA16_00(acc_0_0, vec_batch_0, mat1_0);
    } 
    *out_0 = acc_0_0;
}

static inline void _xa_nn_dot_product_4_rows_4_vecs_aligned
(ae_int64* out_0,
 ae_int64* out_1,
 ae_int64* out_2,
 ae_int64* out_3,
 ae_int64* out_4,
 ae_int64* out_5,
 ae_int64* out_6,
 ae_int64* out_7,
 ae_int64* out_8,
 ae_int64* out_9,
 ae_int64* out_10,
 ae_int64* out_11,
 ae_int64* out_12,
 ae_int64* out_13,
 ae_int64* out_14, 
 ae_int64* out_15,
 ae_int8*    p_mat1,
 ae_int8*    p_mat2,
 ae_int8*    p_mat3,
 ae_int8*    p_mat4,
 ae_int16x8*   p_vec1,    
 ae_int16x8*   p_vec2,
 ae_int16x8*   p_vec3,
 ae_int16x8*   p_vec4,
 WORD32    cols1)
{
    ae_int64 acc_0 = *out_0; 
    ae_int64 acc_1 = *out_1;
    ae_int64 acc_2 = *out_2;
    ae_int64 acc_3 = *out_3;
    ae_int64 acc_4 = *out_4;
    ae_int64 acc_5 = *out_5;
    ae_int64 acc_6 = *out_6;
    ae_int64 acc_7 = *out_7;
    ae_int64 acc_8 = *out_8;
    ae_int64 acc_9 = *out_9;
    ae_int64 acc_10 = *out_10;
    ae_int64 acc_11 = *out_11;
    ae_int64 acc_12 = *out_12;
    ae_int64 acc_13 = *out_13;
    ae_int64 acc_14 = *out_14;
    ae_int64 acc_15 = *out_15;   

    ae_int16x4 vec_batch_0_0_0 = 0, vec_batch_0_0_1 = 0, vec_batch_0_1_0 = 0, vec_batch_0_1_1 = 0;
    ae_int16x4 vec_batch_1_0_0 = 0, vec_batch_1_0_1 = 0, vec_batch_1_1_0 = 0, vec_batch_1_1_1 = 0;

    ae_int8x8 mat1_0_0 = 0, mat1_0_1 = 0, mat1_1_0 = 0, mat1_1_1 = 0;

    int c_itr;
    for(c_itr = 0; c_itr < (cols1 >> 3); c_itr++)
    {
        AE_L16X4X2_IP(vec_batch_0_0_0, vec_batch_0_0_1, p_vec1, 16);
        AE_L16X4X2_IP(vec_batch_0_1_0, vec_batch_0_1_1, p_vec2, 16);
        AE_L16X4X2_IP(vec_batch_1_0_0, vec_batch_1_0_1, p_vec3, 16);
        AE_L16X4X2_IP(vec_batch_1_1_0, vec_batch_1_1_1, p_vec4, 16);

        AE_L8X8_IP(mat1_0_0, (ae_int8x8 *)p_mat1, 8);
        AE_L8X8_IP(mat1_0_1, (ae_int8x8 *)p_mat2, 8);
        AE_L8X8_IP(mat1_1_0, (ae_int8x8 *)p_mat3, 8);
        AE_L8X8_IP(mat1_1_1, (ae_int8x8 *)p_mat4, 8);

        AE_MULA8QW8X16(acc_0, acc_1, acc_2, acc_3, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_0_0_0, vec_batch_0_0_1);
        AE_MULA8QW8X16(acc_4, acc_5, acc_6, acc_7, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_0_1_0, vec_batch_0_1_1);
        AE_MULA8QW8X16(acc_8, acc_9, acc_10, acc_11, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_1_0_0, vec_batch_1_0_1);
        AE_MULA8QW8X16(acc_12, acc_13, acc_14, acc_15, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_1_1_0, vec_batch_1_1_1);
    }
    ae_int8x8 align_mat1_0_0 = 0, align_mat1_0_1 = 0, align_mat1_1_0 = 0, align_mat1_1_1 = 0;
    ae_valignx2 align_vec_batch_0 = AE_LA128_PP(p_vec1);                
    ae_valignx2 align_vec_batch_1 = AE_LA128_PP(p_vec2);  
    ae_valignx2 align_vec_batch_2 = AE_LA128_PP(p_vec3);  
    ae_valignx2 align_vec_batch_3 = AE_LA128_PP(p_vec4);  
    AE_SW_PRIME_64(p_mat1, align_mat1_0_0);
    AE_SW_PRIME_64(p_mat2, align_mat1_0_1);
    AE_SW_PRIME_64(p_mat3, align_mat1_1_0);
    AE_SW_PRIME_64(p_mat4, align_mat1_1_1);

    if ((cols1 & 7) != 0)
    {
        AE_LAV16X4X2_XP(vec_batch_0_0_0, vec_batch_0_0_1, align_vec_batch_0, p_vec1, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_0_1_0, vec_batch_0_1_1, align_vec_batch_1, p_vec2, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_1_0_0, vec_batch_1_0_1, align_vec_batch_2, p_vec3, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_1_1_0, vec_batch_1_1_1, align_vec_batch_3, p_vec4, (cols1 & 7) * 2);

        AE_SW_LA8X8_IP(mat1_0_0, align_mat1_0_0, p_mat1);
        AE_SW_LA8X8_IP(mat1_0_1, align_mat1_0_1, p_mat2);
        AE_SW_LA8X8_IP(mat1_1_0, align_mat1_1_0, p_mat3);
        AE_SW_LA8X8_IP(mat1_1_1, align_mat1_1_1, p_mat4);

        AE_MULA8QW8X16(acc_0, acc_1, acc_2, acc_3, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_0_0_0, vec_batch_0_0_1);
        AE_MULA8QW8X16(acc_4, acc_5, acc_6, acc_7, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_0_1_0, vec_batch_0_1_1);
        AE_MULA8QW8X16(acc_8, acc_9, acc_10, acc_11, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_1_0_0, vec_batch_1_0_1);
        AE_MULA8QW8X16(acc_12, acc_13, acc_14, acc_15, mat1_0_0, mat1_0_1, mat1_1_0, mat1_1_1, vec_batch_1_1_0, vec_batch_1_1_1);
    }   

    *out_0 =  acc_0;
    *out_1 =  acc_1;
    *out_2 =  acc_2;
    *out_3 =  acc_3;
    *out_4 =  acc_4;
    *out_5 =  acc_5;
    *out_6 =  acc_6;
    *out_7 =  acc_7;
    *out_8 =  acc_8;
    *out_9 =  acc_9;
    *out_10 = acc_10;
    *out_11 = acc_11;
    *out_12 = acc_12;
    *out_13 = acc_13;
    *out_14 = acc_14;
    *out_15 = acc_15;
}

static inline void _xa_nn_dot_product_1_row_4_vecs_aligned
(ae_int64* out_0,
 ae_int64* out_1,
 ae_int64* out_2,
 ae_int64* out_3,
 ae_int8*    p_mat1,
 ae_int16x8* p_vec1,    
 ae_int16x8* p_vec2,
 ae_int16x8* p_vec3,
 ae_int16x8* p_vec4,
 WORD32    cols1)
{
    ae_int64 acc_0_0 = *out_0;
    ae_int64 acc_0_1 = *out_1;
    ae_int64 acc_1_0 = *out_2;
    ae_int64 acc_1_1 = *out_3;

    ae_int16x4 vec_batch_0_0_0  = 0, vec_batch_0_0_1  = 0, vec_batch_0_1_0  = 0, vec_batch_0_1_1  = 0, vec_batch_1_0_0  = 0, vec_batch_1_0_1  = 0, vec_batch_1_1_0  = 0, vec_batch_1_1_1  = 0;
    ae_int16x4 mat1_0_0 = 0,  mat1_0_1 = 0;

    int cols1_count = cols1- cols1%8;
    int c_itr;
    for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
    {
        AE_L16X4X2_IP(vec_batch_0_0_0, vec_batch_0_0_1, p_vec1, 16);
        AE_L16X4X2_IP(vec_batch_0_1_0, vec_batch_0_1_1, p_vec2, 16);
        AE_L16X4X2_IP(vec_batch_1_0_0, vec_batch_1_0_1, p_vec3, 16);
        AE_L16X4X2_IP(vec_batch_1_1_0, vec_batch_1_1_1, p_vec4, 16);

        AE_L8X4S_IP(mat1_0_0, (WORD8 *)p_mat1, 4);
        AE_L8X4S_IP(mat1_0_1, (WORD8 *)p_mat1, 4);

        AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_0, mat1_0_0, vec_batch_0_0_0, vec_batch_0_1_0);
        AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_1, mat1_0_1, vec_batch_0_0_1, vec_batch_0_1_1);
        AE_MULAAAA2Q16(acc_1_0, acc_1_1, mat1_0_0, mat1_0_0, vec_batch_1_0_0, vec_batch_1_1_0);
        AE_MULAAAA2Q16(acc_1_0, acc_1_1, mat1_0_1, mat1_0_1, vec_batch_1_0_1, vec_batch_1_1_1);
    }

    ae_valignx2 align_vec_batch_0 = AE_LA128_PP(p_vec1);                
    ae_valignx2 align_vec_batch_1 = AE_LA128_PP(p_vec2);  
    ae_valignx2 align_vec_batch_2 = AE_LA128_PP(p_vec3);  
    ae_valignx2 align_vec_batch_3 = AE_LA128_PP(p_vec4);                  
    ae_valign align_mat_0 = AE_LA64_PP(p_mat1);

    if((cols1 & 7) != 0)
    {
        AE_LAV16X4X2_XP(vec_batch_0_0_0, vec_batch_0_0_1, align_vec_batch_0, p_vec1, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_0_1_0, vec_batch_0_1_1, align_vec_batch_1, p_vec2, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_1_0_0, vec_batch_1_0_1, align_vec_batch_2, p_vec3, (cols1 & 7) * 2);
        AE_LAV16X4X2_XP(vec_batch_1_1_0, vec_batch_1_1_1, align_vec_batch_3, p_vec4, (cols1 & 7) * 2);

        AE_LA8X4S_IP(mat1_0_0, align_mat_0, (WORD8 *)p_mat1);
        AE_LA8X4S_IP(mat1_0_1, align_mat_0, (WORD8 *)p_mat1);

        AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_0, mat1_0_0, vec_batch_0_0_0, vec_batch_0_1_0);
        AE_MULAAAA2Q16(acc_0_0, acc_0_1, mat1_0_1, mat1_0_1, vec_batch_0_0_1, vec_batch_0_1_1);
        AE_MULAAAA2Q16(acc_1_0, acc_1_1, mat1_0_0, mat1_0_0, vec_batch_1_0_0, vec_batch_1_1_0);
        AE_MULAAAA2Q16(acc_1_0, acc_1_1, mat1_0_1, mat1_0_1, vec_batch_1_0_1, vec_batch_1_1_1);               
    }    

    *out_0 =  acc_0_0;
    *out_1 =  acc_0_1;
    *out_2 =  acc_1_0;
    *out_3 =  acc_1_1;

}

static inline void _xa_nn_dot_product_4_rows_1_vec_aligned
(ae_int64* out_0,
 ae_int64* out_1,
 ae_int64* out_2,
 ae_int64* out_3,
 ae_int8*    p_mat1,
 ae_int8*    p_mat2,
 ae_int8*    p_mat3,
 ae_int8*    p_mat4,
 ae_int16x8* p_vec1,    
 WORD32    cols1)
{
    ae_int64 acc_0 = *out_0;
    ae_int64 acc_1 = *out_1;
    ae_int64 acc_2 = *out_2;
    ae_int64 acc_3 = *out_3;

    ae_int16x4 vec_batch_0_0  = 0, vec_batch_0_1 = 0;
    ae_int8x8 mat0_1 = 0,  mat0_2 = 0, mat0_3 = 0,  mat0_4 = 0;

    int cols1_count = cols1 - cols1%8;
    int c_itr;
    for(c_itr = 0; c_itr < (cols1_count >> 3); c_itr++)
    {
        AE_L16X4X2_IP(vec_batch_0_0, vec_batch_0_1, p_vec1, 16);
        AE_L8X8_IP(mat0_1, (ae_int8x8 *)p_mat1, 8);
        AE_L8X8_IP(mat0_2, (ae_int8x8 *)p_mat2, 8);
        AE_L8X8_IP(mat0_3, (ae_int8x8 *)p_mat3, 8);
        AE_L8X8_IP(mat0_4, (ae_int8x8 *)p_mat4, 8);
        AE_MULA8QW8X16(acc_0, acc_1, acc_2, acc_3, mat0_1, mat0_2, mat0_3, mat0_4, vec_batch_0_0, vec_batch_0_1);
    }
    ae_valignx2 align_vec0 = AE_LA128_PP(p_vec1);
    ae_valign align_mat0_1 = AE_LA64_PP(p_mat1), align_mat0_2 = AE_LA64_PP(p_mat2), align_mat0_3 = AE_LA64_PP(p_mat3), align_mat0_4 = AE_LA64_PP(p_mat4);
    if((cols1&7) !=0)
    {
        AE_LAV16X4X2_XP(vec_batch_0_0, vec_batch_0_1, align_vec0, p_vec1, (cols1&7) * 2 );
        AE_LA8X8_IP(mat0_1, align_mat0_1, (ae_int8x8 *)p_mat1);
        AE_LA8X8_IP(mat0_2, align_mat0_2, (ae_int8x8 *)p_mat2);
        AE_LA8X8_IP(mat0_3, align_mat0_3, (ae_int8x8 *)p_mat3);
        AE_LA8X8_IP(mat0_4, align_mat0_4, (ae_int8x8 *)p_mat4);
        AE_MULA8QW8X16(acc_0, acc_1, acc_2, acc_3, mat0_1, mat0_2, mat0_3, mat0_4, vec_batch_0_0, vec_batch_0_1);
    }
    *out_0 =  acc_0;
    *out_1 =  acc_1;
    *out_2 =  acc_2;
    *out_3 =  acc_3;    
}

static inline void _xa_nn_dot_product_1_row_1_vec_aligned
(ae_int64* out_0,
 ae_int8*    p_mat1,
 ae_int16x4* p_vec1,    
 WORD32    cols1)
{
    ae_int64 acc_0_0 = *out_0;
    ae_int16x4 vec_batch_0  = 0;
    ae_int16x4 mat1_0 = 0;

    int cols1_count = cols1 - cols1%4;
    int c_itr;
    for(c_itr = 0; c_itr < (cols1_count >> 2); c_itr++)
    {
        AE_L16X4_IP(vec_batch_0, p_vec1, 8);
        AE_L8X4S_IP(mat1_0, (WORD8 *)p_mat1, 4);
        AE_MULAAAAQ16(acc_0_0, vec_batch_0, mat1_0);
    }
    for(c_itr = cols1_count; c_itr < cols1; c_itr++)
    {
        AE_L16_IP(vec_batch_0, (ae_int16 *)p_vec1, 2);
        AE_L16_IP(mat1_0, (ae_int16 *)p_mat1, 2);
        mat1_0 = AE_MOVDA16(((WORD16)*((WORD8 *)(p_mat1))));
        p_mat1++;
        AE_MULA16_00(acc_0_0, vec_batch_0, mat1_0);
    } 
    *out_0 = acc_0_0;
}

WORD32 xa_nn_matmul_8x16_16(
        WORD16 * __restrict__ p_out,          
        const WORD8 *  __restrict__ p_mat1,   
        const WORD16 * __restrict__ p_vec1,   
        const WORD16 *  __restrict__ p_bias,  
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
    XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
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
    int m_itr, vec_itr;
    /* Assign initial value so this value will be used in trailing loop */
    m_itr = 0;
    vec_itr = 0;

    acc_shift = acc_shift + 32;
    acc_shift = acc_shift > 63 ? 63 : acc_shift < -63 ? -63 : acc_shift;
    bias_shift = bias_shift > 63 ? 63 : bias_shift < -63 ? -63 : bias_shift;

    //Aligned part
    if(((row_stride1 & 7) == 0) && (((unsigned int)p_vec1 & 15) == 0) && (((unsigned int)p_mat1 & 7) == 0) && ((vec_offset & 7) ==0)) 
    {
        for (vec_itr = 0; vec_itr < (vec_count & ~(3)); vec_itr += 4)
        {           
            ae_int16 bias = (0);
            ae_int64 sat_bias = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
            ae_int16 *pbias = (ae_int16 *) p_bias;

            WORD16* p_dst_0 = (WORD16*)p_out + (vec_itr + 0) * out_offset;
            WORD16* p_dst_1 = (WORD16*)p_out + (vec_itr + 1) * out_offset;
            WORD16* p_dst_2 = (WORD16*)p_out + (vec_itr + 2) * out_offset;
            WORD16* p_dst_3 = (WORD16*)p_out + (vec_itr + 3) * out_offset;

            for(m_itr = 0; m_itr < (rows & ~(3)); m_itr += 4)
            {
                ae_int64 acc_0 = 0, acc_1 = 0, acc_2 = 0, acc_3 = 0, acc_4 = 0, acc_5 = 0, acc_6 = 0, acc_7 = 0, acc_8 = 0, acc_9 = 0, acc_10 = 0, acc_11 = 0, acc_12 = 0, acc_13 = 0, acc_14 = 0, acc_15 = 0;
                ae_int16x8 *p_vec_batch_0  = (ae_int16x8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int16x8 *p_vec_batch_1  = (ae_int16x8 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                ae_int16x8 *p_vec_batch_2  = (ae_int16x8 *)(p_vec1 + (vec_itr + 2)*vec_offset);
                ae_int16x8 *p_vec_batch_3  = (ae_int16x8 *)(p_vec1 + (vec_itr + 3)*vec_offset);
                ae_int8 *p_mat1_0_0 = (ae_int8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_int8 *p_mat1_0_1 = (ae_int8 *) &p_mat1[(m_itr+1)*row_stride1];
                ae_int8 *p_mat1_1_0 = (ae_int8 *) &p_mat1[(m_itr+2)*row_stride1];
                ae_int8 *p_mat1_1_1 = (ae_int8 *) &p_mat1[(m_itr+3)*row_stride1];

                _xa_nn_dot_product_4_rows_4_vecs_aligned
                    (&acc_0, &acc_1, &acc_2, &acc_3, &acc_4, &acc_5, &acc_6, &acc_7, &acc_8, &acc_9, &acc_10, &acc_11, &acc_12, &acc_13, &acc_14
                     ,&acc_15, p_mat1_0_0, p_mat1_0_1, p_mat1_1_0, p_mat1_1_1, p_vec_batch_0, p_vec_batch_1 , p_vec_batch_2, p_vec_batch_3, cols1);

                if(p_bias!=NULL)
                {
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_0 = AE_ADD64S(acc_0, sat_bias);
                    acc_4 = AE_ADD64S(acc_4, sat_bias);
                    acc_8 = AE_ADD64S(acc_8, sat_bias);
                    acc_12 = AE_ADD64S(acc_12, sat_bias);
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_1 = AE_ADD64S(acc_1, sat_bias);
                    acc_5 = AE_ADD64S(acc_5, sat_bias);
                    acc_9 = AE_ADD64S(acc_9, sat_bias);
                    acc_13 = AE_ADD64S(acc_13, sat_bias);                  
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_2 = AE_ADD64S(acc_2, sat_bias);
                    acc_6 = AE_ADD64S(acc_6, sat_bias); 
                    acc_10 = AE_ADD64S(acc_10, sat_bias);
                    acc_14 = AE_ADD64S(acc_14, sat_bias); 
                    ae_int16_loadip(bias, pbias, 2);                        
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_3 = AE_ADD64S(acc_3, sat_bias);
                    acc_7 = AE_ADD64S(acc_7, sat_bias);     
                    acc_11 = AE_ADD64S(acc_11, sat_bias);
                    acc_15 = AE_ADD64S(acc_15, sat_bias);      
                }

                ae_int32x2 out_val;
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_0, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_1, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_2, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_3, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_4, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_1, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_5, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_1, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_6, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_1, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_7, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_1, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_8, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_2, out_stride*(sizeof(WORD16)));      
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_9, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_2, out_stride*(sizeof(WORD16)));  
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_10, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_2, out_stride*(sizeof(WORD16)));  
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_11, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_2, out_stride*(sizeof(WORD16)));  
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_12, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_3, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_13, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_3, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_14, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_3, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_15, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_3, out_stride*(sizeof(WORD16)));                                                                                                                                                                                                                         
            }
            //Remaining row
            for(; m_itr < rows; m_itr++)
            {
                ae_int64 acc_0_0 = 0, acc_0_1 = 0, acc_1_0 = 0, acc_1_1 = 0;

                ae_int16x8 *p_vec_batch_0  = (ae_int16x8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int16x8 *p_vec_batch_1  = (ae_int16x8 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                ae_int16x8 *p_vec_batch_2  = (ae_int16x8 *)(p_vec1 + (vec_itr + 2)*vec_offset);
                ae_int16x8 *p_vec_batch_3  = (ae_int16x8 *)(p_vec1 + (vec_itr + 3)*vec_offset);
                ae_int8 *p_mat1_0 = (ae_int8 *) &p_mat1[(m_itr+0)*row_stride1];

                _xa_nn_dot_product_1_row_4_vecs_aligned
                    (&acc_0_0, &acc_0_1, &acc_1_0, &acc_1_1, p_mat1_0, p_vec_batch_0, p_vec_batch_1 , p_vec_batch_2, p_vec_batch_3, cols1);

                if(p_bias!=NULL)
                {
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_0_0 = AE_ADD64S(acc_0_0, sat_bias);
                    acc_0_1 = AE_ADD64S(acc_0_1, sat_bias);
                    acc_1_0 = AE_ADD64S(acc_1_0, sat_bias);
                    acc_1_1 = AE_ADD64S(acc_1_1, sat_bias);                  
                }
                ae_f32x2 out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_0, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_1, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_1, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_1_0, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_2, out_stride*(sizeof(WORD16)));     
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_1_1, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_3, out_stride*(sizeof(WORD16)));                        
            }
        }
        /* Tail loop for vec unroll */
        for(; vec_itr < vec_count; vec_itr++)
        {
            ae_int16 bias = (0);
            ae_int64 sat_bias = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
            ae_int16 *pbias = (ae_int16 *) p_bias;
            WORD16* p_dst_0 = (WORD16*)p_out + (vec_itr + 0) * out_offset;
            for(m_itr = 0; m_itr < (rows & ~(3)); m_itr += 4)
            {
                ae_int64 acc_0 = 0, acc_1 = 0, acc_2 = 0, acc_3 = 0;

                ae_int16x8 *p_vec_batch_0  = (ae_int16x8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int8 *p_mat0_0 = (ae_int8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_int8 *p_mat0_1 = (ae_int8 *) &p_mat1[(m_itr+1)*row_stride1];
                ae_int8 *p_mat1_0 = (ae_int8 *) &p_mat1[(m_itr+2)*row_stride1];
                ae_int8 *p_mat1_1 = (ae_int8 *) &p_mat1[(m_itr+3)*row_stride1];

                _xa_nn_dot_product_4_rows_1_vec_aligned
                    (&acc_0, &acc_1, &acc_2, &acc_3, p_mat0_0, p_mat0_1, p_mat1_0 , p_mat1_1, p_vec_batch_0, cols1);

                if(p_bias != NULL)
                {
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_0 = AE_ADD64S(acc_0, sat_bias);
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_1 = AE_ADD64S(acc_1, sat_bias);
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_2 = AE_ADD64S(acc_2, sat_bias);
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_3 = AE_ADD64S(acc_3, sat_bias);
                }
                ae_f32x2 out_val;
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_0, acc_shift));               
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_1, acc_shift));                               
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_2, acc_shift));                               
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_3, acc_shift));                               
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));            
            }
            for(; m_itr < rows; m_itr++)
            {
                ae_int64 acc_0_0 = 0;
                ae_int16x4 *p_vec_batch_0  = (ae_int16x4 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int8 *p_mat1_0 = (ae_int8 *) &p_mat1[(m_itr+0)*row_stride1];

                _xa_nn_dot_product_1_row_1_vec_aligned(&acc_0_0, p_mat1_0, p_vec_batch_0, cols1);

                if(p_bias != NULL)
                {
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_0_0 = AE_ADD64S(acc_0_0, sat_bias);
                }
                ae_int32 out_val;
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_0, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
            }
        }
    }
    //Unaligned part
    else
    {
        for (vec_itr = 0; vec_itr < (vec_count & ~(3)); vec_itr += 4)
        {           
            ae_int16 bias = (0);
            ae_int64 sat_bias = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
            ae_int16 *pbias = (ae_int16 *) p_bias;

            WORD16* p_dst_0 = (WORD16*)p_out + (vec_itr + 0) * out_offset;
            WORD16* p_dst_1 = (WORD16*)p_out + (vec_itr + 1) * out_offset;
            WORD16* p_dst_2 = (WORD16*)p_out + (vec_itr + 2) * out_offset;
            WORD16* p_dst_3 = (WORD16*)p_out + (vec_itr + 3) * out_offset;

            for(m_itr = 0; m_itr < (rows & ~(3)); m_itr += 4)
            {
                ae_int64 acc_0 = 0, acc_1 = 0, acc_2 = 0, acc_3 = 0, acc_4 = 0, acc_5 = 0, acc_6 = 0, acc_7 = 0, acc_8 = 0, acc_9 = 0, acc_10 = 0, acc_11 = 0, acc_12 = 0, acc_13 = 0, acc_14 = 0, acc_15 = 0;
                ae_int16x8 *p_vec_batch_0  = (ae_int16x8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int16x8 *p_vec_batch_1  = (ae_int16x8 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                ae_int16x8 *p_vec_batch_2  = (ae_int16x8 *)(p_vec1 + (vec_itr + 2)*vec_offset);
                ae_int16x8 *p_vec_batch_3  = (ae_int16x8 *)(p_vec1 + (vec_itr + 3)*vec_offset);
                ae_int8 *p_mat1_0_0 = (ae_int8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_int8 *p_mat1_0_1 = (ae_int8 *) &p_mat1[(m_itr+1)*row_stride1];
                ae_int8 *p_mat1_1_0 = (ae_int8 *) &p_mat1[(m_itr+2)*row_stride1];
                ae_int8 *p_mat1_1_1 = (ae_int8 *) &p_mat1[(m_itr+3)*row_stride1];

                _xa_nn_dot_product_4_rows_4_vecs_unaligned
                    (&acc_0, &acc_1, &acc_2, &acc_3, &acc_4, &acc_5, &acc_6, &acc_7, &acc_8, &acc_9, &acc_10, &acc_11, &acc_12, &acc_13, &acc_14
                     ,&acc_15, p_mat1_0_0, p_mat1_0_1, p_mat1_1_0, p_mat1_1_1, p_vec_batch_0, p_vec_batch_1 , p_vec_batch_2, p_vec_batch_3, cols1);

                if(p_bias!=NULL)
                {
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_0 = AE_ADD64S(acc_0, sat_bias);
                    acc_4 = AE_ADD64S(acc_4, sat_bias);
                    acc_8 = AE_ADD64S(acc_8, sat_bias);
                    acc_12 = AE_ADD64S(acc_12, sat_bias);
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_1 = AE_ADD64S(acc_1, sat_bias);
                    acc_5 = AE_ADD64S(acc_5, sat_bias);
                    acc_9 = AE_ADD64S(acc_9, sat_bias);
                    acc_13 = AE_ADD64S(acc_13, sat_bias);                  
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_2 = AE_ADD64S(acc_2, sat_bias);
                    acc_6 = AE_ADD64S(acc_6, sat_bias); 
                    acc_10 = AE_ADD64S(acc_10, sat_bias);
                    acc_14 = AE_ADD64S(acc_14, sat_bias); 
                    ae_int16_loadip(bias, pbias, 2);                        
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_3 = AE_ADD64S(acc_3, sat_bias);
                    acc_7 = AE_ADD64S(acc_7, sat_bias);     
                    acc_11 = AE_ADD64S(acc_11, sat_bias);
                    acc_15 = AE_ADD64S(acc_15, sat_bias);      
                }

                ae_int32x2 out_val;
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_0, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_1, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_2, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_3, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_4, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_1, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_5, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_1, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_6, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_1, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_7, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_1, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_8, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_2, out_stride*(sizeof(WORD16)));      
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_9, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_2, out_stride*(sizeof(WORD16)));  
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_10, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_2, out_stride*(sizeof(WORD16)));  
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_11, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_2, out_stride*(sizeof(WORD16)));  
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_12, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_3, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_13, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_3, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_14, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_3, out_stride*(sizeof(WORD16))); 
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_15, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_3, out_stride*(sizeof(WORD16)));                                                                                                                                                                                                                         
            }
            //Remaining row
            for(; m_itr < rows; m_itr++)
            {
                ae_int64 acc_0_0 = 0, acc_0_1 = 0, acc_1_0 = 0, acc_1_1 = 0;

                ae_int16x8 *p_vec_batch_0  = (ae_int16x8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int16x8 *p_vec_batch_1  = (ae_int16x8 *)(p_vec1 + (vec_itr + 1)*vec_offset);
                ae_int16x8 *p_vec_batch_2  = (ae_int16x8 *)(p_vec1 + (vec_itr + 2)*vec_offset);
                ae_int16x8 *p_vec_batch_3  = (ae_int16x8 *)(p_vec1 + (vec_itr + 3)*vec_offset);
                ae_int8 *p_mat1_0 = (ae_int8 *) &p_mat1[(m_itr+0)*row_stride1];

                _xa_nn_dot_product_1_row_4_vecs_unaligned
                    (&acc_0_0, &acc_0_1, &acc_1_0, &acc_1_1, p_mat1_0, p_vec_batch_0, p_vec_batch_1 , p_vec_batch_2, p_vec_batch_3, cols1);

                if(p_bias!=NULL)
                {
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_0_0 = AE_ADD64S(acc_0_0, sat_bias);
                    acc_0_1 = AE_ADD64S(acc_0_1, sat_bias);
                    acc_1_0 = AE_ADD64S(acc_1_0, sat_bias);
                    acc_1_1 = AE_ADD64S(acc_1_1, sat_bias);                  
                }
                ae_f32x2 out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_0, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_1, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_1, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_1_0, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_2, out_stride*(sizeof(WORD16)));     
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_1_1, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_3, out_stride*(sizeof(WORD16)));                        
            }
        }
        /* Tail loop for vec unroll */
        for(; vec_itr < vec_count; vec_itr++)
        {
            ae_int16 bias = (0);
            ae_int64 sat_bias = AE_MOVINT64_FROMINT32X2(AE_MOVDA32(0));
            ae_int16 *pbias = (ae_int16 *) p_bias;
            WORD16* p_dst_0 = (WORD16*)p_out + (vec_itr + 0) * out_offset;
            for(m_itr = 0; m_itr < (rows & ~(3)); m_itr += 4)
            {
                ae_int64 acc_0 = 0, acc_1 = 0, acc_2 = 0, acc_3 = 0;

                ae_int16x8 *p_vec_batch_0  = (ae_int16x8 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int8 *p_mat0_0 = (ae_int8 *) &p_mat1[(m_itr+0)*row_stride1];
                ae_int8 *p_mat0_1 = (ae_int8 *) &p_mat1[(m_itr+1)*row_stride1];
                ae_int8 *p_mat1_0 = (ae_int8 *) &p_mat1[(m_itr+2)*row_stride1];
                ae_int8 *p_mat1_1 = (ae_int8 *) &p_mat1[(m_itr+3)*row_stride1];

                _xa_nn_dot_product_4_rows_1_vec_unaligned
                    (&acc_0, &acc_1, &acc_2, &acc_3, p_mat0_0, p_mat0_1, p_mat1_0 , p_mat1_1, p_vec_batch_0, cols1);

                if(p_bias != NULL)
                {
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_0 = AE_ADD64S(acc_0, sat_bias);
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_1 = AE_ADD64S(acc_1, sat_bias);
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_2 = AE_ADD64S(acc_2, sat_bias);
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_3 = AE_ADD64S(acc_3, sat_bias);
                }
                ae_f32x2 out_val;
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_0, acc_shift));               
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_1, acc_shift));                               
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_2, acc_shift));                               
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_3, acc_shift));                               
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));            
            }
            for(; m_itr < rows; m_itr++)
            {
                ae_int64 acc_0_0 = 0;

                ae_int16x4 *p_vec_batch_0  = (ae_int16x4 *)(p_vec1 + (vec_itr + 0)*vec_offset);
                ae_int8 *p_mat1_0 = (ae_int8 *) &p_mat1[(m_itr+0)*row_stride1];

                _xa_nn_dot_product_1_row_1_vec_unaligned(&acc_0_0, p_mat1_0, p_vec_batch_0, cols1);

                if(p_bias != NULL)
                {
                    ae_int16_loadip(bias, pbias, 2);
                    sat_bias = AE_SLAA64S(((ae_int64) bias), bias_shift);
                    acc_0_0 = AE_ADD64S(acc_0_0, sat_bias);
                }
                ae_int32 out_val;
                out_val = AE_ROUND32F64SSYM(AE_SLAA64S(acc_0_0, acc_shift));
                AE_S16_0_XP(AE_SAT16X4(out_val,out_val), (ae_int16*)p_dst_0, out_stride*(sizeof(WORD16)));
            }
        }
    }
    return 0;
}
