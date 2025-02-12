/*******************************************************************************
* Copyright (c) 2018-2025 Cadence Design Systems, Inc.
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

#define ALIGNED_ADDR( addr, align ) \
  (void*)( ( (UWORD32)(addr) + ( (align) - 1 ) ) & ~( (align) - 1 ) )

#define PADDED_SIZE( size, align ) \
  ( ( (size_t)(size) + (align) - 1 ) & ~( (align) - 1 ) )  

#ifdef AE_MULAZB8Q8X8
  #define MAT_VEC_MAC(a0, a1, c0, c1, c2, c3, v0, cz, vz)  AE_MULAZB8Q8X8(a0, a1, c0, c1, c2, c3, v0)
#else
  #define MAT_VEC_MAC(a0, a1, c0, c1, c2, c3, v0, cz, vz) \
  { \
    ae_int16x4 va0, va1; \
    ae_int8x8 d_cz = AE_MOVDA8(cz); \
    AE_SUBW8(va0, va1, v0, AE_MOVDA8(vz)); \
    AE_MULA8Q8X16(a0, a1, c0, c1, c2, c3, va0, va1); \
    AE_MULA8Q8X16(a0, a1, d_cz, d_cz, d_cz, d_cz, AE_NEG16S(va0), AE_NEG16S(va1)); \
  }
#endif

const long long data_sel_pattern[16] = {
 0xffeeddccL, 0xbbaa9988L,
 0xff000000L, 0x00000000L,
 0xffee0000L, 0x00000000L,
 0xffeedd00L, 0x00000000L,
 0xffeeddccL, 0x00000000L,
 0xffeeddccL, 0xbb000000L,
 0xffeeddccL, 0xbbaa0000L,
 0xffeeddccL, 0xbbaa9900L
};

const long long align_load_sel_pattern[16]={
  0xffeeddccL, 0xbbaa9988L,
  0xeeddccbbL, 0xaa998877L,
  0xddccbbaaL, 0x99887766L,
  0xccbbaa99L, 0x88776655L,
  0xbbaa9988L, 0x77665544L,
  0xaa998877L, 0x66554433L,
  0x99887766L, 0x55443322L,
  0x88776655L, 0x44332211L,
};

WORD32 xa_nn_batch_matmul_getsize(
    const WORD32 *const p_mat1_shape,
    const WORD32 *const p_mat2_shape,
    WORD32 mat1_transpose,
    WORD32 mat2_transpose,
    WORD32 mat1_precision,
    WORD32 mat2_precision)
{
   /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((mat1_transpose != 0 && mat1_transpose != 1), -1);
  XA_NNLIB_ARG_CHK_COND((mat2_transpose != 0 && mat2_transpose != 1), -1);

  WORD32 itr;
  for(itr = 0; itr < 5; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_mat1_shape[itr] <= 0 || p_mat2_shape[itr] <= 0), -1);
  }
  WORD32 size = 0;
  WORD32 mat1_elm_size = 1, mat2_elm_size = 1;
  if(mat1_transpose == 1)
  {
    WORD32 mat1_size;
    switch(mat1_precision)
    {
      case 8:
      case -4:
      case -5:
        mat1_elm_size = sizeof(WORD8);
        break;
      case -8:
      case 16:
        mat1_elm_size = sizeof(WORD16);
        break;
      default:
        return -1;
        break;
    }
    mat1_size = p_mat1_shape[0] * p_mat1_shape[1] * p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
    size = mat1_size * mat1_elm_size;
  }
  if(mat2_transpose == 1)
  {
    WORD32 mat2_size;
    switch(mat2_precision)
    {
      case 8:
      case -4:
      case -5:
        mat2_elm_size = sizeof(WORD8);
        break;
      case -8:
      case 16:
        mat2_elm_size = sizeof(WORD16);
        break;
      default:
        return -1;
        break;
    }
    mat2_size = p_mat2_shape[0] * p_mat2_shape[1] * p_mat2_shape[2] * p_mat2_shape[3] * p_mat2_shape[4];
    size += mat2_size * mat2_elm_size;
  }
  
  if((mat2_transpose == 1) && (mat1_transpose == 1) && (mat2_precision==-4)){
    // need for mat2 transpose only.
    WORD32 mat2_size;
    mat2_elm_size = sizeof(WORD8);
    mat2_size = PADDED_SIZE(p_mat2_shape[4],8) * PADDED_SIZE(p_mat2_shape[3],16) + PADDED_SIZE(p_mat2_shape[3],16);
    size = mat2_size * mat2_elm_size;
  }
  return size;
}

static WORD32 xa_nn_asym8sxasym8s_asym8s_mat_transpose(WORD8 * __restrict__ p_out,
                                                        const WORD8 * __restrict__ p_mat,
                                                        WORD32 mat_cols,
                                                        WORD32 mat_rows,
                                                        WORD32 mat_zero_bias,
                                                        WORD32 out_row_off){

    ae_int8x8 transpose_2rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfb73ea62L, 0xd951c840L));
    ae_int8x8 transpose_4rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfbea7362L, 0xd9c85140L));
    ae_int8x8 transpose_8rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfbead9c8L, 0x73625140L));

    ae_int8x8 d0, d1, d2, d3, d4, d5, d6, d7;
    ae_int8x8 d0_1, d1_1, d2_1, d3_1, d4_1, d5_1, d6_1, d7_1;

    ae_int8 *out_ptr_ref = (ae_int8 *)p_out;

    int rows_pad_size = PADDED_SIZE(mat_rows,16);
    WORD8 *p_mat_zbias = p_out + mat_rows*out_row_off;
    for(int i=0;i<rows_pad_size;i++)
    {
        p_mat_zbias[i] = -mat_zero_bias;
    }

    for (int m1_cols = 0; m1_cols < mat_cols; m1_cols += 8){
        int mat_cols_count = (mat_cols - m1_cols) >=8 ? 8: (mat_cols - m1_cols) ;

        ae_int8 *p_mat_r4 = (mat_cols_count > 4)? (ae_int8 *)&p_mat[(m1_cols + 4)*mat_rows] : (ae_int8 *)p_mat_zbias;
        ae_int8 *p_mat_r5 = (mat_cols_count > 5)? (ae_int8 *)&p_mat[(m1_cols + 5)*mat_rows] : (ae_int8 *)p_mat_zbias;
        ae_int8 *p_mat_r6 = (mat_cols_count > 6)? (ae_int8 *)&p_mat[(m1_cols + 6)*mat_rows] : (ae_int8 *)p_mat_zbias;
        ae_int8 *p_mat_r7 = (mat_cols_count > 7)? (ae_int8 *)&p_mat[(m1_cols + 7)*mat_rows] : (ae_int8 *)p_mat_zbias;

        ae_int8 *p_mat_r0 = (ae_int8 *)&p_mat[m1_cols*mat_rows];
        ae_int8 *p_mat_r1 = (mat_cols_count > 1)? (ae_int8 *)&p_mat[(m1_cols +1)* mat_rows] : (ae_int8 *)p_mat_zbias;
        ae_int8 *p_mat_r2 = (mat_cols_count > 2)? (ae_int8 *)&p_mat[(m1_cols + 2)*mat_rows] : (ae_int8 *)p_mat_zbias;
        ae_int8 *p_mat_r3 = (mat_cols_count > 3)? (ae_int8 *)&p_mat[(m1_cols + 3)*mat_rows] : (ae_int8 *)p_mat_zbias;

        ae_int8x8 *out_ptr = (ae_int8x8 *)&out_ptr_ref[m1_cols];
        int m_itr = 0;
        for (; m_itr < (mat_rows & ~(0xF)); m_itr += 16){
          ae_valignx2 valign_mat_p0 = AE_LA128_PP((ae_int8x16 *)p_mat_r0);
          ae_valignx2 valign_mat_p1 = AE_LA128_PP((ae_int8x16 *)p_mat_r1);

          AE_LA8X8X2_IP(d0, d0_1, valign_mat_p0, (ae_int8x16 *)p_mat_r0);
          AE_LA8X8X2_IP(d1, d1_1, valign_mat_p1, (ae_int8x16 *)p_mat_r1);

          valign_mat_p0 = AE_LA128_PP((ae_int8x16 *)p_mat_r2);
          valign_mat_p1 = AE_LA128_PP((ae_int8x16 *)p_mat_r3);
          AE_LA8X8X2_IP(d2, d2_1, valign_mat_p0, (ae_int8x16 *)p_mat_r2);
          AE_LA8X8X2_IP(d3, d3_1, valign_mat_p1, (ae_int8x16 *)p_mat_r3);

          valign_mat_p0 = AE_LA128_PP((ae_int8x16 *)p_mat_r4);
          valign_mat_p1 = AE_LA128_PP((ae_int8x16 *)p_mat_r5);

          AE_LA8X8X2_IP(d4, d4_1, valign_mat_p0, (ae_int8x16 *)p_mat_r4);
          AE_LA8X8X2_IP(d5, d5_1, valign_mat_p1, (ae_int8x16 *)p_mat_r5);

          valign_mat_p0 = AE_LA128_PP((ae_int8x16 *)p_mat_r6);
          valign_mat_p1 = AE_LA128_PP((ae_int8x16 *)p_mat_r7);
          AE_LA8X8X2_IP(d6, d6_1, valign_mat_p0, (ae_int8x16 *)p_mat_r6);
          AE_LA8X8X2_IP(d7, d7_1, valign_mat_p1, (ae_int8x16 *)p_mat_r7);

          AE_DSEL8X8(d0, d1, d0, d1, transpose_2rows_sel);
          AE_DSEL8X8(d2, d3, d2, d3, transpose_2rows_sel);
          AE_DSEL8X8(d4, d5, d4, d5, transpose_2rows_sel);
          AE_DSEL8X8(d6, d7, d6, d7, transpose_2rows_sel);

          AE_DSEL8X8(d0_1, d1_1, d0_1, d1_1, transpose_2rows_sel);
          AE_DSEL8X8(d2_1, d3_1, d2_1, d3_1, transpose_2rows_sel);
          AE_DSEL8X8(d4_1, d5_1, d4_1, d5_1, transpose_2rows_sel);
          AE_DSEL8X8(d6_1, d7_1, d6_1, d7_1, transpose_2rows_sel);
 
          AE_DSEL8X8(d0, d2, d0, d2, transpose_4rows_sel);
          AE_DSEL8X8(d4, d6, d4, d6, transpose_4rows_sel);
          AE_DSEL8X8(d0, d4, d0, d4, transpose_8rows_sel); // r0, r1
          AE_DSEL8X8(d2, d6, d2, d6, transpose_8rows_sel); // r2, r3

          AE_DSEL8X8(d0_1, d2_1, d0_1, d2_1, transpose_4rows_sel);
          AE_DSEL8X8(d4_1, d6_1, d4_1, d6_1, transpose_4rows_sel);
          AE_DSEL8X8(d0_1, d4_1, d0_1, d4_1, transpose_8rows_sel); // r0, r1
          AE_DSEL8X8(d2_1, d6_1, d2_1, d6_1, transpose_8rows_sel); // r2, r3

          AE_DSEL8X8(d1, d3, d1, d3, transpose_4rows_sel);
          AE_DSEL8X8(d5, d7, d5, d7, transpose_4rows_sel);
          AE_DSEL8X8(d1, d5, d1, d5, transpose_8rows_sel); // r4, r5
          AE_DSEL8X8(d3, d7, d3, d7, transpose_8rows_sel); // r6, r7

          AE_DSEL8X8(d1_1, d3_1, d1_1, d3_1, transpose_4rows_sel);
          AE_DSEL8X8(d5_1, d7_1, d5_1, d7_1, transpose_4rows_sel);
          AE_DSEL8X8(d1_1, d5_1, d1_1, d5_1, transpose_8rows_sel); // r4, r5
          AE_DSEL8X8(d3_1, d7_1, d3_1, d7_1, transpose_8rows_sel); // r6, r7

          AE_S8X8_XP(d0, out_ptr , out_row_off); //r0
          AE_S8X8_XP(d4, out_ptr , out_row_off); //r1
          AE_S8X8_XP(d2, out_ptr , out_row_off); //r2
          AE_S8X8_XP(d6, out_ptr , out_row_off); //r3
          AE_S8X8_XP(d1, out_ptr , out_row_off); //r4
          AE_S8X8_XP(d5, out_ptr , out_row_off); //r5
          AE_S8X8_XP(d3, out_ptr , out_row_off); //r6
          AE_S8X8_XP(d7, out_ptr , out_row_off); //r7

          AE_S8X8_XP(d0_1, out_ptr , out_row_off); //r0
          AE_S8X8_XP(d4_1, out_ptr , out_row_off); //r1
          AE_S8X8_XP(d2_1, out_ptr , out_row_off); //r2
          AE_S8X8_XP(d6_1, out_ptr , out_row_off); //r3
          AE_S8X8_XP(d1_1, out_ptr , out_row_off); //r4
          AE_S8X8_XP(d5_1, out_ptr , out_row_off); //r5
          AE_S8X8_XP(d3_1, out_ptr , out_row_off); //r6
          AE_S8X8_XP(d7_1, out_ptr , out_row_off); //r7
        }
        if((mat_rows & 0xF) > 7)
        {
          ae_valign valign_mat_p0 = AE_LA64_PP((ae_int8x8 *)p_mat_r0);
          ae_valign valign_mat_p1 = AE_LA64_PP((ae_int8x8 *)p_mat_r1);

          AE_LA8X8_IP(d0, valign_mat_p0, (ae_int8x8 *)p_mat_r0);
          AE_LA8X8_IP(d1, valign_mat_p1, (ae_int8x8 *)p_mat_r1);

          valign_mat_p0 = AE_LA64_PP((ae_int8x8 *)p_mat_r2);
          valign_mat_p1 = AE_LA64_PP((ae_int8x8 *)p_mat_r3);
          AE_LA8X8_IP(d2, valign_mat_p0, (ae_int8x8 *)p_mat_r2);
          AE_LA8X8_IP(d3, valign_mat_p1, (ae_int8x8 *)p_mat_r3);

          valign_mat_p0 = AE_LA64_PP((ae_int8x8 *)p_mat_r4);
          valign_mat_p1 = AE_LA64_PP((ae_int8x8 *)p_mat_r5);

          AE_LA8X8_IP(d4, valign_mat_p0, (ae_int8x8 *)p_mat_r4);
          AE_LA8X8_IP(d5, valign_mat_p1, (ae_int8x8 *)p_mat_r5);

          valign_mat_p0 = AE_LA64_PP((ae_int8x8 *)p_mat_r6);
          valign_mat_p1 = AE_LA64_PP((ae_int8x8 *)p_mat_r7);
          AE_LA8X8_IP(d6, valign_mat_p0, (ae_int8x8 *)p_mat_r6);
          AE_LA8X8_IP(d7, valign_mat_p1, (ae_int8x8 *)p_mat_r7);

          AE_DSEL8X8(d0, d1, d0, d1, transpose_2rows_sel);
          AE_DSEL8X8(d2, d3, d2, d3, transpose_2rows_sel);
          AE_DSEL8X8(d4, d5, d4, d5, transpose_2rows_sel);
          AE_DSEL8X8(d6, d7, d6, d7, transpose_2rows_sel);

          AE_DSEL8X8(d0, d2, d0, d2, transpose_4rows_sel);
          AE_DSEL8X8(d4, d6, d4, d6, transpose_4rows_sel);
          AE_DSEL8X8(d0, d4, d0, d4, transpose_8rows_sel); // r0, r1
          AE_DSEL8X8(d2, d6, d2, d6, transpose_8rows_sel); // r2, r3

          AE_DSEL8X8(d1, d3, d1, d3, transpose_4rows_sel);
          AE_DSEL8X8(d5, d7, d5, d7, transpose_4rows_sel);
          AE_DSEL8X8(d1, d5, d1, d5, transpose_8rows_sel); // r4, r5
          AE_DSEL8X8(d3, d7, d3, d7, transpose_8rows_sel); // r6, r7

          AE_S8X8_XP(d0, out_ptr , out_row_off); //r0
          AE_S8X8_XP(d4, out_ptr , out_row_off); //r1
          AE_S8X8_XP(d2, out_ptr , out_row_off); //r2
          AE_S8X8_XP(d6, out_ptr , out_row_off); //r3
          AE_S8X8_XP(d1, out_ptr , out_row_off); //r4
          AE_S8X8_XP(d5, out_ptr , out_row_off); //r5
          AE_S8X8_XP(d3, out_ptr , out_row_off); //r6
          AE_S8X8_XP(d7, out_ptr , out_row_off); //r7
        }
        int rem = mat_rows & 7;
        if (rem)
        {
          ae_valign valign_mat_p0 = AE_LA64_PP((ae_int8x8 *)p_mat_r0);
          ae_valign valign_mat_p1 = AE_LA64_PP((ae_int8x8 *)p_mat_r1);

          AE_LA8X8_IP(d0, valign_mat_p0, (ae_int8x8 *)p_mat_r0);
          AE_LA8X8_IP(d1, valign_mat_p1, (ae_int8x8 *)p_mat_r1);

          valign_mat_p0 = AE_LA64_PP((ae_int8x8 *)p_mat_r2);
          valign_mat_p1 = AE_LA64_PP((ae_int8x8 *)p_mat_r3);
          AE_LA8X8_IP(d2, valign_mat_p0, (ae_int8x8 *)p_mat_r2);
          AE_LA8X8_IP(d3, valign_mat_p1, (ae_int8x8 *)p_mat_r3);

          valign_mat_p0 = AE_LA64_PP((ae_int8x8 *)p_mat_r4);
          valign_mat_p1 = AE_LA64_PP((ae_int8x8 *)p_mat_r5);

          AE_LA8X8_IP(d4, valign_mat_p0, (ae_int8x8 *)p_mat_r4);
          AE_LA8X8_IP(d5, valign_mat_p1, (ae_int8x8 *)p_mat_r5);

          valign_mat_p0 = AE_LA64_PP((ae_int8x8 *)p_mat_r6);
          valign_mat_p1 = AE_LA64_PP((ae_int8x8 *)p_mat_r7);
          AE_LA8X8_IP(d6, valign_mat_p0, (ae_int8x8 *)p_mat_r6);
          AE_LA8X8_IP(d7, valign_mat_p1, (ae_int8x8 *)p_mat_r7);

          AE_DSEL8X8(d0, d1, d0, d1, transpose_2rows_sel);
          AE_DSEL8X8(d2, d3, d2, d3, transpose_2rows_sel);
          AE_DSEL8X8(d4, d5, d4, d5, transpose_2rows_sel);
          AE_DSEL8X8(d6, d7, d6, d7, transpose_2rows_sel);
 
          AE_DSEL8X8(d0, d2, d0, d2, transpose_4rows_sel);
          AE_DSEL8X8(d4, d6, d4, d6, transpose_4rows_sel);
          AE_DSEL8X8(d0, d4, d0, d4, transpose_8rows_sel); // r0, r1
          AE_DSEL8X8(d2, d6, d2, d6, transpose_8rows_sel); // r2, r3

          AE_DSEL8X8(d1, d3, d1, d3, transpose_4rows_sel);
          AE_DSEL8X8(d5, d7, d5, d7, transpose_4rows_sel);
          AE_DSEL8X8(d1, d5, d1, d5, transpose_8rows_sel); // r4, r5
          AE_DSEL8X8(d3, d7, d3, d7, transpose_8rows_sel); // r6, r7

          d7 = AE_MOVDA8((WORD8)(-mat_zero_bias)); //r7
          d4 = rem > 1 ? d4 : d7; // r1
          d2 = rem > 2 ? d2 : d7; // r2
          d6 = rem > 3 ? d6 : d7; // r3
          d1 = rem > 4 ? d1 : d7; // r4
          d5 = rem > 5 ? d5 : d7; // r5
          d3 = rem > 6 ? d3 : d7; // r6

          AE_S8X8_XP(d0, out_ptr , out_row_off); //r0
          AE_S8X8_XP(d4, out_ptr , out_row_off); //r1
          AE_S8X8_XP(d2, out_ptr , out_row_off); //r2
          AE_S8X8_XP(d6, out_ptr , out_row_off); //r3
          AE_S8X8_XP(d1, out_ptr , out_row_off); //r4
          AE_S8X8_XP(d5, out_ptr , out_row_off); //r5
          AE_S8X8_XP(d3, out_ptr , out_row_off); //r6
          AE_S8X8_XP(d7, out_ptr , out_row_off); //r7
        }
    }
    return 0;
}

static WORD32 xa_nn_batch_matmul_a8sxa8s_a8s_mat2_aligned_padded_zbias(WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_mat2,
    WORD32 mat1_cols,
    WORD32 mat1_rows,
    WORD32 mat2_cols,
    WORD32 mat2_row_offset,
    WORD32 mat1_zero_bias,
    WORD32 mat2_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias){

    int left_shift, right_shift;

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  right_shift = out_shift;
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift < 0 ? 0 : out_shift;
  right_shift = out_shift > 0 ? 0 : -out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

    ae_int64 biasvc = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-mat2_zero_bias, -mat1_zero_bias));
    AE_MOVZBVCDR(biasvc);
    ae_int8x8 d0,d1,d2,d3,d4,d5,d6,d7;
    ae_int8x8 mat2_col0, mat2_col1, mat2_col2, mat2_col3;
    ae_int8x8 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11;
    ae_int8x8 transpose_2rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfb73ea62L, 0xd951c840L));
    ae_int8x8 transpose_4rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfbea7362L, 0xd9c85140L));
    ae_int8x8 transpose_8rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfbead9c8L, 0x73625140L));

    ae_valignx2 b0, b1;
    b0 = AE_ZALIGN128();
    b1 = AE_ZALIGN128();

    int mat1_row_offset = (mat1_rows << 3) - 8;
    for (int vec_itr=0; vec_itr < mat2_cols; vec_itr += 4){
      ae_int8x8 *mat2_ptr = (ae_int8x8 *)&p_mat2[vec_itr * mat2_row_offset];
      ae_int32x2 r01c0, r23c0, r45c0, r67c0, r01c1, r23c1, r45c1, r67c1, r01c2, r23c2, r45c2, r67c2, r01c3, r23c3, r45c3, r67c3;

      int mat2_cols_rem = 4;
      ae_int8x8 *mat2_ptr0_c = mat2_ptr;
      ae_int8x8 *mat2_ptr1_c = (ae_int8x8 *)((ae_int8 *)mat2_ptr + mat2_row_offset);
      ae_int8x8 *mat2_ptr2_c = (ae_int8x8 *)((ae_int8 *)mat2_ptr + 2*mat2_row_offset);;
      ae_int8x8 *mat2_ptr3_c = (ae_int8x8 *)((ae_int8 *)mat2_ptr + 3*mat2_row_offset);;
      if ((mat2_cols - vec_itr) < 4)
      {
        mat2_cols_rem = mat2_cols & (3);
        mat2_ptr1_c = (ae_int8x8 *)((ae_int8 *)mat2_ptr0_c + mat2_row_offset* (mat2_cols_rem > 1)); // when remaining cols are 2 or 3
        mat2_ptr2_c = (ae_int8x8 *)((ae_int8 *)mat2_ptr0_c + 2*mat2_row_offset* (mat2_cols_rem > 2)); // when remaining cols are 3
        mat2_ptr3_c = mat2_ptr0_c; // duplicate operation
      }

      ae_int8x8 *p_mat1_r0_ref = (ae_int8x8 *)&p_mat1[0];
      ae_int8x8 *p_mat1_r1_ref = (ae_int8x8 *)&p_mat1[mat1_rows];
      ae_int8x8 *p_mat1_r2_ref = (ae_int8x8 *)&p_mat1[2*mat1_rows];
      ae_int8x8 *p_mat1_r3_ref = (ae_int8x8 *)&p_mat1[3*mat1_rows];
      ae_int8x8 *p_mat1_r4_ref = (ae_int8x8 *)&p_mat1[4*mat1_rows];
      ae_int8x8 *p_mat1_r5_ref = (ae_int8x8 *)&p_mat1[5*mat1_rows];
      ae_int8x8 *p_mat1_r6_ref = (ae_int8x8 *)&p_mat1[6*mat1_rows];
      ae_int8x8 *p_mat1_r7_ref = (ae_int8x8 *)&p_mat1[7*mat1_rows];
      
      for (int m1_row = 0; m1_row < mat1_rows; m1_row += 8){
        int mat1_rows_count = (mat1_rows - m1_row) >=8 ? 8: (mat1_rows - m1_row) ;
        r01c0 = r23c0 = r45c0 = r67c0 = r01c1 = r23c1 = r45c1 = r67c1 = r01c2 = r23c2 = r45c2 = r67c2 = r01c3 = r23c3 = r45c3 = r67c3 = 0;

        ae_int8x8 *mat2_ptr0 = mat2_ptr0_c;
        ae_int8x8 *mat2_ptr1 = mat2_ptr1_c;
        ae_int8x8 *mat2_ptr2 = mat2_ptr2_c;
        ae_int8x8 *mat2_ptr3 = mat2_ptr3_c;

        ae_int8x8 *p_mat1_r0 = (ae_int8x8 *)((ae_int8 *)p_mat1_r0_ref + m1_row);
        ae_int8x8 *p_mat1_r1 = (ae_int8x8 *)((ae_int8 *)p_mat1_r1_ref + m1_row);
        ae_int8x8 *p_mat1_r2 = (ae_int8x8 *)((ae_int8 *)p_mat1_r2_ref + m1_row);
        ae_int8x8 *p_mat1_r3 = (ae_int8x8 *)((ae_int8 *)p_mat1_r3_ref + m1_row);
        ae_int8x8 *p_mat1_r4 = (ae_int8x8 *)((ae_int8 *)p_mat1_r4_ref + m1_row);
        ae_int8x8 *p_mat1_r5 = (ae_int8x8 *)((ae_int8 *)p_mat1_r5_ref + m1_row);
        ae_int8x8 *p_mat1_r6 = (ae_int8x8 *)((ae_int8 *)p_mat1_r6_ref + m1_row);
        ae_int8x8 *p_mat1_r7 = (ae_int8x8 *)((ae_int8 *)p_mat1_r7_ref + m1_row);
        
        for (int m_itr = 0; m_itr < (mat1_cols & ~(7)); m_itr += 8){
          
          ae_valign valign_mat1_0 = AE_LA64_PP(p_mat1_r0);
          ae_valign valign_mat1_1 = AE_LA64_PP(p_mat1_r1);
          ae_valign valign_mat1_2 = AE_LA64_PP(p_mat1_r2);
          ae_valign valign_mat1_3 = AE_LA64_PP(p_mat1_r3);
          
          AE_LA8X8_IP(d0, valign_mat1_0, p_mat1_r0);
          AE_LA8X8_IP(d1, valign_mat1_1, p_mat1_r1);
          p_mat1_r0 = (ae_int8x8 *)((ae_int8 *)p_mat1_r0 + mat1_row_offset);
          p_mat1_r1 = (ae_int8x8 *)((ae_int8 *)p_mat1_r1 + mat1_row_offset);

          AE_LA8X8_IP(d2, valign_mat1_2, p_mat1_r2);
          AE_LA8X8_IP(d3, valign_mat1_3, p_mat1_r3);
          p_mat1_r2 = (ae_int8x8 *)((ae_int8 *)p_mat1_r2 + mat1_row_offset);
          p_mat1_r3 = (ae_int8x8 *)((ae_int8 *)p_mat1_r3 + mat1_row_offset);


          valign_mat1_0 = AE_LA64_PP(p_mat1_r4);
          valign_mat1_1 = AE_LA64_PP(p_mat1_r5);
          valign_mat1_2 = AE_LA64_PP(p_mat1_r6);
          valign_mat1_3 = AE_LA64_PP(p_mat1_r7);

          AE_LA8X8_IP(d4, valign_mat1_0, p_mat1_r4);
          AE_LA8X8_IP(d5, valign_mat1_1, p_mat1_r5);
          p_mat1_r4 = (ae_int8x8 *)((ae_int8 *)p_mat1_r4 + mat1_row_offset);
          p_mat1_r5 = (ae_int8x8 *)((ae_int8 *)p_mat1_r5 + mat1_row_offset);

          AE_LA8X8_IP(d6, valign_mat1_2, p_mat1_r6);
          AE_LA8X8_IP(d7, valign_mat1_3, p_mat1_r7);
          p_mat1_r6 = (ae_int8x8 *)((ae_int8 *)p_mat1_r6 + mat1_row_offset);
          p_mat1_r7 = (ae_int8x8 *)((ae_int8 *)p_mat1_r7 + mat1_row_offset);

          AE_L8X8_IP(mat2_col0, mat2_ptr0,8);
          AE_L8X8_IP(mat2_col1, mat2_ptr1,8);
          AE_L8X8_IP(mat2_col2, mat2_ptr2,8);
          AE_L8X8_IP(mat2_col3, mat2_ptr3,8);

          AE_DSEL8X8(tmp0, tmp1, d0, d1, transpose_2rows_sel);
          AE_DSEL8X8(tmp2, tmp3, d2, d3, transpose_2rows_sel);
          AE_DSEL8X8(tmp4, tmp5, d4, d5, transpose_2rows_sel);
          AE_DSEL8X8(tmp6, tmp7, d6, d7, transpose_2rows_sel);
        
          AE_DSEL8X8(tmp8, tmp9, tmp0, tmp2, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp4, tmp6, transpose_4rows_sel);
        
          AE_DSEL8X8(d0, d1, tmp8, tmp10, transpose_8rows_sel); // r0, r1
          AE_DSEL8X8(d2, d3, tmp9, tmp11, transpose_8rows_sel); // r2, r3
        
          AE_DSEL8X8(tmp8, tmp9, tmp1, tmp3, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp5, tmp7, transpose_4rows_sel);
        
          AE_DSEL8X8(d4, d5, tmp8, tmp10, transpose_8rows_sel); // r4, r5
          AE_DSEL8X8(d6, d7, tmp9, tmp11, transpose_8rows_sel); // r6, r7

          // mat row0 to row3 x vec col0 to col3 and mat row4 to row7 x vec col0 to col3 
          MAT_VEC_MAC(r01c0, r23c0, d0, d1, d2, d3, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c1, r23c1, d0, d1, d2, d3, mat2_col1, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c2, r23c2, d0, d1, d2, d3, mat2_col2, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c3, r23c3, d0, d1, d2, d3, mat2_col3, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c0, r67c0, d4, d5, d6, d7, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c1, r67c1, d4, d5, d6, d7, mat2_col1, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c2, r67c2, d4, d5, d6, d7, mat2_col2, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c3, r67c3, d4, d5, d6, d7, mat2_col3, -mat1_zero_bias, -mat2_zero_bias);
        }
        int rem = mat1_cols & 7;
        if (rem)
        {
          p_mat1_r1 = rem > 1 ? p_mat1_r1 : p_mat1_r0;
          p_mat1_r2 = rem > 2 ? p_mat1_r2 : p_mat1_r0;
          p_mat1_r3 = rem > 3 ? p_mat1_r3 : p_mat1_r0;
          p_mat1_r4 = rem > 4 ? p_mat1_r4 : p_mat1_r0;
          p_mat1_r5 = rem > 5 ? p_mat1_r5 : p_mat1_r0;
          p_mat1_r6 = rem > 6 ? p_mat1_r6 : p_mat1_r0;

          ae_valign valign_mat1_0 = AE_LA64_PP(p_mat1_r0);
          ae_valign valign_mat1_1 = AE_LA64_PP(p_mat1_r1);
          ae_valign valign_mat1_2 = AE_LA64_PP(p_mat1_r2);
          ae_valign valign_mat1_3 = AE_LA64_PP(p_mat1_r3);

          AE_LA8X8_IP(d0, valign_mat1_0, p_mat1_r0);
          AE_LA8X8_IP(d1, valign_mat1_1, p_mat1_r1);

          AE_LA8X8_IP(d2, valign_mat1_2, p_mat1_r2);
          AE_LA8X8_IP(d3, valign_mat1_3, p_mat1_r3);

          valign_mat1_0 = AE_LA64_PP(p_mat1_r4);
          valign_mat1_1 = AE_LA64_PP(p_mat1_r5);
          valign_mat1_2 = AE_LA64_PP(p_mat1_r6);

          AE_LA8X8_IP(d4, valign_mat1_0, p_mat1_r4);
          AE_LA8X8_IP(d5, valign_mat1_1, p_mat1_r5);

          AE_LA8X8_IP(d6, valign_mat1_2, p_mat1_r6);

          AE_DSEL8X8(tmp0, tmp1, d0, d1, transpose_2rows_sel);
          AE_DSEL8X8(tmp2, tmp3, d2, d3, transpose_2rows_sel);
        
          AE_DSEL8X8(tmp4, tmp5, d4, d5, transpose_2rows_sel);
          AE_DSEL8X8(tmp6, tmp7, d6, d6, transpose_2rows_sel);
        
          AE_DSEL8X8(tmp8, tmp9, tmp0, tmp2, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp4, tmp6, transpose_4rows_sel);
        
          AE_DSEL8X8(d0, d1, tmp8, tmp10, transpose_8rows_sel); // r0, r1
          AE_DSEL8X8(d2, d3, tmp9, tmp11, transpose_8rows_sel); // r2, r3
        
          AE_DSEL8X8(tmp8, tmp9, tmp1, tmp3, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp5, tmp7, transpose_4rows_sel);
        
          AE_DSEL8X8(d4, d5, tmp8, tmp10, transpose_8rows_sel); // r4, r5
          AE_DSEL8X8(d6, d7, tmp9, tmp11, transpose_8rows_sel); // r6, r7

          AE_L8X8_IP(mat2_col0, mat2_ptr0, 0);
          AE_L8X8_IP(mat2_col1, mat2_ptr1, 0);
          AE_L8X8_IP(mat2_col2, mat2_ptr2, 0);
          AE_L8X8_IP(mat2_col3, mat2_ptr3, 0);

          // mat row0 to row3 x vec col0 to col3 and mat row4 to row7 x vec col0 to col3 
          MAT_VEC_MAC(r01c0, r23c0, d0, d1, d2, d3, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c1, r23c1, d0, d1, d2, d3, mat2_col1, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c2, r23c2, d0, d1, d2, d3, mat2_col2, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c3, r23c3, d0, d1, d2, d3, mat2_col3, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c0, r67c0, d4, d5, d6, d7, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c1, r67c1, d4, d5, d6, d7, mat2_col1, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c2, r67c2, d4, d5, d6, d7, mat2_col2, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c3, r67c3, d4, d5, d6, d7, mat2_col3, -mat1_zero_bias, -mat2_zero_bias);
        }
          ae_int16x4 out_0, out_1, out_2, out_3;

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, r01c0, r23c0, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, r45c0, r67c0, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, r01c1, r23c1, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, r45c1, r67c1, out_multiplier, left_shift, right_shift, out_zero_bias);

          AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

          ae_int8x8 temp_vec0, temp_vec1;
          temp_vec0 = AE_SAT8X8X16(out_0, out_1);
          temp_vec1 = AE_SAT8X8X16(out_2, out_3);

          ae_int8x8 *out_mat_ptr = (ae_int8x8 *)&p_out[vec_itr * mat1_rows + m1_row];
          ae_int8x8 *out_mat_ptr1 = (ae_int8x8 *)&p_out[(vec_itr + 1) * mat1_rows + m1_row];

          AE_SAV8X8X2_XP(temp_vec0, temp_vec0, b0, (ae_int8x16 *)out_mat_ptr, mat1_rows_count);
          AE_SAV8X8X2_XP(temp_vec1, temp_vec1, b1, (ae_int8x16 *)out_mat_ptr1, mat1_rows_count * (mat2_cols_rem > 1));

          AE_SA128POS_FP(b0, out_mat_ptr);
          AE_SA128POS_FP(b1, out_mat_ptr1);

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, r01c2, r23c2, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, r45c2, r67c2, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, r01c3, r23c3, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, r45c3, r67c3, out_multiplier, left_shift, right_shift, out_zero_bias);

          AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

          temp_vec0 = AE_SAT8X8X16(out_0, out_1);
          temp_vec1 = AE_SAT8X8X16(out_2, out_3);

          out_mat_ptr = (ae_int8x8 *)&p_out[(vec_itr + 2) * mat1_rows + m1_row];
          out_mat_ptr1 = (ae_int8x8 *)&p_out[(vec_itr + 3) * mat1_rows + m1_row];

          AE_SAV8X8X2_XP(temp_vec0, temp_vec0, b0, (ae_int8x16 *)out_mat_ptr, mat1_rows_count * (mat2_cols_rem > 2));
          AE_SAV8X8X2_XP(temp_vec1, temp_vec1, b1, (ae_int8x16 *)out_mat_ptr1, mat1_rows_count * (mat2_cols_rem > 3));

          AE_SA128POS_FP(b0, out_mat_ptr);
          AE_SA128POS_FP(b1, out_mat_ptr1);
      }
    }
  return 0;
}

static WORD32 xa_nn_batch_matmul_asym8sxasym8s_asym8s_mat1_transpose(WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_mat2,
    WORD32 mat1_cols,
    WORD32 mat1_rows,
    WORD32 mat2_cols,
    WORD32 mat2_col_offset,
    WORD32 mat1_zero_bias,
    WORD32 mat2_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias,
    WORD32 matAB_T_flag){

    int left_shift, right_shift;

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  right_shift = out_shift;
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift < 0 ? 0 : out_shift;
  right_shift = out_shift > 0 ? 0 : -out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

    ae_int64 biasvc = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-mat2_zero_bias, -mat1_zero_bias));
    AE_MOVZBVCDR(biasvc);
    ae_int8x8 d0,d1,d2,d3,d4,d5,d6,d7;
    ae_int8x8 mat2_col0, mat2_col1, mat2_col2, mat2_col3;
    ae_int8x8 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11;
    ae_int8x8 transpose_2rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfb73ea62L, 0xd951c840L));
    ae_int8x8 transpose_4rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfbea7362L, 0xd9c85140L));
    ae_int8x8 transpose_8rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfbead9c8L, 0x73625140L));

    ae_valignx2 b0, b1;
    b0 = AE_ZALIGN128();
    b1 = AE_ZALIGN128();

    for (int vec_itr=0; vec_itr < mat2_cols; vec_itr += 4){
      ae_int8x8 *mat2_ptr = (ae_int8x8 *)&p_mat2[vec_itr * mat2_col_offset];
      ae_int32x2 r01c0, r23c0, r45c0, r67c0, r01c1, r23c1, r45c1, r67c1, r01c2, r23c2, r45c2, r67c2, r01c3, r23c3, r45c3, r67c3;

      int offset0, offset1, offset2, offset3, offset4, offset5, offset6, offset7; 
      offset0 = (((unsigned int)&p_mat1[0]) & 0x7);
      ae_int8x8 sel_r0 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset0], align_load_sel_pattern[2 * offset0 + 1]));

      offset1 = (((unsigned int)&p_mat1[mat1_rows]) & 0x7);
      ae_int8x8 sel_r1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset1], align_load_sel_pattern[2 * offset1 + 1]));

      offset2 = (((unsigned int)&p_mat1[2*mat1_rows]) & 0x7);
      ae_int8x8 sel_r2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset2], align_load_sel_pattern[2 * offset2 + 1]));

      offset3 = (((unsigned int)&p_mat1[3*mat1_rows]) & 0x7);
      ae_int8x8 sel_r3 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset3], align_load_sel_pattern[2 * offset3 + 1]));

      offset4 = (((unsigned int)&p_mat1[4*mat1_rows]) & 0x7);
      ae_int8x8 sel_r4 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset4], align_load_sel_pattern[2 * offset4 + 1]));

      offset5 = (((unsigned int)&p_mat1[5*mat1_rows]) & 0x7);
      ae_int8x8 sel_r5 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset5], align_load_sel_pattern[2 * offset5 + 1]));

      offset6 = (((unsigned int)&p_mat1[6*mat1_rows]) & 0x7);
      ae_int8x8 sel_r6 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset6], align_load_sel_pattern[2 * offset6 + 1]));

      offset7 = (((unsigned int)&p_mat1[7*mat1_rows]) & 0x7);
      ae_int8x8 sel_r7 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset7], align_load_sel_pattern[2 * offset7 + 1]));

      int mat2_cols_rem = 4;
      ae_int8x8 *mat2_ptr0_c = mat2_ptr;
      ae_int8x8 *mat2_ptr1_c = (ae_int8x8 *)((ae_int8 *)mat2_ptr + mat2_col_offset);
      ae_int8x8 *mat2_ptr2_c = (ae_int8x8 *)((ae_int8 *)mat2_ptr + 2*mat2_col_offset);;
      ae_int8x8 *mat2_ptr3_c = (ae_int8x8 *)((ae_int8 *)mat2_ptr + 3*mat2_col_offset);;
      if ((mat2_cols - vec_itr) < 4)
      {
        mat2_cols_rem = mat2_cols & (3);
        mat2_ptr1_c = (ae_int8x8 *)((ae_int8 *)mat2_ptr0_c + mat2_col_offset* (mat2_cols_rem > 1)); // when remaining cols are 2 or 3
        mat2_ptr2_c = (ae_int8x8 *)((ae_int8 *)mat2_ptr0_c + 2*mat2_col_offset* (mat2_cols_rem > 2)); // when remaining cols are 3
        mat2_ptr3_c = mat2_ptr0_c; // duplicate operation
      }

      ae_int8x8 *p_mat1_r0_ref = (ae_int8x8 *)&p_mat1[0 - offset0];
      ae_int8x8 *p_mat1_r1_ref = (ae_int8x8 *)&p_mat1[mat1_rows - offset1];
      ae_int8x8 *p_mat1_r2_ref = (ae_int8x8 *)&p_mat1[2*mat1_rows - offset2];
      ae_int8x8 *p_mat1_r3_ref = (ae_int8x8 *)&p_mat1[3*mat1_rows - offset3];
      ae_int8x8 *p_mat1_r4_ref = (ae_int8x8 *)&p_mat1[4*mat1_rows - offset4];
      ae_int8x8 *p_mat1_r5_ref = (ae_int8x8 *)&p_mat1[5*mat1_rows - offset5];
      ae_int8x8 *p_mat1_r6_ref = (ae_int8x8 *)&p_mat1[6*mat1_rows - offset6];
      ae_int8x8 *p_mat1_r7_ref = (ae_int8x8 *)&p_mat1[7*mat1_rows - offset7];
      
      for (int m1_row = 0; m1_row < mat1_rows; m1_row += 8){
        int mat1_rows_count = (mat1_rows - m1_row) >=8 ? 8: (mat1_rows - m1_row) ;
        r01c0 = r23c0 = r45c0 = r67c0 = r01c1 = r23c1 = r45c1 = r67c1 = r01c2 = r23c2 = r45c2 = r67c2 = r01c3 = r23c3 = r45c3 = r67c3 = 0;

        ae_int8x8 *mat2_ptr0 = mat2_ptr0_c;
        ae_int8x8 *mat2_ptr1 = mat2_ptr1_c;
        ae_int8x8 *mat2_ptr2 = mat2_ptr2_c;
        ae_int8x8 *mat2_ptr3 = mat2_ptr3_c;

        ae_valign valign_mat2_p0 = AE_LA64_PP(mat2_ptr0);
        ae_valign valign_mat2_p1 = AE_LA64_PP(mat2_ptr1);
        ae_valign valign_mat2_p2 = AE_LA64_PP(mat2_ptr2);
        ae_valign valign_mat2_p3 = AE_LA64_PP(mat2_ptr3);

        ae_int8x8 *p_mat1_r0 = (ae_int8x8 *)((ae_int8 *)p_mat1_r0_ref + m1_row);
        ae_int8x8 *p_mat1_r1 = (ae_int8x8 *)((ae_int8 *)p_mat1_r1_ref + m1_row);
        ae_int8x8 *p_mat1_r2 = (ae_int8x8 *)((ae_int8 *)p_mat1_r2_ref + m1_row);
        ae_int8x8 *p_mat1_r3 = (ae_int8x8 *)((ae_int8 *)p_mat1_r3_ref + m1_row);
        ae_int8x8 *p_mat1_r4 = (ae_int8x8 *)((ae_int8 *)p_mat1_r4_ref + m1_row);
        ae_int8x8 *p_mat1_r5 = (ae_int8x8 *)((ae_int8 *)p_mat1_r5_ref + m1_row);
        ae_int8x8 *p_mat1_r6 = (ae_int8x8 *)((ae_int8 *)p_mat1_r6_ref + m1_row);
        ae_int8x8 *p_mat1_r7 = (ae_int8x8 *)((ae_int8 *)p_mat1_r7_ref + m1_row);
        
        for (int m_itr = 0; m_itr < (mat1_cols & ~(7)); m_itr += 8){
          d0 = AE_L8X8_I(p_mat1_r0, 8);
          AE_L8X8_XP(tmp0, p_mat1_r0, 8*mat1_rows);
          d0 = AE_SEL8X8(tmp0, d0, sel_r0);

          d1 = AE_L8X8_I(p_mat1_r1, 8);
          AE_L8X8_XP(tmp1, p_mat1_r1, 8*mat1_rows);
          d1 = AE_SEL8X8(tmp1, d1, sel_r1);

          d2 = AE_L8X8_I(p_mat1_r2, 8);
          AE_L8X8_XP(tmp2, p_mat1_r2, 8*mat1_rows);
          d2 = AE_SEL8X8(tmp2, d2, sel_r2);

          d3 = AE_L8X8_I(p_mat1_r3, 8);
          AE_L8X8_XP(tmp3, p_mat1_r3, 8*mat1_rows);
          d3 = AE_SEL8X8(tmp3, d3, sel_r3);

          d4 = AE_L8X8_I(p_mat1_r4, 8);
          AE_L8X8_XP(tmp4, p_mat1_r4, 8*mat1_rows);
          d4 = AE_SEL8X8(tmp4, d4, sel_r4);

          d5 = AE_L8X8_I(p_mat1_r5, 8);
          AE_L8X8_XP(tmp5, p_mat1_r5, 8*mat1_rows);
          d5 = AE_SEL8X8(tmp5, d5, sel_r5);

          d6 = AE_L8X8_I(p_mat1_r6, 8);
          AE_L8X8_XP(tmp6, p_mat1_r6, 8*mat1_rows);
          d6 = AE_SEL8X8(tmp6, d6, sel_r6);

          d7 = AE_L8X8_I(p_mat1_r7, 8);
          AE_L8X8_XP(tmp7, p_mat1_r7, 8*mat1_rows);
          d7 = AE_SEL8X8(tmp7, d7, sel_r7);

          AE_LA8X8_IP(mat2_col0, valign_mat2_p0, mat2_ptr0);
          AE_LA8X8_IP(mat2_col1, valign_mat2_p1, mat2_ptr1);
          AE_LA8X8_IP(mat2_col2, valign_mat2_p2, mat2_ptr2);
          AE_LA8X8_IP(mat2_col3, valign_mat2_p3, mat2_ptr3);

          AE_DSEL8X8(tmp0, tmp1, d0, d1, transpose_2rows_sel);
          AE_DSEL8X8(tmp2, tmp3, d2, d3, transpose_2rows_sel);
          AE_DSEL8X8(tmp4, tmp5, d4, d5, transpose_2rows_sel);
          AE_DSEL8X8(tmp6, tmp7, d6, d7, transpose_2rows_sel);
        
          AE_DSEL8X8(tmp8, tmp9, tmp0, tmp2, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp4, tmp6, transpose_4rows_sel);
        
          AE_DSEL8X8(d0, d1, tmp8, tmp10, transpose_8rows_sel); // r0, r1
          AE_DSEL8X8(d2, d3, tmp9, tmp11, transpose_8rows_sel); // r2, r3
        
          AE_DSEL8X8(tmp8, tmp9, tmp1, tmp3, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp5, tmp7, transpose_4rows_sel);
        
          AE_DSEL8X8(d4, d5, tmp8, tmp10, transpose_8rows_sel); // r4, r5
          AE_DSEL8X8(d6, d7, tmp9, tmp11, transpose_8rows_sel); // r6, r7

          // mat row0 to row3 x vec col0 to col3 and mat row4 to row7 x vec col0 to col3 
          MAT_VEC_MAC(r01c0, r23c0, d0, d1, d2, d3, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c1, r23c1, d0, d1, d2, d3, mat2_col1, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c2, r23c2, d0, d1, d2, d3, mat2_col2, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c3, r23c3, d0, d1, d2, d3, mat2_col3, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c0, r67c0, d4, d5, d6, d7, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c1, r67c1, d4, d5, d6, d7, mat2_col1, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c2, r67c2, d4, d5, d6, d7, mat2_col2, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c3, r67c3, d4, d5, d6, d7, mat2_col3, -mat1_zero_bias, -mat2_zero_bias);
        }
        int rem = mat1_cols & 7;
        if (rem)
        {
          p_mat1_r1 = rem > 1 ? p_mat1_r1 : p_mat1_r0;
          p_mat1_r2 = rem > 2 ? p_mat1_r2 : p_mat1_r0;
          p_mat1_r3 = rem > 3 ? p_mat1_r3 : p_mat1_r0;
          p_mat1_r4 = rem > 4 ? p_mat1_r4 : p_mat1_r0;
          p_mat1_r5 = rem > 5 ? p_mat1_r5 : p_mat1_r0;
          p_mat1_r6 = rem > 6 ? p_mat1_r6 : p_mat1_r0;

          d0 = AE_L8X8_I(p_mat1_r0, 8);
          tmp0 = AE_L8X8_I(p_mat1_r0, 0);
          d0 = AE_SEL8X8(tmp0, d0, sel_r0);

          d1 = AE_L8X8_I(p_mat1_r1, 8);
          tmp1 = AE_L8X8_I(p_mat1_r1, 0);
          d1 = AE_SEL8X8(tmp1, d1, sel_r1);

          d2 = AE_L8X8_I(p_mat1_r2, 8);
          tmp2 = AE_L8X8_I(p_mat1_r2, 0);
          d2 = AE_SEL8X8(tmp2, d2, sel_r2);

          d3 = AE_L8X8_I(p_mat1_r3, 8);
          tmp3 = AE_L8X8_I(p_mat1_r3, 0);
          d3 = AE_SEL8X8(tmp3, d3, sel_r3);

          d4 = AE_L8X8_I(p_mat1_r4, 8);
          tmp4 = AE_L8X8_I(p_mat1_r4, 0);
          d4 = AE_SEL8X8(tmp4, d4, sel_r4);

          d5 = AE_L8X8_I(p_mat1_r5, 8);
          tmp5 = AE_L8X8_I(p_mat1_r5, 0);
          d5 = AE_SEL8X8(tmp5, d5, sel_r5);

          d6 = AE_L8X8_I(p_mat1_r6, 8);
          tmp6 = AE_L8X8_I(p_mat1_r6, 0);
          d6 = AE_SEL8X8(tmp6, d6, sel_r6);

          ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(data_sel_pattern[2 * rem], data_sel_pattern[2 * rem + 1]));

          AE_DSEL8X8(tmp0, tmp1, d0, d1, transpose_2rows_sel);
          AE_DSEL8X8(tmp2, tmp3, d2, d3, transpose_2rows_sel);
        
          AE_DSEL8X8(tmp4, tmp5, d4, d5, transpose_2rows_sel);
          AE_DSEL8X8(tmp6, tmp7, d6, d6, transpose_2rows_sel);
        
          AE_DSEL8X8(tmp8, tmp9, tmp0, tmp2, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp4, tmp6, transpose_4rows_sel);
        
          AE_DSEL8X8(d0, d1, tmp8, tmp10, transpose_8rows_sel); // r0, r1
          AE_DSEL8X8(d2, d3, tmp9, tmp11, transpose_8rows_sel); // r2, r3
        
          AE_DSEL8X8(tmp8, tmp9, tmp1, tmp3, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp5, tmp7, transpose_4rows_sel);
        
          AE_DSEL8X8(d4, d5, tmp8, tmp10, transpose_8rows_sel); // r4, r5
          AE_DSEL8X8(d6, d7, tmp9, tmp11, transpose_8rows_sel); // r6, r7

          AE_LA8X8_IP(mat2_col0, valign_mat2_p0, mat2_ptr0);
          AE_LA8X8_IP(mat2_col1, valign_mat2_p1, mat2_ptr1);
          AE_LA8X8_IP(mat2_col2, valign_mat2_p2, mat2_ptr2);
          AE_LA8X8_IP(mat2_col3, valign_mat2_p3, mat2_ptr3);

          ae_int8x8 mat2_bias = AE_MOVDA8((WORD8)(-mat2_zero_bias));

          mat2_col0 = AE_SEL8X8(mat2_col0, mat2_bias, sel1);
          mat2_col1 = AE_SEL8X8(mat2_col1, mat2_bias, sel1);
          mat2_col2 = AE_SEL8X8(mat2_col2, mat2_bias, sel1);
          mat2_col3 = AE_SEL8X8(mat2_col3, mat2_bias, sel1);

          // mat row0 to row3 x vec col0 to col3 and mat row4 to row7 x vec col0 to col3 
          MAT_VEC_MAC(r01c0, r23c0, d0, d1, d2, d3, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c1, r23c1, d0, d1, d2, d3, mat2_col1, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c2, r23c2, d0, d1, d2, d3, mat2_col2, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r01c3, r23c3, d0, d1, d2, d3, mat2_col3, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c0, r67c0, d4, d5, d6, d7, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c1, r67c1, d4, d5, d6, d7, mat2_col1, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c2, r67c2, d4, d5, d6, d7, mat2_col2, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c3, r67c3, d4, d5, d6, d7, mat2_col3, -mat1_zero_bias, -mat2_zero_bias);
        }
        if(matAB_T_flag == 2) // ATxB
        {
          ae_int16x4 out_0, out_1, out_2, out_3;

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, r01c0, r23c0, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, r45c0, r67c0, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, r01c1, r23c1, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, r45c1, r67c1, out_multiplier, left_shift, right_shift, out_zero_bias);

          AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

          ae_int8x8 temp_vec0, temp_vec1;
          temp_vec0 = AE_SAT8X8X16(out_0, out_1);
          temp_vec1 = AE_SAT8X8X16(out_2, out_3);

          ae_int8x8 *out_mat_ptr = (ae_int8x8 *)&p_out[vec_itr * mat1_rows + m1_row];
          ae_int8x8 *out_mat_ptr1 = (ae_int8x8 *)&p_out[(vec_itr + 1) * mat1_rows + m1_row];

          AE_SAV8X8X2_XP(temp_vec0, temp_vec0, b0, (ae_int8x16 *)out_mat_ptr, mat1_rows_count);
          AE_SAV8X8X2_XP(temp_vec1, temp_vec1, b1, (ae_int8x16 *)out_mat_ptr1, mat1_rows_count * (mat2_cols_rem > 1));

          AE_SA128POS_FP(b0, out_mat_ptr);
          AE_SA128POS_FP(b1, out_mat_ptr1);

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, r01c2, r23c2, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, r45c2, r67c2, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, r01c3, r23c3, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, r45c3, r67c3, out_multiplier, left_shift, right_shift, out_zero_bias);

          AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

          temp_vec0 = AE_SAT8X8X16(out_0, out_1);
          temp_vec1 = AE_SAT8X8X16(out_2, out_3);

          out_mat_ptr = (ae_int8x8 *)&p_out[(vec_itr + 2) * mat1_rows + m1_row];
          out_mat_ptr1 = (ae_int8x8 *)&p_out[(vec_itr + 3) * mat1_rows + m1_row];

          AE_SAV8X8X2_XP(temp_vec0, temp_vec0, b0, (ae_int8x16 *)out_mat_ptr, mat1_rows_count * (mat2_cols_rem > 2));
          AE_SAV8X8X2_XP(temp_vec1, temp_vec1, b1, (ae_int8x16 *)out_mat_ptr1, mat1_rows_count * (mat2_cols_rem > 3));

          AE_SA128POS_FP(b0, out_mat_ptr);
          AE_SA128POS_FP(b1, out_mat_ptr1);
        }
        else if(matAB_T_flag == 1) // AxBt
        {
          ae_int16x4 out_0, out_1, out_2, out_3;

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, r01c0, r01c1, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, r01c2, r01c3, out_multiplier, left_shift, right_shift, out_zero_bias);

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, r23c0, r23c1, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, r23c2, r23c3, out_multiplier, left_shift, right_shift, out_zero_bias);

          AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));

          AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

          ae_int8x8 temp_vec0, temp_vec1;
          temp_vec0 = AE_SAT8X8X16(out_0, out_1);
          temp_vec1 = AE_SAT8X8X16(out_2, out_3);

          ae_int8x8 out_dat0 = AE_SEL8X8I(temp_vec0, temp_vec0, 26); // 4 values for row 0
          ae_int8x8 out_dat1 = AE_SEL8X8I(temp_vec0, temp_vec0, 25); // 4 values for row 1

          ae_int8x8 out_dat2 = AE_SEL8X8I(temp_vec1, temp_vec1, 26); // 4 values for row 2
          ae_int8x8 out_dat3 = AE_SEL8X8I(temp_vec1, temp_vec1, 25); // 4 values for row 3

          ae_int8x8 *out_mat_ptr = (ae_int8x8 *)&p_out[m1_row * mat2_cols + vec_itr];
          ae_int8x8 *out_mat_ptr1 = (ae_int8x8 *)&p_out[(m1_row + 1) * mat2_cols + vec_itr];
          ae_int8x8 *out_mat_ptr2 = (ae_int8x8 *)&p_out[(m1_row + 2) * mat2_cols + vec_itr];
          ae_int8x8 *out_mat_ptr3 = (ae_int8x8 *)&p_out[(m1_row + 3) * mat2_cols + vec_itr];

          AE_SAV8X8X2_XP(out_dat0, out_dat0, b0, (ae_int8x16 *)out_mat_ptr, mat2_cols_rem);
          AE_SAV8X8X2_XP(out_dat1, out_dat1, b1, (ae_int8x16 *)out_mat_ptr1, mat2_cols_rem* (mat1_rows_count > 1));
          
          AE_SA128POS_FP(b0, out_mat_ptr);
          AE_SA128POS_FP(b1, out_mat_ptr1);

          AE_SAV8X8X2_XP(out_dat2, out_dat2, b0, (ae_int8x16 *)out_mat_ptr2, mat2_cols_rem* (mat1_rows_count > 2));
          AE_SAV8X8X2_XP(out_dat3, out_dat3, b1, (ae_int8x16 *)out_mat_ptr3, mat2_cols_rem* (mat1_rows_count > 3));

          AE_SA128POS_FP(b0, out_mat_ptr2);
          AE_SA128POS_FP(b1, out_mat_ptr3);

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, r45c0, r45c1, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, r45c2, r45c3, out_multiplier, left_shift, right_shift, out_zero_bias);

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_2, r67c0, r67c1, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_3, r67c2, r67c3, out_multiplier, left_shift, right_shift, out_zero_bias);
          
          AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));

          AE_MINMAX16(out_2, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_3, AE_MOVDA16(-128), AE_MOVDA16(127));

          temp_vec0 = AE_SAT8X8X16(out_0, out_1);
          temp_vec1 = AE_SAT8X8X16(out_2, out_3);

          out_dat0 = AE_SEL8X8I(temp_vec0, temp_vec0, 26); // 4 values for row 4
          out_dat1 = AE_SEL8X8I(temp_vec0, temp_vec0, 25); // 4 values for row 5

          out_dat2 = AE_SEL8X8I(temp_vec1, temp_vec1, 26); // 4 values for row 6
          out_dat3 = AE_SEL8X8I(temp_vec1, temp_vec1, 25); // 4 values for row 7

          out_mat_ptr = (ae_int8x8 *)&p_out[(m1_row + 4)* mat2_cols + vec_itr];
          out_mat_ptr1 = (ae_int8x8 *)&p_out[(m1_row + 5) * mat2_cols + vec_itr];
          out_mat_ptr2 = (ae_int8x8 *)&p_out[(m1_row + 6) * mat2_cols + vec_itr];
          out_mat_ptr3 = (ae_int8x8 *)&p_out[(m1_row + 7) * mat2_cols + vec_itr];

          AE_SAV8X8X2_XP(out_dat0, out_dat0, b0, (ae_int8x16 *)out_mat_ptr, mat2_cols_rem* (mat1_rows_count > 4));
          AE_SAV8X8X2_XP(out_dat1, out_dat1, b1, (ae_int8x16 *)out_mat_ptr1, mat2_cols_rem* (mat1_rows_count > 5));
          
          AE_SA128POS_FP(b0, out_mat_ptr);
          AE_SA128POS_FP(b1, out_mat_ptr1);

          AE_SAV8X8X2_XP(out_dat2, out_dat2, b0, (ae_int8x16 *)out_mat_ptr2, mat2_cols_rem* (mat1_rows_count > 6));
          AE_SAV8X8X2_XP(out_dat3, out_dat3, b1, (ae_int8x16 *)out_mat_ptr3, mat2_cols_rem* (mat1_rows_count > 7));

          AE_SA128POS_FP(b0, out_mat_ptr2);
          AE_SA128POS_FP(b1, out_mat_ptr3);
        }
      }
    }
  return 0;
}

static WORD32 xa_nn_batch_matmul_asym8sxasym8s_asym8s_mat_vec1(WORD8 * __restrict__ p_out,
    const WORD8 * __restrict__ p_mat1,
    const WORD8 * __restrict__ p_mat2,
    WORD32 mat1_cols,
    WORD32 mat1_rows,
    WORD32 mat2_cols,
    WORD32 mat1_zero_bias,
    WORD32 mat2_zero_bias,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias,
    WORD32 matAB_T_flag){

    int left_shift, right_shift;

#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  right_shift = out_shift;
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift < 0 ? 0 : out_shift;
  right_shift = out_shift > 0 ? 0 : -out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

    ae_int64 biasvc = AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(-mat2_zero_bias, -mat1_zero_bias));
    AE_MOVZBVCDR(biasvc);

    ae_int8x8 d0,d1,d2,d3,d4,d5,d6,d7;
    ae_int8x8 mat2_col0;
    ae_int8x8 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11;

    ae_int8x8 transpose_2rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfb73ea62L, 0xd951c840L));
    ae_int8x8 transpose_4rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfbea7362L, 0xd9c85140L));
    ae_int8x8 transpose_8rows_sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0xfbead9c8L, 0x73625140L));
      
    ae_int32x2 r01c0, r23c0, r45c0, r67c0;
    ae_int8x8 *out_mat_ptr = (ae_int8x8 *)p_out;
    ae_valignx2 b0;
    b0 = AE_ZALIGN128();
    int offset0, offset1, offset2, offset3, offset4, offset5, offset6, offset7; 
    offset0 = ((unsigned int)p_mat1 & 0x7);
    ae_int8x8 sel_c0 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset0], align_load_sel_pattern[2 * offset0 + 1]));

    offset1 = ((unsigned int)(p_mat1 + mat1_rows) & 0x7);
    ae_int8x8 sel_c1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset1], align_load_sel_pattern[2 * offset1 + 1]));

    offset2 = ((unsigned int)(p_mat1 + 2*mat1_rows) & 0x7);
    ae_int8x8 sel_c2 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset2], align_load_sel_pattern[2 * offset2 + 1]));

    offset3 = ((unsigned int)(p_mat1 + 3*mat1_rows) & 0x7);
    ae_int8x8 sel_c3 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset3], align_load_sel_pattern[2 * offset3 + 1]));

    offset4 = ((unsigned int)(p_mat1 + 4*mat1_rows) & 0x7);
    ae_int8x8 sel_c4 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset4], align_load_sel_pattern[2 * offset4 + 1]));

    offset5 = ((unsigned int)(p_mat1 + 5*mat1_rows) & 0x7);
    ae_int8x8 sel_c5 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset5], align_load_sel_pattern[2 * offset5 + 1]));

    offset6 = ((unsigned int)(p_mat1 + 6*mat1_rows) & 0x7);
    ae_int8x8 sel_c6 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset6], align_load_sel_pattern[2 * offset6 + 1]));

    offset7 = ((unsigned int)(p_mat1 + 7*mat1_rows) & 0x7);
    ae_int8x8 sel_c7 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(align_load_sel_pattern[2 * offset7], align_load_sel_pattern[2 * offset7 + 1]));

      for (int m1_row = 0; m1_row < mat1_rows; m1_row += 8){

        int mat1_rows_count = (mat1_rows - m1_row) >=8 ? 8: (mat1_rows - m1_row) ;
		    ae_int8x8 *mat2_ptr = (ae_int8x8 *)p_mat2;
        r01c0 = r23c0 = r45c0 = r67c0 = 0;

        ae_valign valign_mat2_p0 = AE_LA64_PP(mat2_ptr);

        ae_int8x8 *p_mat1_c0 = (ae_int8x8 *)&p_mat1[m1_row - offset0];
        ae_int8x8 *p_mat1_c1 = (ae_int8x8 *)&p_mat1[mat1_rows + m1_row - offset1];
        ae_int8x8 *p_mat1_c2 = (ae_int8x8 *)&p_mat1[2*mat1_rows + m1_row - offset2];
        ae_int8x8 *p_mat1_c3 = (ae_int8x8 *)&p_mat1[3*mat1_rows + m1_row - offset3];
        ae_int8x8 *p_mat1_c4 = (ae_int8x8 *)&p_mat1[4*mat1_rows + m1_row - offset4];
        ae_int8x8 *p_mat1_c5 = (ae_int8x8 *)&p_mat1[5*mat1_rows + m1_row - offset5];
        ae_int8x8 *p_mat1_c6 = (ae_int8x8 *)&p_mat1[6*mat1_rows + m1_row - offset6];
        ae_int8x8 *p_mat1_c7 = (ae_int8x8 *)&p_mat1[7*mat1_rows + m1_row - offset7];

        for (int m_itr = 0; m_itr < (mat1_cols & ~(7)); m_itr += 8){
          d0 = AE_L8X8_I(p_mat1_c0, 8);
          AE_L8X8_XP(tmp0, p_mat1_c0, 8*mat1_rows);
          d0 = AE_SEL8X8(tmp0, d0, sel_c0);

          d1 = AE_L8X8_I(p_mat1_c1, 8);
          AE_L8X8_XP(tmp1, p_mat1_c1, 8*mat1_rows);
          d1 = AE_SEL8X8(tmp1, d1, sel_c1);

          d2 = AE_L8X8_I(p_mat1_c2, 8);
          AE_L8X8_XP(tmp2, p_mat1_c2, 8*mat1_rows);
          d2 = AE_SEL8X8(tmp2, d2, sel_c2);

          d3 = AE_L8X8_I(p_mat1_c3, 8);
          AE_L8X8_XP(tmp3, p_mat1_c3, 8*mat1_rows);
          d3 = AE_SEL8X8(tmp3, d3, sel_c3);

          d4 = AE_L8X8_I(p_mat1_c4, 8);
          AE_L8X8_XP(tmp4, p_mat1_c4, 8*mat1_rows);
          d4 = AE_SEL8X8(tmp4, d4, sel_c4);

          d5 = AE_L8X8_I(p_mat1_c5, 8);
          AE_L8X8_XP(tmp5, p_mat1_c5, 8*mat1_rows);
          d5 = AE_SEL8X8(tmp5, d5, sel_c5);

          d6 = AE_L8X8_I(p_mat1_c6, 8);
          AE_L8X8_XP(tmp6, p_mat1_c6, 8*mat1_rows);
          d6 = AE_SEL8X8(tmp6, d6, sel_c6);

          d7 = AE_L8X8_I(p_mat1_c7, 8);
          AE_L8X8_XP(tmp7, p_mat1_c7, 8*mat1_rows);
          d7 = AE_SEL8X8(tmp7, d7, sel_c7);

          AE_LA8X8_IP(mat2_col0, valign_mat2_p0, mat2_ptr);

          AE_DSEL8X8(tmp0, tmp1, d0, d1, transpose_2rows_sel);
          AE_DSEL8X8(tmp2, tmp3, d2, d3, transpose_2rows_sel);
        
          AE_DSEL8X8(tmp4, tmp5, d4, d5, transpose_2rows_sel);
          AE_DSEL8X8(tmp6, tmp7, d6, d7, transpose_2rows_sel);
        
        
          AE_DSEL8X8(tmp8, tmp9, tmp0, tmp2, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp4, tmp6, transpose_4rows_sel);
        
          AE_DSEL8X8(d0, d1, tmp8, tmp10, transpose_8rows_sel); // r0, r1
          AE_DSEL8X8(d2, d3, tmp9, tmp11, transpose_8rows_sel); // r2, r3
        
          AE_DSEL8X8(tmp8, tmp9, tmp1, tmp3, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp5, tmp7, transpose_4rows_sel);
        
          AE_DSEL8X8(d4, d5, tmp8, tmp10, transpose_8rows_sel); // r4, r5
          AE_DSEL8X8(d6, d7, tmp9, tmp11, transpose_8rows_sel); // r6, r7

          // mat row0 to row3 x vec col0  and mat row4 to row7 x vec col0
          MAT_VEC_MAC(r01c0, r23c0, d0, d1, d2, d3, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c0, r67c0, d4, d5, d6, d7, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
        }
        int rem = mat1_cols & 7;
        if (rem)
        {
          p_mat1_c1 = rem > 1 ? p_mat1_c1 : p_mat1_c0;
          p_mat1_c2 = rem > 2 ? p_mat1_c2 : p_mat1_c0;
          p_mat1_c3 = rem > 3 ? p_mat1_c3 : p_mat1_c0;
          p_mat1_c4 = rem > 4 ? p_mat1_c4 : p_mat1_c0;
          p_mat1_c5 = rem > 5 ? p_mat1_c5 : p_mat1_c0;
          p_mat1_c6 = rem > 6 ? p_mat1_c6 : p_mat1_c0;

          d0 = AE_L8X8_I(p_mat1_c0, 8);
          tmp0 = AE_L8X8_I(p_mat1_c0, 0);
          d0 = AE_SEL8X8(tmp0, d0, sel_c0);

          d1 = AE_L8X8_I(p_mat1_c1, 8);
          tmp1 = AE_L8X8_I(p_mat1_c1, 0);
          d1 = AE_SEL8X8(tmp1, d1, sel_c1);

          d2 = AE_L8X8_I(p_mat1_c2, 8);
          tmp2 = AE_L8X8_I(p_mat1_c2, 0);
          d2 = AE_SEL8X8(tmp2, d2, sel_c2);

          d3 = AE_L8X8_I(p_mat1_c3, 8);
          tmp3 = AE_L8X8_I(p_mat1_c3, 0);
          d3 = AE_SEL8X8(tmp3, d3, sel_c3);

          d4 = AE_L8X8_I(p_mat1_c4, 8);
          tmp4 = AE_L8X8_I(p_mat1_c4, 0);
          d4 = AE_SEL8X8(tmp4, d4, sel_c4);

          d5 = AE_L8X8_I(p_mat1_c5, 8);
          tmp5 = AE_L8X8_I(p_mat1_c5, 0);
          d5 = AE_SEL8X8(tmp5, d5, sel_c5);

          d6 = AE_L8X8_I(p_mat1_c6, 8);
          tmp6 = AE_L8X8_I(p_mat1_c6, 0);
          d6 = AE_SEL8X8(tmp6, d6, sel_c6);

          ae_int8x8 sel1 = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(data_sel_pattern[2 * rem], data_sel_pattern[2 * rem + 1]));

          AE_DSEL8X8(tmp0, tmp1, d0, d1, transpose_2rows_sel);
          AE_DSEL8X8(tmp2, tmp3, d2, d3, transpose_2rows_sel);
        
          AE_DSEL8X8(tmp4, tmp5, d4, d5, transpose_2rows_sel);
          AE_DSEL8X8(tmp6, tmp7, d6, d6, transpose_2rows_sel);
        
        
          AE_DSEL8X8(tmp8, tmp9, tmp0, tmp2, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp4, tmp6, transpose_4rows_sel);
        
          AE_DSEL8X8(d0, d1, tmp8, tmp10, transpose_8rows_sel); // r0, r1
          AE_DSEL8X8(d2, d3, tmp9, tmp11, transpose_8rows_sel); // r2, r3
        
          AE_DSEL8X8(tmp8, tmp9, tmp1, tmp3, transpose_4rows_sel);
          AE_DSEL8X8(tmp10, tmp11, tmp5, tmp7, transpose_4rows_sel);
        
          AE_DSEL8X8(d4, d5, tmp8, tmp10, transpose_8rows_sel); // r4, r5
          AE_DSEL8X8(d6, d7, tmp9, tmp11, transpose_8rows_sel); // r6, r7

          AE_LA8X8_IP(mat2_col0, valign_mat2_p0, mat2_ptr);

          ae_int8x8 mat2_bias = AE_MOVDA8((WORD8)(-mat2_zero_bias));

          mat2_col0 = AE_SEL8X8(mat2_col0, mat2_bias, sel1);

          // mat row0 to row3 x vec col0 and mat row4 to row7 x vec col0
          MAT_VEC_MAC(r01c0, r23c0, d0, d1, d2, d3, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
          MAT_VEC_MAC(r45c0, r67c0, d4, d5, d6, d7, mat2_col0, -mat1_zero_bias, -mat2_zero_bias);
        }
          ae_int16x4 out_0, out_1;
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_0, r01c0, r23c0, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(out_1, r45c0, r67c0, out_multiplier, left_shift, right_shift, out_zero_bias);

          AE_MINMAX16(out_0, AE_MOVDA16(-128), AE_MOVDA16(127));
          AE_MINMAX16(out_1, AE_MOVDA16(-128), AE_MOVDA16(127));

          ae_int8x8 temp_vec0;
          temp_vec0 = AE_SAT8X8X16(out_0, out_1);

          AE_SAV8X8X2_XP(temp_vec0, temp_vec0, b0, (ae_int8x16 *)out_mat_ptr, mat1_rows_count);
      }
      AE_SA128POS_FP(b0, out_mat_ptr);
  return 0;
}

WORD32 xa_nn_batch_matmul_asym8sxasym8s_asym8s(
    WORD8 * __restrict__ p_out,
    const WORD32 *const p_out_shape,
    const WORD8 * __restrict__ p_mat1,
    const WORD32 *const p_mat1_shape,
    const WORD8 * __restrict__ p_mat2,
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((mat1_transpose != 0 && mat1_transpose != 1), -1);
  XA_NNLIB_ARG_CHK_COND((mat2_transpose != 0 && mat2_transpose != 1), -1);
  XA_NNLIB_ARG_CHK_COND((mat1_zero_bias < -127 || mat1_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((mat2_zero_bias < -127 || mat2_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  WORD32 itr;
  const WORD8 *p_mat1_final, *p_mat2_final;
  WORD32 p_mat1_final_shape[5], p_mat2_final_shape[5];
  for(itr = 0; itr < 5; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_mat1_shape[itr] <= 0 || p_mat2_shape[itr] <= 0 || p_out_shape[itr] <= 0), -1);
    p_mat1_final_shape[itr] = p_mat1_shape[itr];
    p_mat2_final_shape[itr] = p_mat2_shape[itr];
  }
  p_mat1_final = p_mat1;
  p_mat2_final = p_mat2;

  WORD32 matAB_T_flag = mat1_transpose << 1 | mat2_transpose;
  int ret = 0;
  WORD32 mat1_ext0, mat1_ext1, mat1_ext2;
  mat1_ext0 = p_mat1_final_shape[0] == 1 ? 0 : p_mat1_final_shape[1] * p_mat1_final_shape[2] * p_mat1_final_shape[3] * p_mat1_final_shape[4];
  mat1_ext1 = p_mat1_final_shape[1] == 1 ? 0 : p_mat1_final_shape[2] * p_mat1_final_shape[3] * p_mat1_final_shape[4];
  mat1_ext2 = p_mat1_final_shape[2] == 1 ? 0 : p_mat1_final_shape[3] * p_mat1_final_shape[4];

  WORD32 mat2_ext0, mat2_ext1, mat2_ext2;
  mat2_ext0 = p_mat2_final_shape[0] == 1 ? 0 : p_mat2_final_shape[1] * p_mat2_final_shape[2] * p_mat2_final_shape[3] * p_mat2_final_shape[4];
  mat2_ext1 = p_mat2_final_shape[1] == 1 ? 0 : p_mat2_final_shape[2] * p_mat2_final_shape[3] * p_mat2_final_shape[4];
  mat2_ext2 = p_mat2_final_shape[2] == 1 ? 0 : p_mat2_final_shape[3] * p_mat2_final_shape[4];

  if (matAB_T_flag && (((matAB_T_flag == 2) && (p_mat2_final_shape[3] <= 12)) || (((matAB_T_flag == 1) && (p_mat1_final_shape[3] <= 12)))))
  {
    WORD32 b0, b1, b2;
    if (mat1_transpose)
    { 
      for (b0 = 0; b0 < p_out_shape[0]; b0++)
      {
        const WORD8 *ptr0_mat1 = p_mat1_final + b0 * mat1_ext0;
        const WORD8 *ptr0_mat2 = p_mat2_final + b0 * mat2_ext0;
        for (b1 = 0; b1 < p_out_shape[1]; b1++)
        {
          const WORD8 *ptr1_mat1 = ptr0_mat1 + b1 * mat1_ext1;
          const WORD8 *ptr1_mat2 = ptr0_mat2 + b1 * mat2_ext1;
          for (b2 = 0; b2 < p_out_shape[2]; b2++)
          {
            WORD32 ret = 0;
            const WORD8 *ptr2_mat1 = ptr1_mat1 + b2 * mat1_ext2;
            const WORD8 *ptr2_mat2 = ptr1_mat2 + b2 * mat2_ext2;
            WORD8 *ptr_out = p_out + ((b0 * p_out_shape[1] + b1) * p_out_shape[2] + b2) * p_mat1_final_shape[4] * p_mat2_final_shape[3];
            if (p_mat2_final_shape[3] == 1){
              ret = xa_nn_batch_matmul_asym8sxasym8s_asym8s_mat_vec1(ptr_out,
                                                                    ptr2_mat1,
                                                                    ptr2_mat2,
                                                                    p_mat1_final_shape[3],
                                                                    p_mat1_final_shape[4],
                                                                    p_mat2_final_shape[3],
                                                                    mat1_zero_bias,
                                                                    mat2_zero_bias,
                                                                    out_multiplier,
                                                                    out_shift,
                                                                    out_zero_bias,
                                                                    matAB_T_flag);
            }
            else{
              ret = xa_nn_batch_matmul_asym8sxasym8s_asym8s_mat1_transpose(ptr_out,
                                                                          ptr2_mat1,
                                                                          ptr2_mat2,
                                                                          p_mat1_final_shape[3],
                                                                          p_mat1_final_shape[4],
                                                                          p_mat2_final_shape[3],
                                                                          p_mat1_final_shape[3],
                                                                          mat1_zero_bias,
                                                                          mat2_zero_bias,
                                                                          out_multiplier,
                                                                          out_shift,
                                                                          out_zero_bias,
                                                                          matAB_T_flag);
            }
            if (ret != 0)
              return -1;
          }
        }
      }
    }
    else
    {
      for (b0 = 0; b0 < p_out_shape[0]; b0++)
      {
        const WORD8 *ptr0_mat1 = p_mat1_final + b0 * mat1_ext0;
        const WORD8 *ptr0_mat2 = p_mat2_final + b0 * mat2_ext0;
        for (b1 = 0; b1 < p_out_shape[1]; b1++)
        {
          const WORD8 *ptr1_mat1 = ptr0_mat1 + b1 * mat1_ext1;
          const WORD8 *ptr1_mat2 = ptr0_mat2 + b1 * mat2_ext1;
          for (b2 = 0; b2 < p_out_shape[2]; b2++)
          {
            WORD32 ret = 0;
            const WORD8 *ptr2_mat1 = ptr1_mat1 + b2 * mat1_ext2;
            const WORD8 *ptr2_mat2 = ptr1_mat2 + b2 * mat2_ext2;
            WORD8 *ptr_out = p_out + ((b0 * p_out_shape[1] + b1) * p_out_shape[2] + b2) * p_mat2_final_shape[4] * p_mat1_final_shape[3];
            if (p_mat1_final_shape[3] == 1)
            {
              ret = xa_nn_batch_matmul_asym8sxasym8s_asym8s_mat_vec1(ptr_out,
                                                                    ptr2_mat2,
                                                                    ptr2_mat1,
                                                                    p_mat2_final_shape[3],
                                                                    p_mat2_final_shape[4],
                                                                    p_mat1_final_shape[3],
                                                                    mat2_zero_bias,
                                                                    mat1_zero_bias,
                                                                    out_multiplier,
                                                                    out_shift,
                                                                    out_zero_bias,
                                                                    matAB_T_flag);
            }
            else
            {
              ret = xa_nn_batch_matmul_asym8sxasym8s_asym8s_mat1_transpose(ptr_out,
                                                                          ptr2_mat2,
                                                                          ptr2_mat1,
                                                                          p_mat2_final_shape[3],
                                                                          p_mat2_final_shape[4],
                                                                          p_mat1_final_shape[3],
                                                                          p_mat2_final_shape[3],
                                                                          mat2_zero_bias,
                                                                          mat1_zero_bias,
                                                                          out_multiplier,
                                                                          out_shift,
                                                                          out_zero_bias,
                                                                          matAB_T_flag);
            }
            if (ret != 0)
              return -1;
          }
        }
      }
    }
    return ret;
  }
  else if(matAB_T_flag &&(matAB_T_flag == 3)){
    WORD32 b0, b1, b2;
    int mat2_out_row_off = PADDED_SIZE(p_mat2_final_shape[3],16);
    WORD8 *trans_mat2_ptr = ALIGNED_ADDR((WORD8 *)p_scratch,16);
    for (b0 = 0; b0 < p_out_shape[0]; b0++)
    { 
      const WORD8 *ptr0_mat1 = p_mat1_final + b0 * mat1_ext0;
      const WORD8 *ptr0_mat2 = p_mat2_final + b0 * mat2_ext0;
      for (b1 = 0; b1 < p_out_shape[1]; b1++)
      { 
        const WORD8 *ptr1_mat1 = ptr0_mat1 + b1 * mat1_ext1;
        const WORD8 *ptr1_mat2 = ptr0_mat2 + b1 * mat2_ext1;
        for (b2 = 0; b2 < p_out_shape[2]; b2++)
        { 
          WORD32 ret = 0;
          const WORD8 *ptr2_mat1 = ptr1_mat1 + b2 * mat1_ext2;
          const WORD8 *ptr2_mat2 = ptr1_mat2 + b2 * mat2_ext2;
          WORD8 *ptr_out = p_out + ((b0 * p_out_shape[1] + b1) * p_out_shape[2] + b2) * p_mat1_final_shape[4] * p_mat2_final_shape[4];
          xa_nn_asym8sxasym8s_asym8s_mat_transpose(trans_mat2_ptr,
                                                    ptr2_mat2,
                                                    p_mat2_final_shape[3],
                                                    p_mat2_final_shape[4],
                                                    mat2_zero_bias,
                                                    mat2_out_row_off);

          ret = xa_nn_batch_matmul_a8sxa8s_a8s_mat2_aligned_padded_zbias(ptr_out,
                                                                          ptr2_mat1,
                                                                          trans_mat2_ptr,
                                                                          p_mat1_final_shape[3],
                                                                          p_mat1_final_shape[4],
                                                                          p_mat2_final_shape[4],
                                                                          mat2_out_row_off,
                                                                          mat1_zero_bias,
                                                                          mat2_zero_bias,
                                                                          out_multiplier,
                                                                          out_shift,
                                                                          out_zero_bias);
          if (ret != 0)
            return -1;
        }
      }
    }
    return ret;
  }
  if(mat1_transpose)
  {
    WORD32 mat1_size, ret;
    WORD32 permute_vec[5] = {0, 1, 2, 4, 3};
    p_mat1_final_shape[3] = p_mat1_shape[4];
    p_mat1_final_shape[4] = p_mat1_shape[3];
    ret = xa_nn_transpose_8_8((WORD8 *)p_scratch,
                              p_mat1_final_shape,
                              p_mat1,
                              p_mat1_shape,
                              permute_vec,
                              5,
                              5);
    if(ret != 0)
      return -1;
    p_mat1_final = (const WORD8 *)p_scratch;
    mat1_size = p_mat1_shape[0] * p_mat1_shape[1] * p_mat1_shape[2] * p_mat1_shape[3] * p_mat1_shape[4];
    p_scratch = (VOID *)(p_mat1_final + mat1_size);
  }

  if(mat2_transpose)
  {
    WORD32 ret;
    WORD32 permute_vec[5] = {0, 1, 2, 4, 3};
    p_mat2_final_shape[3] = p_mat2_shape[4];
    p_mat2_final_shape[4] = p_mat2_shape[3];
    ret = xa_nn_transpose_8_8((WORD8 *)p_scratch,
                              p_mat2_final_shape,
                              p_mat2,
                              p_mat2_shape,
                              permute_vec,
                              5,
                              5);
    if(ret != 0)
      return -1;
    p_mat2_final = (const WORD8 *)p_scratch;
  }

  WORD32 accum_depth, mat1_rows, mat2_cols;
  accum_depth = p_mat1_final_shape[4];
  mat1_rows = p_mat1_final_shape[3];
  mat2_cols = p_mat2_final_shape[3];

  WORD32 b0, b1, b2;
  for(b0 = 0; b0 < p_out_shape[0]; b0++)
  {
    const WORD8 *ptr0_mat1 = p_mat1_final + b0 * mat1_ext0;
    const WORD8 *ptr0_mat2 = p_mat2_final + b0 * mat2_ext0;
    for(b1 = 0; b1 < p_out_shape[1]; b1++)
    {
      const WORD8 *ptr1_mat1 = ptr0_mat1 + b1 * mat1_ext1;
      const WORD8 *ptr1_mat2 = ptr0_mat2 + b1 * mat2_ext1;
      for(b2 = 0; b2 < p_out_shape[2]; b2++)
      {
        WORD32 ret = 0;
        const WORD8 *ptr2_mat1 = ptr1_mat1 + b2 * mat1_ext2;
        const WORD8 *ptr2_mat2 = ptr1_mat2 + b2 * mat2_ext2;
        WORD8 *ptr_out = p_out + ((b0 * p_out_shape[1] + b1) * p_out_shape[2] + b2) * mat1_rows * mat2_cols;
        ret = xa_nn_matmul_asym8sxasym8s_asym8s(ptr_out,
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
        if(ret != 0)
          return -1;
      }
    }
  }
  return 0;
}
