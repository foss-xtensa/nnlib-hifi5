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
#include "xa_type_def.h"
#include "xa_nn_common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common.h"

static inline void ComputeInterpolationValuesInteger(
    const WORD32 value, const WORD32 scale_10, const WORD32 shift,
    WORD32 input_size, WORD32* scaled_value, WORD32* lower_bound, WORD32* upper_bound)
{
  *scaled_value = value * scale_10 + shift;

  *lower_bound = XT_MIN(XT_MAX((*scaled_value >> 10), 0), input_size - 1);
  *upper_bound = XT_MIN(((*scaled_value + (1 << 10) - 1) >> 10), input_size - 1);
}

WORD32 xa_nn_resize_bilinear_8_8
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_inp
  ,WORD32  input_batch
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  out_batch
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  out_channels
  ,WORD32  height_scale_10
  ,WORD32  width_scale_10
  ,WORD32  height_shift
  ,WORD32  width_shift
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_batch <= 0 || input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_batch != input_batch || out_channels != input_channels), -1);

  int itr_n, itr_h, itr_w, itr_c;

  int width_off  = input_channels;
  int height_off = input_width * width_off;
  int batch_off  = input_height * height_off;

  WORD8 *ptmp_inp = (WORD8 *)p_inp;
  WORD8 *ptmp_inp_h0, *ptmp_inp_h1;
  WORD8 *ptmp_inp_h0w0, *ptmp_inp_h0w1, *ptmp_inp_h1w0, *ptmp_inp_h1w1;
  ae_int8x8 *ptmp_out0 = (ae_int8x8 *) p_out;
  
  ae_int32x2 d_one_q10 = AE_MOVDA32(1<<10);

  if( ((((unsigned)p_inp) & 7) == 0) && ((((unsigned)p_out) & 7) == 0) && ((input_channels & 7) == 0))
  {
    for(itr_n = 0; itr_n < input_batch; itr_n++)
    {
      for(itr_h = 0; itr_h < out_height; itr_h++)
      {
        WORD32 input_y, y0, y1;
        ComputeInterpolationValuesInteger(itr_h, height_scale_10, height_shift, input_height, &input_y, &y0, &y1);
        ae_int32x2 y_m_y0_q10, one_m_y_m_y0_q10;
        y_m_y0_q10 = AE_MOVDA32(input_y - (y0 << 10));
        one_m_y_m_y0_q10 = AE_SUB32(d_one_q10, y_m_y0_q10);
        ptmp_inp_h0 = ptmp_inp + y0 * height_off;
        ptmp_inp_h1 = ptmp_inp + y1 * height_off;
        for (itr_w = 0; itr_w < out_width; itr_w++)
        {
          WORD32 input_x, x0, x1;
          ComputeInterpolationValuesInteger(itr_w, width_scale_10, width_shift, input_width, &input_x, &x0, &x1);
          ae_int32x2 x_m_x0_q10, one_m_x_m_x0_q10;
          x_m_x0_q10 = AE_MOVDA32(input_x - (x0 << 10));
          one_m_x_m_x0_q10 = AE_SUB32(d_one_q10, x_m_x0_q10);
          ptmp_inp_h0w0 = ptmp_inp_h0 + x0 * width_off;
          ptmp_inp_h0w1 = ptmp_inp_h0 + x1 * width_off;
          ptmp_inp_h1w0 = ptmp_inp_h1 + x0 * width_off;
          ptmp_inp_h1w1 = ptmp_inp_h1 + x1 * width_off;

          ae_int32x2 mul_ll, mul_lu, mul_rl, mul_ru;
          AE_MUL2P32X4(mul_ll, mul_lu, one_m_x_m_x0_q10, one_m_x_m_x0_q10, one_m_y_m_y0_q10, y_m_y0_q10);
          AE_MUL2P32X4(mul_rl, mul_ru, x_m_x0_q10, x_m_x0_q10, one_m_y_m_y0_q10, y_m_y0_q10);
          mul_ll = AE_SLAI32(mul_ll, 7);
          mul_lu = AE_SLAI32(mul_lu, 7);
          mul_rl = AE_SLAI32(mul_rl, 7);
          mul_ru = AE_SLAI32(mul_ru, 7);

#pragma concurrent
          for(itr_c = 0; itr_c < (out_channels >> 3); itr_c++)
          {
            ae_int8x8 d_inp8_ll, d_inp8_lu, d_inp8_rl, d_inp8_ru;
            ae_int16x4 d_inp16_ll0, d_inp16_ll1, d_inp16_lu0, d_inp16_lu1, d_inp16_rl0, d_inp16_rl1, d_inp16_ru0, d_inp16_ru1;
            
            AE_L8X8_IP(d_inp8_ll, (const ae_int8x8 *)ptmp_inp_h0w0, 8);
            AE_L8X8_IP(d_inp8_lu, (const ae_int8x8 *)ptmp_inp_h1w0, 8);
            AE_L8X8_IP(d_inp8_rl, (const ae_int8x8 *)ptmp_inp_h0w1, 8);
            AE_L8X8_IP(d_inp8_ru, (const ae_int8x8 *)ptmp_inp_h1w1, 8);
            
            AE_CVTA16X4X2F8(d_inp16_ll0, d_inp16_ll1, d_inp8_ll,8);
            AE_CVTA16X4X2F8(d_inp16_lu0, d_inp16_lu1, d_inp8_lu,8);
            AE_CVTA16X4X2F8(d_inp16_rl0, d_inp16_rl1, d_inp8_rl,8);
            AE_CVTA16X4X2F8(d_inp16_ru0, d_inp16_ru1, d_inp8_ru,8);

            ae_int32x2 d_out_00, d_out_01, d_out_10, d_out_11;
            AE_MULF2P32X16X4S(d_out_00, d_out_01, mul_ll, mul_ll, d_inp16_ll0);
            AE_MULAF2P32X16X4S(d_out_00, d_out_01, mul_lu, mul_lu, d_inp16_lu0);
            AE_MULAF2P32X16X4S(d_out_00, d_out_01, mul_rl, mul_rl, d_inp16_rl0);
            AE_MULAF2P32X16X4S(d_out_00, d_out_01, mul_ru, mul_ru, d_inp16_ru0);

            AE_MULF2P32X16X4S(d_out_10, d_out_11, mul_ll, mul_ll, d_inp16_ll1);
            AE_MULAF2P32X16X4S(d_out_10, d_out_11, mul_lu, mul_lu, d_inp16_lu1);
            AE_MULAF2P32X16X4S(d_out_10, d_out_11, mul_rl, mul_rl, d_inp16_rl1);
            AE_MULAF2P32X16X4S(d_out_10, d_out_11, mul_ru, mul_ru, d_inp16_ru1);
            
            ae_int16x4 d_out16_0, d_out16_1;
#if TFLITE_SINGLE_ROUNDING
            d_out_00 = AE_SRAI32(d_out_00, 4);
            d_out_01 = AE_SRAI32(d_out_01, 4);
            d_out16_0 = AE_ROUND16X4F32SASYM(d_out_00, d_out_01);

            d_out_10 = AE_SRAI32(d_out_10, 4);
            d_out_11 = AE_SRAI32(d_out_11, 4);
            d_out16_1 = AE_ROUND16X4F32SASYM(d_out_10, d_out_11);
#else
            AE_MULF2P32X4RS(d_out_00, d_out_01, d_out_00, d_out_01, AE_MOVDA32(1 << 11), AE_MOVDA32(1 << 11));
            d_out16_0 = AE_SAT16X4(d_out_00, d_out_01);

            AE_MULF2P32X4RS(d_out_10, d_out_11, d_out_10, d_out_11, AE_MOVDA32(1 << 11), AE_MOVDA32(1 << 11));
            d_out16_1 = AE_SAT16X4(d_out_10, d_out_11);
#endif //TFLITE_SINGLE_ROUNDING

            ae_int8x8 res = AE_SAT8X8X16(d_out16_0, d_out16_1);
            AE_S8X8_IP(res, ptmp_out0, 8);
          }
        }
      }
      ptmp_inp += batch_off;
    }
  }
  else
  {
    int rem_length = (input_channels & 7); 
    for(itr_n = 0; itr_n < input_batch; itr_n++)
    {
      for(itr_h = 0; itr_h < out_height; itr_h++)
      {
        WORD32 input_y, y0, y1;
        ComputeInterpolationValuesInteger(itr_h, height_scale_10, height_shift, input_height, &input_y, &y0, &y1);

        ae_int32x2 y_m_y0_q10, one_m_y_m_y0_q10;
        y_m_y0_q10 = AE_MOVDA32(input_y - (y0 << 10));
        one_m_y_m_y0_q10 = AE_SUB32(d_one_q10, y_m_y0_q10);
        ptmp_inp_h0 = ptmp_inp + y0 * height_off;
        ptmp_inp_h1 = ptmp_inp + y1 * height_off;
        for (itr_w = 0; itr_w < out_width; itr_w++)
        {
          WORD32 input_x, x0, x1;
          ComputeInterpolationValuesInteger(itr_w, width_scale_10, width_shift, input_width, &input_x, &x0, &x1);
          ae_int32x2 x_m_x0_q10, one_m_x_m_x0_q10;
          x_m_x0_q10 = AE_MOVDA32(input_x - (x0 << 10));
          one_m_x_m_x0_q10 = AE_SUB32(d_one_q10, x_m_x0_q10);
          ptmp_inp_h0w0 = ptmp_inp_h0 + x0 * width_off;
          ptmp_inp_h0w1 = ptmp_inp_h0 + x1 * width_off;
          ptmp_inp_h1w0 = ptmp_inp_h1 + x0 * width_off;
          ptmp_inp_h1w1 = ptmp_inp_h1 + x1 * width_off;
          
          ae_valign h0w0_aligner, h0w1_aligner, h1w0_aligner, h1w1_aligner;
          h1w0_aligner = AE_LA64_PP(ptmp_inp_h1w0);
          h1w1_aligner = AE_LA64_PP(ptmp_inp_h1w1);

          ae_int32x2 mul_ll, mul_lu, mul_rl, mul_ru;
          AE_MUL2P32X4(mul_ll, mul_lu, one_m_x_m_x0_q10, one_m_x_m_x0_q10, one_m_y_m_y0_q10, y_m_y0_q10);
          AE_MUL2P32X4(mul_rl, mul_ru, x_m_x0_q10, x_m_x0_q10, one_m_y_m_y0_q10, y_m_y0_q10);
          mul_ll = AE_SLAI32(mul_ll, 7);
          mul_lu = AE_SLAI32(mul_lu, 7);
          mul_rl = AE_SLAI32(mul_rl, 7);
          mul_ru = AE_SLAI32(mul_ru, 7);

          ae_valign out_aligner = AE_ZALIGN64();
          for(itr_c = 0; itr_c < (out_channels >> 3); itr_c++)
          {
            ae_int8x8 d_inp8_ll, d_inp8_lu, d_inp8_rl, d_inp8_ru;
            ae_int16x4 d_inp16_ll0, d_inp16_ll1, d_inp16_lu0, d_inp16_lu1, d_inp16_rl0, d_inp16_rl1, d_inp16_ru0, d_inp16_ru1;
            h0w0_aligner = AE_LA64_PP(ptmp_inp_h0w0);
            AE_LA8X8_IP(d_inp8_ll, h0w0_aligner, (const ae_int8x8 *) ptmp_inp_h0w0);
            h0w1_aligner = AE_LA64_PP(ptmp_inp_h0w1);
            AE_LA8X8_IP(d_inp8_lu, h1w0_aligner, (const ae_int8x8 *) ptmp_inp_h1w0);
            AE_LA8X8_IP(d_inp8_rl, h0w1_aligner, (const ae_int8x8 *) ptmp_inp_h0w1);
            AE_LA8X8_IP(d_inp8_ru, h1w1_aligner, (const ae_int8x8 *) ptmp_inp_h1w1);
            
            AE_CVTA16X4X2F8(d_inp16_ll0, d_inp16_ll1, d_inp8_ll,8);
            AE_CVTA16X4X2F8(d_inp16_lu0, d_inp16_lu1, d_inp8_lu,8);
            AE_CVTA16X4X2F8(d_inp16_rl0, d_inp16_rl1, d_inp8_rl,8);
            AE_CVTA16X4X2F8(d_inp16_ru0, d_inp16_ru1, d_inp8_ru,8);
             
            ae_int32x2 d_out_00, d_out_01, d_out_10, d_out_11;
            AE_MULF2P32X16X4S(d_out_00, d_out_01, mul_ll, mul_ll, d_inp16_ll0);
            AE_MULAF2P32X16X4S(d_out_00, d_out_01, mul_lu, mul_lu, d_inp16_lu0);
            AE_MULAF2P32X16X4S(d_out_00, d_out_01, mul_rl, mul_rl, d_inp16_rl0);
            AE_MULAF2P32X16X4S(d_out_00, d_out_01, mul_ru, mul_ru, d_inp16_ru0);

            AE_MULF2P32X16X4S(d_out_10, d_out_11, mul_ll, mul_ll, d_inp16_ll1);
            AE_MULAF2P32X16X4S(d_out_10, d_out_11, mul_lu, mul_lu, d_inp16_lu1);
            AE_MULAF2P32X16X4S(d_out_10, d_out_11, mul_rl, mul_rl, d_inp16_rl1);
            AE_MULAF2P32X16X4S(d_out_10, d_out_11, mul_ru, mul_ru, d_inp16_ru1);

            ae_int16x4 d_out16_0, d_out16_1;
#if TFLITE_SINGLE_ROUNDING
            d_out_00 = AE_SRAI32(d_out_00, 4);
            d_out_01 = AE_SRAI32(d_out_01, 4);
            d_out16_0 = AE_ROUND16X4F32SASYM(d_out_00, d_out_01);

            d_out_10 = AE_SRAI32(d_out_10, 4);
            d_out_11 = AE_SRAI32(d_out_11, 4);
            d_out16_1 = AE_ROUND16X4F32SASYM(d_out_10, d_out_11);
#else
            AE_MULF2P32X4RS(d_out_00, d_out_01, d_out_00, d_out_01, AE_MOVDA32(1 << 11), AE_MOVDA32(1 << 11));
            d_out16_0 = AE_SAT16X4(d_out_00, d_out_01);

            AE_MULF2P32X4RS(d_out_10, d_out_11, d_out_10, d_out_11, AE_MOVDA32(1 << 11), AE_MOVDA32(1 << 11));
            d_out16_1 = AE_SAT16X4(d_out_10, d_out_11);

#endif //TFLITE_SINGLE_ROUNDING
  
            ae_int8x8 res = AE_SAT8X8X16(d_out16_0, d_out16_1);
            AE_SA8X8_IP(res, out_aligner, ptmp_out0);
          }
          AE_SA64POS_FP(out_aligner, (void *) ptmp_out0);
          if(rem_length)
          {
            ae_valignx2 rem_h0w0_aligner, rem_h0w1_aligner, rem_h1w0_aligner, rem_h1w1_aligner;
            rem_h0w0_aligner = AE_LA128_PP(ptmp_inp_h0w0);
            rem_h0w1_aligner = AE_LA128_PP(ptmp_inp_h0w1);
            rem_h1w0_aligner = AE_LA128_PP(ptmp_inp_h1w0);
            rem_h1w1_aligner = AE_LA128_PP(ptmp_inp_h1w1);
            ae_int8x8 d_inp8_ll=AE_MOVINT8X8_FROMINT16(AE_ZERO16()), d_inp8_lu=AE_MOVINT8X8_FROMINT16(AE_ZERO16()), d_inp8_rl=AE_MOVINT8X8_FROMINT16(AE_ZERO16()), d_inp8_ru=AE_MOVINT8X8_FROMINT16(AE_ZERO16()), extra_8;
            ae_int16x4 d_inp16_ll0, d_inp16_ll1, d_inp16_lu0, d_inp16_lu1, d_inp16_rl0, d_inp16_rl1, d_inp16_ru0, d_inp16_ru1;
            AE_LAV8X8X2_XP(d_inp8_ll, extra_8, rem_h0w0_aligner, (const ae_int8x16 *) ptmp_inp_h0w0, rem_length);
            AE_LAV8X8X2_XP(d_inp8_lu, extra_8, rem_h1w0_aligner, (const ae_int8x16 *) ptmp_inp_h1w0, rem_length);
            AE_LAV8X8X2_XP(d_inp8_rl, extra_8, rem_h0w1_aligner, (const ae_int8x16 *) ptmp_inp_h0w1, rem_length);
            AE_LAV8X8X2_XP(d_inp8_ru, extra_8, rem_h1w1_aligner, (const ae_int8x16 *) ptmp_inp_h1w1, rem_length);

            AE_CVTA16X4X2F8(d_inp16_ll0, d_inp16_ll1, d_inp8_ll,8);
            AE_CVTA16X4X2F8(d_inp16_lu0, d_inp16_lu1, d_inp8_lu,8);
            AE_CVTA16X4X2F8(d_inp16_rl0, d_inp16_rl1, d_inp8_rl,8);
            AE_CVTA16X4X2F8(d_inp16_ru0, d_inp16_ru1, d_inp8_ru,8);

            ae_int32x2 d_out_00, d_out_01, d_out_10=0, d_out_11=0;
            AE_MULF2P32X16X4S(d_out_00, d_out_01, mul_ll, mul_ll, d_inp16_ll0);
            AE_MULAF2P32X16X4S(d_out_00, d_out_01, mul_lu, mul_lu, d_inp16_lu0);
            AE_MULAF2P32X16X4S(d_out_00, d_out_01, mul_rl, mul_rl, d_inp16_rl0);
            AE_MULAF2P32X16X4S(d_out_00, d_out_01, mul_ru, mul_ru, d_inp16_ru0);
        
            if(rem_length > 4)
            {
              AE_MULF2P32X16X4S(d_out_10, d_out_11, mul_ll, mul_ll, d_inp16_ll1);
              AE_MULAF2P32X16X4S(d_out_10, d_out_11, mul_lu, mul_lu, d_inp16_lu1);
              AE_MULAF2P32X16X4S(d_out_10, d_out_11, mul_rl, mul_rl, d_inp16_rl1);
              AE_MULAF2P32X16X4S(d_out_10, d_out_11, mul_ru, mul_ru, d_inp16_ru1);
            }
            ae_int16x4 d_out16_0, d_out16_1=0;
#if TFLITE_SINGLE_ROUNDING
            d_out_00 = AE_SRAI32(d_out_00, 4);
            d_out_01 = AE_SRAI32(d_out_01, 4);
            d_out16_0 = AE_ROUND16X4F32SASYM(d_out_00, d_out_01);
            
            if(rem_length > 4)
            {
              d_out_10 = AE_SRAI32(d_out_10, 4);
              d_out_11 = AE_SRAI32(d_out_11, 4);
              d_out16_1 = AE_ROUND16X4F32SASYM(d_out_10, d_out_11);
            }
#else
            AE_MULF2P32X4RS(d_out_00, d_out_01, d_out_00, d_out_01, AE_MOVDA32(1 << 11), AE_MOVDA32(1 << 11));
            d_out16_0 = AE_SAT16X4(d_out_00, d_out_01);
             
            if(rem_length > 4)
            {
              AE_MULF2P32X4RS(d_out_10, d_out_11, d_out_10, d_out_11, AE_MOVDA32(1 << 11), AE_MOVDA32(1 << 11));
              d_out16_1 = AE_SAT16X4(d_out_10, d_out_11);
            }

#endif //TFLITE_SINGLE_ROUNDING
     
            ae_int8x8 res = AE_SAT8X8X16(d_out16_0, d_out16_1);
            ae_valignx2 rem_out_aligner=AE_ZALIGN128();
            AE_SAV8X8X2_XP(res, extra_8, rem_out_aligner, (ae_int8x16*)ptmp_out0, rem_length);
            AE_SA128POS_FP(rem_out_aligner, (void *)ptmp_out0);

          }
        }
      }
      ptmp_inp += batch_off;
    }
  }
  return 0;
}




