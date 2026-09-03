/*******************************************************************************
* Copyright (c) 2018-2026 Cadence Design Systems, Inc.
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
#include "xa_nnlib_common_fpu.h"
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_err_chk.h"

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_conv2d_depthwise_f16,(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  channels_multiplier,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
    WORD32  inp_data_format,
    WORD32  out_data_format,
    pVOID p_scratch))

DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_dilated_conv2d_depthwise_v2_f16,(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  channels_multiplier,
    WORD32  dilation_height,
    WORD32  dilation_width,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
    WORD32  inp_data_format,
    WORD32  out_data_format,
    pVOID p_scratch,
    const WORD16* out_activation_min,
    const WORD16* out_activation_max,
    xa_dma_cfg_t *p_dma_cfg))

DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_conv2d_depthwise_v2_f16,(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  channels_multiplier,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
    WORD32  inp_data_format,
    WORD32  out_data_format,
    pVOID p_scratch,
    const WORD16* pout_activation_min,
    const WORD16* pout_activation_max,
    xa_dma_cfg_t *p_dma_cfg))

#else /* #if !HAVE_HP_VFPU */

#define DSELHX4(out0, out1, inp0, inp1, dsel){\
  ae_int16x4 out0_tmp, out1_tmp, inp0_tmp, inp1_tmp;\
  inp0_tmp = AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(inp0));\
  inp1_tmp = AE_MOVINT16X4_FROMF16X4(AE_MOVF16X4_FROMHALFX4(inp1));\
  AE_DSEL16X4(out0_tmp, out1_tmp, inp0_tmp, inp1_tmp, dsel);\
  out0 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(out0_tmp));\
  out1 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(out1_tmp));\
}

WORD32 xa_nn_conv2d_depthwise_f16_3x3_with_padding(
    WORD16 *p_out,
    const WORD16 *p_kernel,
    const WORD16 *p_inp,
    const WORD16 *p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 channels_multiplier,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 inp_data_format,
    WORD32 out_data_format,
    VOID *p_scratch,
    const WORD16 *act_min,
    const WORD16 *act_max)
{
  const int32_t outCh = input_channels * channels_multiplier;
  const WORD32 inPitch1U = input_channels;
  const WORD32 inPitch2U = input_channels * input_width;
  const WORD32 outPitch1U = outCh;
  const WORD32 outPitch2U = outCh * out_width;
  const WORD32 kPitch1 = input_channels;

  const xthalf *pInData = (const xthalf *)p_inp;
  xthalf *pOutData = (xthalf *)p_out;
  const xthalf *pCoeffData = (const xthalf *)p_kernel;
  const xthalf *pBiasData = (const xthalf *)p_bias;

  xthalfx4 zeroVec = ZERO_HX4();
  xthalfx4 vec_min = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(
      (act_min != NULL) ? *act_min : (WORD16)0xFC00)));
  xthalfx4 vec_max = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(
      (act_max != NULL) ? *act_max : (WORD16)0x7C00)));

  int32_t ch, x, y, i, j;

  // Compute y-region boundaries
  int y_start_no_pad = (y_padding + y_stride - 1) / y_stride; 
  int y_end_no_pad = (input_height + y_padding - 2) / y_stride;
  if (y_start_no_pad > out_height) y_start_no_pad = out_height;
  if (y_end_no_pad > out_height) y_end_no_pad = out_height;
  if (y_end_no_pad < y_start_no_pad) y_end_no_pad = y_start_no_pad;

  // 8 channels at a time
  for (ch = 0; ch < outCh; ch += 8)
  {
    int32_t remainingCh = XT_MIN((outCh - ch), 8);
    
    // Load bias once per channel 
    xthalfx4 hfvecBias0, hfvecBias1;
    xthalfx8 *pt_bias = (xthalfx8 *)(pBiasData + ch);
    ae_valignx2 bias_align = AE_LA128_PP(pt_bias);
    AE_LAVHX4X2_XP(hfvecBias0, hfvecBias1, bias_align, pt_bias, remainingCh * 2);

    // Load 3x3 kernel once per channel 
    xthalfx4 hfvecCoeff0[3][3];
    xthalfx4 hfvecCoeff1[3][3];
    const xthalf *p_coeff_base = pCoeffData + ch;
    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
        xthalfx8 *pt_coeff = (xthalfx8 *)(p_coeff_base + (i * 3 + j) * kPitch1);
        ae_valignx2 coeff_align = AE_LA128_PP(pt_coeff);
        AE_LAVHX4X2_XP(hfvecCoeff0[i][j], hfvecCoeff1[i][j], coeff_align, pt_coeff, remainingCh * 2);
      }
    }

    // 2 width positions at a time
    for (x = 0; x < out_width; x += 2)
    {
      int rem_width[2];
      int store_bytes[2];
      for (i = 0; i < 2; i++) {
        rem_width[i] = (x + i < out_width) ? remainingCh : 0;
        store_bytes[i] = rem_width[i] * 2;
      }

      // base input column for 2-output group
      int base_col = (x * x_stride) - x_padding;
      // compute column validity
      int col_valid[4];
      for (int c = 0; c < 4; c++) {
        int col_idx = base_col + c * x_stride;
        col_valid[c] = (col_idx >= 0) && (col_idx < input_width);
      }

      // Check if all columns are valid (no horizontal padding needed)
      int all_cols_valid = 1;
      for (int c = 0; c < 4; c++) {
        if (!col_valid[c]) {
          all_cols_valid = 0;
          break;
        }
      }

      // ===== REGION 1: Top padding region (y < y_start_no_pad) =====
      if (y_start_no_pad > 0) {
        xthalfx4 hfvecData0[2][4];
        xthalfx4 hfvecData1[2][4];
        
        // Load first 2 rows with padding check
        for (i = 0; i < 2; i++) {
          int in_row = (i * y_stride) - y_padding;
          int row_valid = (in_row >= 0) && (in_row < input_height);
          const xthalf *p_inp_row = pInData + ch + in_row * inPitch2U;
          
          for (j = 0; j < 4; j++) {
            if (row_valid && col_valid[j]) {
              int col_idx = base_col + j * x_stride;
              xthalfx8 *pt_inp = (xthalfx8 *)(p_inp_row + col_idx * inPitch1U);
              ae_valignx2 align_inp = AE_LA128_PP(pt_inp);
              AE_LAVHX4X2_XP(hfvecData0[i][j], hfvecData1[i][j],align_inp, pt_inp, remainingCh * 2);
            } else {
              hfvecData0[i][j] = zeroVec;
              hfvecData1[i][j] = zeroVec;
            }
          }
        }

        for (y = 0; y < y_start_no_pad; y++) {
          int in_row2 = (y * y_stride + 2 * y_stride) - y_padding;
          int row_valid2 = (in_row2 >= 0) && (in_row2 < input_height);

          // Load 3rd row with padding check
          xthalfx4 hfvecData2_0[4];
          xthalfx4 hfvecData2_1[4];
          const xthalf *p_inp_row2 = pInData + ch + in_row2 * inPitch2U;
          for (j = 0; j < 4; j++) {
            if (row_valid2 && col_valid[j]) {
              int col_idx = base_col + j * x_stride;
              xthalfx8 *pt_inp2 = (xthalfx8 *)(p_inp_row2 + col_idx * inPitch1U);
              ae_valignx2 align2 = AE_LA128_PP(pt_inp2);
              AE_LAVHX4X2_XP(hfvecData2_0[j], hfvecData2_1[j], align2, pt_inp2, remainingCh * 2);
            } else {
              hfvecData2_0[j] = zeroVec;
              hfvecData2_1[j] = zeroVec;
            }
          }

          // Init acc with bias and compute
          xthalfx4 hfvecAcc0[2];
          xthalfx4 hfvecAcc1[2];
          for (i = 0; i < 2; i++) {
            hfvecAcc0[i] = hfvecBias0;
            hfvecAcc1[i] = hfvecBias1;
            int d = i;
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][0], hfvecCoeff1[0][0], hfvecData0[0][d+0], hfvecData1[0][d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][1], hfvecCoeff1[0][1], hfvecData0[0][d+1], hfvecData1[0][d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][2], hfvecCoeff1[0][2], hfvecData0[0][d+2], hfvecData1[0][d+2]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][0], hfvecCoeff1[1][0], hfvecData0[1][d+0], hfvecData1[1][d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][1], hfvecCoeff1[1][1], hfvecData0[1][d+1], hfvecData1[1][d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][2], hfvecCoeff1[1][2], hfvecData0[1][d+2], hfvecData1[1][d+2]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][0], hfvecCoeff1[2][0], hfvecData2_0[d+0], hfvecData2_1[d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][1], hfvecCoeff1[2][1], hfvecData2_0[d+1], hfvecData2_1[d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][2], hfvecCoeff1[2][2], hfvecData2_0[d+2], hfvecData2_1[d+2]);
          }

          // Slide
          for (j = 0; j < 4; j++) {
            hfvecData0[0][j] = hfvecData0[1][j];
            hfvecData1[0][j] = hfvecData1[1][j];
            hfvecData0[1][j] = hfvecData2_0[j];
            hfvecData1[1][j] = hfvecData2_1[j];
          }

          // Store
          ae_valignx2 vaOutData;
          for (i = 0; i < 2; i++) {
            hfvecAcc0[i] = MAX_HX4(MIN_HX4(hfvecAcc0[i], vec_max), vec_min);
            hfvecAcc1[i] = MAX_HX4(MIN_HX4(hfvecAcc1[i], vec_max), vec_min);
            xthalfx8  *pt_out = (xthalfx8 *)(pOutData + ch + y * outPitch2U + (x + i) * outPitch1U);
            AE_SAVHX4X2_XP(hfvecAcc0[i], hfvecAcc1[i], vaOutData, pt_out, store_bytes[i]);
            AE_SA128POS_FP(vaOutData, pt_out);
          }
        }
      }

      // ===== REGION 2: Middle region =====
      // No padding
      if (all_cols_valid && y_end_no_pad > y_start_no_pad) {
        xthalfx4 hfvecData0[2][4];
        xthalfx4 hfvecData1[2][4];
        const xthalf *p_inp_base = pInData + ch + ((y_start_no_pad * y_stride - y_padding) * inPitch2U) + base_col * inPitch1U;
        
        // Load first 2 rows (no padding check)
        for (i = 0; i < 2; i++) {
          const xthalf *p_inp_row = p_inp_base + i * y_stride * inPitch2U;
          for (j = 0; j < 4; j++) {
            xthalfx8 *pt_inp = (xthalfx8 *)(p_inp_row + j * x_stride * inPitch1U);
            ae_valignx2 align_inp = AE_LA128_PP(pt_inp);
            AE_LAVHX4X2_XP(hfvecData0[i][j], hfvecData1[i][j], align_inp, pt_inp, remainingCh * 2);
          }
        }

        for (y = y_start_no_pad; y < y_end_no_pad; y++) {
          // Load 3rd row (no padding check)
          xthalfx4 hfvecData2_0[4];
          xthalfx4 hfvecData2_1[4];
          const xthalf *p_inp_row2 = p_inp_base + (2 * y_stride + (y - y_start_no_pad) * y_stride) * inPitch2U;
          for (j = 0; j < 4; j++) {
            xthalfx8 *pt_inp2 = (xthalfx8 *)(p_inp_row2 + j * x_stride * inPitch1U);
            ae_valignx2 align2 = AE_LA128_PP(pt_inp2);
            AE_LAVHX4X2_XP(hfvecData2_0[j], hfvecData2_1[j], align2, pt_inp2, remainingCh * 2);
          }

          // Init acc and compute
          xthalfx4 hfvecAcc0[2];
          xthalfx4 hfvecAcc1[2];
          for (i = 0; i < 2; i++) {
            hfvecAcc0[i] = hfvecBias0;
            hfvecAcc1[i] = hfvecBias1;
            int d = i;
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][0], hfvecCoeff1[0][0], hfvecData0[0][d+0], hfvecData1[0][d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][1], hfvecCoeff1[0][1], hfvecData0[0][d+1], hfvecData1[0][d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][2], hfvecCoeff1[0][2], hfvecData0[0][d+2], hfvecData1[0][d+2]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][0], hfvecCoeff1[1][0], hfvecData0[1][d+0], hfvecData1[1][d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][1], hfvecCoeff1[1][1], hfvecData0[1][d+1], hfvecData1[1][d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][2], hfvecCoeff1[1][2], hfvecData0[1][d+2], hfvecData1[1][d+2]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][0], hfvecCoeff1[2][0], hfvecData2_0[d+0], hfvecData2_1[d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][1], hfvecCoeff1[2][1], hfvecData2_0[d+1], hfvecData2_1[d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][2], hfvecCoeff1[2][2], hfvecData2_0[d+2], hfvecData2_1[d+2]);
          }

          // Slide
          for (j = 0; j < 4; j++) {
            hfvecData0[0][j] = hfvecData0[1][j];
            hfvecData1[0][j] = hfvecData1[1][j];
            hfvecData0[1][j] = hfvecData2_0[j];
            hfvecData1[1][j] = hfvecData2_1[j];
          }

          // Store
          ae_valignx2 vaOutData;
          for (i = 0; i < 2; i++) {
            hfvecAcc0[i] = MAX_HX4(MIN_HX4(hfvecAcc0[i], vec_max), vec_min);
            hfvecAcc1[i] = MAX_HX4(MIN_HX4(hfvecAcc1[i], vec_max), vec_min);
            xthalfx8 *pt_out = (xthalfx8 *)(pOutData + ch + y * outPitch2U + (x + i) * outPitch1U);
            AE_SAVHX4X2_XP(hfvecAcc0[i], hfvecAcc1[i], vaOutData, pt_out, store_bytes[i]);
            AE_SA128POS_FP(vaOutData, pt_out);
          }
        }
      } 
      // No vertical padding, but horizontal padding
      else if (y_end_no_pad > y_start_no_pad) {
        xthalfx4 hfvecData0[2][4];
        xthalfx4 hfvecData1[2][4];
        
        // Load first 2 rows with horizontal padding check
        for (i = 0; i < 2; i++) {
          int in_row = ((y_start_no_pad + i) * y_stride) - y_padding;
          const xthalf *p_inp_row = pInData + ch + in_row * inPitch2U;
          
          for (j = 0; j < 4; j++) {
            if (col_valid[j]) {
              int col_idx = base_col + j * x_stride;
              xthalfx8 *pt_inp = (xthalfx8 *)(p_inp_row + col_idx * inPitch1U);
              ae_valignx2 align_inp = AE_LA128_PP(pt_inp);
              AE_LAVHX4X2_XP(hfvecData0[i][j], hfvecData1[i][j], align_inp, pt_inp, remainingCh * 2);
            } else {
              hfvecData0[i][j] = zeroVec;
              hfvecData1[i][j] = zeroVec;
            }
          }
        }

        for (y = y_start_no_pad; y < y_end_no_pad; y++) {
          int in_row2 = (y * y_stride + 2 * y_stride) - y_padding;
          
          // Load 3rd row with horizontal padding check
          xthalfx4 hfvecData2_0[4];
          xthalfx4 hfvecData2_1[4];
          const xthalf *p_inp_row2 = pInData + ch + in_row2 * inPitch2U;
          for (j = 0; j < 4; j++) {
            if (col_valid[j]) {
              int col_idx = base_col + j * x_stride;
              xthalfx8 *pt_inp2 = (xthalfx8 *)(p_inp_row2 + col_idx * inPitch1U);
              ae_valignx2 align2 = AE_LA128_PP(pt_inp2);
              AE_LAVHX4X2_XP(hfvecData2_0[j], hfvecData2_1[j], align2, pt_inp2, remainingCh * 2);
            } else {
              hfvecData2_0[j] = zeroVec;
              hfvecData2_1[j] = zeroVec;
            }
          }

          // Init acc and compute
          xthalfx4 hfvecAcc0[2];
          xthalfx4 hfvecAcc1[2];
          for (i = 0; i < 2; i++) {
            hfvecAcc0[i] = hfvecBias0;
            hfvecAcc1[i] = hfvecBias1;
            int d = i;
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][0], hfvecCoeff1[0][0], hfvecData0[0][d+0], hfvecData1[0][d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][1], hfvecCoeff1[0][1], hfvecData0[0][d+1], hfvecData1[0][d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][2], hfvecCoeff1[0][2], hfvecData0[0][d+2], hfvecData1[0][d+2]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][0], hfvecCoeff1[1][0], hfvecData0[1][d+0], hfvecData1[1][d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][1], hfvecCoeff1[1][1], hfvecData0[1][d+1], hfvecData1[1][d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][2], hfvecCoeff1[1][2], hfvecData0[1][d+2], hfvecData1[1][d+2]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][0], hfvecCoeff1[2][0], hfvecData2_0[d+0], hfvecData2_1[d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][1], hfvecCoeff1[2][1], hfvecData2_0[d+1], hfvecData2_1[d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][2], hfvecCoeff1[2][2], hfvecData2_0[d+2], hfvecData2_1[d+2]);
          }

          // Slide
          for (j = 0; j < 4; j++) {
            hfvecData0[0][j] = hfvecData0[1][j];
            hfvecData1[0][j] = hfvecData1[1][j];
            hfvecData0[1][j] = hfvecData2_0[j];
            hfvecData1[1][j] = hfvecData2_1[j];
          }

          // Store
          ae_valignx2 vaOutData;
          for (i = 0; i < 2; i++) {
            hfvecAcc0[i] = MAX_HX4(MIN_HX4(hfvecAcc0[i], vec_max), vec_min);
            hfvecAcc1[i] = MAX_HX4(MIN_HX4(hfvecAcc1[i], vec_max), vec_min);
            xthalfx8 *pt_out = (xthalfx8 *)(pOutData + ch + y * outPitch2U + (x + i) * outPitch1U);
            AE_SAVHX4X2_XP(hfvecAcc0[i], hfvecAcc1[i], vaOutData, pt_out, store_bytes[i]);
            AE_SA128POS_FP(vaOutData, pt_out);
          }
        }
      }

      // ===== REGION 3: Bottom padding region (y >= y_end_no_pad) =====
      if (y_end_no_pad < out_height) {
        xthalfx4 hfvecData0[2][4];
        xthalfx4 hfvecData1[2][4];
        
        // Load first 2 rows with padding check
        for (i = 0; i < 2; i++) {
          int in_row = ((y_end_no_pad + i) * y_stride) - y_padding;
          int row_valid = (in_row >= 0) && (in_row < input_height);
          const xthalf *p_inp_row = pInData + ch + in_row * inPitch2U;
          
          for (j = 0; j < 4; j++) {
            if (row_valid && col_valid[j]) {
              int col_idx = base_col + j * x_stride;
              xthalfx8 *pt_inp = (xthalfx8 *)(p_inp_row + col_idx * inPitch1U);
              ae_valignx2 align_inp = AE_LA128_PP(pt_inp);
              AE_LAVHX4X2_XP(hfvecData0[i][j], hfvecData1[i][j], align_inp, pt_inp, remainingCh * 2);
            } else {
              hfvecData0[i][j] = zeroVec;
              hfvecData1[i][j] = zeroVec;
            }
          }
        }

        for (y = y_end_no_pad; y < out_height; y++) {
          int in_row2 = (y * y_stride + 2 * y_stride) - y_padding;
          int row_valid2 = (in_row2 >= 0) && (in_row2 < input_height);

          // Load 3rd row with padding check
          xthalfx4 hfvecData2_0[4];
          xthalfx4 hfvecData2_1[4];
          const xthalf *p_inp_row2 = pInData + ch + in_row2 * inPitch2U;
          for (j = 0; j < 4; j++) {
            if (row_valid2 && col_valid[j]) {
              int col_idx = base_col + j * x_stride;
              xthalfx8 *pt_inp2 = (xthalfx8 *)(p_inp_row2 + col_idx * inPitch1U);
              ae_valignx2 align2 = AE_LA128_PP(pt_inp2);
              AE_LAVHX4X2_XP(hfvecData2_0[j], hfvecData2_1[j], align2, pt_inp2, remainingCh * 2);
            } else {
              hfvecData2_0[j] = zeroVec;
              hfvecData2_1[j] = zeroVec;
            }
          }

          // Init acc and compute
          xthalfx4 hfvecAcc0[2];
          xthalfx4 hfvecAcc1[2];
          for (i = 0; i < 2; i++) {
            hfvecAcc0[i] = hfvecBias0;
            hfvecAcc1[i] = hfvecBias1;
            int d = i;
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][0], hfvecCoeff1[0][0], hfvecData0[0][d+0], hfvecData1[0][d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][1], hfvecCoeff1[0][1], hfvecData0[0][d+1], hfvecData1[0][d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[0][2], hfvecCoeff1[0][2], hfvecData0[0][d+2], hfvecData1[0][d+2]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][0], hfvecCoeff1[1][0], hfvecData0[1][d+0], hfvecData1[1][d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][1], hfvecCoeff1[1][1], hfvecData0[1][d+1], hfvecData1[1][d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[1][2], hfvecCoeff1[1][2], hfvecData0[1][d+2], hfvecData1[1][d+2]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][0], hfvecCoeff1[2][0], hfvecData2_0[d+0], hfvecData2_1[d+0]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][1], hfvecCoeff1[2][1], hfvecData2_0[d+1], hfvecData2_1[d+1]);
            MADD_HX4X2(hfvecAcc0[i], hfvecAcc1[i], hfvecCoeff0[2][2], hfvecCoeff1[2][2], hfvecData2_0[d+2], hfvecData2_1[d+2]);
          }

          // Slide
          for (j = 0; j < 4; j++) {
            hfvecData0[0][j] = hfvecData0[1][j];
            hfvecData1[0][j] = hfvecData1[1][j];
            hfvecData0[1][j] = hfvecData2_0[j];
            hfvecData1[1][j] = hfvecData2_1[j];
          }

          // Store
          ae_valignx2 vaOutData;
          for (i = 0; i < 2; i++) {
            hfvecAcc0[i] = MAX_HX4(MIN_HX4(hfvecAcc0[i], vec_max), vec_min);
            hfvecAcc1[i] = MAX_HX4(MIN_HX4(hfvecAcc1[i], vec_max), vec_min);
            xthalfx8 *pt_out = (xthalfx8 *)(pOutData + ch + y * outPitch2U + (x + i) * outPitch1U);
            AE_SAVHX4X2_XP(hfvecAcc0[i], hfvecAcc1[i], vaOutData, pt_out, store_bytes[i]);
            AE_SA128POS_FP(vaOutData, pt_out);
          }
        }
      }
    }
  }
  return 0;
}

static void convolve_f16
  (pWORD16 __restrict__ p_out  
  ,const WORD16* __restrict__ p_ker  
  ,const WORD16* __restrict__ p_inp  
  ,WORD16 bias
  ,int input_height
  ,int input_width
  ,int kernel_height
  ,int kernel_width
  ,int actual_out_height      
  ,int actual_out_width  
  ,int out_stride
  ,int x_stride
  ,int y_stride
  ,pWORD64 __restrict__ p_scratch 
  )
{
  int kernel_width_pad = (kernel_width+3)&(~3);

  int i, j, k ,l;
  int output_height = input_height - kernel_height + 1;
  int output_width_for_x_stride_1;

  output_width_for_x_stride_1 = (1 + ((input_width - kernel_width)/1));
  output_width_for_x_stride_1 = ALIGNED_SIZE(output_width_for_x_stride_1, (ALIGNMENT/2));
  if ((actual_out_height - 1) > ((output_height + 1) / (y_stride)))
  {
    return;
  }
  if ((actual_out_width - 1) > ((output_width_for_x_stride_1 + 1) / (x_stride)))
  {
    return;
  }

  xthalfx4 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;

  xthalf *scratch_ptr = (xthalf *)p_scratch;

  xthalfx4 d_inp0, d_inp1, d_inp2, d_inp3, d_inp4;
  xthalfx4 d_inp6543, d_inp5432, d_inp4321;
  xthalfx4 d_ker0, d_ker1, d_ker2;

  xthalfx4 y0, y1, y01, y2, y3, y23, y02, y13;
  
  ae_int16x4 dsel0 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07060504, 0x03020100));
  ae_int16x4 dsel1 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x06050504, 0x04030302));

  if(kernel_width_pad == 12)
  {
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (xthalf *) p_scratch + (i * output_width_for_x_stride_1);
      int temp = output_width_for_x_stride_1 -output_width_for_x_stride_1%8;
      for(j = 0; j < temp; j+=8)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);
        CONST_HX4X2(acc4, acc5, 0);
        CONST_HX4X2(acc6, acc7, 0);
        for(k=0; k < kernel_height; k++)
        {
          ae_int16x4 *pt16x4_inp = (ae_int16x4 *)p_inp;
          AE_ADDCIRC16X4_XC(pt16x4_inp,((sizeof(xthalf)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx8 *pt_inp = (xthalfx8 *)(pt16x4_inp);
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);

          AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
          AE_LHX4X2_XC(d_inp2, d_inp3, pt_inp, 16);
          xthalfx4 *ptx4_inp = (xthalfx4 *)pt_inp;
          AE_LHX4XC(d_inp4, ptx4_inp, sizeof(xthalf)*(input_width-16));
          pt_inp = (xthalfx8 *)(ptx4_inp);

          AE_LHX4IP(d_ker0, pt_ker, 8);
          
          DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

          MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

          DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);

          MADDQ_H(acc4, acc5, d_inp1, d_inp6543, d_ker0);
          MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker0);

          AE_LHX4IP(d_ker1, pt_ker, 8);

          MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);

          DSELHX4(d_inp6543, d_inp5432, d_inp2, d_inp3, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp2, d_inp3);      

          MADDQ_H(acc4, acc5, d_inp2, d_inp6543, d_ker1);
          MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker1);

          AE_LHX4IP(d_ker2, pt_ker, 8);

          MADDQ_H(acc0, acc1, d_inp2, d_inp6543, d_ker2);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker2);

          DSELHX4(d_inp6543, d_inp5432, d_inp3, d_inp4, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp3, d_inp4); 

          MADDQ_H(acc4, acc5, d_inp3, d_inp6543, d_ker2);
          MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker2);
        }
  
        xthalfx4 *scratch_j = (xthalfx4 *)((WORD16 *)scratch_ptr + j);
        
        xthalfx4 out0, out1;
        CONST_HX4X2(out0, out1, 0);
        
        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);   
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);     
        DSELHX4(y02, y13, y01, y23, dsel0);
        DSELHX4(y0, y1, acc4, acc5, dsel0);     
        DSELHX4(y2, y3, acc6, acc7, dsel0);    
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);  
        xthalfx4 y02_1, y13_1;    
        DSELHX4(y02_1, y13_1, y01, y23, dsel0);

        ADD_HX4X2(out0, out1, out0, out1, y02, y02_1);
        ADD_HX4X2(out0, out1, out0, out1, y13, y13_1);

        AE_SHX4IP(out0, scratch_j, 4 * sizeof(xthalf));
        AE_SHX4IP(out1, scratch_j, 4 * sizeof(xthalf));
      }

      for(j=temp; j < output_width_for_x_stride_1; j+=4)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);        
        for(k=0; k < kernel_height; k++)
        {
          ae_int16x4 *pt16x4_inp = (ae_int16x4 *)p_inp;
          AE_ADDCIRC16X4_XC(pt16x4_inp,((sizeof(WORD16)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx8 *pt_inp = (xthalfx8 *)(pt16x4_inp);
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);
          AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
          AE_LHX4X2_XC(d_inp2, d_inp3, pt_inp, sizeof(WORD16)*(input_width-8));

          AE_LHX4IP(d_ker0, pt_ker, 8);
          DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

          MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

          AE_LHX4IP(d_ker1, pt_ker, 8);     
          DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);    
          d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);
          
          MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);   

          AE_LHX4IP(d_ker2, pt_ker, 8);     
          DSELHX4(d_inp6543, d_inp5432, d_inp2, d_inp3, dsel1);    
          d_inp4321 = AE_SELH_4321(d_inp2, d_inp3);
          
          MADDQ_H(acc0, acc1, d_inp2, d_inp6543, d_ker1);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);           

        }
        xthalfx4 *scratch_j = (xthalfx4 *)((WORD16 *)scratch_ptr + j);

        xthalfx4 out0 = CONST_HX4(0);

        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);        
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);
        DSELHX4(y02, y13, y01, y23, dsel0);
        out0 = ADD_HX4(out0, y02);
        out0 = ADD_HX4(out0, y13);

        AE_SHX4IP(out0, scratch_j, 4 * sizeof(xthalf));
      }
    }
  }
  else if(kernel_width_pad == 8)
  {
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (xthalf *) p_scratch + (i * output_width_for_x_stride_1);
      int temp = output_width_for_x_stride_1 -output_width_for_x_stride_1%8;
      for(j = 0; j < temp; j+=8)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);
        CONST_HX4X2(acc4, acc5, 0);
        CONST_HX4X2(acc6, acc7, 0);
        for(k=0; k < kernel_height; k++)
        {
          ae_int16x4 *pt16x4_inp = (ae_int16x4 *)p_inp;
          AE_ADDCIRC16X4_XC(pt16x4_inp,((sizeof(xthalf)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx8 *pt_inp = (xthalfx8 *)(pt16x4_inp);
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);

          AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
          AE_LHX4X2_XC(d_inp2, d_inp3, pt_inp, sizeof(xthalf)*(input_width-8));
          AE_LHX4IP(d_ker0, pt_ker, 8);
          
          DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

          MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

          DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);

          MADDQ_H(acc4, acc5, d_inp1, d_inp6543, d_ker0);
          MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker0);

          AE_LHX4IP(d_ker1, pt_ker, 8);

          MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);

          DSELHX4(d_inp6543, d_inp5432, d_inp2, d_inp3, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp2, d_inp3);      

          MADDQ_H(acc4, acc5, d_inp2, d_inp6543, d_ker1);
          MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker1);
        }
  
        xthalfx4 *scratch_j = (xthalfx4 *)((WORD16 *)scratch_ptr + j);
        
        xthalfx4 out0, out1;
        CONST_HX4X2(out0, out1, 0);
        
        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);   
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);     
        DSELHX4(y02, y13, y01, y23, dsel0);
        DSELHX4(y0, y1, acc4, acc5, dsel0);     
        DSELHX4(y2, y3, acc6, acc7, dsel0);    
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);  
        xthalfx4 y02_1, y13_1;    
        DSELHX4(y02_1, y13_1, y01, y23, dsel0);

        ADD_HX4X2(out0, out1, out0, out1, y02, y02_1);
        ADD_HX4X2(out0, out1, out0, out1, y13, y13_1);

        AE_SHX4IP(out0, scratch_j, 4 * sizeof(xthalf));
        AE_SHX4IP(out1, scratch_j, 4 * sizeof(xthalf));
      }

      for(j=temp; j < output_width_for_x_stride_1; j+=4)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);        
        for(k=0; k < kernel_height; k++)
        {
          ae_int16x4 *pt16x4_inp = (ae_int16x4 *)p_inp;
          AE_ADDCIRC16X4_XC(pt16x4_inp,((sizeof(WORD16)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx8 *pt_inp = (xthalfx8 *)(pt16x4_inp);
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);
          AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
          d_inp2 = AE_LHX4I((xthalfx4 *)pt_inp, 0);
          AE_LHX4IP(d_ker0, pt_ker, 8);

          DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

          MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

          AE_LHX4IP(d_ker1, pt_ker, 8);     
          DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);    
          d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);
          
          MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);         
        }
        xthalfx4 *scratch_j = (xthalfx4 *)((WORD16 *)scratch_ptr + j);

        xthalfx4 out0 = CONST_HX4(0);
        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);        
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);
        DSELHX4(y02, y13, y01, y23, dsel0);
        out0 = ADD_HX4(out0, y02);
        out0 = ADD_HX4(out0, y13);

        AE_SHX4IP(out0, scratch_j, 4 * sizeof(xthalf));
      }
    }
  }
  else if(kernel_width_pad == 4)
  {
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (xthalf *) p_scratch + (i * output_width_for_x_stride_1);
      for(j=0; j < output_width_for_x_stride_1; j+=4)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);        
        for(k=0; k < kernel_height; k++)
        {
          ae_int16x4 *pt16x4_inp = (ae_int16x4 *)p_inp;
          AE_ADDCIRC16X4_XC(pt16x4_inp,((sizeof(WORD16)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx8 *pt_inp = (xthalfx8 *)(pt16x4_inp);
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);
          xthalfx4 *ptx4_inp = (xthalfx4 *)pt_inp;
          AE_LHX4XC(d_inp0, ptx4_inp, 8);
          AE_LHX4XC(d_inp1, ptx4_inp, sizeof(xthalf)*(input_width-4)); 

          AE_LHX4IP(d_ker0, pt_ker, 8);
          DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

          MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);       
        }
        xthalfx4 *scratch_j = (xthalfx4 *)((WORD16 *)scratch_ptr + j);

        xthalfx4 out0 = CONST_HX4(0);
        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);        
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);
        DSELHX4(y02, y13, y01, y23, dsel0);
        out0 = ADD_HX4(out0, y02);
        out0 = ADD_HX4(out0, y13);
        AE_SHX4IP(out0, scratch_j, 4 * sizeof(xthalf));
      }    
    }
  }
  else
  {
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (xthalf *) p_scratch + (i * output_width_for_x_stride_1);
      int temp = output_width_for_x_stride_1 -output_width_for_x_stride_1%8;
      for(j = 0; j < temp; j+=8)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);
        CONST_HX4X2(acc4, acc5, 0);
        CONST_HX4X2(acc6, acc7, 0);    
        for(k=0; k < kernel_height; k++)
        {
          ae_int16x4 *pt16x4_inp = (ae_int16x4 *)p_inp;
          AE_ADDCIRC16X4_XC(pt16x4_inp,((sizeof(xthalf)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx8 *pt_inp = (xthalfx8 *)(pt16x4_inp);
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);
  #pragma no_unroll
          for(l = 0; l < (kernel_width_pad>>3); l++)
          {
            AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
            AE_LHX4X2_I(d_inp2, d_inp3, pt_inp, 0);
            AE_LHX4IP(d_ker0, pt_ker, 8);
            
            DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

            MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

            DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);

            MADDQ_H(acc4, acc5, d_inp1, d_inp6543, d_ker0);
            MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker0);

            AE_LHX4IP(d_ker1, pt_ker, 8);

            MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);

            DSELHX4(d_inp6543, d_inp5432, d_inp2, d_inp3, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp2, d_inp3);      

            MADDQ_H(acc4, acc5, d_inp2, d_inp6543, d_ker1);
            MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker1);
          }
          if(kernel_width_pad&7)
          {
            AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
            d_inp2 = AE_LHX4I((xthalfx4 *)pt_inp, 0);
            d_ker0 = AE_LHX4I(pt_ker, 0);      

            DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

            MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

            DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);

            MADDQ_H(acc4, acc5, d_inp1, d_inp6543, d_ker0);
            MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker0);
          }
        }
  
        xthalfx4 *scratch_j = (xthalfx4 *)((WORD16 *)scratch_ptr + j);
        
        xthalfx4 out0, out1;
        CONST_HX4X2(out0, out1, 0);
        
        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);   
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);     
        DSELHX4(y02, y13, y01, y23, dsel0);
        DSELHX4(y0, y1, acc4, acc5, dsel0);     
        DSELHX4(y2, y3, acc6, acc7, dsel0);    
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);  
        xthalfx4 y02_1, y13_1;    
        DSELHX4(y02_1, y13_1, y01, y23, dsel0);

        ADD_HX4X2(out0, out1, out0, out1, y02, y02_1);
        ADD_HX4X2(out0, out1, out0, out1, y13, y13_1);

        AE_SHX4IP(out0, scratch_j, 4 * sizeof(xthalf));
        AE_SHX4IP(out1, scratch_j, 4 * sizeof(xthalf));
      }

      for(j=temp; j < output_width_for_x_stride_1; j+=4)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);
        for(k=0; k < kernel_height; k++)
        {
          ae_int16x4 *pt16x4_inp = (ae_int16x4 *)p_inp;
          AE_ADDCIRC16X4_XC(pt16x4_inp,((sizeof(WORD16)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx8 *pt_inp = (xthalfx8 *)(pt16x4_inp);
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);
          for(l = 0; l < (kernel_width_pad>>3); l++)
          {
            AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
            d_inp2 = AE_LHX4I((xthalfx4 *)pt_inp, 0);
            AE_LHX4IP(d_ker0, pt_ker, 8);

            DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

            MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

            AE_LHX4IP(d_ker1, pt_ker, 8);     
            DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);    
            d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);
            
            MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);         
          }
          if(kernel_width_pad&7)
          {
            AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
            d_ker0 = AE_LHX4I(pt_ker, 0); 

            DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

            MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);
          }
        }
        xthalfx4 *scratch_j = (xthalfx4 *)((WORD16 *)scratch_ptr + j);

        xthalfx4 out0 = CONST_HX4(0);

        DSELHX4(y0, y1, acc0, acc1, dsel0);
        y01  = ADD_HX4(y0, y1); 
        DSELHX4(y2, y3, acc2, acc3, dsel0);       
        y23  = ADD_HX4(y2, y3);   
        DSELHX4(y02, y13, y01, y23, dsel0);
        out0 = ADD_HX4(out0, y02);
        out0 = ADD_HX4(out0, y13);

        AE_SHX4IP(out0, scratch_j, 4 * sizeof(xthalf));
      }
    }
  }

  /* Here we store output based on strides. For values in a row, values
   * will be picked from it as per 'x_stride'. No need to worry about
   * height dimension, since we took care of it by efficient row
   * accesses. */

  xthalf acc_scratch;
  scratch_ptr = (xthalf *) p_scratch;
  for(i = 0; i < actual_out_height; i++)
  {
    scratch_ptr = (xthalf *) p_scratch + (i * output_width_for_x_stride_1);
    xthalf *out_ptr  = (xthalf *) p_out + (i * out_stride * actual_out_width);
    
    xthalf b0 = AE_LHI(((xthalf *)&bias), 0);

    for(j = 0; j < actual_out_width; j++)
    {
      acc_scratch = AE_LHX(scratch_ptr, (sizeof(xthalf) * (j * x_stride)));
      acc_scratch = ADD_H(acc_scratch, b0);
      AE_SHX(acc_scratch, out_ptr, (sizeof(xthalf) * (j * out_stride)));
    }
  }  
}

#define COPY_KERNEL_TO_SCRATCH(p_out, p_in, kh, kw, kw_pad) \
{ \
  int itr_kh, itr_kw; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    xthalfx4 *pae_in = (xthalfx4 *)(&p_in[itr_kh * kw]); \
    xthalfx4 *pae_out = (xthalfx4 *)(&p_out[itr_kh * kw_pad]); \
    xthalfx4 d_tmp0; \
    ae_valign in_a = AE_LAHX4PP(pae_in); \
_Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < (kw >> 2); itr_kw++) \
    { \
      AE_LAHX4IP(d_tmp0, in_a, pae_in); \
      AE_SHX4IP(d_tmp0, pae_out, 4*sizeof(xthalf)); \
    } \
    if(kw & 3) \
    { \
      AE_LAHX4IP(d_tmp0, in_a, pae_in); \
      ae_int64 d_tmp64 = AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4 (d_tmp0)); \
      d_tmp64 = AE_SRAA64(d_tmp64, 16 * (4 - (kw & 3))); \
      d_tmp64 = AE_SLAA64(d_tmp64, 16 * (4 - (kw & 3))); \
      d_tmp0 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT64(d_tmp64)); \
      AE_SHX4IP(d_tmp0, pae_out, 4*sizeof(xthalf)); \
    } \
  } \
}

WORD32 xa_nn_conv2d_depthwise_nchw_f16(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  channels_multiplier,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
    WORD32  out_data_format,
    pVOID p_scratch)
{
    (VOID) out_data_format;
    WORD16 pad_val = 0;
    xa_nn_dilated_conv2d_depthwise_init
        (p_scratch
        ,input_height
        ,input_width
        ,input_channels
        ,kernel_height
        ,kernel_width
        ,channels_multiplier
        ,1
        ,1
        ,x_stride
        ,y_stride
        ,x_padding
        ,y_padding
        ,out_height
        ,out_width
        ,-2
        ,1
        ,(pVOID)(&pad_val)
        );

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ic, itr_cm, itr_oh;
    int circ_out_height = (p_circ_buf->rows - kernel_height)/y_stride + 1;
    int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
    int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
    int rows_to_add, top_pad, bottom_pad, rows_added;
    int input_row;
    const WORD16 *pt_ker;
    const WORD16 *pt_inp;
    pWORD16 p_inp_circ;
    int i;
    WORD16 *p_kernel_padded = (WORD16 *)(p_state->p_scratch);
    p_kernel_padded = (WORD16 *)ALIGN_PTR(p_kernel_padded, 8);
    pWORD64 p_tmp_out = (pWORD64)(p_kernel_padded + kernel_height_pad * kernel_width_pad);
    p_tmp_out = (pWORD64)ALIGN_PTR(p_tmp_out, 16);

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    WORD16 bias = 0;
    /* Initialize whole scratch for padded kernel to padding value, after this
     we only have to copy actual kernel values, padding area should remain
     untouched */
    xthalfx4 *pae_ker_pad = (xthalfx4 *)p_kernel_padded;

    for(i = 0; i < ((kernel_height_pad * kernel_width_pad) >> 2); i++)
    {
        pae_ker_pad[i] = ZERO_HX4();
    }
    
    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = &p_inp[itr_ic*input_height*input_width];
        for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
        {
            pt_ker = &p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
            COPY_KERNEL_TO_SCRATCH(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
            bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

            CIRC_BUF_ADD_ROWS_INIT(rows_added
                                   ,rows_to_add
                                   ,top_pad
                                   ,bottom_pad
                                   ,input_row
                                   ,input_height
                                   ,input_width
                                   ,kernel_height
                                   ,y_stride
                                   ,x_padding
                                   ,y_padding
                                   ,p_circ_buf
                                   ,pt_inp
                                   );

            for(itr_oh = 0; itr_oh < out_height - (circ_out_height - 1); itr_oh += circ_out_height)
            {
                CIRC_BUF_ADD_ROWS(rows_added
                              ,rows_to_add
                              ,top_pad
                              ,bottom_pad
                              ,input_row
                              ,input_height
                              ,input_width
                              ,circ_out_height
                              ,y_stride
                              ,x_padding
                              ,y_padding
                              ,p_circ_buf
                              ,pt_inp
                              );
                              
                p_inp_circ = (WORD16 *)p_circ_buf->p_curr;
                convolve_f16
                ((&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
                            ,p_kernel_padded
                            ,p_inp_circ
                            ,bias
                            ,p_circ_buf->rows
                            ,p_circ_buf->row_offset
                            ,kernel_height
                            ,kernel_width
                            ,circ_out_height
                            ,out_width
                            ,(input_channels * channels_multiplier)
                            ,x_stride
                            ,y_stride
                            ,p_tmp_out
                            );
            }

            CIRC_BUF_ADD_ROWS(rows_added
                              ,rows_to_add
                              ,top_pad
                              ,bottom_pad
                              ,input_row
                              ,input_height
                              ,input_width
                              ,circ_out_height
                              ,y_stride
                              ,x_padding
                              ,y_padding
                              ,p_circ_buf
                              ,pt_inp
                              );

            p_inp_circ = (WORD16 *)p_circ_buf->p_curr;
            convolve_f16
            ((&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
                        ,p_kernel_padded
                        ,p_inp_circ
                        ,bias
                        ,p_circ_buf->rows
                        ,p_circ_buf->row_offset
                        ,kernel_height
                        ,kernel_width
                        ,(out_height - itr_oh)
                        ,out_width
                        ,(input_channels * channels_multiplier)                      
                        ,x_stride
                        ,y_stride
                        ,p_tmp_out
                        );
        }
    }

    return 0;
}

static inline void conv2d_nhwc_f16
(pWORD16 __restrict__ p_out
 ,const WORD16 *__restrict__ p_ker
 ,const WORD16 *__restrict__ p_inp
 ,const WORD16 *p_bias
 ,int kernel_height
 ,int kernel_width
 ,int out_height
 ,int out_width
 ,int out_channels
 ,int x_stride
 ,int y_stride
 ,pWORD32 __restrict__ p_scratch
 ,const WORD16 *act_min
 ,const WORD16 *act_max
 )
{
    (VOID) x_stride;
    (VOID) p_scratch;
    xthalfx4 vec_min = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(
        (act_min != NULL) ? *act_min : (WORD16)0xFC00)));
    xthalfx4 vec_max = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(
        (act_max != NULL) ? *act_max : (WORD16)0x7C00)));
    WORD32 ker_channels_pad, inp_channels_pad;
    WORD32 itr_oh, itr_ch, itr_kw;
    xthalfx4 *pt_inp0, *pt_inp1, *pt_ker;
    xthalfx8 *out_ptr0, *out_ptr1;

    xthalfx4 d_inp0, d_inp1, d_ker;

    const xthalfx4 *pt_bias;
    ae_valign ker_a;
    ae_valign bias_a;
    ae_valignx2 out0_a, out1_a;

    xthalfx4 d_acc0123, d_acc4567;
    xthalfx4 d_acc0123_1, d_acc4567_1;
    xthalfx4 d_bias0123, d_bias0123_1;

    ker_channels_pad = out_channels;
    inp_channels_pad = (out_channels + 3) & (~3);

    for(itr_oh = 0; itr_oh < (out_height-1); itr_oh+=2)
    {
      out_ptr0 = (xthalfx8 *)(&p_out[itr_oh*out_channels*out_width]);
      out_ptr1 = (xthalfx8 *)(&p_out[(itr_oh+1)*out_channels*out_width]);
      pt_bias = (xthalfx4 *)p_bias;
      bias_a = AE_LAHX4PP(pt_bias);
      out0_a = AE_ZALIGN128();
      out1_a = AE_ZALIGN128();        

      if ((inp_channels_pad%8) == 0)
      {
        for(itr_ch = 0; itr_ch < out_channels; itr_ch+=8)
        {
            xthalfx4 d_ker1;
            ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)p_inp;
            ae_int16x4 *pt16x4_inp1 = (ae_int16x4 *)p_inp;
            AE_ADDCIRC16X4_XC(pt16x4_inp0, (itr_ch + itr_oh*y_stride*kernel_width*inp_channels_pad)*sizeof(xthalf));
            AE_ADDCIRC16X4_XC(pt16x4_inp1, (itr_ch + (itr_oh+1)*y_stride*kernel_width*inp_channels_pad)*sizeof(xthalf));
            xthalfx8 *pt_kerx2 = (xthalfx8 *)(&p_ker[itr_ch]);
            ae_valignx2 ker_ax2 = AE_LA128_PP(pt_kerx2);
            
            AE_LAHX4IP(d_bias0123, bias_a, pt_bias);            
            AE_LAHX4IP(d_bias0123_1, bias_a, pt_bias);            
            d_acc0123 = d_bias0123;
            d_acc4567 = d_bias0123;
            d_acc0123_1 = d_bias0123_1;
            d_acc4567_1 = d_bias0123_1;
			
            xthalfx8 *ptx8_inp0 = (xthalfx8 *)pt16x4_inp0;
            xthalfx8 *ptx8_inp1 = (xthalfx8 *)pt16x4_inp1;
            for(itr_kw = 0; itr_kw < kernel_width * kernel_height; itr_kw++)
            {
                xthalfx4 d_inp01, d_inp11;
                AE_LHX4X2_XC(d_inp0, d_inp01, ptx8_inp0, inp_channels_pad*sizeof(xthalf));
                AE_LHX4X2_XC(d_inp1, d_inp11, ptx8_inp1, inp_channels_pad*sizeof(xthalf));

                AE_LAHX4X2_IP(d_ker, d_ker1, ker_ax2, pt_kerx2);
            
                pt_kerx2 = (xthalfx8 *)((WORD8 *)pt_kerx2 + sizeof(xthalf) * (ker_channels_pad - 8));
                ker_ax2 = AE_LA128_PP(pt_kerx2);
                MADDQ_H(d_acc0123, d_acc4567, d_inp0, d_inp1, d_ker);
                MADDQ_H(d_acc0123_1, d_acc4567_1, d_inp01, d_inp11, d_ker1);
            }
            d_acc0123 = MAX_HX4(MIN_HX4(d_acc0123, vec_max), vec_min);
            d_acc0123_1 = MAX_HX4(MIN_HX4(d_acc0123_1, vec_max), vec_min);
            d_acc4567 = MAX_HX4(MIN_HX4(d_acc4567, vec_max), vec_min);
            d_acc4567_1 = MAX_HX4(MIN_HX4(d_acc4567_1, vec_max), vec_min);
            AE_SAVHX4X2_XP(d_acc0123, d_acc0123_1, out0_a, out_ptr0, (XT_MIN(out_channels-itr_ch, 8) << 1));
            AE_SAVHX4X2_XP(d_acc4567, d_acc4567_1, out1_a, out_ptr1, (XT_MIN(out_channels-itr_ch, 8) << 1));   
        }
      }
      else 
      {
        for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
        {
            ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)p_inp;
            ae_int16x4 *pt16x4_inp1 = (ae_int16x4 *)p_inp;
            AE_ADDCIRC16X4_XC(pt16x4_inp0, (itr_ch + itr_oh*y_stride*kernel_width*inp_channels_pad)*sizeof(xthalf));
            AE_ADDCIRC16X4_XC(pt16x4_inp1, (itr_ch + (itr_oh+1)*y_stride*kernel_width*inp_channels_pad)*sizeof(xthalf));
            pt_inp0 = (xthalfx4 *)pt16x4_inp0;
            pt_inp1 = (xthalfx4 *)pt16x4_inp1;
            pt_ker = (xthalfx4 *)(&p_ker[itr_ch]);
            ker_a = AE_LAHX4PP(pt_ker);
            
            AE_LAHX4IP(d_bias0123, bias_a, pt_bias);            
            d_acc0123 = d_bias0123;
            d_acc4567 = d_bias0123;

            for(itr_kw = 0; itr_kw < kernel_width * kernel_height; itr_kw++)
            {
                AE_LHX4XC(d_inp0, pt_inp0, inp_channels_pad*sizeof(xthalf));
                AE_LHX4XC(d_inp1, pt_inp1, inp_channels_pad*sizeof(xthalf));

                AE_LAHX4IP(d_ker, ker_a, pt_ker);
            
                pt_ker = (xthalfx4 *)((WORD8 *)pt_ker + sizeof(xthalf) * (ker_channels_pad - 4));
                ker_a = AE_LAHX4PP(pt_ker);
                MADDQ_H(d_acc0123, d_acc4567, d_inp0, d_inp1, d_ker);
            }
            d_acc0123 = MAX_HX4(MIN_HX4(d_acc0123, vec_max), vec_min);
            d_acc4567 = MAX_HX4(MIN_HX4(d_acc4567, vec_max), vec_min);
            AE_SAVHX4X2_XP(d_acc0123, ZERO_HX4(), out0_a, out_ptr0, (XT_MIN(out_channels-itr_ch, 4) << 1));
            AE_SAVHX4X2_XP(d_acc4567, ZERO_HX4(), out1_a, out_ptr1, (XT_MIN(out_channels-itr_ch, 4) << 1));   
        }
      }

      AE_SA128POS_FP(out0_a, out_ptr0);
      AE_SA128POS_FP(out1_a, out_ptr1);
    }
    if(itr_oh < out_height)
    {
        out_ptr0 = (xthalfx8 *)(&p_out[itr_oh*out_channels*out_width]);
        pt_bias = (const xthalfx4 *)p_bias;
        bias_a = AE_LAHX4PP(pt_bias);   
        out0_a = AE_ZALIGN128();     
        for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
        {
            ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)p_inp;
            AE_ADDCIRC16X4_XC(pt16x4_inp0, (itr_ch + itr_oh*y_stride*kernel_width*inp_channels_pad)*sizeof(xthalf));
            pt_inp0 = (xthalfx4 *)pt16x4_inp0;
            pt_ker = (xthalfx4 *)(&p_ker[itr_ch]);
            ker_a = AE_LAHX4PP(pt_ker);
            AE_LAHX4IP(d_bias0123, bias_a, pt_bias);
            d_acc0123 = d_bias0123;

            for(itr_kw = 0; itr_kw < kernel_width * kernel_height; itr_kw++)
            {
                AE_LHX4XC(d_inp0, pt_inp0, inp_channels_pad*sizeof(xthalf));
                AE_LAHX4IP(d_ker, ker_a, pt_ker);                 

                pt_ker = (xthalfx4 *)((WORD8 *)pt_ker + sizeof(xthalf) * (ker_channels_pad - 4));
                ker_a = AE_LAHX4PP(pt_ker);
                MADD_HX4(d_acc0123, d_ker, d_inp0);
            }
            d_acc0123 = MAX_HX4(MIN_HX4(d_acc0123, vec_max), vec_min);
            AE_SAVHX4X2_XP(d_acc0123, ZERO_HX4(), out0_a, out_ptr0, (XT_MIN(out_channels-itr_ch, 4) << 1));
        }
        AE_SA128POS_FP(out0_a, out_ptr0);
    }
}

static void xa_nn_conv2d_depthwise_nhwc_f16
(pWORD16 __restrict__ p_out
 ,const WORD16 *__restrict__ p_kernel
 ,const WORD16 *__restrict__ p_inp
 ,const WORD16 *__restrict__ p_bias
 ,WORD32  input_height
 ,WORD32  input_width
 ,WORD32  input_channels
 ,WORD32  kernel_height
 ,WORD32  kernel_width
 ,WORD32  channels_multiplier
 ,WORD32  x_stride
 ,WORD32  y_stride
 ,WORD32  x_padding
 ,WORD32  y_padding
 ,WORD32  out_height
 ,WORD32  out_width
 ,WORD32  out_data_format
 ,pVOID p_scratch
 ,const WORD16 *act_min
 ,const WORD16 *act_max
)
{
    (VOID) out_data_format;
    WORD16 pad_val = 0;
    xa_nn_dilated_conv2d_depthwise_init
        (p_scratch
         ,input_height
         ,input_width
         ,input_channels
         ,kernel_height
         ,kernel_width
         ,channels_multiplier
         ,1
         ,1
         ,x_stride
         ,y_stride
         ,x_padding
         ,y_padding
         ,out_height
         ,out_width
         ,-2
         ,0
         ,(pVOID)(&pad_val)
        );

    xa_nn_circ_buf_t *p_state = (xa_nn_circ_buf_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = p_state;
    int itr_ow;
    int cols_to_add, left_pad, right_pad, cols_added;
    int input_col;
    const WORD16 *pt_inp;
    pWORD16 p_inp_circ;

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    pt_inp = (const WORD16 *)p_inp;

    CIRC_BUF_ADD_COLS_INIT(cols_added
            ,cols_to_add
            ,left_pad
            ,right_pad
            ,input_col
            ,input_height
            ,input_width
            ,input_channels
            ,kernel_height
            ,kernel_width
            ,channels_multiplier
            ,x_stride
            ,x_padding
            ,y_padding
            ,out_height
            ,p_circ_buf
            ,pt_inp
            );

    for(itr_ow = 0; itr_ow < out_width; itr_ow++)
    {
        CIRC_BUF_ADD_COLS(cols_added
                ,cols_to_add
                ,left_pad
                ,right_pad
                ,input_col
                ,input_height
                ,input_width
                ,input_channels
                ,kernel_height
                ,kernel_width
                ,channels_multiplier
                ,x_stride
                ,x_padding
                ,y_padding
                ,out_height
                ,p_circ_buf
                ,pt_inp
                );

        p_inp_circ = (WORD16 *)p_circ_buf->p_curr;

        conv2d_nhwc_f16
            ((pWORD16)(&p_out[itr_ow*input_channels*channels_multiplier])
             ,p_kernel
             ,p_inp_circ
             ,p_bias
             ,kernel_height
             ,kernel_width
             ,out_height
             ,out_width
             ,(input_channels * channels_multiplier)
             ,x_stride
             ,y_stride
             ,p_scratch
             ,act_min
             ,act_max
            );
    }
}

WORD32 xa_nn_conv2d_depthwise_f16(
        WORD16* __restrict__ p_out,
        const WORD16* __restrict__ p_kernel,
        const WORD16* __restrict__ p_inp,
        const WORD16* __restrict__ p_bias,
        WORD32  input_height,
        WORD32  input_width,
        WORD32  input_channels,
        WORD32  kernel_height,
        WORD32  kernel_width,
        WORD32  channels_multiplier,
        WORD32  x_stride,
        WORD32  y_stride,
        WORD32  x_padding,
        WORD32  y_padding,
        WORD32  out_height,
        WORD32  out_width,
        WORD32  inp_data_format,
        WORD32  out_data_format,
        pVOID p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

    if(inp_data_format == 0){ 
      if (kernel_height == 3 && kernel_width == 3 && x_stride == 1 && y_stride == 1){
        xa_nn_conv2d_depthwise_f16_3x3_with_padding(
          p_out, p_kernel, p_inp, p_bias,
          input_height, input_width, input_channels,
          kernel_height, kernel_width, channels_multiplier,
          x_stride, y_stride, x_padding, y_padding,
          out_height, out_width, inp_data_format, out_data_format, p_scratch,
          NULL, NULL);
      }
      else{
        xa_nn_conv2d_depthwise_nhwc_f16
            (p_out
             ,p_kernel
             ,p_inp
             ,p_bias
             ,input_height
             ,input_width
             ,input_channels
             ,kernel_height
             ,kernel_width
             ,channels_multiplier
             ,x_stride
             ,y_stride
             ,x_padding
             ,y_padding
             ,out_height
             ,out_width
             ,out_data_format
             ,p_scratch
             ,NULL
             ,NULL);      
      }
    }
    else if(inp_data_format == 1)
    {
        xa_nn_conv2d_depthwise_nchw_f16(
                p_out,
                p_kernel,
                p_inp,
                p_bias,
                input_height,
                input_width,
                input_channels,
                kernel_height,
                kernel_width,
                channels_multiplier,
                x_stride,
                y_stride,
                x_padding,
                y_padding,
                out_height,
                out_width,
                out_data_format,
                p_scratch);
    }
    return 0;
}


static WORD32 gcd_f16(WORD32 a, WORD32 b)
{
    while (a != b)
    {
        if (a > b)
        {
            return gcd_f16(a - b, b);
        }
        else
        {
            return gcd_f16(a, b - a);
        }
    }
    return a;
}

static void xa_nn_dilated_conv2d_depthwise_nhwc_f16
    (WORD16 *__restrict__ p_out
    ,const WORD16 *__restrict__ p_kernel
    ,const WORD16 *__restrict__ p_inp
    ,const WORD16 *__restrict__ p_bias
    ,WORD32  input_height
    ,WORD32  input_width
    ,WORD32  input_channels
    ,WORD32  kernel_height
    ,WORD32  kernel_width
    ,WORD32  channels_multiplier
    ,WORD32  dilation_height
    ,WORD32  dilation_width
    ,WORD32  x_stride
    ,WORD32  y_stride
    ,WORD32  x_padding
    ,WORD32  y_padding
    ,WORD32  out_height
    ,WORD32  out_width
    ,WORD32  out_data_format
    ,pVOID p_scratch
    )
{
    (VOID) out_data_format;

    WORD16 pad_val = 0;
    xa_nn_dilated_conv2d_depthwise_init
        (p_scratch
        ,input_height
        ,input_width
        ,input_channels
        ,kernel_height
        ,kernel_width
        ,channels_multiplier
        ,dilation_height
        ,dilation_width
        ,x_stride
        ,y_stride
        ,x_padding
        ,y_padding
        ,out_height
        ,out_width
        ,-2
        ,0
        ,(pVOID)(&pad_val)
        );

    xa_nn_circ_buf_t *p_state = (xa_nn_circ_buf_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = p_state;

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    int itr_ow;
    int itr_dh, itr_dw;
    int cols_to_add, left_pad, right_pad, cols_added;
    int input_col;
    const WORD16 *pt_inp;
    WORD16 *p_inp_circ;

    pt_inp = (const WORD16 *)p_inp;

    WORD32 dh_count, dw_count;
    WORD32 y_padding_dh, x_padding_dw;
    WORD32 x_stride_dw;
    WORD32 out_height_dh, out_width_dw;
    WORD32 rem_dh, rem_dw;
    WORD32 gcd_h, gcd_w;
    WORD32 y_stride_circ_buf;

    gcd_h = gcd_f16(dilation_height, y_stride);
    gcd_w = gcd_f16(dilation_width, x_stride);
    dh_count = dilation_height / gcd_h;
    dw_count = dilation_width / gcd_w;
    y_padding_dh = y_padding;
    out_height_dh = out_height / dh_count;
    out_width_dw = out_width / dw_count;
    rem_dh = out_height - out_height_dh * dh_count;
    y_stride_circ_buf = y_stride / gcd_h;

    for(itr_dh = 0; itr_dh < dh_count; itr_dh++, rem_dh--)
    {
        x_padding_dw = x_padding;
        x_stride_dw = x_stride * dw_count;
        rem_dw = out_width - out_width_dw * dw_count;

        WORD32 out_height_dh_cur = out_height_dh + (rem_dh > 0 ? 1 : 0);
        for(itr_dw = 0; itr_dw < dw_count; itr_dw++, rem_dw--)
        {
            WORD32 out_width_dw_cur = out_width_dw + (rem_dw > 0 ? 1 : 0);
            DILATED_CIRC_BUF_ADD_COLS_INIT(
                    cols_added,
                    cols_to_add,
                    left_pad,
                    right_pad,
                    input_col,
                    input_height,
                    input_width,
                    input_channels,
                    kernel_height,
                    kernel_width,
                    channels_multiplier,
                    dilation_height,
                    dilation_width,
                    x_stride_dw,
                    y_stride_circ_buf,
                    x_padding_dw,
                    y_padding_dh,
                    out_height_dh_cur,
                    p_circ_buf,
                    pt_inp);

            for(itr_ow = 0; itr_ow < out_width_dw_cur; itr_ow++)
            {
                WORD16 *pt_out = (WORD16 *)&p_out[(itr_dh * out_width + itr_dw + itr_ow * dw_count) * input_channels * channels_multiplier];
                DILATED_CIRC_BUF_ADD_COLS(
                        cols_added,
                        cols_to_add,
                        left_pad,
                        right_pad,
                        input_col,
                        input_height,
                        input_width,
                        input_channels,
                        kernel_height,
                        kernel_width,
                        channels_multiplier,
                        dilation_height,
                        dilation_width,
                        x_stride_dw,
                        y_stride_circ_buf,
                        x_padding_dw,
                        y_padding_dh,
                        out_height_dh_cur,
                        p_circ_buf,
                        pt_inp);

                p_inp_circ = (WORD16 *)p_circ_buf->p_curr;

                conv2d_nhwc_f16
                    (pt_out
                    ,p_kernel
                    ,p_inp_circ
                    ,p_bias
                    ,kernel_height
                    ,kernel_width
                    ,out_height_dh_cur
                    ,out_width * dh_count
                    ,(input_channels * channels_multiplier)
                    ,x_stride
                    ,y_stride_circ_buf
                    ,NULL
                    ,NULL
                    ,NULL
                    );
            }
            x_padding_dw -= x_stride;
        }
        y_padding_dh -= y_stride;
    }
}

WORD32 xa_nn_dilated_conv2d_depthwise_v2_f16(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  channels_multiplier,
    WORD32  dilation_height,
    WORD32  dilation_width,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
    WORD32  inp_data_format,
    WORD32  out_data_format,
    pVOID p_scratch,
    const WORD16* out_activation_min,
    const WORD16* out_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
    (VOID) p_dma_cfg;
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((dilation_height <= 0 || dilation_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

    xa_nn_dilated_conv2d_depthwise_nhwc_f16(
            p_out,
            p_kernel,
            p_inp,
            p_bias,
            input_height,
            input_width,
            input_channels,
            kernel_height,
            kernel_width,
            channels_multiplier,
            dilation_height,
            dilation_width,
            x_stride,
            y_stride,
            x_padding,
            y_padding,
            out_height,
            out_width,
            out_data_format,
            p_scratch);

    /* Apply output activation min/max clamp */
    {
        int total_out = out_height * out_width * input_channels * channels_multiplier;
        xthalfx4 *pt_out = (xthalfx4 *)p_out;
        xthalfx4 vec_min = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(
                (out_activation_min != NULL) ? *out_activation_min : (WORD16)0xFC00)));
        xthalfx4 vec_max = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(
                (out_activation_max != NULL) ? *out_activation_max : (WORD16)0x7C00)));
        int i;
        xthalfx8 *pt_out8 = (xthalfx8 *)pt_out;
        /* Process 8 elements at a time */
        for (i = 0; i < (total_out >> 3); i++)
        {
            xthalfx4 v0, v1;
            AE_LHX4X2_IP(v0, v1, pt_out8, sizeof(xthalfx8));
            v0 = MAX_HX4(MIN_HX4(v0, vec_max), vec_min);
            v1 = MAX_HX4(MIN_HX4(v1, vec_max), vec_min);
            pt_out8 -= 1;
            AE_SHX4X2_IP(v0, v1, pt_out8, sizeof(xthalfx8));
        }
        /* Process remaining elements (up to 7) using aligned SAV */
        int rem = (total_out & 7);
        if (rem > 0)
        {
            xthalfx4 v0, v1;
            ae_valignx2 va_ld = AE_LA128_PP(pt_out8);
            AE_LAHX4X2_IP(v0, v1, va_ld, pt_out8);
            v0 = MAX_HX4(MIN_HX4(v0, vec_max), vec_min);
            v1 = MAX_HX4(MIN_HX4(v1, vec_max), vec_min);
            pt_out8 -= 1;
            ae_valignx2 va_st = AE_ZALIGN128();
            AE_SAVHX4X2_XP(v0, v1, va_st, pt_out8, rem * sizeof(xthalf));
            AE_SA128POS_FP(va_st, pt_out8);
        }
    }

    return 0;
}

WORD32 xa_nn_conv2d_depthwise_v2_f16(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  channels_multiplier,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
    WORD32  inp_data_format,
    WORD32  out_data_format,
    pVOID p_scratch,
    const WORD16 *pout_activation_min,
    const WORD16 *pout_activation_max,
    xa_dma_cfg_t *p_dma_cfg)
{
    (VOID) p_dma_cfg;
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

    if (kernel_height == 3 && kernel_width == 3 && x_stride == 1 && y_stride == 1) {
        xa_nn_conv2d_depthwise_f16_3x3_with_padding(
            p_out, p_kernel, p_inp, p_bias,
            input_height, input_width, input_channels,
            kernel_height, kernel_width, channels_multiplier,
            x_stride, y_stride, x_padding, y_padding,
            out_height, out_width, inp_data_format, out_data_format, p_scratch,
            pout_activation_min, pout_activation_max);
    } else {
        xa_nn_conv2d_depthwise_nhwc_f16(
            p_out, p_kernel, p_inp, p_bias,
            input_height, input_width, input_channels,
            kernel_height, kernel_width, channels_multiplier,
            x_stride, y_stride, x_padding, y_padding,
            out_height, out_width, out_data_format, p_scratch,
            pout_activation_min, pout_activation_max);
    }

    return 0;
}

#endif /* #if !HAVE_HP_VFPU */
