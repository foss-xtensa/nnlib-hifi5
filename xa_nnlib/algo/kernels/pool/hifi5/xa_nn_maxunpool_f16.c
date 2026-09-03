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
#include "xa_nnlib_common.h"
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_maxpool_state.h"
#include "xa_nnlib_err_chk.h"
#include <string.h>

/* =====================================================================
 * xa_nn_maxpoolId_v2_f16_nhwc
 * NHWC f16 max-pool with packed argmax indices, VALID padding only.
 * No scratch memory needed; state kept in on-stack accumulator arrays.
 * Vectorised over input_channels (groups of 4 f16 per xthalfx4 lane).
 * Outer loops unrolled HEIGHT_UNROLL x WIDTH_UNROLL output pixels.
 * ID encoding: p_out_Id[oh,ow,c] = (ky_best << 4) | (kx_best & 0xF)
 * ===================================================================== */

#define MAXPOOLID_H_UNROLL 2
#define MAXPOOLID_W_UNROLL 2

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(void, xa_nn_maxpoolId_v2_f16_nhwc,(
    WORD16  * __restrict__ p_out,
    UWORD8  * __restrict__ p_out_Id,
const WORD16  * __restrict__ p_inp,
    WORD32   input_height,
    WORD32   input_width,
    WORD32   input_channels,
    WORD32   kernel_height,
    WORD32   kernel_width,
    WORD32   x_stride,
    WORD32   y_stride,
    WORD32   out_height,
    WORD32   out_width,
    WORD16   act_min_bits,
    WORD16   act_max_bits))
#else /* #if !HAVE_HP_VFPU */
void xa_nn_maxpoolId_v2_f16_nhwc(
    WORD16  * __restrict__ p_out,
    UWORD8  * __restrict__ p_out_Id,
const WORD16  * __restrict__ p_inp,
    WORD32   input_height,
    WORD32   input_width,
    WORD32   input_channels,
    WORD32   kernel_height,
    WORD32   kernel_width,
    WORD32   x_stride,
    WORD32   y_stride,
    WORD32   out_height,
    WORD32   out_width,
    WORD16   act_min_bits,
    WORD16   act_max_bits)
{
  int itr_ic, itr_oh, itr_ow;
  int i, j, ky, kx;

  /* Broadcast fp16 bit patterns to all 4 SIMD lanes */
  xthalfx4 v_neg_inf = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16((WORD16)(UWORD16)0xFC00u));
  xthalfx4 v_act_min = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(act_min_bits));
  xthalfx4 v_act_max = AE_MOVXTHALFX4_FROMINT16X4(AE_MOVDA16(act_max_bits));

  /* Pre-compute stride constants - depend only on function parameters. */
  int stride_kx = input_channels;
  int stride_ky = input_width * input_channels;
  int stride_i  = y_stride * input_width * input_channels;
  int stride_j  = x_stride * input_channels;
  int n_iters   = kernel_height * kernel_width;

  /* Precalculate loop_params. Buffers are small enough to declare on stack */
  int loop_params_offset[256];
  UWORD8 loop_params_id[256];
  {
    int lp = 0;
    for (ky = 0; ky < kernel_height; ky++) {
      #pragma no_unroll
      for (kx = 0; kx < kernel_width; kx++) {
        int cur_off = ky * stride_ky + kx * stride_kx;
        int next_kx = kx + 1, next_ky = ky;
        if (next_kx >= kernel_width) { next_kx = 0; next_ky++; }
        int next_off = (next_ky < kernel_height) ?
          next_ky * stride_ky + next_kx * stride_kx : cur_off;
        loop_params_id[lp] = (ky << 4) | (kx & 0xF);
        loop_params_offset[lp] = next_off - cur_off - 8;
        lp++;
      }
    }
  }

  /* Outer loop: process channels in groups of 4 */
  for (itr_ic = 0; itr_ic < input_channels; itr_ic += 8)
  {
    int rem_chan  = XT_MIN(8, input_channels - itr_ic);

    for (itr_oh = 0; itr_oh < out_height; itr_oh += MAXPOOLID_H_UNROLL)
    {
      for (itr_ow = 0; itr_ow < out_width; itr_ow += MAXPOOLID_W_UNROLL)
      {
        /* Per-lane accumulators for H_UNROLL x W_UNROLL output pixels.
         * Dimension [2]: two channel groups of 4 (total 8 channels per tile). */
        xthalfx4   max_val[MAXPOOLID_H_UNROLL][MAXPOOLID_W_UNROLL][2];
#if XCHAL_HAVE_HIFI5S
        ae_int8x8 idx_val[MAXPOOLID_H_UNROLL][MAXPOOLID_W_UNROLL];
#else
        ae_int16x4 idx_val[MAXPOOLID_H_UNROLL][MAXPOOLID_W_UNROLL][2];
#endif

        /* Initialise: max = -inf, index = 0 */
        #pragma unroll
        for (i = 0; i < MAXPOOLID_H_UNROLL; i++) {
          #pragma unroll
          for (j = 0; j < MAXPOOLID_W_UNROLL; j++) {
            max_val[i][j][0] = max_val[i][j][1] = v_neg_inf;
#if XCHAL_HAVE_HIFI5S
            idx_val[i][j]    = AE_MOVDA8(0);
#else
            idx_val[i][j][0] = idx_val[i][j][1] = AE_MOVDA16(0);
#endif
          }
        }

        /* ---- Kernel accumulation ---- */
        /* Base pointer for (ky=0, kx=0) anchored at output pixel (itr_oh, itr_ow). */
        xthalf *p_base =
          (xthalf *)p_inp
          + (itr_oh * y_stride * input_width + itr_ow * x_stride)
            * input_channels
          + itr_ic;

        /* H_UNROLL x W_UNROLL pointers, indexed [i*MAXPOOLID_W_UNROLL+j];
         * advanced per iter by delta (accounts for AE_LAHX4X2_IP auto-increment). */
        xthalfx8 * __restrict__ p_load_v8[MAXPOOLID_H_UNROLL * MAXPOOLID_W_UNROLL];
        #pragma unroll
        for (i = 0; i < MAXPOOLID_H_UNROLL; i++) {
          #pragma unroll
          for (j = 0; j < MAXPOOLID_W_UNROLL; j++) {
            p_load_v8[i * MAXPOOLID_W_UNROLL + j] =
              (xthalfx8 *)(p_base + i * stride_i + j * stride_j);
          }
        }

        /* Loop param pointers */
        WORD32 * __restrict__ ploop_params_offset = loop_params_offset;
#if XCHAL_HAVE_HIFI5S
        ae_int8 * __restrict__ ploop_params_id = (ae_int8 *)loop_params_id;
#else
        WORD8 * __restrict__ ploop_params_id = (WORD8 *)loop_params_id;
#endif

        for (int iter = 0; iter < n_iters; iter++) {
#if XCHAL_HAVE_HIFI5S
          ae_int8x8 v_cur_id;
          AE_L8_IP(v_cur_id, ploop_params_id, 1);
#else
          WORD16 v_cur_id_s = (WORD16)(*ploop_params_id++);
          ae_int16x4 v_cur_id = AE_MOVDA16(v_cur_id_s);
#endif
          int delta = *ploop_params_offset++;

          #pragma unroll
          for (i = 0; i < MAXPOOLID_H_UNROLL; i++) {
            #pragma unroll
            for (j = 0; j < MAXPOOLID_W_UNROLL; j++) {
              xthalfx4 cur_val0, cur_val1;
              /*Load and pointer increment*/
              ae_valignx2 align_src = AE_LA128_PP(p_load_v8[i * MAXPOOLID_W_UNROLL + j]);
              AE_LAHX4X2_IP(cur_val0, cur_val1, align_src, p_load_v8[i * MAXPOOLID_W_UNROLL + j]);
              p_load_v8[i * MAXPOOLID_W_UNROLL + j] =
                (xthalfx8 *)((xthalf *)p_load_v8[i * MAXPOOLID_W_UNROLL + j] + delta);

              xtbool4 cond0 = OLT_HX4(max_val[i][j][0], cur_val0);
              xtbool4 cond1 = OLT_HX4(max_val[i][j][1], cur_val1);
              max_val[i][j][0] = MAX_HX4(max_val[i][j][0], cur_val0);
              max_val[i][j][1] = MAX_HX4(max_val[i][j][1], cur_val1);
#if XCHAL_HAVE_HIFI5S
              xtbool8 cond = AE_JOINB8B4(cond0, cond1);
              AE_MOVT8X8(idx_val[i][j], v_cur_id, cond);
#else
              AE_MOVT16X4(idx_val[i][j][0], v_cur_id, cond0);
              AE_MOVT16X4(idx_val[i][j][1], v_cur_id, cond1);
#endif
            }
          }
        } /* iter */

        /* ---- Fused activation clamp + output write ---- */
        #pragma unroll
        for (i = 0; i < MAXPOOLID_H_UNROLL; i++) {
          int oh = itr_oh + i;
          if (oh >= out_height) break;

          #pragma unroll
          for (j = 0; j < MAXPOOLID_W_UNROLL; j++) {
            int ow = itr_ow + j;
            if (ow >= out_width) break;

            /* Clamp both channel groups to activation range (does not change index) */
            max_val[i][j][0] = MAX_HX4(max_val[i][j][0], v_act_min);
            max_val[i][j][0] = MIN_HX4(max_val[i][j][0], v_act_max);
            max_val[i][j][1] = MAX_HX4(max_val[i][j][1], v_act_min);
            max_val[i][j][1] = MIN_HX4(max_val[i][j][1], v_act_max);

            WORD16 * __restrict__ p_out_curr = (WORD16 *)p_out
              + (oh * out_width + ow) * input_channels + itr_ic;
            UWORD8 * __restrict__ p_id_curr  = p_out_Id
              + (oh * out_width + ow) * input_channels + itr_ic;

            ae_valignx2 va_f16 = AE_ZALIGN128();
            xthalfx8 *p_out_hx8 = (xthalfx8 *)p_out_curr;
            AE_SAVHX4X2_XP(max_val[i][j][0], max_val[i][j][1],
                     va_f16, p_out_hx8,
                     rem_chan * sizeof(xthalf));
            AE_SA128POS_FP(va_f16, p_out_hx8);
#if XCHAL_HAVE_HIFI5S
            ae_int8x8 packed_id0 = idx_val[i][j]; /* already packed in low bytes */
#else
            ae_int8x8 sel = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x0e0c0a08, 0x06040200));
            ae_int8x8 packed_id0 = AE_SEL8X8( 
              AE_MOVINT8X8_FROMINT16X4(idx_val[i][j][0]), 
              AE_MOVINT8X8_FROMINT16X4(idx_val[i][j][1]), sel);
#endif
            ae_valignx2 va_id = AE_ZALIGN128();
            ae_int8x16 *p_id_16 = (ae_int8x16 *)p_id_curr;
            AE_SAV8X8X2_XP(packed_id0, packed_id0, va_id, p_id_16, rem_chan);
            AE_SA128POS_FP(va_id, p_id_16);
          } /* j (width unroll) */
        } /* i (height unroll) */
      } /* itr_ow */
    } /* itr_oh */
  } /* itr_ic */
}
#endif /* #if !HAVE_HP_VFPU */



#if !HAVE_HP_VFPU

DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_maxpoolId_v2_f16,(
  WORD16 *__restrict__ p_out,
  UWORD8 *__restrict__ p_out_Id,
  const WORD16 *__restrict__ p_inp ,
  WORD32  input_height,
  WORD32  input_width,
  WORD32  input_channels,
  WORD32  kernel_height,
  WORD32  kernel_width,
  WORD32  x_stride,
  WORD32  y_stride,
  WORD32  x_padding,
  WORD32  y_padding,
  WORD32  out_height,
  WORD32  out_width,
  WORD32  inp_data_format,
  WORD32  out_data_format,
  VOID   *p_scratch,
  WORD16 *pout_activation_min,
  WORD16 *pout_activation_max,
  xa_dma_cfg_t *p_dma_cfg))

#else

WORD32 xa_nn_maxpoolId_v2_f16(
  WORD16 *__restrict__ p_out,
  UWORD8 *__restrict__ p_out_Id,
  const WORD16 *__restrict__ p_inp ,
  WORD32  input_height,
  WORD32  input_width,
  WORD32  input_channels,
  WORD32  kernel_height,
  WORD32  kernel_width,
  WORD32  x_stride,
  WORD32  y_stride,
  WORD32  x_padding,
  WORD32  y_padding,
  WORD32  out_height,
  WORD32  out_width,
  WORD32  inp_data_format,
  WORD32  out_data_format,
  VOID   *p_scratch,
  WORD16 *pout_activation_min,
  WORD16 *pout_activation_max,
  xa_dma_cfg_t *p_dma_cfg)
{
  (void)p_scratch;
  (void)p_dma_cfg;

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out,    -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_Id, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp,    -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_width  > input_width),  -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  /* VALID padding only */
  XA_NNLIB_ARG_CHK_COND((y_padding != 0 || x_padding != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  /* NHWC format only */
  XA_NNLIB_ARG_CHK_COND((inp_data_format != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);
  /* 4-bit ID field: kernel dims must be 1..16 */
  XA_NNLIB_ARG_CHK_COND((kernel_height > 16 || kernel_width > 16), -1);
  /* Confirm the geometry is truly VALID (no right/bottom overhang) */
  {
    int right_pad  = (out_width  - 1) * x_stride + kernel_width  - input_width;
    int bottom_pad = (out_height - 1) * y_stride + kernel_height - input_height;
    XA_NNLIB_ARG_CHK_COND((right_pad > 0 || bottom_pad > 0), -1);
  }

  /* Derive f16 activation bounds (NULL -> -inf / +inf) */
  WORD16 act_min_bits = pout_activation_min
            ? *pout_activation_min
            : (WORD16)(UWORD16)0xFC00u; /* f16 -inf */
  WORD16 act_max_bits = pout_activation_max
            ? *pout_activation_max
            : (WORD16)(UWORD16)0x7C00u; /* f16 +inf */

  xa_nn_maxpoolId_v2_f16_nhwc(p_out, p_out_Id, p_inp,
                input_height, input_width, input_channels,
                kernel_height, kernel_width,
                x_stride, y_stride,
                out_height, out_width,
                act_min_bits, act_max_bits);
  return 0;
}

#endif

WORD32 xa_nn_maxunpool_f16(
  WORD16* __restrict__ p_out,
  const WORD16* __restrict__ p_inp,
  const UWORD8* __restrict__ p_inp_Id,
  WORD32  input_height,
  WORD32  input_width,
  WORD32  input_channels,
  WORD32  kernel_height,
  WORD32  kernel_width,
  WORD32  x_stride,
  WORD32  y_stride,
  WORD32  x_padding,
  WORD32  y_padding,
  WORD32  out_height,
  WORD32  out_width,
  WORD32  inp_data_format,
  WORD32  out_data_format,
  VOID   *p_scratch )
{
  XA_NNLIB_ARG_CHK_PTR(p_out,    -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp,    -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_Id, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_height > 16), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_width  <= 0 || kernel_width  > 16), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_width > 2048), -1);
  XA_NNLIB_ARG_CHK_COND((inp_data_format != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

  memset(p_out, 0,
        (WORD32)out_height * out_width * input_channels * (WORD32)sizeof(WORD16));

  WORD32 lenHW       = input_height * input_width;
  WORD32 ic_bytes = (WORD32)input_channels * (WORD32)sizeof(WORD16);
  /* Byte advance for p_base when moving to the next input column */
  WORD32 x_step = (WORD32)x_stride * ic_bytes;
  /* Byte advance at end of each input row: jump y_stride rows down, reset to column 0 */
  WORD32 h_step = ((WORD32)y_stride * out_width
          - (WORD32)(input_width - 1) * x_stride) * ic_bytes;

  /* Constants for vectorised offset computation (out_width <= 2048 guaranteed) */
  ae_int16x4 v_ow16 = AE_MOVDA16((WORD16)out_width);
  ae_int16x4 v_ic16 = AE_MOVDA16((WORD16)ic_bytes);

  /* Selectors for AE_DSEL16X4(q0, q1, d0, d1, d2).
   * d2.n[10:8] -> q0.n: 7=d0.3, 6=d0.2, 5=d0.1, 4=d0.0, 3=d1.3, ...
   * d2.n[2:0]  -> q1.n: same encoding.
   * lane3=MSB=ch0. Only lane0 of q0/q1 used by AE_S16_0_X. d0=d1=vN. */
  ae_int16x4 sel_ch01 = AE_MOVDA16(0x0706); /* q0.0=d0.3=ch0, q1.0=d0.2=ch1 */
  ae_int16x4 sel_ch23 = AE_MOVDA16(0x0504); /* q0.0=d0.1=ch2, q1.0=d0.0=ch3 */

  WORD32 c, inHW;

  /* SIMD loop: 8 channels per iteration (two ae_int16x4 = 16 bytes) */
  for (c = 0; c < (input_channels & ~7); c += 8)
  {
    const WORD16 * __restrict__ pInp  = p_inp    + c;
    const UWORD8 * __restrict__ pIdx  = p_inp_Id + c;
    WORD16       * __restrict__ p_base = p_out   + c;
    int           w      = 0;

    for (inHW = 0; inHW < lenHW; inHW++)
    {
      /* Unaligned load of 8 input WORD16 values */
      ae_int16x4 v0, v1;
      {
        ae_int16x4 *ptmp = (ae_int16x4 *)pInp;
        ae_valign va = AE_LA64_PP(ptmp);
        AE_LA16X4_IP(v0, va, ptmp);
        AE_LA16X4_IP(v1, va, ptmp);
        /* ptmp advanced by 8 WORD16; adjust for input_channels stride */
        pInp = (const WORD16 *)ptmp + (input_channels - 8);
      }

      /* Load 8 index bytes (unaligned) */
      ae_int8x8 vi;
      {
        ae_int8x8 *pitmp = (ae_int8x8 *)pIdx;
        ae_valign va_b = AE_LA64_PP(pitmp);
        AE_LA8X8_IP(vi, va_b, pitmp);
        pIdx = (const UWORD8 *)pitmp + (input_channels - 8);
      }

      /* Extract nibbles: kx = bits[3:0], ky = bits[7:4].
       * No AE_AND8X8: reinterpret as ae_int16x4, apply AE_AND16 with
       * mask 0x0F0F (clears upper nibble of each byte), cast back. */
      ae_int8x8 v_kx_b = AE_MOVINT8X8_FROMINT16X4(
        AE_AND16(AE_MOVINT16X4_FROMINT8X8(vi), AE_MOVDA16(0x0F0F)));
      ae_int8x8 v_ky_b = AE_SRLI8(vi, 4);

      /* Zero-extend to ae_int16x4 pairs.
       * dh: channels 0-3 (high bytes of ae_int8x8; MSB byte = channel 0)
       * dl: channels 4-7 (low bytes) */
      ae_int16x4 v_kx_dh, v_kx_dl, v_ky_dh, v_ky_dl;
      AE_ADDW8U(v_kx_dh, v_kx_dl, v_kx_b, AE_MOVDA8(0));
      AE_ADDW8U(v_ky_dh, v_ky_dl, v_ky_b, AE_MOVDA8(0));

      /* Compute off[k] = (ky[k]*ow + kx[k]) * ic_bytes + 2k.
       * Step 1: (ky*ow+kx) in 16-bit: 4 MULA16X4 + 2 SAT16X4.
       * Step 2: *ic_bytes in 32-bit: 2 MUL16X4 + 4 ADD32. */
      ae_int32x2 mh0 = AE_ZERO32(), mh1 = AE_ZERO32();
      ae_int32x2 ml0 = AE_ZERO32(), ml1 = AE_ZERO32();
      AE_MULA16X4(mh0, mh1, v_ky_dh, v_ow16);
      AE_MULA16X4(ml0, ml1, v_ky_dl, v_ow16);
      AE_MULA16X4(mh0, mh1, v_kx_dh, AE_MOVDA16(1));
      AE_MULA16X4(ml0, ml1, v_kx_dl, AE_MOVDA16(1));
      ae_int16x4 v_prod_dh = AE_SAT16X4(mh0, mh1);
      ae_int16x4 v_prod_dl = AE_SAT16X4(ml0, ml1);

      ae_int32x2 r0, r1, r2, r3;
      AE_MUL16X4(r0, r1, v_prod_dh, v_ic16);
      AE_MUL16X4(r2, r3, v_prod_dl, v_ic16);
      r0 = AE_ADD32(r0, AE_MOVDA32X2( 0,  2));
      r1 = AE_ADD32(r1, AE_MOVDA32X2( 4,  6));
      r2 = AE_ADD32(r2, AE_MOVDA32X2( 8, 10));
      r3 = AE_ADD32(r3, AE_MOVDA32X2(12, 14));

      WORD32 off[8];
      off[0] = AE_MOVAD32_H(r0);
      off[1] = AE_MOVAD32_L(r0);
      off[2] = AE_MOVAD32_H(r1);
      off[3] = AE_MOVAD32_L(r1);
      off[4] = AE_MOVAD32_H(r2);
      off[5] = AE_MOVAD32_L(r2);
      off[6] = AE_MOVAD32_H(r3);
      off[7] = AE_MOVAD32_L(r3);

      /* Scatter: use DSEL16X4 to route each lane to position 0 (read by AE_S16_0_X)
       * without a scalar round-trip.  4 DSEL ops replace 8 MOVAD16 + 8 MOVDA16. */
      ae_int16x4 t0, t1, t2, t3, t4, t5, t6, t7;
      AE_DSEL16X4(t0, t1, v0, v0, sel_ch01); /* t0[0]=ch0, t1[0]=ch1 */
      AE_DSEL16X4(t2, t3, v0, v0, sel_ch23); /* t2[0]=ch2, t3[0]=ch3 */
      AE_DSEL16X4(t4, t5, v1, v1, sel_ch01); /* t4[0]=ch4, t5[0]=ch5 */
      AE_DSEL16X4(t6, t7, v1, v1, sel_ch23); /* t6[0]=ch6, t7[0]=ch7 */

      ae_int16 * __restrict__ pbase16 = (ae_int16 *)p_base;
      AE_S16_0_X(t0, pbase16, off[0]);
      AE_S16_0_X(t1, pbase16, off[1]);
      AE_S16_0_X(t2, pbase16, off[2]);
      AE_S16_0_X(t3, pbase16, off[3]);
      AE_S16_0_X(t4, pbase16, off[4]);
      AE_S16_0_X(t5, pbase16, off[5]);
      AE_S16_0_X(t6, pbase16, off[6]);
      AE_S16_0_X(t7, pbase16, off[7]);

      /* Advance p_base to next input position */
      if (++w == input_width) {
        w = 0;
        p_base = (WORD16 *)((UWORD8 *)p_base + h_step);
      } else {
        p_base = (WORD16 *)((UWORD8 *)p_base + x_step);
      }
    }
  }

  /* Tail: remaining channels (input_channels % 8 != 0) */
  for (; c < input_channels; c++)
  {
    WORD32 w = 0;
    WORD32 ow_ic    = out_width * input_channels;
    WORD32 x_step_s = x_stride * input_channels;
    WORD32 h_step_s = ((WORD32)y_stride * out_width
             - (WORD32)(input_width - 1) * x_stride) * input_channels;
    const UWORD8 *pIdx  = p_inp_Id + c;
    const WORD16 *pInp  = p_inp    + c;
    WORD16       *p_base = p_out   + c;

    for (inHW = 0; inHW < lenHW; inHW++)
    {
      UWORD8 idx = *pIdx;  pIdx += input_channels;
      WORD32 kx  = idx & 0x0F;
      WORD32 ky  = (idx >> 4) & 0x0F;
      p_base[ky * ow_ic + kx * input_channels] = *pInp;
      pInp += input_channels;
      if (++w == input_width) { 
        w = 0; p_base += h_step_s; 
      } else {
        p_base += x_step_s;
      }
    }
  }

  return 0;
}
