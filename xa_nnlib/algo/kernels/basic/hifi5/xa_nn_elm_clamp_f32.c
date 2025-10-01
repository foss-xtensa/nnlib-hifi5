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
#include "xa_nn_basic_state.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common_macros_hifi5.h"


#ifdef AE_LAVSX2X2_XP
  #define AE_SW_LAVSX2X2_XP(d1, d2, va, ptr, off)  AE_LAVSX2X2_XP(d1, d2, va, ptr, off)
#else
  #define AE_SW_LAVSX2X2_XP(d1, d2, va, ptr, off) \
  { \
    ae_int16x4 d_out16_0, d_out16_1; \
    ae_int16x8 *ptr_16x8 = (ae_int16x8 *)ptr; \
    AE_LAV16X4X2_XP(d_out16_0, d_out16_1, va, ptr_16x8, off); \
    d_out16_0 = AE_SEL16_2301(d_out16_0, d_out16_0); \
    d_out16_1 = AE_SEL16_2301(d_out16_1, d_out16_1); \
    d1 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_out16_0)); \
    d2 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_out16_1)); \
    ptr = (xtfloatx4 *)ptr_16x8; \
  }
#endif
#ifdef AE_SAVSX2X2_XP
  #define AE_SW_SAVSX2X2_XP(d1, d2, va, ptr, off)  AE_SAVSX2X2_XP(d1, d2, va, ptr, off)
#else
  #define AE_SW_SAVSX2X2_XP(d1, d2, va, ptr, off) \
  { \
    ae_int16x4 d_in16_0, d_in16_1; \
    ae_int16x8 *ptr_16x8 = (ae_int16x8 *)ptr; \
    d_in16_0 = AE_MOVINT16X4_FROMINT32X2(AE_MOVINT32X2_FROMXTFLOATX2(d1)); \
    d_in16_1 = AE_MOVINT16X4_FROMINT32X2(AE_MOVINT32X2_FROMXTFLOATX2(d2)); \
    d_in16_0 = AE_SEL16_2301(d_in16_0, d_in16_0); \
    d_in16_1 = AE_SEL16_2301(d_in16_1, d_in16_1); \
    AE_SAV16X4X2_XP(d_in16_0, d_in16_1, va, ptr_16x8, off); \
    ptr = (xtfloatx4 *)ptr_16x8; \
  }
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_clamp_f32xf32xf32_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp,
                const FLOAT32 *p_min,
                const FLOAT32 *p_max,
                WORD32 num_elm
              )
           )
#else
WORD32 xa_nn_elm_clamp_f32xf32xf32_f32(FLOAT32 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp,
                               const FLOAT32 * __restrict__ p_min,
                               const FLOAT32 * __restrict__ p_max,
                               WORD32 num_elm)
{

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_min, -1);
    XA_NNLIB_ARG_CHK_PTR(p_max, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_min, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_max, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    int i;
    xtfloatx4 *inp = (xtfloatx4 *)p_inp;
    xtfloatx4 *min = (xtfloatx4 *)p_min;
    xtfloatx4 *max = (xtfloatx4 *)p_max;
    xtfloatx4 *out =  (xtfloatx4 *)p_out;

    xtfloatx2 x1, x1_1, d_min, d_min_1, d_max, d_max_1, y, y_1;

    if(((((unsigned)p_out)&0xF) == 0) && ((((unsigned)p_inp)&0xF) == 0) && ((((unsigned)p_min)&0xF) == 0) && ((((unsigned)p_max)&0xF) == 0))
    {
      for(i=0;i < num_elm>>2;i++)
      {
        AE_LSX2X2_IP(x1, x1_1, inp, 4 * sizeof(FLOAT32));
        AE_LSX2X2_IP(d_min, d_min_1, min, 4*sizeof(FLOAT32));
        AE_LSX2X2_IP(d_max, d_max_1, max, 4*sizeof(FLOAT32));
        y = MAX_SX2(x1, d_min);
        y_1 = MAX_SX2(x1_1, d_min_1);
        y = MIN_SX2(y, d_max);
        y_1 = MIN_SX2(y_1, d_max_1);
        AE_SSX2X2_IP(y, y_1, out,  4*sizeof(FLOAT32));
      }
    }
    else
    {
      ae_valignx2 inp_a, min_a, max_a, out_a;

      inp_a = AE_LA128_PP(inp);
      min_a = AE_LA128_PP(min);
      max_a = AE_LA128_PP(max);
      out_a = AE_ZALIGN128();
      /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
      for(i=0;i < num_elm>>2;i++)
      {
        AE_LASX2X2_IP(x1, x1_1, inp_a, inp);
        AE_LASX2X2_IP(d_min, d_min_1, min_a, min);
        AE_LASX2X2_IP(d_max, d_max_1, max_a, max);
        y = MAX_SX2(x1, d_min);
        y_1 = MAX_SX2(x1_1, d_min_1);
        y = MIN_SX2(y, d_max);
        y_1 = MIN_SX2(y_1, d_max_1);
        AE_SASX2X2_IP(y, y_1, out_a, out);
      }
      AE_SA128POS_FP(out_a, out);
    }
    // Remainder Loop
    int rem = num_elm & 3;
    if (rem)
    {
        ae_valignx2 inp_a, min_a, max_a, out_a;

        inp_a = AE_LA128_PP(inp);
        min_a = AE_LA128_PP(min);
        max_a = AE_LA128_PP(max);
        out_a = AE_ZALIGN128();

        AE_SW_LAVSX2X2_XP(x1, x1_1, inp_a, inp, rem* sizeof(FLOAT32));
        AE_SW_LAVSX2X2_XP(d_min, d_min_1, min_a, min, rem* sizeof(FLOAT32));
        AE_SW_LAVSX2X2_XP(d_max, d_max_1, max_a, max, rem* sizeof(FLOAT32));

        y = MAX_SX2(x1, d_min);
        y_1 = MAX_SX2(x1_1, d_min_1);
        y = MIN_SX2(y, d_max);
        y_1 = MIN_SX2(y_1, d_max_1);
        AE_SW_SAVSX2X2_XP(y, y_1, out_a, out,rem* sizeof(FLOAT32));
        AE_SA128POS_FP(out_a, out);  
    }
    return 0;
}
#endif
