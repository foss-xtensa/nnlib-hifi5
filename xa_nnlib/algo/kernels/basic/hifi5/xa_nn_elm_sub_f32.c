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
#include "xa_type_def.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nn_common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nn_basic_state.h"
#include "xa_nnlib_kernels_api.h"

#if HAVE_VFPU
#define AE_SW_LAVSX2X2_XP(out0, out1, align_out, p_out, off) \
{ \
  ae_int16x4 d_out16_0, d_out16_1; \
  AE_LAV16X4X2_XP(d_out16_0, d_out16_1, align_out, (ae_int16x8 *)p_out, off); \
  d_out16_0 = AE_SEL16_2301(d_out16_0, d_out16_0); \
  d_out16_1 = AE_SEL16_2301(d_out16_1, d_out16_1); \
  out0 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_out16_0)); \
  out1 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_out16_1)); \
}

#define AE_SW_SAVSX2X2_XP(in0, in1, align_in, p_out, off) \
{ \
  ae_int32x2 d_in32_0, d_in32_1; \
  ae_int16x4 d_in16_0, d_in16_1; \
  d_in32_0 = AE_MOVINT32X2_FROMXTFLOATX2(in0); \
  d_in32_1 = AE_MOVINT32X2_FROMXTFLOATX2(in1); \
  d_in16_0 = AE_MOVINT16X4_FROMINT32X2(d_in32_0); \
  d_in16_1 = AE_MOVINT16X4_FROMINT32X2(d_in32_1); \
  d_in16_0 = AE_SEL16_2301(d_in16_0, d_in16_0); \
  d_in16_1 = AE_SEL16_2301(d_in16_1, d_in16_1); \
  AE_SAV16X4X2_XP(d_in16_0, d_in16_1, align_in, (ae_int16x8 *)p_out, off); \
}
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_sub_f32xf32_f32,
             (
                FLOAT32 *y,
                const FLOAT32 *x1,
                const FLOAT32 *x2,
                WORD32 N
              )
           )
#else
WORD32 xa_nn_elm_sub_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp1,
                               const FLOAT32 * __restrict__ p_inp2,
                               WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    int i;
    xtfloatx2 *inp1 = (xtfloatx2 *)p_inp1;
    xtfloatx2 *inp2 = (xtfloatx2 *)p_inp2;
    xtfloatx2 *out =  (xtfloatx2 *)p_out;
    xtfloatx2 x1, x2, y;

    if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
    {
        for(i=0;i < num_elm>>1;i++)
        {
            XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
            y = XT_SUB_SX2(x1, x2);
            XT_SSX2IP( y, out,  2*sizeof(FLOAT32));
        }
    }
    else
    {
        ae_valign inp1_a, inp2_a, out_a;

        inp1_a = XT_LASX2PP(inp1);
        inp2_a = XT_LASX2PP(inp2);
        out_a = AE_ZALIGN64();
        /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
        for(i=0;i < num_elm>>1;i++)
        {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            y = XT_SUB_SX2(x1, x2);
            XT_SASX2IP(y, out_a, out);
        }
        XT_SASX2POSFP(out_a, out);
    }
    // Remainder Loop
    if (num_elm & 1)
    {
        xtfloat a1, a2, a;
        XT_LSIP(a1, (xtfloat *)inp1, 0);
        XT_LSIP(a2, (xtfloat *)inp2, 0);
        a = XT_SUB_S(a1, a2);
        XT_SSI(a, (xtfloat *)out, 0);
    }

    return 0;
}
#endif

#if HAVE_VFPU
static void internal_elm_sub_broadcast_2D_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                    const    FLOAT32 * __restrict__ p_inp1,
                    const    FLOAT32 * __restrict__ p_inp2,
                             WORD32  out_lc,
                             WORD32  in_lc,
                             xtbool  sign_flag)
{
  int i, j;

  xtfloatx4  *__restrict__ p_a = (xtfloatx4 *)p_inp1;
  xtfloatx4  *__restrict__ p_b = (xtfloatx4 *)p_inp2;
  xtfloatx4  *__restrict__  p_c = (xtfloatx4 *)p_out;

  int num_simd4_ops;
  int num_scalar_ops;

  if(out_lc == 1)
  {
    num_simd4_ops = in_lc >> 2;
    num_scalar_ops = in_lc & 3;
  }
  else
  {
    num_simd4_ops = (in_lc >> 3) << 1;
    num_scalar_ops = in_lc & 7;
  }

  xtfloatx2 x1_0, x1_1, x2_0, x2_1, y_0, y_1;

  for(i = 0; i < out_lc; i++)
  {
    /* For computing inp2 - inp1 */
    if(sign_flag)
    {
      p_a = (xtfloatx4 *)&p_inp1[i * in_lc];
      p_b = (xtfloatx4 *)p_inp2;
    }
    else    /* For computing inp1 - inp2 */
    {
      p_b = (xtfloatx4 *)&p_inp1[i * in_lc];
      p_a = (xtfloatx4 *)p_inp2;
    }
    p_c = (xtfloatx4 *)&p_out[i * in_lc];

    ae_valignx2 vinp1, vinp2, out_a = AE_ZALIGN128();
    vinp1 = AE_LA128_PP(p_a);
    vinp2 = AE_LA128_PP(p_b);
    for(j = 0; j < num_simd4_ops; j++)
    {
      AE_LASX2X2_IP(x1_0, x1_1, vinp1, p_a);
      AE_LASX2X2_IP(x2_0, x2_1, vinp2, p_b);
      SUB_SX2X2(y_0, y_1, x2_0, x2_1, x1_0, x1_1);
      AE_SASX2X2_IP(y_0, y_1, out_a, p_c);
    }
    AE_SA128POS_FP(out_a, p_c);
  }

  if(num_scalar_ops != 0)
  {
    WORD32 rem0, rem1;
    rem0 = num_scalar_ops >= 4 ? 16 : num_scalar_ops << 2;
    rem1 = num_scalar_ops <= 4 ? 0 : (num_scalar_ops - 4) << 2;

    xtfloatx2 x1_2, x1_3, x2_2, x2_3, y_2, y_3;
    ae_valignx2 vinp1, vinp2, out_a;
    /* For computing inp2 - inp1 */
    if(sign_flag)
    {
      p_b = (xtfloatx4 *)&p_inp2[num_simd4_ops << 2];
      vinp2 = AE_LA128_PP(p_b);

      AE_SW_LAVSX2X2_XP(x2_0, x2_1, vinp2, p_b, rem0);
      AE_SW_LAVSX2X2_XP(x2_2, x2_3, vinp2, p_b, rem1);
      for(i = 0; i < out_lc; i++)
      {
        p_a = (xtfloatx4 *)&p_inp1[i * in_lc + (num_simd4_ops << 2)];
        p_c = (xtfloatx4 *)&p_out[i * in_lc + (num_simd4_ops << 2)];
        vinp1 = AE_LA128_PP(p_a);
        out_a = AE_ZALIGN128();

        AE_SW_LAVSX2X2_XP(x1_0, x1_1, vinp1, p_a, rem0);
        AE_SW_LAVSX2X2_XP(x1_2, x1_3, vinp1, p_a, rem1);
        SUB_SX2X2(y_0, y_1, x2_0, x2_1, x1_0, x1_1);
        SUB_SX2X2(y_2, y_3, x2_2, x2_3, x1_2, x1_3);
        AE_SW_SAVSX2X2_XP(y_0, y_1, out_a, p_c, rem0);
        AE_SW_SAVSX2X2_XP(y_2, y_3, out_a, p_c, rem1);
        AE_SA128POS_FP(out_a, p_c);
      }
    }
    else    /* For computing inp1 - inp2 */
    {
      p_b = (xtfloatx4 *)&p_inp2[num_simd4_ops << 2];
      vinp2 = AE_LA128_PP(p_b);

      AE_SW_LAVSX2X2_XP(x2_0, x2_1, vinp2, p_b, rem0);
      AE_SW_LAVSX2X2_XP(x2_2, x2_3, vinp2, p_b, rem1);
      for(i = 0; i < out_lc; i++)
      {
        p_a = (xtfloatx4 *)&p_inp1[i * in_lc + (num_simd4_ops << 2)];
        p_c = (xtfloatx4 *)&p_out[i * in_lc + (num_simd4_ops << 2)];
        vinp1 = AE_LA128_PP(p_a);
        out_a = AE_ZALIGN128();

        AE_SW_LAVSX2X2_XP(x1_0, x1_1, vinp1, p_a, rem0);
        AE_SW_LAVSX2X2_XP(x1_2, x1_3, vinp1, p_a, rem1);
        SUB_SX2X2(y_0, y_1, x1_0, x1_1, x2_0, x2_1);
        SUB_SX2X2(y_2, y_3, x1_2, x1_3, x2_2, x2_3);
        AE_SW_SAVSX2X2_XP(y_0, y_1, out_a, p_c, rem0);
        AE_SW_SAVSX2X2_XP(y_2, y_3, out_a, p_c, rem1);
        AE_SA128POS_FP(out_a, p_c);
      }
    }
  }
}

static void internal_elm_sub_broadcast_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                    const    FLOAT32 * __restrict__ p_inp1,
                    const    FLOAT32 * __restrict__ p_inp2,
                             WORD32  num_elm,
                             xtbool  sign_flag)
{
  int i;
  xtfloatx4  * __restrict__ p_a = (xtfloatx4 *)p_inp1;
  xtfloat  * __restrict__ p_b = (xtfloat *)p_inp2;
  xtfloatx4  *__restrict__  p_c =  (xtfloatx4 *)p_out;

  const int num_simd4_ops = num_elm >> 2;
  const int num_scalar_ops = num_elm & 3;

  xtfloatx2 x1_0, x1_1, x2, y_0, y_1;
  x2 = XT_LSI(p_b, 0);

  /* For computing inp2 - inp1 */
  if(sign_flag){
    if(((((unsigned)p_a) & 15) == 0) && ((((unsigned)p_c) & 15) == 0))
    {
      for(i=0; i<num_simd4_ops; i++)
      {
        AE_LSX2X2_IP(x1_0, x1_1, p_a, 4 * sizeof(FLOAT32));
        SUB_SX2X2(y_0, y_1, x2, x2, x1_0, x1_1);
        AE_SSX2X2_IP(y_0, y_1, p_c, 4 * sizeof(FLOAT32));
      }
    }
    else
    {
      ae_valignx2 inp1_a, out_a;
      inp1_a = AE_LA128_PP(p_a);
      out_a = AE_ZALIGN128();
      for(i=0; i<num_simd4_ops; i++)
      {
        AE_LASX2X2_IP(x1_0, x1_1, inp1_a, p_a);
        SUB_SX2X2(y_0, y_1, x2, x2, x1_0, x1_1);
        AE_SASX2X2_IP(y_0, y_1, out_a, p_c);
      }
      AE_SA128POS_FP(out_a, p_c);
    }
    if(num_scalar_ops !=0)
    {
      ae_valignx2 inp1_a, out_a;
      inp1_a = AE_LA128_PP(p_a);
      out_a = AE_ZALIGN128();
      AE_SW_LAVSX2X2_XP(x1_0, x1_1, inp1_a, p_a, num_scalar_ops << 2);
      SUB_SX2X2(y_0, y_1, x2, x2, x1_0, x1_1);
      AE_SW_SAVSX2X2_XP(y_0, y_1, out_a, p_c, num_scalar_ops << 2);
      AE_SA128POS_FP(out_a, p_c);
    }
  }
  /* For computing inp1 - inp2 */
  else
  {
    if(((((unsigned)p_a) & 15) == 0) && ((((unsigned)p_c) & 15) == 0))
    {
      for(i=0; i<num_simd4_ops; i++)
      {
        AE_LSX2X2_IP(x1_0, x1_1, (xtfloatx4 *)p_a, 4 * sizeof(FLOAT32));
        SUB_SX2X2(y_0, y_1, x1_0, x1_1, x2, x2);
        AE_SSX2X2_IP(y_0, y_1, (xtfloatx4 *)p_c, 4 * sizeof(FLOAT32));
      }
    }
    else
    {
      ae_valignx2 inp1_a, out_a;
      inp1_a = AE_LA128_PP(p_a);
      out_a = AE_ZALIGN128();
      for(i=0; i<num_simd4_ops; i++)
      {
        AE_LASX2X2_IP(x1_0, x1_1, inp1_a, (xtfloatx4 *)p_a);
        SUB_SX2X2(y_0, y_1, x1_0, x1_1, x2, x2);
        AE_SASX2X2_IP(y_0, y_1, out_a, (xtfloatx4 *)p_c);
      }
      AE_SA128POS_FP(out_a, (xtfloatx4 *)p_c);
    }
    if(num_scalar_ops !=0)
    {
      ae_valignx2 inp1_a, out_a;
      inp1_a = AE_LA128_PP(p_a);
      out_a = AE_ZALIGN128();
      AE_SW_LAVSX2X2_XP(x1_0, x1_1, inp1_a, p_a, num_scalar_ops << 2);
      SUB_SX2X2(y_0, y_1, x1_0, x1_1, x2, x2);
      AE_SW_SAVSX2X2_XP(y_0, y_1, out_a, p_c, num_scalar_ops << 2);
      AE_SA128POS_FP(out_a, p_c);
    }
  }
}
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_sub_broadcast_4D_f32xf32_f32,
             (
                      FLOAT32 * p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const FLOAT32 * p_inp2,
                      const WORD32 *const p_inp2_shape
              )
           )
#else
WORD32 xa_nn_elm_sub_broadcast_4D_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const FLOAT32 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);

  /* Check shapes */
  int i;
  xtbool sign_flag;
  for(i = 0; i < 4; i++)
  {
    if((p_inp1_shape[i] != p_inp2_shape[i] && p_inp1_shape[i] != 1 && p_inp2_shape[i] != 1) ||
       (p_out_shape[i] != (p_inp1_shape[i] > p_inp2_shape[i] ? p_inp1_shape[i] : p_inp2_shape[i])))
    {
      return -1;
    }
  }

  WORD32 inp1_strides[4], inp2_strides[4];
  inp1_strides[3] = 1;
  inp2_strides[3] = 1;
  for(i = 2; i >= 0; i--)
  {
    ae_int32x2 d_str, d_shape;
    d_str = AE_MOVDA32X2(inp1_strides[i + 1], inp2_strides[i + 1]);
    d_shape = AE_MOVDA32X2(p_inp1_shape[i + 1], p_inp2_shape[i + 1]);
    d_str = AE_MULP32X2(d_str, d_shape);
    inp1_strides[i] = AE_MOVAD32_H(d_str);
    inp2_strides[i] = AE_MOVAD32_L(d_str);
  }

  int need_broadcast = 0;
  int inp1_const = 1, inp2_const = 1;
  for(i = 0; i < 4; i++)
  {
    if(p_inp1_shape[i] != p_inp2_shape[i])
    {
      if(p_inp1_shape[i] == 1)
        inp1_strides[i] = 0;
      else
        inp2_strides[i] = 0;

      need_broadcast = 1;
    }
    if(p_inp1_shape[i] != 1)
      inp1_const &= 0;
    if(p_inp2_shape[i] != 1)
      inp2_const &= 0;
  }
  int itr0, itr1, itr2;

  FLOAT32 *p_out_tmp = p_out;
  const FLOAT32 *__restrict__ p_inp1_tmp = p_inp1;
  const FLOAT32 *__restrict__ p_inp2_tmp = p_inp2;
  if(need_broadcast == 0)
  {
    sign_flag = 0;
    internal_elm_sub_broadcast_2D_f32xf32_f32(
                p_out,
                p_inp1,
                p_inp2,
                1,
                p_out_shape[0] * inp1_strides[0],
                sign_flag);
  }
  else if(inp1_strides[3] == inp2_strides[3])
  {
    WORD32 in_lc, out_lc;
    sign_flag = 0;
    in_lc = p_out_shape[2] * p_out_shape[3];
    out_lc = 1;
    if(inp1_strides[2] == 0)
    {
      const FLOAT32 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
      sign_flag = 1;
      int tmp_strides[2];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];

      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];

      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      in_lc = p_out_shape[3];
      out_lc = p_out_shape[2];
    }
    else if(inp2_strides[2] == 0)
    {
      in_lc = p_out_shape[3];
      out_lc = p_out_shape[2];
    }

    for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const FLOAT32 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const FLOAT32 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        internal_elm_sub_broadcast_2D_f32xf32_f32(
            p_out_tmp,
            p_inp1_tmp0,
            p_inp2_tmp0,
            out_lc,
            in_lc,
            sign_flag);
        p_out_tmp += in_lc * out_lc;
        p_inp1_tmp0 += inp1_strides[1];
        p_inp2_tmp0 += inp2_strides[1];
      }
      p_inp1_tmp += inp1_strides[0];
      p_inp2_tmp += inp2_strides[0];
    }
  }
  else if(inp1_const == 1 || inp2_const == 1)
  {
    sign_flag = 0;
    if(inp1_strides[3] == 0)
    {
      sign_flag = 1;
      const FLOAT32 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
    }
    internal_elm_sub_broadcast_f32xf32_f32(
        p_out_tmp,
        p_inp1_tmp,
        p_inp2_tmp,
        p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3],
        sign_flag);
  }
  else
  {
    sign_flag = 0;
    if(inp1_strides[3] == 0)
    {
      const FLOAT32 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
      sign_flag = 1;
      int tmp_strides[3];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];
      tmp_strides[2] = inp1_strides[2];

      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];
      inp1_strides[2] = inp2_strides[2];

      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      inp2_strides[2] = tmp_strides[2];
    }
    for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const FLOAT32 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const FLOAT32 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        const FLOAT32 *__restrict__ p_inp1_tmp1 = p_inp1_tmp0;
        const FLOAT32 *__restrict__ p_inp2_tmp1 = p_inp2_tmp0;
        for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
        {
          {
            internal_elm_sub_broadcast_f32xf32_f32(
                p_out_tmp,
                p_inp1_tmp1,
                p_inp2_tmp1,
                p_out_shape[3],
                sign_flag);
          }
          p_out_tmp += p_out_shape[3];
          p_inp1_tmp1 += inp1_strides[2];
          p_inp2_tmp1 += inp2_strides[2];
        }
        p_inp1_tmp0 += inp1_strides[1];
        p_inp2_tmp0 += inp2_strides[1];
      }
      p_inp1_tmp += inp1_strides[0];
      p_inp2_tmp += inp2_strides[0];
    }
  }
  return 0;
}
#endif
