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
/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common.h"

#ifdef AE_LAVSX2X2_XP
  #define AE_SW_LAVSX2X2_XP(d1, d2, va, ptr, off)  AE_LAVSX2X2_XP(d1, d2, va, ptr, off)
#else
  #define AE_SW_LAVSX2X2_XP(d1, d2, va, ptr, off) \
  { \
    ae_int16x4 d_out16_0, d_out16_1; \
    AE_LAV16X4X2_XP(d_out16_0, d_out16_1, va, (ae_int16x8 *)ptr, off); \
    d_out16_0 = AE_SEL16_2301(d_out16_0, d_out16_0); \
    d_out16_1 = AE_SEL16_2301(d_out16_1, d_out16_1); \
    d1 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_out16_0)); \
    d2 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_out16_1)); \
  }
#endif
#ifdef AE_SAVSX2X2_XP
  #define AE_SW_SAVSX2X2_XP(d1, d2, va, ptr, off)  AE_SAVSX2X2_XP(d1, d2, va, ptr, off)
#else
  #define AE_SW_SAVSX2X2_XP(d1, d2, va, ptr, off) \
  { \
    ae_int16x4 d_in16_0, d_in16_1; \
    d_in16_0 = AE_MOVINT16X4_FROMINT32X2(AE_MOVINT32X2_FROMXTFLOATX2(d1)); \
    d_in16_1 = AE_MOVINT16X4_FROMINT32X2(AE_MOVINT32X2_FROMXTFLOATX2(d2)); \
    d_in16_0 = AE_SEL16_2301(d_in16_0, d_in16_0); \
    d_in16_1 = AE_SEL16_2301(d_in16_1, d_in16_1); \
    AE_SAV16X4X2_XP(d_in16_0, d_in16_1, va, (ae_int16x8 *)ptr, off); \
  }
#endif
#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_add_f32xf32_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp1,
                const FLOAT32 *p_inp2,
                WORD32 num_elm
              )
           )
#else
WORD32 xa_nn_elm_add_f32xf32_f32(FLOAT32 * __restrict__ p_out,
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

  int i = 0;
  xtfloatx4 *inp1 = (xtfloatx4 *)p_inp1;
  xtfloatx4 *inp2 = (xtfloatx4 *)p_inp2;
  xtfloatx4 *out =  (xtfloatx4 *)p_out;
  xtfloatx2 x1, x2, x3, x4, y1, y2, y3, y4;
  xtfloatx2 z1, z2, z3, z4;

  if(((((unsigned)p_out)&15) == 0) && ((((unsigned)p_inp1)&15) == 0) && ((((unsigned)p_inp2)&15) == 0))
  {
#pragma no_unroll
    for(i = 0; i < (num_elm >> 3); i++)
    {
      AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
      AE_LSX2X2_IP(x3, x4, inp1, 4*sizeof(FLOAT32));
      AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));
      AE_LSX2X2_IP(y3, y4, inp2, 4*sizeof(FLOAT32));
      ADD_SX2X2(z1, z2, x1, x2, y1, y2);
      ADD_SX2X2(z3, z4, x3, x4, y3, y4);
      AE_SSX2X2_IP(z1, z2, out, 4*sizeof(FLOAT32));
      AE_SSX2X2_IP(z3, z4, out, 4*sizeof(FLOAT32));
    }
  }
  else
  {
    ae_valignx2 inp1_a, inp2_a, out_a;

    inp1_a = AE_LA128_PP(inp1);
    inp2_a = AE_LA128_PP(inp2);
    out_a = AE_ZALIGN128();

    /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
#pragma no_unroll
    for(i = 0; i < (num_elm >> 3); i++)
    {
      AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
      AE_LASX2X2_IP(x3, x4, inp1_a, inp1);
      AE_LASX2X2_IP(y1, y2, inp2_a, inp2);
      AE_LASX2X2_IP(y3, y4, inp2_a, inp2);
      ADD_SX2X2(z1, z2, x1, x2, y1, y2);
      ADD_SX2X2(z3, z4, x3, x4, y3, y4);
      AE_SASX2X2_IP(z1, z2, out_a, out);
      AE_SASX2X2_IP(z3, z4, out_a, out);
    }
    AE_SA128POS_FP(out_a, out);
  }
  // Remainder Loop
  int rem_itr = num_elm & (7);
  for(i = 0; i < rem_itr; i++)
  {
    xtfloat a1, a2, a;
    AE_LSIP(a1, (xtfloat *)inp1, sizeof(FLOAT32));
    AE_LSIP(a2, (xtfloat *)inp2, sizeof(FLOAT32));
    a = ADD_S(a1, a2);
    AE_SSIP(a, (xtfloat *)out, sizeof(FLOAT32));
  }

  return 0;
}
#endif

#if HAVE_VFPU
static void internal_elm_add_broadcast_2D_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                    const    FLOAT32 * __restrict__ p_inp1,
                    const    FLOAT32 * __restrict__ p_inp2,
                             WORD32  out_lc,
                             WORD32  in_lc)
{
  int i, j;

  xtfloatx4  * __restrict__ p_a = (xtfloatx4 *)p_inp1;
  xtfloatx4  * __restrict__ p_b = (xtfloatx4 *)p_inp2;
  xtfloatx4  *__restrict__  p_c =  (xtfloatx4 *)p_out;

  int num_simd4_ops;
  int num_scalar_ops;

  num_simd4_ops = in_lc >> 2;
  num_scalar_ops = in_lc & 3;

  xtfloatx2 x1, x2, y1,y2,out1,out2;

  for(i = 0; i < out_lc; i++)
  {
    p_a = (xtfloatx4 *)&p_inp1[i * in_lc];
    p_b = (xtfloatx4 *)p_inp2;
    p_c = (xtfloatx4 *)&p_out[i * in_lc];
    if(((((unsigned)p_a)&0xF) == 0) && ((((unsigned)p_b)&0xF) == 0) && ((((unsigned)p_c)&0xF) == 0))
    {
      for(j = 0; j < num_simd4_ops; j++)
      {
        AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
        AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));
        ADD_SX2X2(out1, out2, x1, x2, y1, y2);
        AE_SSX2X2_IP(out1, out2, p_c, 4 * sizeof(FLOAT32)); 
      }
    }
    else
    {
      ae_valignx2  vinp1, vinp2,out_a = AE_ZALIGN128();
      vinp1 = AE_LA128_PP(p_a);
      vinp2 = AE_LA128_PP(p_b);
      for(j = 0; j < num_simd4_ops; j++)
      {
        AE_LASX2X2_IP(x1, x2, vinp1, p_a);
        AE_LASX2X2_IP(y1, y2, vinp2, p_b);
        ADD_SX2X2(out1, out2, x1, x2, y1, y2);
        AE_SASX2X2_IP(out1, out2, out_a, p_c); 
      }
      AE_SA128POS_FP(out_a, (xtfloatx4 *)p_c);
    }
    if(num_scalar_ops !=0)
    {
      ae_valignx2  vinp1, vinp2,out_a = AE_ZALIGN128();
      vinp1 = AE_LA128_PP(p_a);
      vinp2 = AE_LA128_PP(p_b);
      AE_SW_LAVSX2X2_XP(x1, x2, vinp1, (xtfloatx4 *)p_a, num_scalar_ops* sizeof(FLOAT32));
      AE_SW_LAVSX2X2_XP(y1, y2, vinp2, (xtfloatx4 *)p_b, num_scalar_ops* sizeof(FLOAT32));
      ADD_SX2X2(out1, out2, x1, x2, y1, y2);
      AE_SW_SAVSX2X2_XP(out1, out2, out_a, (xtfloatx4 *)p_c,num_scalar_ops* sizeof(FLOAT32));
      AE_SA128POS_FP(out_a, (xtfloatx4 *)p_c);
    }
  }
}

static void internal_elm_add_broadcast_f32xf32_f32(FLOAT32 * __restrict__ p_out,
                    const    FLOAT32 * __restrict__ p_inp1,
                    const    FLOAT32 * __restrict__ p_inp2,
                             WORD32  num_elm)
{
  int i;
  xtfloatx4  * __restrict__ p_a = (xtfloatx4 *)p_inp1;
  xtfloatx4  * __restrict__ p_b = (xtfloatx4 *)p_inp2;
  xtfloatx4  *__restrict__  p_c =  (xtfloatx4 *)p_out;

  const int num_simd4_ops = num_elm >> 2;
  const int num_scalar_ops = num_elm & 3;

  xtfloatx2 x1, x2, y, out1, out2;
  y = AE_LSI((xtfloat *)p_b, 0);

  if(((((unsigned)p_a)&0xF) == 0) && ((((unsigned)p_c)&0xF) == 0))
  {
    for(i=0; i<num_simd4_ops; i++)
    {
      AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
      ADD_SX2X2(out1, out2, x1, x2, y, y);
      AE_SSX2X2_IP(out1, out2, p_c, 4 * sizeof(FLOAT32)); 
    }
  }
  else
  {
    ae_valignx2  vinp1, out_a = AE_ZALIGN128();
    vinp1 = AE_LA128_PP(p_a);
    for(i = 0; i < num_simd4_ops; i++)
    {
      AE_LASX2X2_IP(x1, x2, vinp1, p_a);
      ADD_SX2X2(out1, out2, x1, x2, y, y);
      AE_SASX2X2_IP(out1, out2, out_a, p_c); 
    }
    AE_SA128POS_FP(out_a, (xtfloatx4 *)p_c);
  }
  if(num_scalar_ops !=0)
  {
    ae_valignx2  vinp1, out_a = AE_ZALIGN128();
    vinp1 = AE_LA128_PP(p_a);
    AE_SW_LAVSX2X2_XP(x1, x2, vinp1, (xtfloatx4 *)p_a, num_scalar_ops* sizeof(FLOAT32));
    ADD_SX2X2(out1, out2, x1, x2, y, y);
    AE_SW_SAVSX2X2_XP(out1, out2, out_a, (xtfloatx4 *)p_c,num_scalar_ops* sizeof(FLOAT32));
    AE_SA128POS_FP(out_a, (xtfloatx4 *)p_c);
  }
}
#endif
#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_add_broadcast_4D_f32xf32_f32,
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

WORD32 xa_nn_elm_add_broadcast_4D_f32xf32_f32(FLOAT32 * __restrict__ p_out,
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
    internal_elm_add_broadcast_2D_f32xf32_f32(
                p_out,
                p_inp1,
                p_inp2,
                1,
                p_out_shape[0] * inp1_strides[0]);
  }
  else if(inp1_strides[3] == inp2_strides[3])
  {
    WORD32 in_lc, out_lc;
    in_lc = p_out_shape[2] * p_out_shape[3];
    out_lc = 1;
    if(inp1_strides[2] == 0)
    {
      const FLOAT32 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
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
        internal_elm_add_broadcast_2D_f32xf32_f32(
            p_out_tmp,
            p_inp1_tmp0,
            p_inp2_tmp0,
            out_lc,
            in_lc);
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
    if(inp1_strides[3] == 0)
    {
      const FLOAT32 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
    }
    internal_elm_add_broadcast_f32xf32_f32(
        p_out_tmp,
        p_inp1_tmp,
        p_inp2_tmp,
        p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3]);
  }
  else
  {
    if(inp1_strides[3] == 0)
    {
      const FLOAT32 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
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
            internal_elm_add_broadcast_f32xf32_f32(
                p_out_tmp,
                p_inp1_tmp1,
                p_inp2_tmp1,
                p_out_shape[3]);
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
