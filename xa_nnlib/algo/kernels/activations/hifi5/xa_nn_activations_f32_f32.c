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
/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"
#include "xa_type_def.h"
#include "../../../ndsp/hifi5/include/NatureDSP_Signal_math.h"
#include "xa_nnlib_err_chk.h"
#include <math.h>

#define LIMIT_SX2(out, inp, min, max){\
        out = MAX_SX2(min, inp); \
        out = MIN_SX2(max, out); \
}

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_activation_min_max_f32_f32,(
            FLOAT32 *  p_out,
    const   FLOAT32 *  p_vec,
            FLOAT32    activation_min,
            FLOAT32    activation_max,
            WORD32     vec_length))
#else
/*xa_nn_vec_activation_min_max_f32_f32()
 * inp: p_vec: 4 byte aligned pointer
 * out: p_out: 4 byte aligned pointer */

WORD32 xa_nn_vec_activation_min_max_f32_f32(FLOAT32 * __restrict__ p_out,
           const  FLOAT32 * __restrict__ p_vec,
                  FLOAT32 activation_min,
                  FLOAT32 activation_max,
                  WORD32  vec_length)
{
    int i, N0;
    xtfloatx2 x, y, l, m, min, max;
    xtfloat z;
    xtfloatx4 *pi, *po;
    ae_valignx2 align_inp, align_out; 

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

    pi = (xtfloatx4 *)p_vec;
    po = (xtfloatx4 *)p_out;

    min  = (xtfloatx2) activation_min;
    max  = (xtfloatx2) activation_max;

    align_inp = AE_LA128_PP(pi);
    align_out = AE_ZALIGN128();

    if(activation_max == INFINITY)
    {
        if (vec_length<=7)
        {
            __Pragma("no_unroll")
            for (i=0; i<vec_length; i++)
            {
                AE_LSIP(z, (xtfloat *)pi, sizeof(xtfloat));
                z = MAX_S(min, z);
                AE_SSIP(z, (xtfloat *)po, sizeof(xtfloat));
            }
            return 0;
        }

        AE_LASX2X2_IP(x, y, align_inp, pi);
        AE_LASX2X2_IP(l, m, align_inp, pi);
        x = MAX_SX2(min, x);
        y = MAX_SX2(min, y);
        l = MAX_SX2(min, l);
        m = MAX_SX2(min, m);
        AE_SASX2X2_IP(x, y, align_out, po);
        AE_SASX2X2_IP(l, m, align_out, po);
        AE_SA128POS_FP(align_out, po);
        N0=((vec_length-1)&7)+1;
        vec_length-=N0;
        if (vec_length<=0) return 0;
        pi=(xtfloatx4 *)(p_vec+N0);
        po=(xtfloatx4 *)(p_out+N0);
        align_inp=AE_LA128_PP(pi);
        for (i=0; i<(vec_length>>3); i++)
        {
            AE_LASX2X2_IP(x, y, align_inp, pi);
            AE_LASX2X2_IP(l, m, align_inp, pi);
            x = MAX_SX2(min, x);
            y = MAX_SX2(min, y);
            l = MAX_SX2(min, l);
            m = MAX_SX2(min, m);
            AE_SASX2X2_IP(x, y, align_out, po);
            AE_SASX2X2_IP(l, m, align_out, po);
        }
        AE_SA128POS_FP(align_out, po);

    }
    else
    {
        if (vec_length<=7)
        {
            __Pragma("no_unroll")
            for (i=0; i<vec_length; i++)
            {
                AE_LSIP(z, (xtfloat *)pi, sizeof(xtfloat));
                z = MAX_S(min, z);
                z = MIN_S(max, z);
                AE_SSIP(z, (xtfloat *)po, sizeof(xtfloat));
            }
            return 0;
        }

        AE_LASX2X2_IP(x, y, align_inp, pi);
        AE_LASX2X2_IP(l, m, align_inp, pi);
        LIMIT_SX2(x, x, min, max)
        LIMIT_SX2(y, y, min, max)
        LIMIT_SX2(l, l, min, max)
        LIMIT_SX2(m, m, min, max)
        AE_SASX2X2_IP(x, y, align_out, po);
        AE_SASX2X2_IP(l, m, align_out, po);
        AE_SA128POS_FP(align_out, po);
        N0=((vec_length-1)&7)+1;
        vec_length-=N0;
        if (vec_length<=0) return 0;
        pi=(xtfloatx4 *)(p_vec+N0);
        po=(xtfloatx4 *)(p_out+N0);
        align_inp=AE_LA128_PP(pi);
        for (i=0; i<(vec_length>>3); i++)
        {
            AE_LASX2X2_IP(x, y, align_inp, pi);
            AE_LASX2X2_IP(l, m, align_inp, pi);
            LIMIT_SX2(x, x, min, max)
            LIMIT_SX2(y, y, min, max)
            LIMIT_SX2(l, l, min, max)
            LIMIT_SX2(m, m, min, max)
            AE_SASX2X2_IP(x, y, align_out, po);
            AE_SASX2X2_IP(l, m, align_out, po);
        }
        AE_SA128POS_FP(align_out, po);
    }

    return 0;
}
#endif

#if HAVE_VFPU

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_sigmoid_f32_f32,(
    FLOAT32       *  p_out,
    const FLOAT32 *  p_vec,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_sigmoid_f32_f32(
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  xa_nnlib_vec_sigmoidf(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_tanh_f32_f32,(
    FLOAT32       *  p_out,
    const FLOAT32 *  p_vec,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_tanh_f32_f32(
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  xa_nnlib_vec_tanhf(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_relu_std_f32_f32,(
    FLOAT32       *  p_out,
    const FLOAT32 *  p_vec,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_relu_std_f32_f32(
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
    xa_nn_vec_activation_min_max_f32_f32(p_out, p_vec, 0, INFINITY, vec_length);
    return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_relu_f32_f32,(
    FLOAT32       *  p_out,
    const FLOAT32 *  p_vec,
    FLOAT32       threshold,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_relu_f32_f32(
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    FLOAT32       threshold,                   /* threshold, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  xa_nnlib_vec_reluf(p_out, p_vec, threshold, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_relu1_f32_f32,(
    FLOAT32       *  p_out,
    const FLOAT32 *  p_vec,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_relu1_f32_f32(
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  xa_nnlib_vec_reluf(p_out, p_vec, 1.0f, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_relu6_f32_f32,(
    FLOAT32       *  p_out,
    const FLOAT32 *  p_vec,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_relu6_f32_f32(
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  xa_nnlib_vec_reluf(p_out, p_vec, 6.0f, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_softmax_f32_f32,(
    FLOAT32       *  p_out,
    const FLOAT32 *  p_vec,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_softmax_f32_f32(
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  xa_nnlib_vec_softmaxf(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */
#endif

