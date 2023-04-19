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
#include "xa_type_def.h"
#include "common.h"
#include "xa_nnlib_err_chk.h"

#define LIMIT(out, inp, min, max){\
        out = AE_MIN8(inp, max);\
        out = AE_MAX8(out, min);\
}

#define MAX_WORD8 127
#define MIN_WORD8 -128

/*
 * inp: p_vec: 1 byte aligned input pointer
 * out: p_out: no alignment needed for output pointer*/
WORD32 xa_nn_vec_activation_min_max_8_8(WORD8 * __restrict__ p_out,
                                      const  WORD8 * __restrict__ p_vec,
                                      int    activation_min,
                                      int    activation_max,
                                      WORD32 vec_length)
{
    int i;
    ae_int8x8 x, y, min, max;
    ae_valignx2 align_src, align_dst;

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_vec, -1);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((activation_max < activation_min), -1);

    WORD8 *p_o = p_out;
    WORD8 *p_v = (WORD8 *)p_vec;

    min  = AE_MOVDA8(activation_min);
    max  = AE_MOVDA8(activation_max);

    align_src = AE_LA128_PP((ae_int8x16 *)p_v);
    align_dst = AE_ZALIGN128();

    if((activation_max >= (int)MAX_WORD8) && (activation_min <= (int)MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 4); i++)
        {
            AE_LA8X8X2_IP(x, y, align_src, (ae_int8x16 *)p_v);

            AE_SA8X8X2_IP(x, y, align_dst, (ae_int8x16 *)p_o);
        }
        int rem_itr = (vec_length & 15);
        {
            AE_LAV8X8X2_XP(x, y, align_src, (ae_int8x16 *)p_v, rem_itr);

            AE_SAV8X8X2_XP(x, y, align_dst, (ae_int8x16 *)p_o, rem_itr);
        }
        AE_SA128POS_FP(align_dst, p_o);
    }
    else if((activation_max < (int)MAX_WORD8) && (activation_min <= MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 4); i++)
        {
            AE_LA8X8X2_IP(x, y, align_src, (ae_int8x16 *)p_v);

            x = AE_MIN8(x, max);
            y = AE_MIN8(y, max);

            AE_SA8X8X2_IP(x, y, align_dst, (ae_int8x16 *)p_o);
        }
        int rem_itr = (vec_length & 15);
        {
            AE_LAV8X8X2_XP(x, y, align_src, (ae_int8x16 *)p_v, rem_itr);

            x = AE_MIN8(x, max);
            y = AE_MIN8(y, max);

            AE_SAV8X8X2_XP(x, y, align_dst, (ae_int8x16 *)p_o, rem_itr);
        }
        AE_SA128POS_FP(align_dst, p_o);
    }
    else if((activation_max >= (int)MAX_WORD8) && (activation_min > MIN_WORD8))
    {
        for(i=0; i<(vec_length >> 4); i++)
        {
            AE_LA8X8X2_IP(x, y, align_src, (ae_int8x16 *)p_v);

            x = AE_MAX8(x, min);
            y = AE_MAX8(y, min);

            AE_SA8X8X2_IP(x, y, align_dst, (ae_int8x16 *)p_o);
        }
        int rem_itr = (vec_length & 15);
        {
            AE_LAV8X8X2_XP(x, y, align_src, (ae_int8x16 *)p_v, rem_itr);

            x = AE_MAX8(x, min);
            y = AE_MAX8(y, min);

            AE_SAV8X8X2_XP(x, y, align_dst, (ae_int8x16 *)p_o, rem_itr);
        }
        AE_SA128POS_FP(align_dst, p_o);
    }
    else
    {
        for(i=0; i<(vec_length >> 4); i++)
        {
            AE_LA8X8X2_IP(x, y, align_src, (ae_int8x16 *)p_v);

            LIMIT(x, x, min, max)
            LIMIT(y, y, min, max)

            AE_SA8X8X2_IP(x, y, align_dst, (ae_int8x16 *)p_o);
        }
        int rem_itr = (vec_length & 15);
        {
            AE_LAV8X8X2_XP(x, y, align_src, (ae_int8x16 *)p_v, rem_itr);

            LIMIT(x, x, min, max)
            LIMIT(y, y, min, max)

            AE_SAV8X8X2_XP(x, y, align_dst, (ae_int8x16 *)p_o, rem_itr);
        }
        AE_SA128POS_FP(align_dst, p_o);
    }

    return 0;
}

/*
 * ReLU 8-bit:
 */
WORD32 xa_nn_vec_relu_8_8(
    WORD8        * __restrict__ p_out,
    const WORD8  * __restrict__ p_vec,
    WORD8       threshold,
    WORD32       vec_length)
{
    xa_nn_vec_activation_min_max_8_8( p_out,
                                      p_vec,
                                      0,
                                      threshold,
                                      vec_length);
    return 0;
}

/*
 * ReLU Standard 8-bit:
 */
WORD32 xa_nn_vec_relu_std_8_8(
    WORD8        * __restrict__ p_out,
    const WORD8  * __restrict__ p_vec,
    WORD32       vec_length)
{
    xa_nn_vec_activation_min_max_8_8( p_out,
                                      p_vec,
                                      0,
                                      MAX_WORD8,
                                      vec_length);
  return 0;
}
