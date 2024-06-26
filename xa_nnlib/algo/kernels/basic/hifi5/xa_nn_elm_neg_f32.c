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
#include "xa_nnlib_common.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_neg_f32_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp,
                WORD32 num_elm
              )
           )
#else
WORD32 xa_nn_elm_neg_f32_f32(FLOAT32 * __restrict__ p_out,
                             const FLOAT32 * __restrict__ p_inp,
                             WORD32 num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

  int i = 0;
  xtfloatx4 *inp = (xtfloatx4 *)p_inp;
  xtfloatx4 *out = (xtfloatx4 *)p_out;
  xtfloatx2 x1, x2, x3, x4, y1, y2, y3, y4;

  if(((((unsigned)p_out) & 15) == 0) && ((((unsigned)p_inp) & 15) == 0))
  {
#pragma no_unroll
    for(i = 0; i < (num_elm >> 3); i++)
    {
      AE_LSX2X2_IP(x1, x2, inp, 4*sizeof(FLOAT32));
      AE_LSX2X2_IP(x3, x4, inp, 4*sizeof(FLOAT32));
      NEG_SX2X2(y1, y2, x1, x2);
      NEG_SX2X2(y3, y4, x3, x4);
      AE_SSX2X2_IP(y1, y2, out, 4*sizeof(FLOAT32));
      AE_SSX2X2_IP(y3, y4, out, 4*sizeof(FLOAT32));
    }
  }
  else
  {
    ae_valignx2 inp_a, out_a;

    inp_a = AE_LA128_PP(inp);
    out_a = AE_ZALIGN128();

#pragma no_unroll
    for(i = 0; i < (num_elm >> 3); i++)
    {
      AE_LASX2X2_IP(x1, x2, inp_a, inp);
      AE_LASX2X2_IP(x3, x4, inp_a, inp);
      NEG_SX2X2(y1, y2, x1, x2);
      NEG_SX2X2(y3, y4, x3, x4);
      AE_SASX2X2_IP(y1, y2, out_a, out);
      AE_SASX2X2_IP(y3, y4, out_a, out);
    }
    AE_SA128POS_FP(out_a, out);
  }

  // Remainder Loop
  int rem_itr = num_elm & (7);
  for(i = 0; i < rem_itr; i++)
  {
    xtfloat a1, a;
    AE_LSIP(a1, (xtfloat *)inp, sizeof(FLOAT32));
    a = NEG_S(a1);
    AE_SSIP(a, (xtfloat *)out, sizeof(FLOAT32));
  }

  return 0;
}
#endif
