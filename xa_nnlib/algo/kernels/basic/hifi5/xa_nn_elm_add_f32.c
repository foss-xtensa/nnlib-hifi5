/*******************************************************************************
* Copyright (c) 2022 Cadence Design Systems, Inc.
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
#include "common_fpu.h"
#include "xa_nnlib_common.h"

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
