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

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_memset_f32_f32,
             (
                FLOAT32 * __restrict__ p_out,
                FLOAT32 val,
                WORD32 num_elm
              )
           )
#else

WORD32 xa_nn_memset_f32_f32(FLOAT32 * __restrict__ p_out,
                                FLOAT32 val,
                                WORD32 num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

  FLOAT32 valueArray[4] = {val, val, val, val};

  int i;
  xtfloatx4 *out =  (xtfloatx4 *)p_out;
  xtfloatx4 *inp  =  (xtfloatx4 *) valueArray;
  xtfloatx2 x1, x2;//, y1, y2, y3, y4;

  //Loading input values
  ae_valignx2 inp_a;
  if( (((unsigned)valueArray) & 15) == 0)
  {
        AE_LSX2X2_IP(x1, x2, inp, 4*sizeof(FLOAT32));
  }
  else
     {
    inp_a = AE_LA128_PP(inp);
    AE_LASX2X2_IP(x1, x2, inp_a, inp);
     }
    
  if( (((unsigned)p_out) & 15) == 0)
  {
#pragma loop_count factor=4
    for(i=0;i < num_elm>>2;i++)
    {
        AE_SSX2X2_IP(x1, x2, out, 4*sizeof(FLOAT32));
    }
  }
  else
  {
      ae_valignx2 out_a;

        out_a = AE_ZALIGN128();
#pragma concurrent
#pragma loop_count factor=4
        for(i=0;i < num_elm>>2;i++)
        {
            AE_SASX2X2_IP(x1, x2, out_a, out);
        }
        AE_SA128POS_FP(out_a, out);
  }
  // Remainder Loop
  // i <<= 2;
#pragma loop_count min=0,max=3
    
 xtfloat a;
 xtfloat *out_t = (xtfloat *)out;
 xtfloat *inp_val = (xtfloat *) valueArray;
 AE_LSIP(a, inp_val, sizeof(FLOAT32));

  for(i = 0; i < (num_elm&3) ; i++)
  {
    AE_SSIP(a, out_t, sizeof(FLOAT32));
  }

  return 0;
}

#endif /* !HAVE_VFPU */
