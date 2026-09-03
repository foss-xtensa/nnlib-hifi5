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
#include <stddef.h>
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_bcast_macro.h"

#if HAVE_HP_VFPU

static void internal_elm_mul_broadcast_2D_f16xf16_f16(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32  out_lc = args->out_lc;
  WORD32  in_lc = args->in_lc;

  int i, j;
  
  xthalf * __restrict__ p_inp1_f16 = (xthalf*)p_inp1;
  xthalf * __restrict__ p_inp2_f16 = (xthalf*)p_inp2;
  xthalf *__restrict__ p_out_f16 = (xthalf*)p_out;

  xthalfx8  * __restrict__ p_a = (xthalfx8 *)p_inp1;
  xthalfx8  * __restrict__ p_b = (xthalfx8 *)p_inp2;
  xthalfx8  *__restrict__  p_c =  (xthalfx8 *)p_out;

  int num_simd8_ops;
  int num_scalar_ops;

  num_simd8_ops = in_lc >> 3;
  num_scalar_ops = in_lc & 7;

  xthalfx4 x1, x2, y1, y2, out1, out2;
 
  for(i = 0; i < out_lc; i++)
  {
    p_a = (xthalfx8 *)&p_inp1_f16[i * in_lc];
    p_b = (xthalfx8 *)p_inp2_f16;
    p_c = (xthalfx8 *)&p_out_f16[i * in_lc];
    if(((((unsigned)p_a)&0xF) == 0) && ((((unsigned)p_b)&0xF) == 0) && ((((unsigned)p_c)&0xF) == 0))
    {
      for(j = 0; j < num_simd8_ops; j++)
      {
        AE_LHX4X2_IP(x1, x2, p_a, 8 * sizeof(xthalf));
        AE_LHX4X2_IP(y1, y2, p_b, 8 * sizeof(xthalf));
        out1 = MUL_HX4(x1, y1);
        out2 = MUL_HX4(x2, y2);

        AE_SHX4X2_IP(out1, out2, p_c, 8 * sizeof(xthalf));
      }
    }
    else
    {
      ae_valignx2  vinp1, vinp2, out_a = AE_ZALIGN128();
      vinp1 = AE_LA128_PP(p_a);
      vinp2 = AE_LA128_PP(p_b);
      for(j = 0; j < num_simd8_ops; j++)
      {
        AE_LAHX4X2_IP(x1, x2, vinp1, p_a);
        AE_LAHX4X2_IP(y1, y2, vinp2, p_b);
        MUL_HX4X2(out1, out2, x1, x2, y1, y2);
        AE_SAHX4X2_IP(out1, out2, out_a, p_c);
      }
      AE_SA128POS_FP(out_a, p_c);
    }
    if(num_scalar_ops != 0)
    {
      ae_valignx2  vinp1, vinp2, out_a = AE_ZALIGN128();
      vinp1 = AE_LA128_PP(p_a);
      vinp2 = AE_LA128_PP(p_b);
      AE_LAVHX4X2_XP(x1, x2, vinp1, p_a, num_scalar_ops * sizeof(xthalf));
      AE_LAVHX4X2_XP(y1, y2, vinp2, p_b, num_scalar_ops * sizeof(xthalf));
      MUL_HX4X2(out1, out2, x1, x2, y1, y2);
      AE_SAVHX4X2_XP(out1, out2, out_a, p_c, num_scalar_ops * sizeof(xthalf));
      AE_SA128POS_FP(out_a, p_c);
    }
  }
}

static void internal_elm_mul_broadcast_f16xf16_f16(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32  num_elm = args->num_elm;
  
  int i;
  xthalfx8  * __restrict__ p_a = (xthalfx8 *)p_inp1;
  xthalfx8  * __restrict__ p_b = (xthalfx8 *)p_inp2;
  xthalfx8  *__restrict__  p_c =  (xthalfx8 *)p_out;

  const int num_simd8_ops = num_elm >> 3;
  const int num_scalar_ops = num_elm & 7;

  xthalfx4 x1, x2, y1, y2, out1, out2;
  xthalf *pf16_b = (xthalf *)p_b;
  WORD16 y_scalar = *((WORD16*)pf16_b);
  y1 = AE_MOVHALFX4_FROMF16X4(AE_MOVF16X4_FROMINT16X4(AE_MOVDA16(y_scalar)));
  y2 = y1;
        
  if(((((unsigned)p_a)&0xF) == 0) && ((((unsigned)p_c)&0xF) == 0))
  {
    for(i = 0; i < num_simd8_ops; i++)
    {
      AE_LHX4X2_IP(x1, x2, p_a, 8 * sizeof(xthalf));
      MUL_HX4X2(out1, out2, x1, x2, y1, y2);
      AE_SHX4X2_IP(out1, out2, p_c, 8 * sizeof(xthalf));
    }

  }
  else
  {
    ae_valignx2  vinp1, out_a = AE_ZALIGN128();
    vinp1 = AE_LA128_PP(p_a);
    for(i = 0; i < num_simd8_ops; i++)
    {
      AE_LAHX4X2_IP(x1, x2, vinp1, p_a);
      MUL_HX4X2(out1, out2, x1, x2, y1, y2);
      AE_SAHX4X2_IP(out1, out2, out_a, p_c);
    }
    AE_SA128POS_FP(out_a, p_c);
  }
  if(num_scalar_ops != 0)
  {
    ae_valignx2  vinp1, out_a = AE_ZALIGN128();
    vinp1 = AE_LA128_PP(p_a);
    AE_LAVHX4X2_XP(x1, x2, vinp1, p_a, num_scalar_ops * sizeof(xthalf));
    MUL_HX4X2(out1, out2, x1, x2, y1, y2);
    AE_SAVHX4X2_XP(out1, out2, out_a, p_c, num_scalar_ops * sizeof(xthalf));
    AE_SA128POS_FP(out_a, p_c);
  }
}
#endif

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_mul_broadcast_4D_f16xf16_f16,
             (
                      WORD16 * p_out,
                      const WORD32 *const p_out_shape,
                      const WORD16 * p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const WORD16 * p_inp2,
                      const WORD32 *const p_inp2_shape
              )
           )
#else           
WORD32 xa_nn_elm_mul_broadcast_4D_f16xf16_f16(WORD16 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                      const WORD16 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const WORD16 * __restrict__ p_inp2,
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
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);

  bcast_args_t args = {0};
  args.out_elm_size = args.inp_elm_size = 2;
  args.multiplier_sign = 1;

  return CALL_BCAST(internal_elm_mul_broadcast_2D_f16xf16_f16, 
            internal_elm_mul_broadcast_f16xf16_f16,
            p_out,
            p_out_shape,
            p_inp1,
            p_inp1_shape,
            p_inp2,
            p_inp2_shape,
            &args);
}
#endif
