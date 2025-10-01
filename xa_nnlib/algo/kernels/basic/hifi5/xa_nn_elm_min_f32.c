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
#include "xa_nnlib_common_bcast_macro.h"


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
             WORD32, xa_nn_elm_min_f32xf32_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp1,
                const FLOAT32 *p_inp2,
                WORD32 num_elm
              )
           )
#else
WORD32 xa_nn_elm_min_f32xf32_f32(FLOAT32 * __restrict__ p_out,
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
    xtfloatx4 *inp1 = (xtfloatx4 *)p_inp1;
    xtfloatx4 *inp2 = (xtfloatx4 *)p_inp2;
    xtfloatx4 *out =  (xtfloatx4 *)p_out;
    xtfloatx2 x1, x1_1, x2, x2_1, y, y_1;

    if(((((unsigned)p_out)&0xF) == 0) && ((((unsigned)p_inp1)&0xF) == 0) && ((((unsigned)p_inp2)&0xF) == 0))
    {
      for(i=0;i < num_elm>>2;i++)
      {
        AE_LSX2X2_IP(x1, x1_1, inp1, 4*sizeof(FLOAT32));
        AE_LSX2X2_IP(x2, x2_1, inp2, 4*sizeof(FLOAT32));
        y = MIN_SX2(x1, x2);
        y_1 = MIN_SX2(x1_1, x2_1);
        AE_SSX2X2_IP(y, y_1, out,  4*sizeof(FLOAT32));
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
      for(i=0;i < num_elm>>2;i++)
      {
        AE_LASX2X2_IP(x1, x1_1, inp1_a, inp1);
        AE_LASX2X2_IP(x2, x2_1, inp2_a, inp2);
        y = MIN_SX2(x1, x2);
        y_1 = MIN_SX2(x1_1, x2_1);
        AE_SASX2X2_IP(y, y_1, out_a, out);
      }
      AE_SA128POS_FP(out_a, out);
    }
    // Remainder Loop
    int rem  = num_elm & 3;
    if(rem)
    {
      ae_valignx2 inp1_a, inp2_a, out_a;
      inp1_a = AE_LA128_PP(inp1);
      inp2_a = AE_LA128_PP(inp2);
      out_a = AE_ZALIGN128();
      AE_SW_LAVSX2X2_XP(x1, x1_1, inp1_a, inp1, rem* sizeof(FLOAT32));
      AE_SW_LAVSX2X2_XP(x2, x2_1, inp2_a, inp2, rem* sizeof(FLOAT32));
      y = MIN_SX2(x1, x2);
      y_1 = MIN_SX2(x1_1, x2_1);
      AE_SW_SAVSX2X2_XP(y, y_1, out_a, out,rem* sizeof(FLOAT32));
      AE_SA128POS_FP(out_a, out);
    }
    return 0;
}
#endif

#if HAVE_VFPU
static void internal_elm_min_2D_Bcast_f32xf32_f32(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32  out_lc = args->out_lc;
  WORD32  in_lc = args->in_lc;
  int i, j;

  FLOAT32 * __restrict__ p_inp1_f32 = (FLOAT32*)p_inp1;
  FLOAT32 * __restrict__ p_inp2_f32 = (FLOAT32*)p_inp2;
  FLOAT32 *__restrict__ p_out_f32 = (FLOAT32*)p_out;
  xtfloatx4  * __restrict__ p_a = (xtfloatx4 *)p_inp1;
  xtfloatx4  * __restrict__ p_b = (xtfloatx4 *)p_inp2; 
  xtfloatx4  *__restrict__  p_c =  (xtfloatx4 *)p_out;

  int num_simd4_ops;
  int num_scalar_ops;

  num_simd4_ops = in_lc >> 2;
  num_scalar_ops = in_lc & 3;

  xtfloatx2 x1, x2, y1, y2, out1, out2;
 
  for(i = 0; i < out_lc; i++)
  {
    p_a = (xtfloatx4 *)&p_inp1_f32[i * in_lc];
    p_b = (xtfloatx4 *)p_inp2_f32;
    p_c = (xtfloatx4 *)&p_out_f32[i * in_lc];
    if(((((unsigned)p_a)&0xF) == 0) && ((((unsigned)p_b)&0xF) == 0) && ((((unsigned)p_c)&0xF) == 0))
    {
      for(j = 0; j < num_simd4_ops; j++)
      {
        AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
        AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));
        out1= MIN_SX2(x1,y1);
        out2= MIN_SX2(x2,y2);
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
        out1= MIN_SX2(x1,y1);
        out2= MIN_SX2(x2,y2);
        AE_SASX2X2_IP(out1, out2, out_a, p_c); 
      }
      AE_SA128POS_FP(out_a, p_c);
    }
    if(num_scalar_ops !=0)
    {
      ae_valignx2  vinp1, vinp2,out_a = AE_ZALIGN128();
      vinp1 = AE_LA128_PP(p_a);
      vinp2 = AE_LA128_PP(p_b);
      AE_SW_LAVSX2X2_XP(x1, x2, vinp1, p_a, num_scalar_ops* sizeof(FLOAT32));
      AE_SW_LAVSX2X2_XP(y1, y2, vinp2, p_b, num_scalar_ops* sizeof(FLOAT32));
      out1= MIN_SX2(x1,y1);
      out2= MIN_SX2(x2,y2);
      AE_SW_SAVSX2X2_XP(out1, out2, out_a, p_c,num_scalar_ops* sizeof(FLOAT32));
      AE_SA128POS_FP(out_a, p_c);
    }
  }
}

static void internal_elm_min_Bcast_f32xf32_f32(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32  num_elm = args->num_elm;
  int i;
  xtfloatx4  * __restrict__ p_a = (xtfloatx4 *)p_inp1;
  xtfloatx4  * __restrict__ p_b = (xtfloatx4 *)p_inp2; 
  xtfloatx4  *__restrict__  p_c =  (xtfloatx4 *)p_out;

  const int num_simd4_ops = num_elm >> 2;
  const int num_scalar_ops = num_elm & 3;

  xtfloatx2 x1, x1_1, x2, y1, y1_1;
  xtfloat *pfloat_b = (xtfloat *)p_b;
  x2 = AE_MOVXTFLOATX2_FROMXTFLOAT(AE_LSI(pfloat_b, 0));

  if(((((unsigned)p_a)&0xF) == 0) && ((((unsigned)p_c)&0xF) == 0))
  {
    for(i=0; i<num_simd4_ops; i++)
    {
      AE_LSX2X2_IP(x1, x1_1, p_a, 4 * sizeof(FLOAT32));
      y1= MIN_SX2(x1,x2);
      y1_1= MIN_SX2(x1_1,x2);
      AE_SSX2X2_IP(y1, y1_1, p_c, 4 * sizeof(FLOAT32)); 
    }
  }
  else
  {
      ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
      inp1_a = AE_LA128_PP(p_a);
      for(i=0; i<num_simd4_ops; i++)
      {
        AE_LASX2X2_IP(x1, x1_1, inp1_a, p_a);
        y1= MIN_SX2(x1,x2);
        y1_1= MIN_SX2(x1_1,x2);
        AE_SASX2X2_IP(y1,y1_1, out_a, p_c);
      }
      AE_SA128POS_FP(out_a, p_c);
  }
  if(num_scalar_ops!=0){
    ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
    inp1_a = AE_LA128_PP(p_a);
    AE_SW_LAVSX2X2_XP(x1, x1_1, inp1_a, p_a, num_scalar_ops* sizeof(FLOAT32));
    y1= MIN_SX2(x1,x2);
    y1_1= MIN_SX2(x1_1,x2);
    AE_SW_SAVSX2X2_XP(y1, y1_1, out_a, p_c,num_scalar_ops* sizeof(FLOAT32));
    AE_SA128POS_FP(out_a, p_c);  
  }
}
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_min_4D_Bcast_f32xf32_f32,
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
WORD32 xa_nn_elm_min_4D_Bcast_f32xf32_f32(FLOAT32 * __restrict__ p_out,
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

  bcast_args_t args = {0};
  args.out_elm_size = args.inp_elm_size = 4;
  args.multiplier_sign = 1;
  return CALL_BCAST(internal_elm_min_2D_Bcast_f32xf32_f32, 
            internal_elm_min_Bcast_f32xf32_f32,
            p_out,
            p_out_shape,
            p_inp1,
            p_inp1_shape,
            p_inp2,
            p_inp2_shape,
            &args);
}
#endif