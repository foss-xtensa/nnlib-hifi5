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

#define SW_DIV_SX2X2(vd_d0, vd_d1, vs_d0, vs_d1, vr_d0, vr_d1) \
{ \
    xtfloatx2 f0_d1, f0_d0, f1_d1, f1_d0, f2_d1, f2_d0, f22_d1, f22_d0, f3_d0, f3_d1, f4_d0, f4_d1, f5_d0, f5_d1, f55_d0, f55_d1, f6_d0, f6_d1, f7_d0, f7_d1, f8_d0, f8_d1; \
    f3_d0 = DIV0_SX2(vr_d0); \
    f3_d1 = DIV0_SX2(vr_d1); \
    f4_d0 = NEXP01_SX2(vr_d0); \
    f4_d1 = NEXP01_SX2(vr_d1); \
    CONST_SX2X2(f5_d1, f5_d0, 1); \
    MADDN_SX2(f5_d1, f4_d1, f3_d1); \
    MADDN_SX2(f5_d0, f4_d0, f3_d0); \
    MOV_SX2X2(f6_d1, f6_d0, f3_d1, f3_d0); \
    MOV_SX2X2(f7_d1, f7_d0, vr_d1, vr_d0); \
    f2_d0 = NEXP01_SX2(vs_d0); \
    f2_d1 = NEXP01_SX2(vs_d1); \
    MADDN_SX2(f6_d1, f5_d1, f6_d1); \
    MADDN_SX2(f6_d0, f5_d0, f6_d0); \
    CONST_SX2X2(f55_d1, f55_d0, 1); \
    CONST_SX2X2(f0_d1, f0_d0, 0); \
    MOV_SX2X2(f8_d1, f8_d0, f2_d1, f2_d0); \
    MADDN_SX2(f55_d1, f4_d1, f6_d1); \
    MADDN_SX2(f55_d0, f4_d0, f6_d0); \
    MSUBN_SX2(f0_d1, f8_d1, f3_d1); \
    MSUBN_SX2(f0_d0, f8_d0, f3_d0); \
    MKDADJ_SX2(f7_d0, vs_d0); \
    MKDADJ_SX2(f7_d1, vs_d1); \
    MADDN_SX2(f6_d1, f55_d1, f6_d1); \
    MADDN_SX2(f6_d0, f55_d0, f6_d0); \
    MSUBN_SX2(f8_d1, f4_d1, f0_d1); \
    MSUBN_SX2(f8_d0, f4_d0, f0_d0); \
    CONST_SX2X2(f1_d1, f1_d0, 1); \
    MADDN_SX2(f1_d1, f4_d1, f6_d1); \
    MADDN_SX2(f1_d0, f4_d0, f6_d0); \
    MSUBN_SX2(f0_d1, f8_d1, f6_d1); \
    MSUBN_SX2(f0_d0, f8_d0, f6_d0); \
    NEG_SX2X2(f22_d1, f22_d0, f2_d1, f2_d0); \
    MADDN_SX2(f6_d1, f1_d1, f6_d1); \
    MADDN_SX2(f6_d0, f1_d0, f6_d0); \
    MADDN_SX2(f22_d1, f4_d1, f0_d1); \
    MADDN_SX2(f22_d0, f4_d0, f0_d0); \
    ADDEXPM_SX2(f0_d0, f7_d0); \
    ADDEXPM_SX2(f0_d1, f7_d1); \
    ADDEXP_SX2(f6_d0, f7_d0); \
    ADDEXP_SX2(f6_d1, f7_d1); \
    DIVN_SX2(f0_d1, f22_d1, f6_d1); \
    DIVN_SX2(f0_d0, f22_d0, f6_d0); \
    MOV_SX2X2(vd_d1, vd_d0, f0_d1, f0_d0); \
}

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_div_f32xf32_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp1,
                const FLOAT32 *p_inp2,
                WORD32 num_elm
              )
           )
#else
WORD32 xa_nn_elm_div_f32xf32_f32(FLOAT32 * __restrict__ p_out,
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
        y = XT_DIV_SX2(x1, x2);
        XT_SASX2IP(y, out_a, out);
    }
    XT_SASX2POSFP(out_a, out);

    // Remainder Loop
    xtfloat *inp1_temp = (xtfloat *)inp1;
    xtfloat *inp2_temp = (xtfloat *)inp2;
    if (num_elm & 1)
    {
        xtfloat a1, a2, a;
        XT_LSIP(a1, inp1_temp, 0);
        XT_LSIP(a2, inp2_temp, 0);
        a = XT_DIV_S(a1, a2);
        XT_SSI(a, (xtfloat *)out, 0);
    }

    return 0;
}
#endif

#if HAVE_VFPU
static void internal_elm_div_broadcast_2D_f32xf32_f32(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32 out_lc = args->out_lc;
  WORD32 in_lc = args->in_lc;
  xtbool sign_flag = args->sign_flag;
  int i, j;

  xtfloatx4  * __restrict__ p_a = (xtfloatx4 *)p_inp1;
  xtfloatx4  * __restrict__ p_b = (xtfloatx4 *)p_inp2; 
  xtfloatx4  *__restrict__  p_c =  (xtfloatx4 *)p_out;
  FLOAT32 * __restrict__ p_inp1_f32 = (FLOAT32*)p_inp1;
  FLOAT32 * __restrict__ p_inp2_f32 = (FLOAT32*)p_inp2;
  FLOAT32 *__restrict__ p_out_f32 = (FLOAT32*)p_out;

  int num_simd4_ops;
  int num_scalar_ops;

  num_simd4_ops = in_lc >> 2;
  num_scalar_ops = in_lc & 3;

  xtfloatx2 x1, x1_1, x2, x2_1, y = ZERO_SX2() ,y_1= ZERO_SX2();

  /* For computing inp2 / inp1 */   
  if(AE_MOVAB(sign_flag)){  
    for(i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx4 *)&p_inp1_f32[i * in_lc];
      p_b = (xtfloatx4 *)p_inp2_f32;
      p_c = (xtfloatx4 *)&p_out_f32[i * in_lc];
      if(((((unsigned)p_a)&0xF) == 0) && ((((unsigned)p_b)&0xF) == 0) && ((((unsigned)p_c)&0xF) == 0))
      {
        for(j = 0; j < num_simd4_ops; j++)
        {
          AE_LSX2X2_IP(x1, x1_1, p_a, 4 * sizeof(FLOAT32));
          AE_LSX2X2_IP(x2, x2_1, p_b, 4 * sizeof(FLOAT32));
          SW_DIV_SX2X2(y, y_1, x2, x2_1, x1, x1_1);
          AE_SSX2X2_IP(y, y_1, p_c, 4 * sizeof(FLOAT32)); 
        }
      }
      else
      {
        ae_valignx2 vinp1, vinp2, out_a = AE_ZALIGN128();
        vinp1 = AE_LA128_PP(p_a);
        vinp2 = AE_LA128_PP(p_b);
        for(j = 0; j < num_simd4_ops; j++)
        {
          AE_LASX2X2_IP(x1, x1_1, vinp1, p_a);
          AE_LASX2X2_IP(x2, x2_1, vinp2, p_b);
          SW_DIV_SX2X2(y, y_1, x2, x2_1, x1, x1_1);
          AE_SASX2X2_IP(y, y_1, out_a, p_c); 
        }
        AE_SA128POS_FP(out_a, p_c);
      }
      if(num_scalar_ops !=0)
      {
        ae_valignx2 vinp1, vinp2, out_a = AE_ZALIGN128();
        vinp1 = AE_LA128_PP(p_a);
        vinp2 = AE_LA128_PP(p_b);
        AE_SW_LAVSX2X2_XP(x1, x1_1, vinp1, p_a, num_scalar_ops* sizeof(FLOAT32));
        AE_SW_LAVSX2X2_XP(x2, x2_1, vinp2, p_b, num_scalar_ops* sizeof(FLOAT32));
        SW_DIV_SX2X2(y, y_1, x2, x2_1, x1, x1_1);
        AE_SW_SAVSX2X2_XP(y, y_1, out_a, p_c,num_scalar_ops* sizeof(FLOAT32));
        AE_SA128POS_FP(out_a, (xtfloatx4 *)p_c);
      }      
    }
  }
  /* For computing inp1 / inp2 */   
  else
  {
    for(i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx4 *)&p_inp1_f32[i * in_lc];
      p_b = (xtfloatx4 *)p_inp2_f32;
      p_c = (xtfloatx4 *)&p_out_f32[i * in_lc];
      if(((((unsigned)p_a)&0xF) == 0) && ((((unsigned)p_b)&0xF) == 0) && ((((unsigned)p_c)&0xF) == 0))
      {
        for(j = 0; j < num_simd4_ops; j++)
        {
          AE_LSX2X2_IP(x1, x1_1, p_a, 4 * sizeof(FLOAT32));
          AE_LSX2X2_IP(x2, x2_1, p_b, 4 * sizeof(FLOAT32));
          SW_DIV_SX2X2(y, y_1, x1, x1_1, x2, x2_1);
          AE_SSX2X2_IP(y, y_1, p_c, 4 * sizeof(FLOAT32));
        }
      }
      else
      {
        ae_valignx2 vinp1, vinp2, out_a = AE_ZALIGN128();
        vinp1 = AE_LA128_PP(p_a);
        vinp2 = AE_LA128_PP(p_b);

        for(j = 0; j < num_simd4_ops; j++)
        {
          AE_LASX2X2_IP(x1, x1_1, vinp1, p_a);
          AE_LASX2X2_IP(x2, x2_1, vinp2, p_b);
          SW_DIV_SX2X2(y, y_1, x1, x1_1, x2, x2_1);
          AE_SASX2X2_IP(y, y_1, out_a, p_c); 
        }
        AE_SA128POS_FP(out_a, p_c);
      }
      if(num_scalar_ops !=0)
      {
        ae_valignx2 vinp1, vinp2, out_a = AE_ZALIGN128();
        vinp1 = AE_LA128_PP(p_a);
        vinp2 = AE_LA128_PP(p_b);
        AE_SW_LAVSX2X2_XP(x1, x1_1, vinp1, p_a, num_scalar_ops* sizeof(FLOAT32));
        AE_SW_LAVSX2X2_XP(x2, x2_1, vinp2, p_b, num_scalar_ops* sizeof(FLOAT32));
        SW_DIV_SX2X2(y, y_1, x1, x1_1, x2, x2_1);
        AE_SW_SAVSX2X2_XP(y, y_1, out_a, p_c,num_scalar_ops* sizeof(FLOAT32));
        AE_SA128POS_FP(out_a, p_c);
      }      
    }  
  }
}

static void internal_elm_div_broadcast_f32xf32_f32(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32  num_elm = args->num_elm;
  xtbool  sign_flag = args->sign_flag;
  
  int i;
  xtfloatx4  * __restrict__ p_a = (xtfloatx4 *)p_inp1;
  xtfloatx4  * __restrict__ p_b = (xtfloatx4 *)p_inp2; 
  xtfloatx4  *__restrict__  p_c =  (xtfloatx4 *)p_out;

  const int num_simd4_ops = num_elm >> 2;
  const int num_scalar_ops = num_elm & 3;

  xtfloatx2 x1,x1_1, x2, y= ZERO_SX2(),y_1= ZERO_SX2();
  xtfloat *pfloat_b = (xtfloat *)p_b;
  x2 = AE_MOVXTFLOATX2_FROMXTFLOAT(AE_LSI(pfloat_b, 0));
        
  /* For computing inp2 - inp1 */      
  if(AE_MOVAB(sign_flag)){
    if(((((unsigned)p_a)&0xF) == 0) && ((((unsigned)p_c)&0xF) == 0))
    {
      for(i=0; i<num_simd4_ops; i++)
      {
        AE_LSX2X2_IP(x1, x1_1, p_a, 4 * sizeof(FLOAT32));
        SW_DIV_SX2X2(y, y_1, x2, x2, x1, x1_1);
        AE_SSX2X2_IP(y, y_1, p_c, 4 * sizeof(FLOAT32));
      }
    }
    else
    {
      ae_valignx2 vinp1, out_a = AE_ZALIGN128();
      vinp1 = AE_LA128_PP(p_a);
      for(i=0; i<num_simd4_ops; i++)
      {
        AE_LASX2X2_IP(x1, x1_1, vinp1, p_a);
        SW_DIV_SX2X2(y, y_1, x2, x2, x1, x1_1);
        AE_SASX2X2_IP(y, y_1, out_a, p_c); 
      }
      AE_SA128POS_FP(out_a, p_c);
    }  
    if(num_scalar_ops !=0)
    {
      ae_valignx2 vinp1, out_a = AE_ZALIGN128();
      vinp1 = AE_LA128_PP(p_a);
      AE_SW_LAVSX2X2_XP(x1, x1_1, vinp1, p_a, num_scalar_ops* sizeof(FLOAT32));
      SW_DIV_SX2X2(y, y_1, x2, x2, x1, x1_1);
      AE_SW_SAVSX2X2_XP(y, y_1, out_a, p_c,num_scalar_ops* sizeof(FLOAT32));
      AE_SA128POS_FP(out_a, p_c);
    }
  }
  /* For computing inp1 - inp2 */   
  else
  {
    if(((((unsigned)p_a)&0xF) == 0) && ((((unsigned)p_c)&0xF) == 0))
    {
      for(i=0; i<num_simd4_ops; i++)
      {
        AE_LSX2X2_IP(x1, x1_1, p_a, 4 * sizeof(FLOAT32));
        SW_DIV_SX2X2(y, y_1, x1, x1_1, x2, x2);
        AE_SSX2X2_IP(y, y_1, p_c, 4 * sizeof(FLOAT32));
      }
    }
    else
    {
      ae_valignx2 vinp1, out_a = AE_ZALIGN128();
      vinp1 = AE_LA128_PP(p_a);
      for(i=0; i<num_simd4_ops; i++)
      {
        AE_LASX2X2_IP(x1, x1_1, vinp1, p_a);
        SW_DIV_SX2X2(y, y_1, x1, x1_1, x2, x2);
        AE_SASX2X2_IP(y, y_1, out_a, p_c); 
      }
      AE_SA128POS_FP(out_a, p_c);
    }
    if(num_scalar_ops !=0)
    {
      ae_valignx2 vinp1, out_a = AE_ZALIGN128();
      vinp1 = AE_LA128_PP(p_a);
      AE_SW_LAVSX2X2_XP(x1, x1_1, vinp1, p_a, num_scalar_ops* sizeof(FLOAT32));
      SW_DIV_SX2X2(y, y_1, x1, x1_1, x2, x2);
      AE_SW_SAVSX2X2_XP(y, y_1, out_a, p_c,num_scalar_ops* sizeof(FLOAT32));
      AE_SA128POS_FP(out_a, p_c);
    }    
  }
}
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_div_broadcast_4D_f32xf32_f32,
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
WORD32 xa_nn_elm_div_broadcast_4D_f32xf32_f32(FLOAT32 * __restrict__ p_out,
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

  return CALL_BCAST(internal_elm_div_broadcast_2D_f32xf32_f32, 
            internal_elm_div_broadcast_f32xf32_f32,
            p_out,
            p_out_shape,
            p_inp1,
            p_inp1_shape,
            p_inp2,
            p_inp2_shape,
            &args);
}
#endif
