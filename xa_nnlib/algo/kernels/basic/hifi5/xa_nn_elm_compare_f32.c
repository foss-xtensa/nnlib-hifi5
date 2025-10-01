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
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common_bcast_macro.h"

#ifdef AE_LAVSX2X2_XP
  #define AE_SW_LAVSX2X2_XP(d1, d2, va, ptr, off)  AE_LAVSX2X2_XP(d1, d2, va, ptr, off)
#else
  #define AE_SW_LAVSX2X2_XP(d1, d2, va, ptr, off) \
  { \
    ae_int16x4 d_out16_0, d_out16_1; \
    ae_int16x8 *ptr_temp = (ae_int16x8 *)ptr;\
    AE_LAV16X4X2_XP(d_out16_0, d_out16_1, va, ptr_temp, off); \
    d_out16_0 = AE_SEL16_2301(d_out16_0, d_out16_0); \
    d_out16_1 = AE_SEL16_2301(d_out16_1, d_out16_1); \
    d1 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_out16_0)); \
    d2 = AE_MOVXTFLOATX2_FROMINT32X2(AE_MOVINT32X2_FROMINT16X4(d_out16_1)); \
    ptr = (xtfloatx4 *)ptr_temp; \
  }
#endif

#if XCHAL_HAVE_HIFI5S
#define AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4) \
{ \
    xtbool4 check34; \
    xtbool4 check12; \
    xtbool8 check; \
    check12 = AE_JOINB4B2(check1, check2); \
    check34 = AE_JOINB4B2(check3, check4); \
    check = AE_JOINB(check12, check34); \
    AE_MOVT8X8(result, ones, check); \
}

#define AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4) \
{ \
    xtbool4 check34; \
    xtbool4 check12; \
    xtbool8 check; \
    check12 = AE_JOINB4B2(check1, check2); \
    check34 = AE_JOINB4B2(check3, check4); \
    check = AE_JOINB(check12, check34); \
    AE_MOVF8X8(result, ones, check); \
}

#else
#define AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4) \
{ \
    int val = (AE_MOVAB2(check1) << 6) + (AE_MOVAB2(check2) << 4) + (AE_MOVAB2(check3) << 2) + AE_MOVAB2(check4) ; \
    ae_int8x8 temp; \
    AE_MOVT8X16_L(temp, result, result, ones, val); \
}

#define AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4) \
{ \
    int val = (AE_MOVAB2(check1) << 6) + (AE_MOVAB2(check2) << 4) + (AE_MOVAB2(check3) << 2) + AE_MOVAB2(check4) ; \
    ae_int8x8 temp; \
    AE_MOVT8X16_L(temp, result, ones, result, val); \
}

#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_compare_f32xf32_f32,
             (
                WORD8 *y,
                const FLOAT32 *x1,
                const FLOAT32 *x2,
                WORD32 N,
                compare_ops_t kernel_type
              )
           )
#else
WORD32 xa_nn_elm_compare_f32xf32_f32(WORD8 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp1,
                               const FLOAT32 * __restrict__ p_inp2,
                               WORD32 num_elm,
                               compare_ops_t kernel_type)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0) || (kernel_type < 0) || (kernel_type > 5), -1);
    int i;
    xtfloatx4 *inp1 = (xtfloatx4 *)p_inp1;
    xtfloatx4 *inp2 = (xtfloatx4 *)p_inp2;
    ae_int8x8 *out = (ae_int8x8 *)p_out;
    xtfloatx2 x1, x2, y1, y2;

    ae_int8x8 result;
    ae_int8x8 ones = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x01010101L, 0x01010101L));
    
    if(kernel_type == COMPARE_GREATEREQUAL)
    {  
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&0xF) == 0) && ((((unsigned)p_inp2)&0xF) == 0))
      {
          for(i=0;i < num_elm>>3;i++)
          {
              result =  AE_MOVDA8(0);
              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));
              
              xtbool2 check1 = XT_OLE_SX2(y1, x1);
              xtbool2 check2 = XT_OLE_SX2(y2, x2);

              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));

              xtbool2 check3 = XT_OLE_SX2(y1, x1);
              xtbool2 check4 = XT_OLE_SX2(y2, x2);

              AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
              AE_S8X8_IP(result, out, 8);
          }
      }
      else
      {
        ae_valignx2 inp1_a, inp2_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(inp1);
        inp2_a = AE_LA128_PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>3;i++)
          {
            result =  AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check1 = XT_OLE_SX2(y1, x1);
            xtbool2 check2 = XT_OLE_SX2(y2, x2);

            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check3 = XT_OLE_SX2(y1, x1);
            xtbool2 check4 = XT_OLE_SX2(y2, x2);
            
            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, out);
          }
          AE_SA64POS_FP(su, out);
      }
      int rem = num_elm & 7;
      ae_int8x16 *out_8x16 = (ae_int8x16 *)out;
      if (rem)
      {
          int rem1 = rem >= 4 ? 4 : rem;
          int rem2 = rem >= 4 ? rem - 4 : 0;
          result =  AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(inp1);
          inp2_a = AE_LA128_PP(inp2);
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem1* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem1* sizeof(FLOAT32));

          xtbool2 check1 = XT_OLE_SX2(y1, x1);
          xtbool2 check2 = XT_OLE_SX2(y2, x2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem2* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem2* sizeof(FLOAT32));

          xtbool2 check3 = XT_OLE_SX2(y1, x1);
          xtbool2 check4 = XT_OLE_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, out_8x16, rem);
          AE_SA128POS_FP(out_a, out_8x16);
      }
      out = (ae_int8x8*)out_8x16;
    }
    else if(kernel_type == COMPARE_GREATER)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&0xF) == 0) && ((((unsigned)p_inp2)&0xF) == 0))
      {
          for(i=0;i < num_elm>>3;i++)
          {
              result =  AE_MOVDA8(0);
              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));
              
              xtbool2 check1 = XT_OLT_SX2(y1, x1);
              xtbool2 check2 = XT_OLT_SX2(y2, x2);

              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));

              xtbool2 check3 = XT_OLT_SX2(y1, x1);
              xtbool2 check4 = XT_OLT_SX2(y2, x2);

              AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
              AE_S8X8_IP(result, out, 8);
          }
      }
      else
      {
        ae_valignx2 inp1_a, inp2_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(inp1);
        inp2_a = AE_LA128_PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>3;i++)
          {
            result =  AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check1 = XT_OLT_SX2(y1, x1);
            xtbool2 check2 = XT_OLT_SX2(y2, x2);

            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check3 = XT_OLT_SX2(y1, x1);
            xtbool2 check4 = XT_OLT_SX2(y2, x2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, out);
          }
          AE_SA64POS_FP(su, out);
      }
      int rem = num_elm & 7;
      ae_int8x16 *out_8x16 = (ae_int8x16 *)out;
      if (rem)
      {
          int rem1 = rem >= 4 ? 4 : rem;
          int rem2 = rem >= 4 ? rem - 4 : 0;
          result =  AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(inp1);
          inp2_a = AE_LA128_PP(inp2);
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem1* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem1* sizeof(FLOAT32));

          xtbool2 check1 = XT_OLT_SX2(y1, x1);
          xtbool2 check2 = XT_OLT_SX2(y2, x2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem2* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem2* sizeof(FLOAT32));

          xtbool2 check3 = XT_OLT_SX2(y1, x1);
          xtbool2 check4 = XT_OLT_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SAV8X8X2_XP(result, result, out_a, out_8x16, rem);
          AE_SA128POS_FP(out_a, out_8x16);
      }
      out = (ae_int8x8*)out_8x16;
    }
    else if(kernel_type == COMPARE_LESSEREQUAL)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&0xF) == 0) && ((((unsigned)p_inp2)&0xF) == 0))
      {
          for(i=0;i < num_elm>>3;i++)
          {
              result =  AE_MOVDA8(0);
              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));
              
              xtbool2 check1 = XT_OLE_SX2(x1, y1);
              xtbool2 check2 = XT_OLE_SX2(x2, y2);

              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));

              xtbool2 check3 = XT_OLE_SX2(x1, y1);
              xtbool2 check4 = XT_OLE_SX2(x2, y2);

              AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
              AE_S8X8_IP(result, out, 8);
          }
      }
      else
      {
        ae_valignx2 inp1_a, inp2_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(inp1);
        inp2_a = AE_LA128_PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>3;i++)
          {
            result =  AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check1 = XT_OLE_SX2(x1, y1);
            xtbool2 check2 = XT_OLE_SX2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check3 = XT_OLE_SX2(x1, y1);
            xtbool2 check4 = XT_OLE_SX2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, out);
          }
          AE_SA64POS_FP(su, out);
      }
      int rem = num_elm & 7;
      ae_int8x16 *out_8x16 = (ae_int8x16 *)out;
      if (rem)
      {
          int rem1 = rem >= 4 ? 4 : rem;
          int rem2 = rem >= 4 ? rem - 4 : 0;
          result =  AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(inp1);
          inp2_a = AE_LA128_PP(inp2);
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem1* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem1* sizeof(FLOAT32));

          xtbool2 check1 = XT_OLE_SX2(x1, y1);
          xtbool2 check2 = XT_OLE_SX2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem2* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem2* sizeof(FLOAT32));

          xtbool2 check3 = XT_OLE_SX2(x1, y1);
          xtbool2 check4 = XT_OLE_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SAV8X8X2_XP(result, result, out_a, out_8x16, rem);
          AE_SA128POS_FP(out_a, out_8x16);
      }
      out = (ae_int8x8*)out_8x16;
    }
    else if(kernel_type == COMPARE_LESSER)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&0xF) == 0) && ((((unsigned)p_inp2)&0xF) == 0))
      {
          for(i=0;i < num_elm>>3;i++)
          {
              result =  AE_MOVDA8(0);
              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));
              
              xtbool2 check1 = XT_OLT_SX2(x1, y1);
              xtbool2 check2 = XT_OLT_SX2(x2, y2);

              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));

              xtbool2 check3 = XT_OLT_SX2(x1, y1);
              xtbool2 check4 = XT_OLT_SX2(x2, y2);
              
              AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
              AE_S8X8_IP(result, out, 8);
          }
      }
      else
      {
        ae_valignx2 inp1_a, inp2_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(inp1);
        inp2_a = AE_LA128_PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>3;i++)
          {
            result =  AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check1 = XT_OLT_SX2(x1, y1);
            xtbool2 check2 = XT_OLT_SX2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check3 = XT_OLT_SX2(x1, y1);
            xtbool2 check4 = XT_OLT_SX2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, out);
          }
          AE_SA64POS_FP(su, out);
      }
      int rem = num_elm & 7;
      ae_int8x16 *out_8x16 = (ae_int8x16 *)out;
      if (rem)
      {
          int rem1 = rem >= 4 ? 4 : rem;
          int rem2 = rem >= 4 ? rem - 4 : 0;
          result =  AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(inp1);
          inp2_a = AE_LA128_PP(inp2);
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem1* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem1* sizeof(FLOAT32));

          xtbool2 check1 = XT_OLT_SX2(x1, y1);
          xtbool2 check2 = XT_OLT_SX2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem2* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem2* sizeof(FLOAT32));

          xtbool2 check3 = XT_OLT_SX2(x1, y1);
          xtbool2 check4 = XT_OLT_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SAV8X8X2_XP(result, result, out_a, out_8x16, rem);
          AE_SA128POS_FP(out_a, out_8x16);
      }
      out = (ae_int8x8*)out_8x16;
    }
    else if(kernel_type == COMPARE_EQUAL)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&0xF) == 0) && ((((unsigned)p_inp2)&0xF) == 0))
      {
          for(i=0;i < num_elm>>3;i++)
          {
              result =  AE_MOVDA8(0);
              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));
              
              xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
              xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));

              xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
              xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

              AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
              AE_S8X8_IP(result, out, 8);
          }
      }
      else
      {
        ae_valignx2 inp1_a, inp2_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(inp1);
        inp2_a = AE_LA128_PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>3;i++)
          {
            result =  AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, out);
          }
          AE_SA64POS_FP(su, out);
      }
      int rem = num_elm & 7;
      ae_int8x16 *out_8x16 = (ae_int8x16 *)out;
      if (rem)
      {
          int rem1 = rem >= 4 ? 4 : rem;
          int rem2 = rem >= 4 ? rem - 4 : 0;
          result =  AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(inp1);
          inp2_a = AE_LA128_PP(inp2);
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem1* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem1* sizeof(FLOAT32));

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem2* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem2* sizeof(FLOAT32));

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SAV8X8X2_XP(result, result, out_a, out_8x16, rem);
          AE_SA128POS_FP(out_a, out_8x16);
      }
      out = (ae_int8x8*)out_8x16;
    }
    else if(kernel_type == COMPARE_NOTEQUAL)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&0xF) == 0) && ((((unsigned)p_inp2)&0xF) == 0))
      {
          for(i=0;i < num_elm>>3;i++)
          {
              result =  AE_MOVDA8(0);
              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));
              
              xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
              xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

              AE_LSX2X2_IP(x1, x2, inp1, 4*sizeof(FLOAT32));
              AE_LSX2X2_IP(y1, y2, inp2, 4*sizeof(FLOAT32));

              xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
              xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

              AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);
              AE_S8X8_IP(result, out, 8);
          }
      }
      else
      {
        ae_valignx2 inp1_a, inp2_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(inp1);
        inp2_a = AE_LA128_PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>3;i++)
          {
            result =  AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, inp1);
            AE_LASX2X2_IP(y1, y2, inp2_a, inp2);

            xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);
            
            AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, out);
          }
          AE_SA64POS_FP(su, out);
      }
      int rem = num_elm & 7;
      ae_int8x16 *out_8x16 = (ae_int8x16 *)out;
      if (rem)
      {
          int rem1 = rem >= 4 ? 4 : rem;
          int rem2 = rem >= 4 ? rem - 4 : 0;
          result =  AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(inp1);
          inp2_a = AE_LA128_PP(inp2);
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem1* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem1* sizeof(FLOAT32));

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, inp1, rem2* sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, inp2, rem2* sizeof(FLOAT32));

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);
          AE_SAV8X8X2_XP(result, result, out_a, out_8x16, rem);
          AE_SA128POS_FP(out_a, out_8x16);
      }
      out = (ae_int8x8*)out_8x16;
    }
    return 0;
}
#endif

#if HAVE_VFPU
static void internal_elm_greater_lesser_equal_broadcast_2D_f32xf32_f32(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32 out_lc = args->out_lc;
  WORD32 in_lc = args->in_lc;
  xtbool sign_flag = args->sign_flag;
  compare_ops_t kernel_type = args->kernel_type;
  FLOAT32 *p_inp1_f32 = (FLOAT32*) p_inp1;
  FLOAT32 *p_inp2_f32 = (FLOAT32*) p_inp2;
  int i, j;
  int num_simd2_ops;
  int num_scalar_ops;

  num_simd2_ops = in_lc >> 3;
  num_scalar_ops = in_lc & 7;

  xtfloatx4  * __restrict__ p_a = (xtfloatx4 *)p_inp1;
  xtfloatx4 * __restrict__ p_b = (xtfloatx4 *)p_inp2;
  ae_int8x8 *p_c = (ae_int8x8 *)p_out;
  xtfloatx2 x1, x2, y1, y2;

  ae_int8x8 result;
  ae_int8x8 ones = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x01010101L, 0x01010101L));

  /* For computing inp2 - inp1 */
  if (AE_MOVAB(sign_flag))
  {
    for (i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx4 *)&p_inp1_f32[i * in_lc];
      p_b = (xtfloatx4 *)p_inp2_f32;

      if (kernel_type == COMPARE_GREATEREQUAL)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = XT_OLE_SX2(x1, y1);
            xtbool2 check2 = XT_OLE_SX2(x2, y2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = XT_OLE_SX2(x1, y1);
            xtbool2 check4 = XT_OLE_SX2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = XT_OLE_SX2(x1, y1);
            xtbool2 check2 = XT_OLE_SX2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = XT_OLE_SX2(x1, y1);
            xtbool2 check4 = XT_OLE_SX2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLE_SX2(x1, y1);
          xtbool2 check2 = XT_OLE_SX2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLE_SX2(x1, y1);
          xtbool2 check4 = XT_OLE_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
      else if (kernel_type == COMPARE_GREATER)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = XT_OLT_SX2(x1, y1);
            xtbool2 check2 = XT_OLT_SX2(x2, y2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = XT_OLT_SX2(x1, y1);
            xtbool2 check4 = XT_OLT_SX2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = XT_OLT_SX2(x1, y1);
            xtbool2 check2 = XT_OLT_SX2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = XT_OLT_SX2(x1, y1);
            xtbool2 check4 = XT_OLT_SX2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLT_SX2(x1, y1);
          xtbool2 check2 = XT_OLT_SX2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLT_SX2(x1, y1);
          xtbool2 check4 = XT_OLT_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
      else if (kernel_type == COMPARE_LESSEREQUAL)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = XT_OLE_SX2(y1, x1);
            xtbool2 check2 = XT_OLE_SX2(y2, x2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = XT_OLE_SX2(y1, x1);
            xtbool2 check4 = XT_OLE_SX2(y2, x2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = XT_OLE_SX2(y1, x1);
            xtbool2 check2 = XT_OLE_SX2(y2, x2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = XT_OLE_SX2(y1, x1);
            xtbool2 check4 = XT_OLE_SX2(y2, x2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLE_SX2(y1, x1);
          xtbool2 check2 = XT_OLE_SX2(y2, x2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLE_SX2(y1, x1);
          xtbool2 check4 = XT_OLE_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
      else if (kernel_type == COMPARE_LESSER)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = XT_OLT_SX2(y1, x1);
            xtbool2 check2 = XT_OLT_SX2(y2, x2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = XT_OLT_SX2(y1, x1);
            xtbool2 check4 = XT_OLT_SX2(y2, x2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = XT_OLT_SX2(y1, x1);
            xtbool2 check2 = XT_OLT_SX2(y2, x2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = XT_OLT_SX2(y1, x1);
            xtbool2 check4 = XT_OLT_SX2(y2, x2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLT_SX2(y1, x1);
          xtbool2 check2 = XT_OLT_SX2(y2, x2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLT_SX2(y1, x1);
          xtbool2 check4 = XT_OLT_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
      else if (kernel_type == COMPARE_EQUAL)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
      else if (kernel_type == COMPARE_NOTEQUAL)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
    }
  }
  /* For computing inp1 - inp2 */
  else
  {
    for (i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx4 *)&p_inp1_f32[i * in_lc];
      p_b = (xtfloatx4 *)p_inp2_f32;

      if (kernel_type == COMPARE_GREATEREQUAL)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = XT_OLE_SX2(y1, x1);
            xtbool2 check2 = XT_OLE_SX2(y2, x2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = XT_OLE_SX2(y1, x1);
            xtbool2 check4 = XT_OLE_SX2(y2, x2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = XT_OLE_SX2(y1, x1);
            xtbool2 check2 = XT_OLE_SX2(y2, x2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = XT_OLE_SX2(y1, x1);
            xtbool2 check4 = XT_OLE_SX2(y2, x2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLE_SX2(y1, x1);
          xtbool2 check2 = XT_OLE_SX2(y2, x2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLE_SX2(y1, x1);
          xtbool2 check4 = XT_OLE_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
      else if (kernel_type == COMPARE_GREATER)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = XT_OLT_SX2(y1, x1);
            xtbool2 check2 = XT_OLT_SX2(y2, x2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = XT_OLT_SX2(y1, x1);
            xtbool2 check4 = XT_OLT_SX2(y2, x2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = XT_OLT_SX2(y1, x1);
            xtbool2 check2 = XT_OLT_SX2(y2, x2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = XT_OLT_SX2(y1, x1);
            xtbool2 check4 = XT_OLT_SX2(y2, x2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLT_SX2(y1, x1);
          xtbool2 check2 = XT_OLT_SX2(y2, x2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLT_SX2(y1, x1);
          xtbool2 check4 = XT_OLT_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
      else if (kernel_type == COMPARE_LESSEREQUAL)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = XT_OLE_SX2(x1, y1);
            xtbool2 check2 = XT_OLE_SX2(x2, y2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = XT_OLE_SX2(x1, y1);
            xtbool2 check4 = XT_OLE_SX2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = XT_OLE_SX2(x1, y1);
            xtbool2 check2 = XT_OLE_SX2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = XT_OLE_SX2(x1, y1);
            xtbool2 check4 = XT_OLE_SX2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLE_SX2(x1, y1);
          xtbool2 check2 = XT_OLE_SX2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLE_SX2(x1, y1);
          xtbool2 check4 = XT_OLE_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
      else if (kernel_type == COMPARE_LESSER)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = XT_OLT_SX2(x1, y1);
            xtbool2 check2 = XT_OLT_SX2(x2, y2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = XT_OLT_SX2(x1, y1);
            xtbool2 check4 = XT_OLT_SX2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = XT_OLT_SX2(x1, y1);
            xtbool2 check2 = XT_OLT_SX2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = XT_OLT_SX2(x1, y1);
            xtbool2 check4 = XT_OLT_SX2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLT_SX2(x1, y1);
          xtbool2 check2 = XT_OLT_SX2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLT_SX2(x1, y1);
          xtbool2 check4 = XT_OLT_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
      else if (kernel_type == COMPARE_EQUAL)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
      else if (kernel_type == COMPARE_NOTEQUAL)
      {
        if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_b) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
        {
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));
            AE_LSX2X2_IP(y1, y2, p_b, 4 * sizeof(FLOAT32));

            xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);
            AE_S8X8_IP(result, p_c, 8);
          }
        }
        else
        {
          ae_valignx2 inp1_a, inp2_a;
          ae_valign su = AE_ZALIGN64();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          for (j = 0; j < num_simd2_ops; j++)
          {
            result = AE_MOVDA8(0);
            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_LASX2X2_IP(x1, x2, inp1_a, p_a);
            AE_LASX2X2_IP(y1, y2, inp2_a, p_b);

            xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
            xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

            AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);
            AE_SA8X8_IP(result, su, p_c);
          }
          AE_SA64POS_FP(su, p_c);
        }
        if (num_scalar_ops)
        {
          int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
          int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
          result = AE_MOVDA8(0);
          ae_valignx2 inp1_a, inp2_a, out_a = AE_ZALIGN128();
          inp1_a = AE_LA128_PP(p_a);
          inp2_a = AE_LA128_PP(p_b);
          ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem1 * sizeof(FLOAT32));

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));
          AE_SW_LAVSX2X2_XP(y1, y2, inp2_a, p_b, rem2 * sizeof(FLOAT32));

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);

          AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
          AE_SA128POS_FP(out_a, p8x16_c);
          p_c = (ae_int8x8 *)p8x16_c;
        }
      }
    }
  }
}

static void internal_elm_greater_lesser_equal_broadcast_f32xf32_f32(void * __restrict__ p_out,
                    const    void * __restrict__ p_inp1,
                    const    void * __restrict__ p_inp2,
                    bcast_args_t* args)
{
  WORD32 num_elm = args->num_elm;
  xtbool sign_flag = args->sign_flag;
  compare_ops_t kernel_type = args->kernel_type;

  int i;
  xtfloatx4  * __restrict__ p_a = (xtfloatx4 *)p_inp1;
  xtfloatx2  * __restrict__ p_b = (xtfloatx2 *)p_inp2; 

  ae_int8x8 *p_c = (ae_int8x8 *)p_out;
  xtfloatx2 x1, x2, y1, y2;

  ae_int8x8 result;
  ae_int8x8 ones = AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x01010101L, 0x01010101L));
  
  const int num_simd2_ops = num_elm >> 3;
  const int num_scalar_ops = num_elm & 7;

  y1 = AE_MOVXTFLOATX2_FROMXTFLOAT(AE_LSI((xtfloat*)p_b, 0));
  y2 = AE_MOVXTFLOATX2_FROMXTFLOAT(AE_LSI((xtfloat*)p_b, 0));

  /* For computing inp2 - inp1 */
  if (AE_MOVAB(sign_flag))
  {
    if (kernel_type == COMPARE_GREATEREQUAL)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLE_SX2(x1, y1);
          xtbool2 check2 = XT_OLE_SX2(x2, y2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLE_SX2(x1, y1);
          xtbool2 check4 = XT_OLE_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = XT_OLE_SX2(x1, y1);
          xtbool2 check2 = XT_OLE_SX2(x2, y2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = XT_OLE_SX2(x1, y1);
          xtbool2 check4 = XT_OLE_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);
        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = XT_OLE_SX2(x1, y1);
        xtbool2 check2 = XT_OLE_SX2(x2, y2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = XT_OLE_SX2(x1, y1);
        xtbool2 check4 = XT_OLE_SX2(x2, y2);

        AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
    else if (kernel_type == COMPARE_GREATER)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLT_SX2(x1, y1);
          xtbool2 check2 = XT_OLT_SX2(x2, y2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLT_SX2(x1, y1);
          xtbool2 check4 = XT_OLT_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = XT_OLT_SX2(x1, y1);
          xtbool2 check2 = XT_OLT_SX2(x2, y2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = XT_OLT_SX2(x1, y1);
          xtbool2 check4 = XT_OLT_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);
        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = XT_OLT_SX2(x1, y1);
        xtbool2 check2 = XT_OLT_SX2(x2, y2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = XT_OLT_SX2(x1, y1);
        xtbool2 check4 = XT_OLT_SX2(x2, y2);

        AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
    else if (kernel_type == COMPARE_LESSEREQUAL)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLE_SX2(y1, x1);
          xtbool2 check2 = XT_OLE_SX2(y2, x2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLE_SX2(y1, x1);
          xtbool2 check4 = XT_OLE_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = XT_OLE_SX2(y1, x1);
          xtbool2 check2 = XT_OLE_SX2(y2, x2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = XT_OLE_SX2(y1, x1);
          xtbool2 check4 = XT_OLE_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);
        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = XT_OLE_SX2(y1, x1);
        xtbool2 check2 = XT_OLE_SX2(y2, x2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = XT_OLE_SX2(y1, x1);
        xtbool2 check4 = XT_OLE_SX2(y2, x2);

        AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
    else if (kernel_type == COMPARE_LESSER)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLT_SX2(y1, x1);
          xtbool2 check2 = XT_OLT_SX2(y2, x2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLT_SX2(y1, x1);
          xtbool2 check4 = XT_OLT_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = XT_OLT_SX2(y1, x1);
          xtbool2 check2 = XT_OLT_SX2(y2, x2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = XT_OLT_SX2(y1, x1);
          xtbool2 check4 = XT_OLT_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);
        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = XT_OLT_SX2(y1, x1);
        xtbool2 check2 = XT_OLT_SX2(y2, x2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = XT_OLT_SX2(y1, x1);
        xtbool2 check4 = XT_OLT_SX2(y2, x2);

        AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
    else if (kernel_type == COMPARE_EQUAL)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
        xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
        xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

        AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
    else if (kernel_type == COMPARE_NOTEQUAL)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);
        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
        xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
        xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

        AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
  }
  /* For computing inp1 - inp2 */
  else
  {
    if (kernel_type == COMPARE_GREATEREQUAL)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLE_SX2(y1, x1);
          xtbool2 check2 = XT_OLE_SX2(y2, x2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLE_SX2(y1, x1);
          xtbool2 check4 = XT_OLE_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = XT_OLE_SX2(y1, x1);
          xtbool2 check2 = XT_OLE_SX2(y2, x2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = XT_OLE_SX2(y1, x1);
          xtbool2 check4 = XT_OLE_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);
        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = XT_OLE_SX2(y1, x1);
        xtbool2 check2 = XT_OLE_SX2(y2, x2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = XT_OLE_SX2(y1, x1);
        xtbool2 check4 = XT_OLE_SX2(y2, x2);

        AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
    else if (kernel_type == COMPARE_GREATER)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLT_SX2(y1, x1);
          xtbool2 check2 = XT_OLT_SX2(y2, x2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLT_SX2(y1, x1);
          xtbool2 check4 = XT_OLT_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = XT_OLT_SX2(y1, x1);
          xtbool2 check2 = XT_OLT_SX2(y2, x2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = XT_OLT_SX2(y1, x1);
          xtbool2 check4 = XT_OLT_SX2(y2, x2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);
        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = XT_OLT_SX2(y1, x1);
        xtbool2 check2 = XT_OLT_SX2(y2, x2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = XT_OLT_SX2(y1, x1);
        xtbool2 check4 = XT_OLT_SX2(y2, x2);

        AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
    else if (kernel_type == COMPARE_LESSEREQUAL)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLE_SX2(x1, y1);
          xtbool2 check2 = XT_OLE_SX2(x2, y2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLE_SX2(x1, y1);
          xtbool2 check4 = XT_OLE_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = XT_OLE_SX2(x1, y1);
          xtbool2 check2 = XT_OLE_SX2(x2, y2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = XT_OLE_SX2(x1, y1);
          xtbool2 check4 = XT_OLE_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);
        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = XT_OLE_SX2(x1, y1);
        xtbool2 check2 = XT_OLE_SX2(x2, y2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = XT_OLE_SX2(x1, y1);
        xtbool2 check4 = XT_OLE_SX2(x2, y2);

        AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
    else if (kernel_type == COMPARE_LESSER)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = XT_OLT_SX2(x1, y1);
          xtbool2 check2 = XT_OLT_SX2(x2, y2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = XT_OLT_SX2(x1, y1);
          xtbool2 check4 = XT_OLT_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = XT_OLT_SX2(x1, y1);
          xtbool2 check2 = XT_OLT_SX2(x2, y2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = XT_OLT_SX2(x1, y1);
          xtbool2 check4 = XT_OLT_SX2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);
        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = XT_OLT_SX2(x1, y1);
        xtbool2 check2 = XT_OLT_SX2(x2, y2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = XT_OLT_SX2(x1, y1);
        xtbool2 check4 = XT_OLT_SX2(x2, y2);

        AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
    else if (kernel_type == COMPARE_EQUAL)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);
        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
        xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
        xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

        AE_SW_MOVT_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
    else if (kernel_type == COMPARE_NOTEQUAL)
    {
      if (((((unsigned)p_a) & 0xF) == 0) && ((((unsigned)p_c) & 7) == 0))
      {
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_LSX2X2_IP(x1, x2, p_a, 4 * sizeof(FLOAT32));

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);
          AE_S8X8_IP(result, p_c, 8);
        }
      }
      else
      {
        ae_valignx2 inp1_a;
        ae_valign su = AE_ZALIGN64();
        inp1_a = AE_LA128_PP(p_a);
        for (i = 0; i < num_simd2_ops; i++)
        {
          result = AE_MOVDA8(0);
          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_LASX2X2_IP(x1, x2, inp1_a, p_a);

          xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
          xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

          AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);
          AE_SA8X8_IP(result, su, p_c);
        }
        AE_SA64POS_FP(su, p_c);
      }
      if (num_scalar_ops)
      {
        int rem1 = num_scalar_ops >= 4 ? 4 : num_scalar_ops;
        int rem2 = num_scalar_ops >= 4 ? num_scalar_ops - 4 : 0;
        result = AE_MOVDA8(0);
        ae_valignx2 inp1_a, out_a = AE_ZALIGN128();
        ae_int8x16 *p8x16_c = (ae_int8x16 *)p_c;
        inp1_a = AE_LA128_PP(p_a);
        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem1 * sizeof(FLOAT32));

        xtbool2 check1 = xtfloatx2_EQ_xtfloatx2(x1, y1);
        xtbool2 check2 = xtfloatx2_EQ_xtfloatx2(x2, y2);

        AE_SW_LAVSX2X2_XP(x1, x2, inp1_a, p_a, rem2 * sizeof(FLOAT32));

        xtbool2 check3 = xtfloatx2_EQ_xtfloatx2(x1, y1);
        xtbool2 check4 = xtfloatx2_EQ_xtfloatx2(x2, y2);

        AE_SW_MOVF_4B2(result, ones, check1, check2, check3, check4);

        AE_SAV8X8X2_XP(result, result, out_a, p8x16_c, num_scalar_ops);
        AE_SA128POS_FP(out_a, p8x16_c);
        p_c = (ae_int8x8 *)p8x16_c;
      }
    }
  }
}
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_compare_broadcast_4D_f32xf32_f32,
             (
                      WORD8 * p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const FLOAT32 * p_inp2,
                      const WORD32 *const p_inp2_shape,
                      compare_ops_t kernel_type
              )
           )
#else           
WORD32 xa_nn_elm_compare_broadcast_4D_f32xf32_f32(WORD8 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const FLOAT32 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape,
                      compare_ops_t kernel_type)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((kernel_type < 0) || (kernel_type > 5), -1);

  bcast_args_t args = {0};
  args.inp_elm_size = 4;
  args.out_elm_size = 1;
  args.multiplier_sign = 1;
  args.kernel_type = kernel_type;

  return CALL_BCAST(internal_elm_greater_lesser_equal_broadcast_2D_f32xf32_f32, 
            internal_elm_greater_lesser_equal_broadcast_f32xf32_f32,
            p_out,
            p_out_shape,
            p_inp1,
            p_inp1_shape,
            p_inp2,
            p_inp2_shape,
            &args);
}
#endif
