/*******************************************************************************
* Copyright (c) 2018-2020 Cadence Design Systems, Inc.
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
//#include "xa_nn_basic_state.h"

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out, inp, inp1, multiplier, l_shift, right_shift, out_off) \
    AE_MUL2P32X4S(inp, inp1, inp, inp1, l_shift, l_shift); \
    AE_MULF2P32X4RAS(inp, inp1, inp, inp1, AE_MOVDA32(multiplier), AE_MOVDA32(multiplier)); \
    inp = AE_SRAA32SYMS(inp, right_shift);\
    inp1 = AE_SRAA32SYMS(inp1, right_shift);\
    out = AE_SAT16X4(inp, inp1); \
    out = AE_ADD16S(AE_MOVDA16(out_off), out); \
    AE_MINMAX16(out, AE_MOVDA16(-128), AE_MOVDA16(127)); 

#define PACK_32X2(dst1, src1, src2) \
dst1 = AE_SEL8X8(AE_MOVINT8X8_FROMINT16X4(src1), AE_MOVINT8X8_FROMINT16X4(src2), AE_MOVINT8X8_FROMINT32X2(AE_MOVDA32X2(0x0e0c0a08, 0x06040200)));

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2(inp, multiplier, left_shift, right_shift) \
    inp = AE_SLAA32(inp, left_shift); \
    inp = AE_MULFP32X2RAS(inp, multiplier); \
    inp = AE_SRAA32SYMS(inp, right_shift);

WORD32 xa_nn_elm_quantize_asym16s_asym8s(WORD8 * __restrict__ p_out,
                                    const WORD16 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  WORD8 *out = p_out;
  WORD16 *p_i = (WORD16 *)p_inp;

  int left_shift, right_shift;
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;

  ae_valignx2 align_inp = AE_LA128_PP(p_inp);
  ae_valign align_dst = AE_ZALIGN64();
  
  ae_int16x4 d_inp_zero_bias = AE_MOVDA16(inp_zero_bias);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);
  ae_int32x2 l_mult = AE_MOVDA32(1 << left_shift);
  
  ae_int16x4 out_0, out_1; 

  for(i = 0; i < (num_elm >> 3); i++)
  {
    ae_int16x4 d_inp0, d_inp1;
    ae_int32x2 d_inp32_0, d_inp32_1;
    ae_int32x2 d_inp32_2, d_inp32_3;
    AE_LA16X4X2_IP(d_inp0, d_inp1, align_inp, (ae_int16x8 *)p_i);
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias); 
    AE_SUBW16(d_inp32_2, d_inp32_3, d_inp1, d_inp_zero_bias);
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, d_inp32_0, d_inp32_1, d_out_multiplier, l_mult, right_shift, out_zero_bias);
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_1, d_inp32_2, d_inp32_3, d_out_multiplier, l_mult, right_shift, out_zero_bias);
    
    ae_int8x8 out32_0; 
    PACK_32X2(out32_0, out_0, out_1);
    
    AE_SA8X8_IP(out32_0, align_dst, (ae_int8x8 *)out);
  }
  AE_SA64POS_FP(align_dst, out);
    
  /*Remainder loop*/
  for(i = 0; i < (num_elm & 7); i++)
  {
    ae_int16x4 d_inp0;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_L16_IP(d_inp0, (ae_int16 *)p_i, 2);
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias); 
    MULTIPLYBYQUANTIZEDMULTIPLIER_X4(out_0, d_inp32_0, d_inp32_1, d_out_multiplier, l_mult, right_shift, out_zero_bias);
    
    ae_int8x8 out32_0; 
    PACK_32X2(out32_0, out_0, out_0);
    
    AE_S8_0_IP(out32_0, (ae_int8 *)out, 1);
  }

  return 0;
}

WORD32 xa_nn_elm_quantize_asym16s_asym32s(WORD32 * __restrict__ p_out,
                                    const WORD16 * __restrict__ p_inp,
                                    WORD32  inp_zero_bias,
                                    WORD32  out_zero_bias,
                                    WORD32  out_shift,
                                    WORD32  out_multiplier,
                                    WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
  //XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
  XA_NNLIB_ARG_CHK_COND(((inp_zero_bias < -32768) || (inp_zero_bias > 32767)), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 31)), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  int i;
  WORD32 *p_o = p_out;
  WORD16 *p_i = (WORD16 *)p_inp;

  int left_shift, right_shift;
  left_shift  = (out_shift < 0)?0:out_shift;
  right_shift = (out_shift > 0)?0:-out_shift;

  ae_valign align_inp = AE_LA64_PP(p_inp);
  ae_valign align_dst = AE_ZALIGN64();

  ae_int16x4 d_inp_zero_bias = AE_MOVDA16(inp_zero_bias);
  ae_int32x2 d_out_multiplier = AE_MOVDA32(out_multiplier);
  ae_int32x2 d_out_zero_bias = AE_MOVDA32(out_zero_bias);
  ae_int32x2 d_out0, d_out1;

  for(i = 0; i < (num_elm >> 2); i++)
  {
    ae_int16x4 d_inp0;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_LA16X4_IP(d_inp0, align_inp, (ae_int16x4 *)p_i);
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias);

    MULTIPLYBYQUANTIZEDMULTIPLIER_X2(d_inp32_0, d_out_multiplier, left_shift, right_shift);
    MULTIPLYBYQUANTIZEDMULTIPLIER_X2(d_inp32_1, d_out_multiplier, left_shift, right_shift);
    d_out0 = AE_ADD32S(d_inp32_0, d_out_zero_bias);
    d_out1 = AE_ADD32S(d_inp32_1, d_out_zero_bias);
    AE_SA32X2_IP(d_out0, align_dst, (ae_int32x2*)p_o);
    AE_SA32X2_IP(d_out1, align_dst, (ae_int32x2*)p_o);

  }
  AE_SA64POS_FP(align_dst, p_o);

  /*Remainder loop*/
  for(i = 0; i < (num_elm & 3); i++)
  {
    ae_int16x4 d_inp0;
    ae_int32x2 d_inp32_0, d_inp32_1;
    AE_L16_IP(d_inp0, (ae_int16 *)p_i, 2);
    AE_SUBW16(d_inp32_0, d_inp32_1, d_inp0, d_inp_zero_bias);
    MULTIPLYBYQUANTIZEDMULTIPLIER_X2(d_inp32_0, d_out_multiplier, left_shift, right_shift);
    d_out0 = AE_ADD32S(d_inp32_0, d_out_zero_bias);
    AE_S32_L_IP(d_out0, (ae_int32*)p_o, 4);
  }
  return 0;
}
