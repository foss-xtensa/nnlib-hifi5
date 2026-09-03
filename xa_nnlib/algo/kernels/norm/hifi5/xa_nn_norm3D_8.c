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
#include "xa_nnlib_common.h"
#define USHRT_MAX  65535
#define ZERO32   AE_ZERO32()
#define ZERO64  AE_ZERO64() 
//(ae_int64)(1)
#define SW_MOVDA32(a) AE_MOVDA32X2(a, a)
#define ZERO16   AE_ZERO16()

#define SW_SLAA32S_INT32X2_INT32X2(inp1, inp2) AE_MOVINT32X2_FROMF32X2(AE_SLAA32S(AE_MOVF32X2_FROMINT32X2(inp1), inp2))

WORD32 xa_nn_norm_calc_3D_8_nhwc(
    WORD16 * p_out /*Noram data: 2D -> iw*ih, or scalar*/ , 
    WORD8 * p_outnsa /*NSA data: 2D -> iw*ih, or scalar*/ ,
    const WORD8 * p_inp /*3D -> iw*ih*ic */,
    int input_height, int input_width, int input_channels, 
    int accross_depth_flag,
    int out_shift, /*sumSquareShift*/
    const UWORD16 *prsqrt, int rsqrt_table_len)
{
  /* NULL pointer check */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(prsqrt, -1);
  /* Basic Parameter checks */  
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0 || input_channels <= 0),-1);
  XA_NNLIB_ARG_CHK_COND(rsqrt_table_len <= 0,-1);
  XA_NNLIB_ARG_CHK_COND((out_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((accross_depth_flag != 0) && (accross_depth_flag != 1), -1);
  int out_rshift = -out_shift;
  
  if(accross_depth_flag == 0) /* Calc norm data for entire 3D input */
  {
    WORD32 i;
    WORD32 inp_len = input_height*input_width*input_channels;
    WORD32 lc = inp_len >> 4;
    WORD32 remc = inp_len & 15;
    
    ae_int8x16 *ptr_inp = (ae_int8x16 *)p_inp;
    ae_valignx2 a_inp = AE_LA128_PP(ptr_inp);
    ae_int8x8 d_inp1, d_inp2;
    /*ae_int32x2*/ ae_int64 acc_64_1=ZERO64, acc_64_2=ZERO64; // try acc_64_3,4 and see if there's more optimization on the compiler's side
    
    for(i = 0; i < lc; i++)
    {
      AE_LA8X8X2_IP(d_inp1, d_inp2, a_inp, ptr_inp);
      AE_MULAAAA2Q8(acc_64_1, acc_64_2, d_inp1, d_inp1);
      AE_MULAAAA2Q8(acc_64_1, acc_64_2, d_inp2, d_inp2);
    }
    if(remc & 15)
    {
      AE_LAV8X8X2_XP(d_inp1, d_inp2, a_inp, ptr_inp, inp_len & 15); 
      AE_MULAAAA2Q8(acc_64_1, acc_64_2, d_inp1, d_inp1);
      AE_MULAAAA2Q8(acc_64_1, acc_64_2, d_inp2, d_inp2);
    }
    ae_f64 accF_1 = AE_MOVF64_FROMINT64(acc_64_1); ae_f64 accF_2 = AE_MOVF64_FROMINT64(acc_64_2);
    accF_1 = AE_ADD64S(accF_1, accF_2);
    accF_2 = AE_SLAA64S(accF_1, 32-out_rshift);
    ae_int32x2 acc1 = AE_MOVINT32X2_FROMF32X2(AE_ROUND32X2F64SASYM(accF_2, accF_2)); //equivalent to AE_MINMAX32(acc_64,0,32) as acc_64s are always positive
    WORD32 nsaShift = AE_NSAZ32_L(acc1);
    if(AE_MOVAB2(AE_EQ32(acc1,ZERO32)))
    {
      nsaShift = 31;
    }
    nsaShift = 15 - nsaShift + 1;
    nsaShift = (nsaShift<0) ? 0 : nsaShift;
    acc1 = AE_MOVINT32X2_FROMF32X2(AE_SRAA32RS(AE_MOVF32X2_FROMINT32X2(acc1), nsaShift));
    AE_MINMAX32(acc1, SW_MOVDA32(0), SW_MOVDA32(rsqrt_table_len-1));
    p_out[0] = prsqrt[AE_MOVAD32_H(acc1)];
    p_outnsa[0] = nsaShift + out_rshift;

  }
  else /* Calc norm data across depth dimension only */
  {
    WORD32 ihw, ic;
    WORD32 ilc = input_channels >> 4;
    WORD32 iremc = input_channels & 15;
    WORD32 olc = input_height * input_width;
    
    ae_int8x16 *ptr_inp = (ae_int8x16 *)p_inp;
    //ae_int16 * ptr_out = (ae_int16 *)p_out;
    
    ae_valignx2 a_inp = AE_LA128_PP(ptr_inp);
    ae_int8x8  d_inp1, d_inp2;   
    
    for(ihw = 0; ihw < olc; ihw++)
    {
        ae_int64 acc_64_1=ZERO64, acc_64_2=ZERO64;
        
        for(ic = 0; ic < ilc; ic++)
        {
          AE_LA8X8X2_IP(d_inp1, d_inp2, a_inp, ptr_inp);
          AE_MULAAAA2Q8(acc_64_1, acc_64_2, d_inp1, d_inp1);
          AE_MULAAAA2Q8(acc_64_1, acc_64_2, d_inp2, d_inp2);
        }
        if(iremc)
        {
          AE_LAV8X8X2_XP(d_inp1, d_inp2, a_inp, ptr_inp, iremc);
          AE_MULAAAA2Q8(acc_64_1, acc_64_2, d_inp1, d_inp1);
          AE_MULAAAA2Q8(acc_64_1, acc_64_2, d_inp2, d_inp2);
        }
    
    ae_f64 accF_1 = AE_MOVF64_FROMINT64(acc_64_1); ae_f64 accF_2 = AE_MOVF64_FROMINT64(acc_64_2);
    accF_1 = AE_ADD64S(accF_1, accF_2); 
    accF_2 = AE_SLAA64S(accF_1, 32-out_rshift); 
    ae_int32x2 acc1 = AE_MOVINT32X2_FROMF32X2(AE_ROUND32X2F64SASYM(accF_2, accF_2)); 
    WORD32 nsaShift = AE_NSAZ32_L(acc1);
    if(AE_MOVAB2(AE_EQ32(acc1,ZERO32)))
    {
      nsaShift = 31;
    }
    nsaShift = 15 - nsaShift + 1;
    nsaShift = (nsaShift<0) ? 0 : nsaShift;
    //ae_int32 one_nsaShift = (nsaShift>0 ? (1 << (nsaShift - 1)) : 0);
    acc1 = AE_MOVINT32X2_FROMF32X2(AE_SRAA32RS(AE_MOVF32X2_FROMINT32X2(acc1), nsaShift));
    AE_MINMAX32(acc1, SW_MOVDA32(0), SW_MOVDA32(rsqrt_table_len-1));
    p_out[ihw] = prsqrt[AE_MOVAD32_H(acc1)];
    p_outnsa[ihw] = nsaShift + out_rshift;
      }
  }

  return 0;
}


WORD32 xa_nn_norm_apply_3D_8_nhwc(
    WORD8 * p_out, 
    const WORD8 * p_inp, /*3D -> iw*ih*ic */
    WORD16 *p_inp_normdata,
    const WORD8 *p_inp_nsadata,
    int input_height, int input_width, int input_channels,
    int accross_depth_flag,
    int per_chan_flag,
    WORD16 * p_out_multiplier,
    WORD32 out_shift,
    WORD32 rsqrt_shift
)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_normdata, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_multiplier, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_nsadata, -1);

  /* Pointer Alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_normdata, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_nsadata, sizeof(WORD8), -1);

  /* Param Checks*/
  XA_NNLIB_ARG_CHK_COND((out_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((rsqrt_shift < 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((accross_depth_flag != 0) && (accross_depth_flag != 1), -1);
  XA_NNLIB_ARG_CHK_COND((per_chan_flag != 0) && (per_chan_flag != 1), -1);

  int out_rshift = -out_shift;
  if(accross_depth_flag == 0)
  {
    int ic;
    UWORD16 norm_factor = p_inp_normdata[0];
    ae_int32x2 d_norm = SW_MOVDA32(norm_factor);
    WORD8  nsaShift    = p_inp_nsadata[0];
    WORD8 finalShift  = out_rshift + ((nsaShift + 1) >> 1) + rsqrt_shift;
  
    const signed char * ptr_inp = p_inp;
    ae_valign a_inp;
    a_inp = AE_LA64_PP(ptr_inp);
    ae_int8x8 * ptr_out = (ae_int8x8 *)p_out;
    
    ae_int8x8 d_out;
    ae_int16x4 d_inp1, d_inp2, d_out_mult1, d_out_mult2, d_mult;
    ae_int64 norm_inp11_1, norm_inp11_2, norm_inp12_1, norm_inp12_2, norm_inp21_1, norm_inp21_2, norm_inp22_1, norm_inp22_2;
    ae_int64 out11_1, out11_2, out12_1, out12_2, out21_1, out21_2, out22_1, out22_2;
    ae_int32x2 sat_out11, sat_out12, sat_out21, sat_out22;
    ae_int16x4 sat_out1, sat_out2;
    ae_int32x2 scaled_inp11, scaled_inp12, scaled_inp21, scaled_inp22;

    ae_valign a_out = AE_ZALIGN64();
    ae_f32x2 d_scale1 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 d_scale2 = AE_MOVF32X2_FROMINT32X2(ZERO32);

    if(per_chan_flag == 0)
    {
      const ae_int16 * ptr_out_multiplier = (const ae_int16 *)p_out_multiplier;
      AE_L16_IP(d_mult, ptr_out_multiplier, 2);

      WORD32 input_size=input_height*input_width*input_channels;
      WORD32 nsa_mult_factor = (nsaShift & 0x1) ? 46341 : (1<<15);
      ae_int32x2 d_nsa_multiplier = SW_MOVDA32(nsa_mult_factor);

      AE_MULF2P32X16X4S(d_scale1, d_scale2, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult));
      ae_int16x4 d_scale = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale1), AE_MOVINT32X2_FROMF32X2(d_scale2));

      ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));

//#pragma concurrent
      for(ic = 0; ic < (input_size>>3); ic++)
      {
        norm_inp11_1 = round_cnst;
        norm_inp11_2 = round_cnst;
        norm_inp12_1 = round_cnst;
        norm_inp12_2 = round_cnst;
        norm_inp21_1 = round_cnst;
        norm_inp21_2 = round_cnst;
        norm_inp22_1 = round_cnst;
        norm_inp22_2 = round_cnst;
        AE_LA8X4S_IP(d_inp1, a_inp, ptr_inp);
        AE_LA8X4S_IP(d_inp2, a_inp, ptr_inp);
        
        AE_MUL16X4S(scaled_inp11, scaled_inp12, d_inp1, d_scale);
        AE_MUL16X4S(scaled_inp21, scaled_inp22, d_inp2, d_scale);

        AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
        AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
        AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
        AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);
        AE_MULA32_HH(norm_inp21_1, scaled_inp21, d_norm);
        AE_MULA32_LL(norm_inp21_2, scaled_inp21, d_norm);
        AE_MULA32_HH(norm_inp22_1, scaled_inp22, d_norm);
        AE_MULA32_LL(norm_inp22_2, scaled_inp22, d_norm);

        out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
        out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
        out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
        out12_2 = AE_SRAA64(norm_inp12_2, finalShift);
        out21_1 = AE_SRAA64(norm_inp21_1, finalShift);
        out21_2 = AE_SRAA64(norm_inp21_2, finalShift);
        out22_1 = AE_SRAA64(norm_inp22_1, finalShift);
        out22_2 = AE_SRAA64(norm_inp22_2, finalShift);

        sat_out11 = AE_SAT32X2(out11_1, out11_2);
        sat_out12 = AE_SAT32X2(out12_1, out12_2);
        sat_out21 = AE_SAT32X2(out21_1, out21_2);
        sat_out22 = AE_SAT32X2(out22_1, out22_2);

        sat_out1 = AE_SAT16X4(sat_out11, sat_out12);
        sat_out2 = AE_SAT16X4(sat_out21, sat_out22);

        d_out = AE_SAT8X8X16(sat_out1, sat_out2);
        AE_SA8X8_IP(d_out, a_out, ptr_out);
      }
      AE_SA64POS_FP(a_out, ptr_out);
      WORD32 rem_size = input_size & 7;
      const ae_int8x16 *ptr8x16_inp = (const ae_int8x16 * )ptr_inp;
      ae_int8x16 *ptr8x16_out = (ae_int8x16 *)ptr_out;
      
      ae_f16x4 d_inp1_t, d_inp2_t;
      if(rem_size)
      {
        ae_valignx2 a_inpx2, a_outx2;
        a_inpx2 = AE_LA128_PP(ptr8x16_inp);
        a_outx2 = AE_ZALIGN128();
        ae_int8x8 d_inp, temp;
        
        norm_inp11_1 = round_cnst;
        norm_inp11_2 = round_cnst;
        norm_inp12_1 = round_cnst;
        norm_inp12_2 = round_cnst;
        norm_inp21_1 = round_cnst;
        norm_inp21_2 = round_cnst;
        norm_inp22_1 = round_cnst;
        norm_inp22_2 = round_cnst;

        AE_LAV8X8X2_XP(d_inp, temp, a_inpx2, ptr8x16_inp, rem_size);
        AE_CVTI16X4X2F8(d_inp1_t, d_inp2_t, d_inp, 0);

        AE_MUL16X4S(scaled_inp11, scaled_inp12, AE_MOVINT16X4_FROMF16X4(d_inp1_t), d_scale);
        AE_MUL16X4S(scaled_inp21, scaled_inp22, AE_MOVINT16X4_FROMF16X4(d_inp2_t), d_scale);

        AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
        AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
        AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
        AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);
        AE_MULA32_HH(norm_inp21_1, scaled_inp21, d_norm);
        AE_MULA32_LL(norm_inp21_2, scaled_inp21, d_norm);
        AE_MULA32_HH(norm_inp22_1, scaled_inp22, d_norm);
        AE_MULA32_LL(norm_inp22_2, scaled_inp22, d_norm);

        out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
        out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
        out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
        out12_2 = AE_SRAA64(norm_inp12_2, finalShift);
        out21_1 = AE_SRAA64(norm_inp21_1, finalShift);
        out21_2 = AE_SRAA64(norm_inp21_2, finalShift);
        out22_1 = AE_SRAA64(norm_inp22_1, finalShift);
        out22_2 = AE_SRAA64(norm_inp22_2, finalShift);

        sat_out11 = AE_SAT32X2(out11_1, out11_2);
        sat_out12 = AE_SAT32X2(out12_1, out12_2);
        sat_out21 = AE_SAT32X2(out21_1, out21_2);
        sat_out22 = AE_SAT32X2(out22_1, out22_2);

        sat_out1 = AE_SAT16X4(sat_out11, sat_out12);
        sat_out2 = AE_SAT16X4(sat_out21, sat_out22);

        d_out = AE_SAT8X8X16(sat_out1, sat_out2);
        AE_SAV8X8X2_XP(d_out, temp, a_outx2, ptr8x16_out, rem_size);
        AE_SA128POS_FP(a_outx2, ptr8x16_out);
      }
      ptr_out = (ae_int8x8*)ptr8x16_out;
      ptr_inp = (const signed char*)ptr8x16_inp;
    }
    else
    {
      ae_int16x4 d_scale1, d_scale2;
      ae_f32x2 d_scale11, d_scale12, d_scale21, d_scale22;
      WORD32 nsa_mult_factor = (nsaShift & 0x1) ? 46341 : (1<<15);
      ae_int32x2 d_nsa_multiplier = SW_MOVDA32(nsa_mult_factor);

      ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));

        for(int ihw = 0; ihw < input_height * input_width; ihw++)
        {
          ae_int16x8 *ptr_out_mult = (ae_int16x8 *)p_out_multiplier;
          ae_valignx2 a_out_mult = AE_LA128_PP(ptr_out_mult);

//#pragma concurrent
          for(ic = 0; ic < (input_channels>>3); ic++)
          {
             norm_inp11_1 = round_cnst;
             norm_inp11_2 = round_cnst;
             norm_inp12_1 = round_cnst;
             norm_inp12_2 = round_cnst;
             norm_inp21_1 = round_cnst;
             norm_inp21_2 = round_cnst;
             norm_inp22_1 = round_cnst;
             norm_inp22_2 = round_cnst;

             AE_LA8X4S_IP(d_inp1, a_inp, ptr_inp);
             AE_LA8X4S_IP(d_inp2, a_inp, ptr_inp);
             AE_LA16X4X2_IP(d_out_mult1, d_out_mult2, a_out_mult, ptr_out_mult);

             AE_MULF2P32X16X4S(d_scale11, d_scale12, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult1));
             AE_MULF2P32X16X4S(d_scale21, d_scale22, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult2));
             d_scale1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale11), AE_MOVINT32X2_FROMF32X2(d_scale12));
             d_scale2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale21), AE_MOVINT32X2_FROMF32X2(d_scale22));

             AE_MUL16X4S(scaled_inp11, scaled_inp12, d_inp1, d_scale1);  
             AE_MUL16X4S(scaled_inp21, scaled_inp22, d_inp2, d_scale2);

             AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
             AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
             AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
             AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);
             AE_MULA32_HH(norm_inp21_1, scaled_inp21, d_norm);
             AE_MULA32_LL(norm_inp21_2, scaled_inp21, d_norm);
             AE_MULA32_HH(norm_inp22_1, scaled_inp22, d_norm);
             AE_MULA32_LL(norm_inp22_2, scaled_inp22, d_norm);

             out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
             out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
             out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
             out12_2 = AE_SRAA64(norm_inp12_2, finalShift);
             out21_1 = AE_SRAA64(norm_inp21_1, finalShift);
             out21_2 = AE_SRAA64(norm_inp21_2, finalShift);
             out22_1 = AE_SRAA64(norm_inp22_1, finalShift);
             out22_2 = AE_SRAA64(norm_inp22_2, finalShift);
     
             sat_out11 = AE_SAT32X2(out11_1, out11_2);
             sat_out12 = AE_SAT32X2(out12_1, out12_2);
             sat_out21 = AE_SAT32X2(out21_1, out21_2);
             sat_out22 = AE_SAT32X2(out22_1, out22_2);
     
             sat_out1 = AE_SAT16X4(sat_out11, sat_out12);
             sat_out2 = AE_SAT16X4(sat_out21, sat_out22);
     
             d_out = AE_SAT8X8X16(sat_out1, sat_out2);
     
             AE_SA8X8_IP(d_out, a_out, ptr_out);
          }
          AE_SA64POS_FP(a_out, ptr_out);
          WORD32 rem_channels = input_channels & 7;
          
          const ae_int8x16 * ptr8x16_inp = (const ae_int8x16 * )ptr_inp;
          ae_int8x16 *ptr8x16_out = (ae_int8x16 *)ptr_out;
          
          ae_f16x4 d_inp1_t, d_inp2_t;
          if(rem_channels)
          {
            ae_valignx2 a_inpx2, a_outx2;
            a_inpx2 = AE_LA128_PP(ptr8x16_inp);
            a_outx2 = AE_ZALIGN128();
            ae_int8x8 d_inp, temp1;
             norm_inp11_1 = round_cnst;
             norm_inp11_2 = round_cnst;
             norm_inp12_1 = round_cnst;
             norm_inp12_2 = round_cnst;
             norm_inp21_1 = round_cnst;
             norm_inp21_2 = round_cnst;
             norm_inp22_1 = round_cnst;
             norm_inp22_2 = round_cnst;
            
            AE_LAV8X8X2_XP(d_inp, temp1, a_inpx2, ptr8x16_inp, rem_channels);
            AE_CVTI16X4X2F8(d_inp1_t, d_inp2_t, d_inp, 0);
            AE_LAV16X4X2_XP(d_out_mult1, d_out_mult2, a_out_mult, ptr_out_mult, rem_channels*2);

            AE_MULF2P32X16X4S(d_scale11, d_scale12, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult1));
            AE_MULF2P32X16X4S(d_scale21, d_scale22, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult2));
            d_scale1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale11), AE_MOVINT32X2_FROMF32X2(d_scale12));
            d_scale2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale21), AE_MOVINT32X2_FROMF32X2(d_scale22));

            AE_MUL16X4S(scaled_inp11, scaled_inp12, AE_MOVINT16X4_FROMF16X4(d_inp1_t), d_scale1);
            AE_MUL16X4S(scaled_inp21, scaled_inp22, AE_MOVINT16X4_FROMF16X4(d_inp2_t), d_scale2);
    
            AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
            AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
            AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
            AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);
            AE_MULA32_HH(norm_inp21_1, scaled_inp21, d_norm);
            AE_MULA32_LL(norm_inp21_2, scaled_inp21, d_norm);
            AE_MULA32_HH(norm_inp22_1, scaled_inp22, d_norm);
            AE_MULA32_LL(norm_inp22_2, scaled_inp22, d_norm);
    
            out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
            out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
            out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
            out12_2 = AE_SRAA64(norm_inp12_2, finalShift);
            out21_1 = AE_SRAA64(norm_inp21_1, finalShift);
            out21_2 = AE_SRAA64(norm_inp21_2, finalShift);
            out22_1 = AE_SRAA64(norm_inp22_1, finalShift);
            out22_2 = AE_SRAA64(norm_inp22_2, finalShift);
    
            sat_out11 = AE_SAT32X2(out11_1, out11_2);
            sat_out12 = AE_SAT32X2(out12_1, out12_2);
            sat_out21 = AE_SAT32X2(out21_1, out21_2);
            sat_out22 = AE_SAT32X2(out22_1, out22_2);
    
            sat_out1 = AE_SAT16X4(sat_out11, sat_out12);
            sat_out2 = AE_SAT16X4(sat_out21, sat_out22);
    
            d_out = AE_SAT8X8X16(sat_out1, sat_out2);
            AE_SAV8X8X2_XP(d_out, temp1, a_outx2, ptr8x16_out, rem_channels);
            AE_SA128POS_FP(a_outx2, ptr8x16_out);
          }
          ptr_out = (ae_int8x8*)ptr8x16_out;
          ptr_inp = (const signed char*)ptr8x16_inp;
          a_inp = AE_LA64_PP(ptr_inp);
        }
    
    }

  }
  else 
  {
    int ic;
    const ae_int16 * ptr_inp_normdata = (const ae_int16 *)p_inp_normdata;
        
    const signed char * ptr_inp = p_inp;
    ae_int8x8 * ptr_out = (ae_int8x8 *)p_out;
    WORD8 * ptr_nsa_shift = (WORD8 *)p_inp_nsadata;

    ae_int8x8 d_out;
    ae_int16x4 d_inp1, d_inp2, d_norm_factor, d_out_mult1, d_out_mult2;
    ae_int64 norm_inp11_1, norm_inp11_2, norm_inp12_1, norm_inp12_2, norm_inp21_1, norm_inp21_2, norm_inp22_1, norm_inp22_2;
    ae_int32x2 scaled_inp11, scaled_inp12, scaled_inp21, scaled_inp22;
    ae_int32x2 d_norm = ZERO32;
    ae_int64 out11_1, out11_2, out12_1, out12_2, out21_1, out21_2, out22_1, out22_2;
    ae_int32x2 sat_out11, sat_out12, sat_out21, sat_out22;
    ae_int16x4 sat_out1, sat_out2;

    ae_valign a_inp;
    ae_valign a_out = AE_ZALIGN64();
    WORD8 nsaShift, finalShift;
    WORD32 nsa_mult_factor;
    ae_int32x2 d_nsa_multiplier;
    ae_f32x2 d_norm_t;
    ae_f32x2 temp = AE_MOVF32X2_FROMINT32X2(ZERO32);
    
    ae_f32x2 d_scale11, d_scale12, d_scale21, d_scale22;

    if(per_chan_flag == 0)
    {
      ae_int16x4 d_mult = AE_MOVDA16(p_out_multiplier[0]);
      ae_int16x4 d_scale;

        for(int ihw = 0; ihw < input_height * input_width; ihw++)
        {
          AE_L16_IP(d_norm_factor, ptr_inp_normdata, 2);
          AE_CVTI32X4F16U(d_norm_t, temp, d_norm_factor, 0);
          d_norm = AE_MOVINT32X2_FROMF32X2(d_norm_t);
          a_inp = AE_LA64_PP(ptr_inp);
          nsaShift    = *(ptr_nsa_shift++);
          finalShift  = out_rshift + ((nsaShift + 1) >> 1) + rsqrt_shift;
  
          nsa_mult_factor = (nsaShift & 0x1) ? 46341 : (1<<15);
          d_nsa_multiplier = SW_MOVDA32(nsa_mult_factor);
  
          AE_MULF2P32X16X4S(d_scale11, d_scale12, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult));
          d_scale = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale11), AE_MOVINT32X2_FROMF32X2(d_scale12));
  
          ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));

#pragma concurrent          
          for(ic = 0; ic < (input_channels>>3); ic++)
          {
            norm_inp11_1 = round_cnst;
            norm_inp11_2 = round_cnst;
            norm_inp12_1 = round_cnst;
            norm_inp12_2 = round_cnst;
            norm_inp21_1 = round_cnst;
            norm_inp21_2 = round_cnst;
            norm_inp22_1 = round_cnst;
            norm_inp22_2 = round_cnst;

            AE_LA8X4S_IP(d_inp1, a_inp, ptr_inp);
            AE_LA8X4S_IP(d_inp2, a_inp, ptr_inp);
            
            AE_MUL16X4S(scaled_inp11, scaled_inp12, d_inp1, d_scale);
            AE_MUL16X4S(scaled_inp21, scaled_inp22, d_inp2, d_scale);
    
            AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
            AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
            AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
            AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);
            AE_MULA32_HH(norm_inp21_1, scaled_inp21, d_norm);
            AE_MULA32_LL(norm_inp21_2, scaled_inp21, d_norm);
            AE_MULA32_HH(norm_inp22_1, scaled_inp22, d_norm);
            AE_MULA32_LL(norm_inp22_2, scaled_inp22, d_norm);
    
            out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
            out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
            out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
            out12_2 = AE_SRAA64(norm_inp12_2, finalShift);
            out21_1 = AE_SRAA64(norm_inp21_1, finalShift);
            out21_2 = AE_SRAA64(norm_inp21_2, finalShift);
            out22_1 = AE_SRAA64(norm_inp22_1, finalShift);
            out22_2 = AE_SRAA64(norm_inp22_2, finalShift);
    
            sat_out11 = AE_SAT32X2(out11_1, out11_2);
            sat_out12 = AE_SAT32X2(out12_1, out12_2);
            sat_out21 = AE_SAT32X2(out21_1, out21_2);
            sat_out22 = AE_SAT32X2(out22_1, out22_2);
    
            sat_out1 = AE_SAT16X4(sat_out11, sat_out12);
            sat_out2 = AE_SAT16X4(sat_out21, sat_out22);
    
            d_out = AE_SAT8X8X16(sat_out1, sat_out2);
            AE_SA8X8_IP(d_out, a_out, ptr_out);
          }
          AE_SA64POS_FP(a_out, ptr_out);
          WORD32 rem_channels = input_channels & 7;
          
          const ae_int8x16 * ptr8x16_inp = (const ae_int8x16 * )ptr_inp;
          ae_int8x16 *ptr8x16_out = (ae_int8x16 *)ptr_out;
          
          ae_f16x4 d_inp1_t, d_inp2_t;
          if(rem_channels)
          {
            ae_valignx2 a_inpx2, a_outx2;
            a_inpx2 = AE_LA128_PP(ptr8x16_inp);
            a_outx2 = AE_ZALIGN128();
            ae_int8x8 d_inp, temp1;
            norm_inp11_1 = round_cnst;
            norm_inp11_2 = round_cnst;
            norm_inp12_1 = round_cnst;
            norm_inp12_2 = round_cnst;
            norm_inp21_1 = round_cnst;
            norm_inp21_2 = round_cnst;
            norm_inp22_1 = round_cnst;
            norm_inp22_2 = round_cnst;
            
            AE_LAV8X8X2_XP(d_inp, temp1, a_inpx2, ptr8x16_inp, rem_channels);
            AE_CVTI16X4X2F8(d_inp1_t, d_inp2_t, d_inp, 0);
            
            AE_MUL16X4S(scaled_inp11, scaled_inp12, AE_MOVINT16X4_FROMF16X4(d_inp1_t), d_scale);
            AE_MUL16X4S(scaled_inp21, scaled_inp22, AE_MOVINT16X4_FROMF16X4(d_inp2_t), d_scale);
    
            AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
            AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
            AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
            AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);
            AE_MULA32_HH(norm_inp21_1, scaled_inp21, d_norm);
            AE_MULA32_LL(norm_inp21_2, scaled_inp21, d_norm);
            AE_MULA32_HH(norm_inp22_1, scaled_inp22, d_norm);
            AE_MULA32_LL(norm_inp22_2, scaled_inp22, d_norm);
    
            out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
            out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
            out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
            out12_2 = AE_SRAA64(norm_inp12_2, finalShift);
            out21_1 = AE_SRAA64(norm_inp21_1, finalShift);
            out21_2 = AE_SRAA64(norm_inp21_2, finalShift);
            out22_1 = AE_SRAA64(norm_inp22_1, finalShift);
            out22_2 = AE_SRAA64(norm_inp22_2, finalShift);
            sat_out11 = AE_SAT32X2(out11_1, out11_2);
            sat_out12 = AE_SAT32X2(out12_1, out12_2);
            sat_out21 = AE_SAT32X2(out21_1, out21_2);
            sat_out22 = AE_SAT32X2(out22_1, out22_2);
    
            sat_out1 = AE_SAT16X4(sat_out11, sat_out12);
            sat_out2 = AE_SAT16X4(sat_out21, sat_out22);
    
            d_out = AE_SAT8X8X16(sat_out1, sat_out2);
            AE_SAV8X8X2_XP(d_out, temp1, a_outx2, ptr8x16_out, rem_channels);
            AE_SA128POS_FP(a_outx2, ptr8x16_out);
          }
          ptr_out = (ae_int8x8*)ptr8x16_out;
          ptr_inp = (const signed char*)ptr8x16_inp;
        }
    }
    else
    {
        for(int ihw = 0; ihw < input_height * input_width; ihw++)
        {
          ae_int16x8 *ptr_out_mult = (ae_int16x8 *)p_out_multiplier;
          ae_valignx2 a_out_mult = AE_LA128_PP(ptr_out_mult);
          ae_int16x4 d_scale1, d_scale2;

          AE_L16_IP(d_norm_factor, ptr_inp_normdata, 2);
          AE_CVTI32X4F16U(d_norm_t, temp, d_norm_factor, 0);
          d_norm = AE_MOVINT32X2_FROMF32X2(d_norm_t);
          nsaShift    = *(ptr_nsa_shift++);
          finalShift  = out_rshift + ((nsaShift + 1) >> 1) + rsqrt_shift;
  
          nsa_mult_factor = (nsaShift & 0x1) ? 46341 : (1<<15);
          d_nsa_multiplier = SW_MOVDA32(nsa_mult_factor);

          ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));

          a_inp = AE_LA64_PP(ptr_inp);
#pragma concurrent          
          for(ic = 0; ic < (input_channels>>3); ic++)
          {
            norm_inp11_1 = round_cnst;
            norm_inp11_2 = round_cnst;
            norm_inp12_1 = round_cnst;
            norm_inp12_2 = round_cnst;
            norm_inp21_1 = round_cnst;
            norm_inp21_2 = round_cnst;
            norm_inp22_1 = round_cnst;
            norm_inp22_2 = round_cnst;
            AE_LA8X4S_IP(d_inp1, a_inp, ptr_inp);
            AE_LA8X4S_IP(d_inp2, a_inp, ptr_inp);
            AE_LA16X4X2_IP(d_out_mult1, d_out_mult2, a_out_mult, ptr_out_mult);

            AE_MULF2P32X16X4S(d_scale11, d_scale12, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult1));
            AE_MULF2P32X16X4S(d_scale21, d_scale22, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult2));
            d_scale1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale11), AE_MOVINT32X2_FROMF32X2(d_scale12));
            d_scale2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale21), AE_MOVINT32X2_FROMF32X2(d_scale22));

            AE_MUL16X4S(scaled_inp11, scaled_inp12, d_inp1, d_scale1);
            AE_MUL16X4S(scaled_inp21, scaled_inp22, d_inp2, d_scale2);
    
            AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
            AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
            AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
            AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);
            AE_MULA32_HH(norm_inp21_1, scaled_inp21, d_norm);
            AE_MULA32_LL(norm_inp21_2, scaled_inp21, d_norm);
            AE_MULA32_HH(norm_inp22_1, scaled_inp22, d_norm);
            AE_MULA32_LL(norm_inp22_2, scaled_inp22, d_norm);
    
            out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
            out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
            out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
            out12_2 = AE_SRAA64(norm_inp12_2, finalShift);
            out21_1 = AE_SRAA64(norm_inp21_1, finalShift);
            out21_2 = AE_SRAA64(norm_inp21_2, finalShift);
            out22_1 = AE_SRAA64(norm_inp22_1, finalShift);
            out22_2 = AE_SRAA64(norm_inp22_2, finalShift);
    
            sat_out11 = AE_SAT32X2(out11_1, out11_2);
            sat_out12 = AE_SAT32X2(out12_1, out12_2);
            sat_out21 = AE_SAT32X2(out21_1, out21_2);
            sat_out22 = AE_SAT32X2(out22_1, out22_2);
    
            sat_out1 = AE_SAT16X4(sat_out11, sat_out12);
            sat_out2 = AE_SAT16X4(sat_out21, sat_out22);
    
            d_out = AE_SAT8X8X16(sat_out1, sat_out2);
            AE_SA8X8_IP(d_out, a_out, ptr_out);
          }
          AE_SA64POS_FP(a_out, ptr_out);
          WORD32 rem_channels = input_channels & 7;
          
          const ae_int8x16 * ptr8x16_inp = (const ae_int8x16 * )ptr_inp;
          ae_int8x16 *ptr8x16_out = (ae_int8x16 *)ptr_out;
          
          ae_f16x4 d_inp1_t, d_inp2_t;
          if(rem_channels)
          {
            ae_valignx2 a_inpx2, a_outx2;
            a_inpx2 = AE_LA128_PP(ptr8x16_inp);
            a_outx2 = AE_ZALIGN128();
            ae_int8x8 d_inp, temp1;
            norm_inp11_1 = round_cnst;
            norm_inp11_2 = round_cnst;
            norm_inp12_1 = round_cnst;
            norm_inp12_2 = round_cnst;
            norm_inp21_1 = round_cnst;
            norm_inp21_2 = round_cnst;
            norm_inp22_1 = round_cnst;
            norm_inp22_2 = round_cnst;
            
            AE_LAV8X8X2_XP(d_inp, temp1, a_inpx2, ptr8x16_inp, rem_channels);
            AE_CVTI16X4X2F8(d_inp1_t, d_inp2_t, d_inp, 0);
            AE_LAV16X4X2_XP(d_out_mult1, d_out_mult2, a_out_mult, ptr_out_mult, rem_channels*2);
            AE_MULF2P32X16X4S(d_scale11, d_scale12, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult1));
            AE_MULF2P32X16X4S(d_scale21, d_scale22, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_out_mult2));
            d_scale1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale11), AE_MOVINT32X2_FROMF32X2(d_scale12));
            d_scale2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale21), AE_MOVINT32X2_FROMF32X2(d_scale22));

            AE_MUL16X4S(scaled_inp11, scaled_inp12, AE_MOVINT16X4_FROMF16X4(d_inp1_t), d_scale1);
            AE_MUL16X4S(scaled_inp21, scaled_inp22, AE_MOVINT16X4_FROMF16X4(d_inp2_t), d_scale2);
    
            AE_MULA32_HH(norm_inp11_1, scaled_inp11, d_norm);
            AE_MULA32_LL(norm_inp11_2, scaled_inp11, d_norm);
            AE_MULA32_HH(norm_inp12_1, scaled_inp12, d_norm);
            AE_MULA32_LL(norm_inp12_2, scaled_inp12, d_norm);
            AE_MULA32_HH(norm_inp21_1, scaled_inp21, d_norm);
            AE_MULA32_LL(norm_inp21_2, scaled_inp21, d_norm);
            AE_MULA32_HH(norm_inp22_1, scaled_inp22, d_norm);
            AE_MULA32_LL(norm_inp22_2, scaled_inp22, d_norm);
    
            out11_1 = AE_SRAA64(norm_inp11_1, finalShift);
            out11_2 = AE_SRAA64(norm_inp11_2, finalShift);
            out12_1 = AE_SRAA64(norm_inp12_1, finalShift);
            out12_2 = AE_SRAA64(norm_inp12_2, finalShift);
            out21_1 = AE_SRAA64(norm_inp21_1, finalShift);
            out21_2 = AE_SRAA64(norm_inp21_2, finalShift);
            out22_1 = AE_SRAA64(norm_inp22_1, finalShift);
            out22_2 = AE_SRAA64(norm_inp22_2, finalShift);
    
            sat_out11 = AE_SAT32X2(out11_1, out11_2);
            sat_out12 = AE_SAT32X2(out12_1, out12_2);
            sat_out21 = AE_SAT32X2(out21_1, out21_2);
            sat_out22 = AE_SAT32X2(out22_1, out22_2);
    
            sat_out1 = AE_SAT16X4(sat_out11, sat_out12);
            sat_out2 = AE_SAT16X4(sat_out21, sat_out22);
    
            d_out = AE_SAT8X8X16(sat_out1, sat_out2);
            AE_SAV8X8X2_XP(d_out, temp1, a_outx2, ptr8x16_out, rem_channels);
            AE_SA128POS_FP(a_outx2, ptr8x16_out);
          }
          ptr_out = (ae_int8x8*)ptr8x16_out;
          ptr_inp = (const signed char*)ptr8x16_inp;
        }
    }     
  }
  return 0;
}

