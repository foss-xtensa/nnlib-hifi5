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

#define SW_MOVDA32(a) AE_MOVDA32X2(a, a)
#define ZERO32 AE_ZERO32()
#define ZERO64   AE_ZERO64()

WORD32 xa_nn_norm_calc_3D_16_nhwc(
    UWORD16 * p_outnorm /*Norm data: 2D -> iw*ih, or scalar*/ , 
    WORD8 * p_outnsa /*NSA data: 2D -> iw*ih, or scalar*/ , 
    const WORD16 * p_inp /*3D -> iw*ih*ic */,
    int input_height, int input_width, int input_channels, 
    int accross_depth_flag,
    int out_shift, /*sumSquareShift*/
    const UWORD16 *prsqrt, int rsqrt_table_len /* rsqrt table */)
{

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_outnorm, -1);
  XA_NNLIB_ARG_CHK_PTR(p_outnsa, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(prsqrt, -1);

  /* Pointer Alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_outnorm, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(prsqrt, sizeof(WORD16), -1);

  /* Param Checks*/
  XA_NNLIB_ARG_CHK_COND((out_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((rsqrt_table_len <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((accross_depth_flag != 0) && (accross_depth_flag != 1), -1);

  int out_rshift = -out_shift;
  if(rsqrt_table_len>32768) { rsqrt_table_len = 32768;}

  if(accross_depth_flag == 0) /* Calc norm data for entire 3D input */
  {
    WORD32 i;
    WORD32 inp_len = input_height*input_width*input_channels;
    WORD32 lc = inp_len >> 3;
    WORD32 remc = inp_len & 7;
    
    ae_int16x8 *ptr_inp = (ae_int16x8 *)p_inp;
    ae_valignx2 a_inp = AE_LA128_PP(ptr_inp);
    ae_int16x4 d_inp1, d_inp2;
    ae_int64 acc1=ZERO64, acc2=ZERO64;
    ae_int64 acc, norm64;
    ae_int64 one_out_rshifted = int32_rtor_ae_int64(1 << (out_rshift - 1));
    ae_int64 norm_min = ZERO64;
    ae_int64 norm64_max = int32_rtor_ae_int64(2147483647);
    ae_int64 norm32_max = int32_rtor_ae_int64(rsqrt_table_len-1);
#pragma concurrent    
    for(i = 0; i < lc; i++)
    {
      AE_LA16X4X2_IP(d_inp1, d_inp2, a_inp, ptr_inp);
      AE_MULAAAA2Q16(acc1, acc2, d_inp1, d_inp2, d_inp1, d_inp2);
    }
    if(remc)
    {
      AE_LAV16X4X2_XP(d_inp1, d_inp2, a_inp, ptr_inp, remc*2);
      AE_MULAAAA2Q16(acc1, acc2, d_inp1, d_inp2, d_inp1, d_inp2);
    }
    acc = AE_ADD64(acc1, acc2);
    
    norm64 = AE_ADD64(acc, one_out_rshifted); 
    norm64 = AE_SRAA64(norm64, out_rshift);
    
    norm64 = AE_MIN64(AE_MAX64(norm64,norm_min),norm64_max);
    
    ae_int32x2 norm32x2 = AE_SAT32X2(norm64, norm64);
    WORD32 nsaShift = AE_NSAZ32_L(norm32x2);
    if(AE_MOVAB(AE_EQ64(norm64,ZERO64)))
    {
      nsaShift = 31;
    }
    nsaShift = 15 - nsaShift + 1;
    nsaShift = (nsaShift<0) ? 0 : nsaShift;
    ae_int64 one_nsaShift = int32_rtor_ae_int64(nsaShift>0 ? (1 << (nsaShift - 1)) : 0);
    norm64 = AE_ADD64(norm64, one_nsaShift);  
    norm64 = AE_SRAA64(norm64, nsaShift);
    norm64 = AE_MIN64(AE_MAX64(norm64,norm_min),norm32_max);
    
    norm32x2 = AE_MOVINT32X2_FROMINT64(norm64);
    p_outnorm[0] = prsqrt[AE_MOVAD32_L(norm32x2)];
    p_outnsa[0] = nsaShift + out_rshift;
  }
  else
  {
    WORD32 i, j;
    WORD32 lc = input_channels >> 3;
    WORD32 remc = input_channels & 7;
    
    ae_int16x8 *ptr_inp = (ae_int16x8 *)p_inp;
    ae_valignx2 a_inp = AE_LA128_PP(ptr_inp);
    ae_int16x4 d_inp1, d_inp2;
    ae_int64 acc1, acc2;
    ae_int64 acc, norm64;
    ae_int64 one_out_rshifted = int32_rtor_ae_int64(out_rshift > 0 ? (1 << (out_rshift - 1)) : 0);
    ae_int64 norm_min = ZERO64;
    ae_int64 norm64_max = int32_rtor_ae_int64(2147483647);
    ae_int64 norm32_max = int32_rtor_ae_int64(rsqrt_table_len-1);
    ae_int32x2 norm32x2;
    
    for(j = 0; j < (input_height * input_width); j++)
    {
      acc1=ZERO64, acc2=ZERO64;
#pragma concurrent
      for(i = 0; i < lc; i++)
      {
        AE_LA16X4X2_IP(d_inp1, d_inp2, a_inp, ptr_inp);
        AE_MULAAAA2Q16(acc1, acc2, d_inp1, d_inp2, d_inp1, d_inp2);
      }
      if(remc)
      {
        AE_LAV16X4X2_XP(d_inp1, d_inp2, a_inp, ptr_inp, remc*2);
        AE_MULAAAA2Q16(acc1, acc2, d_inp1, d_inp2, d_inp1, d_inp2);
      }
      acc = AE_ADD64(acc1, acc2);
      
      norm64 = AE_ADD64(acc, one_out_rshifted); 
      norm64 = AE_SRAA64(norm64, out_rshift);
      
      norm64 = AE_MIN64(AE_MAX64(norm64,norm_min),norm64_max);
      
      norm32x2 = AE_SAT32X2(norm64, norm64);
      WORD32 nsaShift = AE_NSAZ32_L(norm32x2);
      if(AE_MOVAB(AE_EQ64(norm64,ZERO64)))
      {
        nsaShift = 31;
      }
      nsaShift = 15 - nsaShift + 1;
      nsaShift = (nsaShift<0) ? 0 : nsaShift;
      ae_int64 one_nsaShift = int32_rtor_ae_int64(nsaShift>0 ? (1 << (nsaShift - 1)) : 0);
      norm64 = AE_ADD64(norm64, one_nsaShift);  
      norm64 = AE_SRAA64(norm64, nsaShift);
      norm64 = AE_MIN64(AE_MAX64(norm64,norm_min),norm32_max);
      norm32x2 = AE_MOVINT32X2_FROMINT64(norm64);
      p_outnorm[j] = prsqrt[AE_MOVAD32_L(norm32x2)];
      p_outnsa[j] = nsaShift + out_rshift;
    }
  }

  return 0;
}

WORD32 xa_nn_norm_apply_3D_16_nhwc(
    WORD16 * p_out, 
    const WORD16 * p_inp, /*3D -> iw*ih*ic */
    const UWORD16 *p_inp_normdata,
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
  XA_NNLIB_ARG_CHK_COND((out_shift > 15 ) || (out_shift < -15), -1);
  XA_NNLIB_ARG_CHK_COND((rsqrt_shift < 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_height <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((accross_depth_flag != 0) && (accross_depth_flag != 1), -1);
  XA_NNLIB_ARG_CHK_COND((per_chan_flag != 0) && (per_chan_flag != 1), -1);

  int out_rshift = -out_shift;

  if(accross_depth_flag == 0){
    UWORD16 norm_factor = p_inp_normdata[0];
    ae_int32x2 d_norm = SW_MOVDA32(norm_factor);
    
    WORD8  nsaShift    = p_inp_nsadata[0];
    WORD8 finalShift  = out_rshift + ((nsaShift + 1) >> 1) + rsqrt_shift;
    
    const ae_int16x8 * ptr_inp = (const ae_int16x8 *) p_inp;
    ae_valignx2 a_inp = AE_LA128_PP(ptr_inp);
    
    ae_int16x4 d_inp1, d_inp2;
    ae_int16x4 d_mult1, d_mult2;
    ae_int32x2 scaled_inp11, scaled_inp12, scaled_inp21, scaled_inp22;
    ae_int64 norm_inp11_1, norm_inp11_2, norm_inp12_1, norm_inp12_2, norm_inp21_1, norm_inp21_2, norm_inp22_1, norm_inp22_2;
    ae_int64 out11_1, out11_2, out12_1, out12_2, out21_1, out21_2, out22_1, out22_2;
    ae_int32x2 sat_out11, sat_out12, sat_out21, sat_out22;
    ae_int16x4 sat_out1, sat_out2;
    
    ae_int16x8 *ptr_out = (ae_int16x8 *)p_out;
    ae_valignx2 a_out = AE_ZALIGN128();
    if(per_chan_flag == 0)
    {
      WORD32 i;
      WORD32 size = input_height*input_width*input_channels;
      
      WORD16 out_mul     = p_out_multiplier[0];      
      WORD16 finalScale = ((nsaShift & 0x1) ? (((int64_t) 46341 * out_mul) >> 15) : out_mul);
      ae_int16x4 d_scale = AE_MOVDA16(finalScale);

      ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));
      for(i = 0; i < (size >> 3); i++)
      {
        norm_inp11_1 = round_cnst;
        norm_inp11_2 = round_cnst;
        norm_inp12_1 = round_cnst;
        norm_inp12_2 = round_cnst;
        norm_inp21_1 = round_cnst;
        norm_inp21_2 = round_cnst;
        norm_inp22_1 = round_cnst;
        norm_inp22_2 = round_cnst;

        AE_LA16X4X2_IP(d_inp1, d_inp2, a_inp, ptr_inp);
        
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
        
        AE_SA16X4X2_IP(sat_out1, sat_out2, a_out, ptr_out);
      }
      WORD32 rem_size = size & 7;
      if(rem_size)
      {
        norm_inp11_1 = round_cnst;
        norm_inp11_2 = round_cnst;
        norm_inp12_1 = round_cnst;
        norm_inp12_2 = round_cnst;
        norm_inp21_1 = round_cnst;
        norm_inp21_2 = round_cnst;
        norm_inp22_1 = round_cnst;
        norm_inp22_2 = round_cnst;

        AE_LAV16X4X2_XP(d_inp1, d_inp2, a_inp, ptr_inp, rem_size*2);
        
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
        
        AE_SAV16X4X2_XP(sat_out1, sat_out2, a_out, ptr_out, rem_size*2);
      }
      AE_SA128POS_FP(a_out, ptr_out);
    }
    else
    {
      int ihw, ic;     
      
      ae_int16x4 d_scale1, d_scale2;
	  ae_f32x2 d_scale11 = AE_MOVF32X2_FROMINT32X2(ZERO32);
      ae_f32x2 d_scale12 = AE_MOVF32X2_FROMINT32X2(ZERO32);
      ae_f32x2 d_scale21 = AE_MOVF32X2_FROMINT32X2(ZERO32);
      ae_f32x2 d_scale22 = AE_MOVF32X2_FROMINT32X2(ZERO32);
      
      WORD32 nsa_mult_factor = (nsaShift & 0x1) ? 46341 : (1<<15);
      ae_int32x2 d_nsa_multiplier = SW_MOVDA32(nsa_mult_factor);
    
      ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));
      for(ihw = 0; ihw < input_height * input_width; ihw++)
      {
        ae_int16x8 *ptr_mult = (ae_int16x8 *)p_out_multiplier;
        ae_valignx2 a_mult = AE_LA128_PP(ptr_mult);
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

          AE_LA16X4X2_IP(d_inp1, d_inp2, a_inp, ptr_inp);
          AE_LA16X4X2_IP(d_mult1, d_mult2, a_mult, ptr_mult);
          
          AE_MULF2P32X16X4S(d_scale11, d_scale12, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult1));
          AE_MULF2P32X16X4S(d_scale21, d_scale22, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult2));
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
          
          AE_SA16X4X2_IP(sat_out1, sat_out2, a_out, ptr_out);
        }
        WORD32 rem_channels = input_channels & 7;
        if(rem_channels)
        {
          norm_inp11_1 = round_cnst;
          norm_inp11_2 = round_cnst;
          norm_inp12_1 = round_cnst;
          norm_inp12_2 = round_cnst;
          norm_inp21_1 = round_cnst;
          norm_inp21_2 = round_cnst;
          norm_inp22_1 = round_cnst;
          norm_inp22_2 = round_cnst;

          AE_LAV16X4X2_XP(d_inp1, d_inp2, a_inp, ptr_inp, rem_channels*2);
          AE_LAV16X4X2_XP(d_mult1, d_mult2, a_mult, ptr_mult, rem_channels*2);
          
          AE_MULF2P32X16X4S(d_scale11, d_scale12, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult1));
          AE_MULF2P32X16X4S(d_scale21, d_scale22, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult2));
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
          
          AE_SAV16X4X2_XP(sat_out1, sat_out2, a_out, ptr_out, rem_channels*2);
        }
        AE_SA128POS_FP(a_out, ptr_out);
      }
    }
  } 
  else {
    int ihw, ic;
      
    const ae_int16x8 * ptr_inp = (const ae_int16x8 *)p_inp;
    ae_valignx2 a_inp = AE_LA128_PP(ptr_inp);
    
    ae_int16 * ptr_norm = (ae_int16 *)p_inp_normdata;      
    WORD8 * ptr_nsa_shift = (WORD8 *)p_inp_nsadata;
    
    ae_int16x4 d_norm16x4;
    ae_int32x2 d_norm = ZERO32;
    ae_f32x2 temp = AE_MOVF32X2_FROMINT32X2(ZERO32);
    WORD8 nsaShift, finalShift;
    WORD32 nsa_mult_factor;
    ae_int32x2 d_nsa_multiplier;
    ae_int16x4 d_inp1, d_inp2;
    
    ae_int32x2 scaled_inp11, scaled_inp12, scaled_inp21, scaled_inp22;
    ae_int64 norm_inp11_1, norm_inp11_2, norm_inp12_1, norm_inp12_2, norm_inp21_1, norm_inp21_2, norm_inp22_1, norm_inp22_2;
    ae_int64 out11_1, out11_2, out12_1, out12_2, out21_1, out21_2, out22_1, out22_2;
    ae_int32x2 sat_out11, sat_out12, sat_out21, sat_out22;
    ae_int16x4 sat_out1, sat_out2;
    
    ae_int16x8 * ptr_out = (ae_int16x8 *)p_out;
    ae_valignx2 a_out = AE_ZALIGN128();
    
    if(per_chan_flag == 0)
    {
      ae_int16x4 d_mult = AE_MOVDA16(p_out_multiplier[0]);
      ae_int16x4 d_scale;
      
      ae_f32x2 d_scale1 = AE_MOVF32X2_FROMINT32X2(ZERO32);
      ae_f32x2 d_scale2 = AE_MOVF32X2_FROMINT32X2(ZERO32);
      ae_f32x2 d_norm_t;
      for(ihw = 0; ihw < input_height * input_width; ihw++)
      {
        AE_L16_IP(d_norm16x4, ptr_norm, 2);
        AE_CVTI32X4F16U(d_norm_t, temp, d_norm16x4, 0);

        nsaShift    = *(ptr_nsa_shift++);
        finalShift  = out_rshift + ((nsaShift + 1) >> 1) + rsqrt_shift;
        
        nsa_mult_factor = (nsaShift & 0x1) ? 46341 : (1<<15);
        d_nsa_multiplier = SW_MOVDA32(nsa_mult_factor);
        
        AE_MULF2P32X16X4S(d_scale1, d_scale2, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult));
        d_scale = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(d_scale1), AE_MOVINT32X2_FROMF32X2(d_scale2));

        ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));
        d_norm = AE_MOVINT32X2_FROMF32X2(d_norm_t);
        for(ic = 0; ic < (input_channels >> 3); ic++)
        {
          norm_inp11_1 = round_cnst;
          norm_inp11_2 = round_cnst;
          norm_inp12_1 = round_cnst;
          norm_inp12_2 = round_cnst;
          norm_inp21_1 = round_cnst;
          norm_inp21_2 = round_cnst;
          norm_inp22_1 = round_cnst;
          norm_inp22_2 = round_cnst;

          AE_LA16X4X2_IP(d_inp1, d_inp2, a_inp, ptr_inp);
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

          AE_SA16X4X2_IP(sat_out1, sat_out2, a_out, ptr_out);          
        }
        WORD32 rem_channels = input_channels & 7;
        if(rem_channels)
        {
          norm_inp11_1 = round_cnst;
          norm_inp11_2 = round_cnst;
          norm_inp12_1 = round_cnst;
          norm_inp12_2 = round_cnst;
          norm_inp21_1 = round_cnst;
          norm_inp21_2 = round_cnst;
          norm_inp22_1 = round_cnst;
          norm_inp22_2 = round_cnst;

          AE_LAV16X4X2_XP(d_inp1, d_inp2, a_inp, ptr_inp, rem_channels*2);
          
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

          AE_SAV16X4X2_XP(sat_out1, sat_out2, a_out, ptr_out, rem_channels*2);          
        }
        AE_SA128POS_FP(a_out, ptr_out);
      }
    }
    else
    {
      ae_int16x8 * ptr_mult;
      ae_valignx2 a_mult;
      
      
      ae_f32x2 d_scale11 = AE_MOVF32X2_FROMINT32X2(ZERO32);
      ae_f32x2 d_scale12 = AE_MOVF32X2_FROMINT32X2(ZERO32);
      ae_f32x2 d_scale21 = AE_MOVF32X2_FROMINT32X2(ZERO32);
      ae_f32x2 d_scale22 = AE_MOVF32X2_FROMINT32X2(ZERO32);
      ae_int16x4 d_scale1, d_scale2;
      ae_int16x4 d_mult1, d_mult2;

      ae_f32x2 d_norm_t;
      for(ihw = 0; ihw < input_height * input_width; ihw++)
      {
        AE_L16_IP(d_norm16x4, ptr_norm, 2);
        AE_CVTI32X4F16U(d_norm_t, temp, d_norm16x4, 0);

        nsaShift    = *(ptr_nsa_shift++);
        finalShift  = out_rshift + ((nsaShift + 1) >> 1) + rsqrt_shift;
        
        nsa_mult_factor = (nsaShift & 0x1) ? 46341 : (1<<15);
        d_nsa_multiplier = SW_MOVDA32(nsa_mult_factor);
        
        ptr_mult = (ae_int16x8 *)p_out_multiplier;
        a_mult = AE_LA128_PP(ptr_mult);

        ae_int64 round_cnst = AE_SLAA64(AE_MOVINT64_FROMINT32X2(AE_MOVDA32X2(0, 1)), (finalShift-1));

        d_norm = AE_MOVINT32X2_FROMF32X2(d_norm_t);
        for(ic = 0; ic < (input_channels >> 3); ic++)
        {
          norm_inp11_1 = round_cnst;
          norm_inp11_2 = round_cnst;
          norm_inp12_1 = round_cnst;
          norm_inp12_2 = round_cnst;
          norm_inp21_1 = round_cnst;
          norm_inp21_2 = round_cnst;
          norm_inp22_1 = round_cnst;
          norm_inp22_2 = round_cnst;

          AE_LA16X4X2_IP(d_inp1, d_inp2, a_inp, ptr_inp);
          AE_LA16X4X2_IP(d_mult1, d_mult2, a_mult, ptr_mult);
          
          AE_MULF2P32X16X4S(d_scale11, d_scale12, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult1));
          AE_MULF2P32X16X4S(d_scale21, d_scale22, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult2));
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

          AE_SA16X4X2_IP(sat_out1, sat_out2, a_out, ptr_out);          
        }
        WORD32 rem_channels = input_channels & 7;
        if(rem_channels)
        {
          norm_inp11_1 = round_cnst;
          norm_inp11_2 = round_cnst;
          norm_inp12_1 = round_cnst;
          norm_inp12_2 = round_cnst;
          norm_inp21_1 = round_cnst;
          norm_inp21_2 = round_cnst;
          norm_inp22_1 = round_cnst;
          norm_inp22_2 = round_cnst;

          AE_LAV16X4X2_XP(d_inp1, d_inp2, a_inp, ptr_inp, rem_channels*2);
          AE_LAV16X4X2_XP(d_mult1, d_mult2, a_mult, ptr_mult, rem_channels*2);
          
          AE_MULF2P32X16X4S(d_scale11, d_scale12, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult1));
          AE_MULF2P32X16X4S(d_scale21, d_scale22, AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF32X2_FROMINT32X2(d_nsa_multiplier), AE_MOVF16X4_FROMINT16X4(d_mult2));
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

          AE_SAV16X4X2_XP(sat_out1, sat_out2, a_out, ptr_out, rem_channels*2);          
        }
        AE_SA128POS_FP(a_out, ptr_out);
      }
    }  
  }

  return 0;
}

