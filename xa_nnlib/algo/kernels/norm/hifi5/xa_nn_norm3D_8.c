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
#define USHRT_MAX  65535

#define ZERO32   AE_ZERO32()
#define SW_MOVDA32(a) AE_MOVDA32X2(a, a)
#define ZERO16   AE_ZERO16()

#define SW_SLAA32S_INT32X2_INT32X2(inp1, inp2) AE_MOVINT32X2_FROMF32X2(AE_SLAA32S(AE_MOVF32X2_FROMINT32X2(inp1), inp2))

WORD32 xa_nn_norm_calc_3D_8_nhwc(
    WORD16 * p_out /*Noram data: 2D -> iw*ih, or scalar*/ , 
    const WORD8 * p_inp /*3D -> iw*ih*ic */,
    int input_height, int input_width, int input_channels, 
    int accross_depth_flag,
    int out_shift, /*sumSquareShift*/
    const UWORD16 *prsqrt, int rsqrt_shift, int rsqrt_table_len, /* rsqrt table */
    const UWORD16 *precip, int recip_shift) /* recip table */
{

  /* NULL pointer check */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(prsqrt, -1);
  XA_NNLIB_ARG_CHK_PTR(precip, -1);
  /* Basic Parameter checks */  
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0 || input_channels <= 0),-1);
  XA_NNLIB_ARG_CHK_COND(rsqrt_table_len <= 0,-1);
  XA_NNLIB_ARG_CHK_COND((out_shift > 0), -1);
  XA_NNLIB_ARG_CHK_COND((accross_depth_flag != 0) && (accross_depth_flag != 1), -1);
  int out_rshift = -out_shift;
  /* Shift checks */
  XA_NNLIB_ARG_CHK_COND((recip_shift-out_rshift < 0), -1);
  XA_NNLIB_ARG_CHK_COND((recip_shift-rsqrt_shift < 0), -1);
  XA_NNLIB_ARG_CHK_COND((recip_shift< 0), -1);
  
  if(accross_depth_flag == 0) /* Calc norm data for entire 3D input */
  {
    WORD32 i;
    WORD32 inp_len = input_height*input_width*input_channels;
    WORD32 lc = inp_len >> 4;
    WORD32 remc = inp_len & 15;
    
    ae_int8x16 *ptr_inp = (ae_int8x16 *)p_inp;
    ae_valignx2 a_inp = AE_LA128_PP(ptr_inp);
    ae_int8x8 d_inp1, d_inp2;
    ae_int32x2 acc1=ZERO32, acc2=ZERO32;
    
    for(i = 0; i < lc; i++)
    {
      AE_LA8X8X2_IP(d_inp1, d_inp2, a_inp, ptr_inp);
      AE_MULA8Q8X8(acc1, acc2, d_inp1, d_inp1, d_inp1, d_inp1, d_inp1);
      AE_MULA8Q8X8(acc1, acc2, d_inp2, d_inp2, d_inp2, d_inp2, d_inp2);
    }
    if(remc & 15)
    {
      AE_LAV8X8X2_XP(d_inp1, d_inp2, a_inp, ptr_inp, inp_len & 15);
      AE_MULA8Q8X8(acc1, acc2, d_inp1, d_inp1, d_inp1, d_inp1, d_inp1);
      AE_MULA8Q8X8(acc1, acc2, d_inp2, d_inp2, d_inp2, d_inp2, d_inp2);
    }
    
    acc1 = AE_MOVINT32X2_FROMF32X2(AE_SRAA32RS(AE_MOVF32X2_FROMINT32X2(acc1), out_rshift));
    AE_MINMAX32(acc1, SW_MOVDA32(0), SW_MOVDA32(rsqrt_table_len-1));
    p_out[0] = prsqrt[AE_MOVAD32_H(acc1)];
  }
  else /* Calc norm data across depth dimension only */
  {
    WORD32 ihw, ic;
    WORD32 ilc = input_channels >> 4;
    WORD32 iremc = input_channels & 15;
    WORD32 olc = input_height * input_width;
    
    ae_int8x16 *ptr_inp = (ae_int8x16 *)p_inp;
    ae_int16 * ptr_out = (ae_int16 *)p_out;
    
    ae_valignx2 a_inp = AE_LA128_PP(ptr_inp);

    ae_int16x4 res16;    
    ae_int32x2 norm32_t, recip_dmax, table_idx32;
    ae_int32x2 table_idx32_sftd;
    ae_int64 table_idx64, res64;
    ae_int32x2 res32, res32_sftd;
    ae_int32x2 d_sqrt32;
    WORD32 max;
    ae_int8x8  d_inp1, d_inp2;   
    
    for(ihw = 0; ihw < olc; ihw++)
    {
        ae_int32x2 acc1 = ZERO32, acc2 = ZERO32;
        
        ae_int16x4 maxval16 = ZERO16;
        
        for(ic = 0; ic < ilc; ic++)
        {
          AE_LA8X8X2_IP(d_inp1, d_inp2, a_inp, ptr_inp);
          AE_MULA8Q8X8(acc1, acc2, d_inp1, d_inp1, d_inp1, d_inp1, d_inp1);
          AE_MULA8Q8X8(acc1, acc2, d_inp2, d_inp2, d_inp2, d_inp2, d_inp2);

          ae_int16x4 d_inp1_0, d_inp1_1, d_inp2_0, d_inp2_1;
          AE_SUBW8(d_inp1_0, d_inp1_1, d_inp1, AE_MOVDA8(0));
          AE_SUBW8(d_inp2_0, d_inp2_1, d_inp2, AE_MOVDA8(0));
          ae_int16x4 absval1_0, absval1_1;
          ae_int16x4 absval2_0, absval2_1;
          absval1_0 = AE_MOVINT16X4_FROMF16X4(AE_ABS16S(AE_MOVF16X4_FROMINT16X4(d_inp1_0)));
          absval1_1 = AE_MOVINT16X4_FROMF16X4(AE_ABS16S(AE_MOVF16X4_FROMINT16X4(d_inp1_1)));
          absval2_0 = AE_MOVINT16X4_FROMF16X4(AE_ABS16S(AE_MOVF16X4_FROMINT16X4(d_inp2_0)));
          absval2_1 = AE_MOVINT16X4_FROMF16X4(AE_ABS16S(AE_MOVF16X4_FROMINT16X4(d_inp2_1)));

          maxval16 = AE_MAX16(maxval16, absval1_0);
          maxval16 = AE_MAX16(maxval16, absval1_1);
          maxval16 = AE_MAX16(maxval16, absval2_0);
          maxval16 = AE_MAX16(maxval16, absval2_1);
        }
        if(iremc)
        {
          AE_LAV8X8X2_XP(d_inp1, d_inp2, a_inp, ptr_inp, iremc);
          AE_MULA8Q8X8(acc1, acc2, d_inp1, d_inp1, d_inp1, d_inp1, d_inp1);
          AE_MULA8Q8X8(acc1, acc2, d_inp2, d_inp2, d_inp2, d_inp2, d_inp2);

          ae_int16x4 d_inp1_0, d_inp1_1, d_inp2_0, d_inp2_1;
          AE_SUBW8(d_inp1_0, d_inp1_1, d_inp1, AE_MOVDA8(0));
          AE_SUBW8(d_inp2_0, d_inp2_1, d_inp2, AE_MOVDA8(0));
          ae_int16x4 absval1_0, absval1_1;
          ae_int16x4 absval2_0, absval2_1;
          absval1_0 = AE_MOVINT16X4_FROMF16X4(AE_ABS16S(AE_MOVF16X4_FROMINT16X4(d_inp1_0)));
          absval1_1 = AE_MOVINT16X4_FROMF16X4(AE_ABS16S(AE_MOVF16X4_FROMINT16X4(d_inp1_1)));
          absval2_0 = AE_MOVINT16X4_FROMF16X4(AE_ABS16S(AE_MOVF16X4_FROMINT16X4(d_inp2_0)));
          absval2_1 = AE_MOVINT16X4_FROMF16X4(AE_ABS16S(AE_MOVF16X4_FROMINT16X4(d_inp2_1)));

          maxval16 = AE_MAX16(maxval16, absval1_0);
          maxval16 = AE_MAX16(maxval16, absval1_1);
          maxval16 = AE_MAX16(maxval16, absval2_0);
          maxval16 = AE_MAX16(maxval16, absval2_1);
        }
        {
          ae_int16x4 temp2;
          ae_int32x2 temp3 = ZERO32, temp4 = ZERO32;
          temp2 = maxval16;

          ae_f32x2 temp3_t, temp4_t;
          AE_CVTI32X4F16(temp3_t, temp4_t, temp2, 0);
          temp3 = AE_MOVINT32X2_FROMF32X2(temp3_t);
          temp4 = AE_MOVINT32X2_FROMF32X2(temp4_t);
          temp4 = AE_MAX32(temp3, temp4);

          temp3 = AE_SEL32_LH(temp4, temp4);
          temp3 = AE_MAX32(temp3, temp4);

          max = AE_MOVAD32_L(temp3);
        }
        
        norm32_t = AE_MOVINT32X2_FROMF32X2(AE_SRAA32RS(AE_MOVF32X2_FROMINT32X2(acc1), out_rshift));
        AE_MINMAX32(norm32_t,SW_MOVDA32(0),SW_MOVDA32(USHRT_MAX));
        recip_dmax = SW_MOVDA32(precip[max]);
        
        table_idx64 = AE_MUL32_HH(norm32_t,recip_dmax);
        table_idx32 = AE_SATU32X2(table_idx64, table_idx64);
        table_idx32_sftd = AE_MOVINT32X2_FROMF32X2(AE_SRAA32RS(AE_MOVF32X2_FROMINT32X2(table_idx32),  (recip_shift - out_rshift)));
        AE_MINMAX32(table_idx32_sftd, SW_MOVDA32(0), SW_MOVDA32(USHRT_MAX));
        
        table_idx64 = AE_MUL32_HH(table_idx32_sftd,recip_dmax);
        table_idx32 = AE_SATU32X2(table_idx64, table_idx64);
        table_idx32_sftd = AE_MOVINT32X2_FROMF32X2(AE_SRAA32RS(AE_MOVF32X2_FROMINT32X2(table_idx32),  (recip_shift - rsqrt_shift)));
        AE_MINMAX32(table_idx32_sftd, SW_MOVDA32(0), SW_MOVDA32(rsqrt_table_len-1));
        
        d_sqrt32 = SW_MOVDA32(prsqrt[AE_MOVAD32_H(table_idx32_sftd)]);
        res64 = AE_MUL32_HH(d_sqrt32, recip_dmax);
        res32 = AE_SATU32X2(res64, res64);
        res32_sftd = AE_MOVINT32X2_FROMF32X2(AE_SRAA32RS(AE_MOVF32X2_FROMINT32X2(res32), recip_shift));
        res16 = AE_SAT16X4(res32_sftd, res32_sftd);
        
        AE_S16_0_IP(res16, ptr_out, 2);
      }
  }

  return 0;
}


WORD32 xa_nn_norm_apply_3D_8_nhwc(
    WORD8 * p_out, 
    const WORD8 * p_inp, /*3D -> iw*ih*ic */
    WORD16 *p_inp_normdata,
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

  /* Pointer Alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_normdata, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_multiplier, sizeof(WORD16), -1);

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
    int ih, iw, ic;
    ae_int16x4 norm_factor = AE_MOVDA16(0-p_inp_normdata[0]);
    
    const signed char * ptr_inp = p_inp;
    ae_int8x8 * ptr_out = (ae_int8x8 *)p_out;
    
    ae_int8x8 d_out;
    ae_int16x4 d_inp1, d_inp2, d_out_mult1, d_out_mult2, tmp1, tmp2, out1, out2;
    ae_int32x2 tmp11, tmp12, tmp21, tmp22;
    ae_int32x2 out11, out12, out21, out22;
    ae_f32x2 shifted_tmp11 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_tmp12 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_tmp21 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_tmp22 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_out11 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_out12 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_out21 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_out22 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    
    ae_valign a_inp;
    ae_valign a_out = AE_ZALIGN64();
    
    ae_int32x2 one_out_shifted = SW_SLAA32S_INT32X2_INT32X2(SW_MOVDA32(-1), (31 - out_rshift));
    ae_int32x2 one_rsqrt_shifted = SW_SLAA32S_INT32X2_INT32X2(SW_MOVDA32(-1), (31 - rsqrt_shift));
    
    if(per_chan_flag == 0)
    {
      d_out_mult1 = AE_MOVDA16(0-p_out_multiplier[0]);
      WORD32 input_size=input_height*input_width*input_channels;
      a_inp = AE_LA64_PP(ptr_inp);
#pragma concurrent
      for(ic = 0; ic < (input_size>>3); ic++)
      {
        AE_LA8X4S_IP(d_inp1, a_inp, ptr_inp);
        AE_LA8X4S_IP(d_inp2, a_inp, ptr_inp);
        
        AE_MUL16X4(tmp11, tmp12, d_inp1, d_out_mult1);
        AE_MUL16X4(tmp21, tmp22, d_inp2, d_out_mult1);
        
        AE_MULF2P32X4RAS(shifted_tmp11, shifted_tmp12, AE_MOVF32X2_FROMINT32X2(tmp11), AE_MOVF32X2_FROMINT32X2(tmp12), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
        AE_MULF2P32X4RAS(shifted_tmp21, shifted_tmp22, AE_MOVF32X2_FROMINT32X2(tmp21), AE_MOVF32X2_FROMINT32X2(tmp22), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
        
        tmp1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp11), AE_MOVINT32X2_FROMF32X2(shifted_tmp12));
        tmp2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp21), AE_MOVINT32X2_FROMF32X2(shifted_tmp22));
        
        AE_MUL16X4(out11, out12, tmp1, norm_factor);
        AE_MUL16X4(out21, out22, tmp2, norm_factor);
        
        AE_MULF2P32X4RAS(shifted_out11, shifted_out12, AE_MOVF32X2_FROMINT32X2(out11), AE_MOVF32X2_FROMINT32X2(out12), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
        AE_MULF2P32X4RAS(shifted_out21, shifted_out22, AE_MOVF32X2_FROMINT32X2(out21), AE_MOVF32X2_FROMINT32X2(out22), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
        
        out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out11), AE_MOVINT32X2_FROMF32X2(shifted_out12));
        out2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out21), AE_MOVINT32X2_FROMF32X2(shifted_out22));
        
        d_out = AE_SAT8X8X16(out1, out2);
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
        
        AE_LAV8X8X2_XP(d_inp, temp, a_inpx2, ptr8x16_inp, rem_size);
        AE_CVTI16X4X2F8(d_inp1_t, d_inp2_t, d_inp, 0);
        
        AE_MUL16X4(tmp11, tmp12, AE_MOVINT16X4_FROMF16X4(d_inp1_t), d_out_mult1);
        AE_MUL16X4(tmp21, tmp22, AE_MOVINT16X4_FROMF16X4(d_inp2_t), d_out_mult1);
        
        AE_MULF2P32X4RAS(shifted_tmp11, shifted_tmp12, AE_MOVF32X2_FROMINT32X2(tmp11), AE_MOVF32X2_FROMINT32X2(tmp12), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
        AE_MULF2P32X4RAS(shifted_tmp21, shifted_tmp22, AE_MOVF32X2_FROMINT32X2(tmp21), AE_MOVF32X2_FROMINT32X2(tmp22), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
        
        tmp1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp11), AE_MOVINT32X2_FROMF32X2(shifted_tmp12));
        tmp2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp21), AE_MOVINT32X2_FROMF32X2(shifted_tmp22));
        
        AE_MUL16X4(out11, out12, tmp1, norm_factor);
        AE_MUL16X4(out21, out22, tmp2, norm_factor);
        
        AE_MULF2P32X4RAS(shifted_out11, shifted_out12, AE_MOVF32X2_FROMINT32X2(out11), AE_MOVF32X2_FROMINT32X2(out12), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
        AE_MULF2P32X4RAS(shifted_out21, shifted_out22, AE_MOVF32X2_FROMINT32X2(out21), AE_MOVF32X2_FROMINT32X2(out22), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
        
        out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out11), AE_MOVINT32X2_FROMF32X2(shifted_out12));
        out2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out21), AE_MOVINT32X2_FROMF32X2(shifted_out22));
        
        d_out = AE_SAT8X8X16(out1, out2);
        AE_SAV8X8X2_XP(d_out, temp, a_outx2, ptr8x16_out, rem_size);
        AE_SA128POS_FP(a_outx2, ptr8x16_out);
      }
      ptr_out = (ae_int8x8*)ptr8x16_out;
      ptr_inp = (const signed char*)ptr8x16_inp;
    }
    else
    {
      for(ih = 0; ih < input_height; ih++)
      {
        for(iw = 0; iw < input_width; iw++)
        {
          ae_int16x8 *ptr_out_mult = (ae_int16x8 *)p_out_multiplier;
          ae_valignx2 a_out_mult = AE_LA128_PP(ptr_out_mult);
          a_inp = AE_LA64_PP(ptr_inp);
#pragma concurrent
          for(ic = 0; ic < (input_channels>>3); ic++)
          {
            AE_LA8X4S_IP(d_inp1, a_inp, ptr_inp);
            AE_LA8X4S_IP(d_inp2, a_inp, ptr_inp);
            AE_LA16X4X2_IP(d_out_mult1, d_out_mult2, a_out_mult, ptr_out_mult);
            
            d_out_mult1 = AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(d_out_mult1)));
            d_out_mult2 = AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(d_out_mult2)));
            
            AE_MUL16X4(tmp11, tmp12, d_inp1, d_out_mult1);
            AE_MUL16X4(tmp21, tmp22, d_inp2, d_out_mult2);
            
            AE_MULF2P32X4RAS(shifted_tmp11, shifted_tmp12, AE_MOVF32X2_FROMINT32X2(tmp11), AE_MOVF32X2_FROMINT32X2(tmp12), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            AE_MULF2P32X4RAS(shifted_tmp21, shifted_tmp22, AE_MOVF32X2_FROMINT32X2(tmp21), AE_MOVF32X2_FROMINT32X2(tmp22), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            
            tmp1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp11), AE_MOVINT32X2_FROMF32X2(shifted_tmp12));
            tmp2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp21), AE_MOVINT32X2_FROMF32X2(shifted_tmp22));
            
            AE_MUL16X4(out11, out12, tmp1, norm_factor);
            AE_MUL16X4(out21, out22, tmp2, norm_factor);
            
            AE_MULF2P32X4RAS(shifted_out11, shifted_out12, AE_MOVF32X2_FROMINT32X2(out11), AE_MOVF32X2_FROMINT32X2(out12), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            AE_MULF2P32X4RAS(shifted_out21, shifted_out22, AE_MOVF32X2_FROMINT32X2(out21), AE_MOVF32X2_FROMINT32X2(out22), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            
            out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out11), AE_MOVINT32X2_FROMF32X2(shifted_out12));
            out2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out21), AE_MOVINT32X2_FROMF32X2(shifted_out22));
            
            d_out = AE_SAT8X8X16(out1, out2);
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
            ae_int8x8 d_inp, temp;
            
            AE_LAV8X8X2_XP(d_inp, temp, a_inpx2, ptr8x16_inp, rem_channels);
            AE_CVTI16X4X2F8(d_inp1_t, d_inp2_t, d_inp, 0);
            AE_LAV16X4X2_XP(d_out_mult1, d_out_mult2, a_out_mult, ptr_out_mult, rem_channels*2);
            
            d_out_mult1 = AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(d_out_mult1)));
            d_out_mult2 = AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(d_out_mult2)));
            
            AE_MUL16X4(tmp11, tmp12, AE_MOVINT16X4_FROMF16X4(d_inp1_t), d_out_mult1);
            AE_MUL16X4(tmp21, tmp22, AE_MOVINT16X4_FROMF16X4(d_inp2_t), d_out_mult2);
            
            AE_MULF2P32X4RAS(shifted_tmp11, shifted_tmp12, AE_MOVF32X2_FROMINT32X2(tmp11), AE_MOVF32X2_FROMINT32X2(tmp12), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            AE_MULF2P32X4RAS(shifted_tmp21, shifted_tmp22, AE_MOVF32X2_FROMINT32X2(tmp21), AE_MOVF32X2_FROMINT32X2(tmp22), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            
            tmp1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp11), AE_MOVINT32X2_FROMF32X2(shifted_tmp12));
            tmp2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp21), AE_MOVINT32X2_FROMF32X2(shifted_tmp22));
            
            AE_MUL16X4(out11, out12, tmp1, norm_factor);
            AE_MUL16X4(out21, out22, tmp2, norm_factor);
            
            AE_MULF2P32X4RAS(shifted_out11, shifted_out12, AE_MOVF32X2_FROMINT32X2(out11), AE_MOVF32X2_FROMINT32X2(out12), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            AE_MULF2P32X4RAS(shifted_out21, shifted_out22, AE_MOVF32X2_FROMINT32X2(out21), AE_MOVF32X2_FROMINT32X2(out22), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            
            out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out11), AE_MOVINT32X2_FROMF32X2(shifted_out12));
            out2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out21), AE_MOVINT32X2_FROMF32X2(shifted_out22));
            
            d_out = AE_SAT8X8X16(out1, out2);
            AE_SAV8X8X2_XP(d_out, temp, a_outx2, ptr8x16_out, rem_channels);
            AE_SA128POS_FP(a_outx2, ptr8x16_out);
          }
          ptr_out = (ae_int8x8*)ptr8x16_out;
          ptr_inp = (const signed char*)ptr8x16_inp;
        }
      }
    
    } 

  } 
  else 
  {
    int ih, iw, ic;
    const ae_int16 * ptr_inp_normdata = (const ae_int16 *)p_inp_normdata;
        
    const signed char * ptr_inp = p_inp;
    ae_int8x8 * ptr_out = (ae_int8x8 *)p_out;
    
    ae_int8x8 d_out;
    ae_int16x4 d_inp1, d_inp2, d_norm_factor, d_out_mult1, d_out_mult2, tmp1, tmp2, out1, out2;
    ae_int32x2 tmp11, tmp12, tmp21, tmp22;
    ae_int32x2 out11, out12, out21, out22;
    ae_f32x2 shifted_tmp11 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_tmp12 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_tmp21 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_tmp22 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_out11 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_out12 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_out21 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    ae_f32x2 shifted_out22 = AE_MOVF32X2_FROMINT32X2(ZERO32);
    
    ae_valign a_inp;
    ae_valign a_out = AE_ZALIGN64();
    
    ae_int32x2 one_out_shifted = SW_SLAA32S_INT32X2_INT32X2(SW_MOVDA32(-1), (31 - out_rshift));
    ae_int32x2 one_rsqrt_shifted = SW_SLAA32S_INT32X2_INT32X2(SW_MOVDA32(-1), (31 - rsqrt_shift));
    
    if(per_chan_flag == 0)
    {
      d_out_mult1 = AE_MOVDA16(0-p_out_multiplier[0]);
      for(ih = 0; ih < input_height; ih++)
      {
        for(iw = 0; iw < input_width; iw++)
        {
          AE_L16_IP(d_norm_factor, ptr_inp_normdata, 2);
          d_norm_factor = AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(d_norm_factor)));
          a_inp = AE_LA64_PP(ptr_inp);
#pragma concurrent          
          for(ic = 0; ic < (input_channels>>3); ic++)
          {
            AE_LA8X4S_IP(d_inp1, a_inp, ptr_inp);
            AE_LA8X4S_IP(d_inp2, a_inp, ptr_inp);
            
            AE_MUL16X4(tmp11, tmp12, d_inp1, d_out_mult1);
            AE_MUL16X4(tmp21, tmp22, d_inp2, d_out_mult1);
            
            AE_MULF2P32X4RAS(shifted_tmp11, shifted_tmp12, AE_MOVF32X2_FROMINT32X2(tmp11), AE_MOVF32X2_FROMINT32X2(tmp12), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            AE_MULF2P32X4RAS(shifted_tmp21, shifted_tmp22, AE_MOVF32X2_FROMINT32X2(tmp21), AE_MOVF32X2_FROMINT32X2(tmp22), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            
            tmp1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp11), AE_MOVINT32X2_FROMF32X2(shifted_tmp12));
            tmp2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp21), AE_MOVINT32X2_FROMF32X2(shifted_tmp22));
            
            AE_MUL16X4(out11, out12, tmp1, d_norm_factor);
            AE_MUL16X4(out21, out22, tmp2, d_norm_factor);
            
            AE_MULF2P32X4RAS(shifted_out11, shifted_out12, AE_MOVF32X2_FROMINT32X2(out11), AE_MOVF32X2_FROMINT32X2(out12), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            AE_MULF2P32X4RAS(shifted_out21, shifted_out22, AE_MOVF32X2_FROMINT32X2(out21), AE_MOVF32X2_FROMINT32X2(out22), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            
            out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out11), AE_MOVINT32X2_FROMF32X2(shifted_out12));
            out2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out21), AE_MOVINT32X2_FROMF32X2(shifted_out22));
            
            d_out = AE_SAT8X8X16(out1, out2);
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
            ae_int8x8 d_inp, temp;
            
            AE_LAV8X8X2_XP(d_inp, temp, a_inpx2, ptr8x16_inp, rem_channels);
            AE_CVTI16X4X2F8(d_inp1_t, d_inp2_t, d_inp, 0);
            
            AE_MUL16X4(tmp11, tmp12, AE_MOVINT16X4_FROMF16X4(d_inp1_t), d_out_mult1);
            AE_MUL16X4(tmp21, tmp22, AE_MOVINT16X4_FROMF16X4(d_inp2_t), d_out_mult1);
            
            AE_MULF2P32X4RAS(shifted_tmp11, shifted_tmp12, AE_MOVF32X2_FROMINT32X2(tmp11), AE_MOVF32X2_FROMINT32X2(tmp12), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            AE_MULF2P32X4RAS(shifted_tmp21, shifted_tmp22, AE_MOVF32X2_FROMINT32X2(tmp21), AE_MOVF32X2_FROMINT32X2(tmp22), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            
            tmp1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp11), AE_MOVINT32X2_FROMF32X2(shifted_tmp12));
            tmp2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp21), AE_MOVINT32X2_FROMF32X2(shifted_tmp22));
            
            AE_MUL16X4(out11, out12, tmp1, d_norm_factor);
            AE_MUL16X4(out21, out22, tmp2, d_norm_factor);
            
            AE_MULF2P32X4RAS(shifted_out11, shifted_out12, AE_MOVF32X2_FROMINT32X2(out11), AE_MOVF32X2_FROMINT32X2(out12), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            AE_MULF2P32X4RAS(shifted_out21, shifted_out22, AE_MOVF32X2_FROMINT32X2(out21), AE_MOVF32X2_FROMINT32X2(out22), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            
            out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out11), AE_MOVINT32X2_FROMF32X2(shifted_out12));
            out2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out21), AE_MOVINT32X2_FROMF32X2(shifted_out22));
            
            d_out = AE_SAT8X8X16(out1, out2);
            AE_SAV8X8X2_XP(d_out, temp, a_outx2, ptr8x16_out, rem_channels);
            AE_SA128POS_FP(a_outx2, ptr8x16_out);
          }
          ptr_out = (ae_int8x8*)ptr8x16_out;
          ptr_inp = (const signed char*)ptr8x16_inp;
        }
      }
    }
    else
    {
      for(ih = 0; ih < input_height; ih++)
      {
        for(iw = 0; iw < input_width; iw++)
        {
          ae_int16x8 *ptr_out_mult = (ae_int16x8 *)p_out_multiplier;
          ae_valignx2 a_out_mult = AE_LA128_PP(ptr_out_mult);
          AE_L16_IP(d_norm_factor, ptr_inp_normdata, 2);
          d_norm_factor = AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(d_norm_factor)));
          a_inp = AE_LA64_PP(ptr_inp);
#pragma concurrent          
          for(ic = 0; ic < (input_channels>>3); ic++)
          {
            AE_LA8X4S_IP(d_inp1, a_inp, ptr_inp);
            AE_LA8X4S_IP(d_inp2, a_inp, ptr_inp);
            AE_LA16X4X2_IP(d_out_mult1, d_out_mult2, a_out_mult, ptr_out_mult);
            
            d_out_mult1 = AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(d_out_mult1)));
            d_out_mult2 = AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(d_out_mult2)));
            
            AE_MUL16X4(tmp11, tmp12, d_inp1, d_out_mult1);
            AE_MUL16X4(tmp21, tmp22, d_inp2, d_out_mult2);
            
            AE_MULF2P32X4RAS(shifted_tmp11, shifted_tmp12, AE_MOVF32X2_FROMINT32X2(tmp11), AE_MOVF32X2_FROMINT32X2(tmp12), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            AE_MULF2P32X4RAS(shifted_tmp21, shifted_tmp22, AE_MOVF32X2_FROMINT32X2(tmp21), AE_MOVF32X2_FROMINT32X2(tmp22), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            
            tmp1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp11), AE_MOVINT32X2_FROMF32X2(shifted_tmp12));
            tmp2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp21), AE_MOVINT32X2_FROMF32X2(shifted_tmp22));
            
            AE_MUL16X4(out11, out12, tmp1, d_norm_factor);
            AE_MUL16X4(out21, out22, tmp2, d_norm_factor);
            
            AE_MULF2P32X4RAS(shifted_out11, shifted_out12, AE_MOVF32X2_FROMINT32X2(out11), AE_MOVF32X2_FROMINT32X2(out12), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            AE_MULF2P32X4RAS(shifted_out21, shifted_out22, AE_MOVF32X2_FROMINT32X2(out21), AE_MOVF32X2_FROMINT32X2(out22), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            
            out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out11), AE_MOVINT32X2_FROMF32X2(shifted_out12));
            out2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out21), AE_MOVINT32X2_FROMF32X2(shifted_out22));
            
            d_out = AE_SAT8X8X16(out1, out2);
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
            ae_int8x8 d_inp, temp;
            
            AE_LAV8X8X2_XP(d_inp, temp, a_inpx2, ptr8x16_inp, rem_channels);
            AE_CVTI16X4X2F8(d_inp1_t, d_inp2_t, d_inp, 0);
            AE_LAV16X4X2_XP(d_out_mult1, d_out_mult2, a_out_mult, ptr_out_mult, rem_channels*2);
            
            d_out_mult1 = AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(d_out_mult1)));
            d_out_mult2 = AE_MOVINT16X4_FROMF16X4(AE_NEG16S(AE_MOVF16X4_FROMINT16X4(d_out_mult2)));
            
            AE_MUL16X4(tmp11, tmp12, AE_MOVINT16X4_FROMF16X4(d_inp1_t), d_out_mult1);
            AE_MUL16X4(tmp21, tmp22, AE_MOVINT16X4_FROMF16X4(d_inp2_t), d_out_mult2);
            
            AE_MULF2P32X4RAS(shifted_tmp11, shifted_tmp12, AE_MOVF32X2_FROMINT32X2(tmp11), AE_MOVF32X2_FROMINT32X2(tmp12), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            AE_MULF2P32X4RAS(shifted_tmp21, shifted_tmp22, AE_MOVF32X2_FROMINT32X2(tmp21), AE_MOVF32X2_FROMINT32X2(tmp22), AE_MOVF32X2_FROMINT32X2(one_out_shifted), AE_MOVF32X2_FROMINT32X2(one_out_shifted));
            
            tmp1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp11), AE_MOVINT32X2_FROMF32X2(shifted_tmp12));
            tmp2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_tmp21), AE_MOVINT32X2_FROMF32X2(shifted_tmp22));
            
            AE_MUL16X4(out11, out12, tmp1, d_norm_factor);
            AE_MUL16X4(out21, out22, tmp2, d_norm_factor);
            
            AE_MULF2P32X4RAS(shifted_out11, shifted_out12, AE_MOVF32X2_FROMINT32X2(out11), AE_MOVF32X2_FROMINT32X2(out12), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            AE_MULF2P32X4RAS(shifted_out21, shifted_out22, AE_MOVF32X2_FROMINT32X2(out21), AE_MOVF32X2_FROMINT32X2(out22), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted), AE_MOVF32X2_FROMINT32X2(one_rsqrt_shifted));
            
            out1 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out11), AE_MOVINT32X2_FROMF32X2(shifted_out12));
            out2 = AE_SAT16X4(AE_MOVINT32X2_FROMF32X2(shifted_out21), AE_MOVINT32X2_FROMF32X2(shifted_out22));
            
            d_out = AE_SAT8X8X16(out1, out2);
            AE_SAV8X8X2_XP(d_out, temp, a_outx2, ptr8x16_out, rem_channels);
            AE_SA128POS_FP(a_outx2, ptr8x16_out);
          }
          ptr_out = (ae_int8x8*)ptr8x16_out;
          ptr_inp = (const signed char*)ptr8x16_inp;
        }
      }
    }     
  }

  return 0;
}

