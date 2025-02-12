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
#include "xa_nn_common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_hifi_isa_compat.h"
#define AE_SW_LAV32X2X2_XP(out0, out1, align_out, p_out, off) \
{ \
  ae_int16x4 d_out16_0, d_out16_1; \
  AE_LAV16X4X2_XP(d_out16_0, d_out16_1, align_out, (ae_int16x8 *)p_out, off); \
  d_out16_0 = AE_SEL16_2301(d_out16_0, d_out16_0); \
  d_out16_1 = AE_SEL16_2301(d_out16_1, d_out16_1); \
  out0 = AE_MOVINT32X2_FROMINT16X4(d_out16_0); \
  out1 = AE_MOVINT32X2_FROMINT16X4(d_out16_1); \
}

 

#define AE_SW_SAV32X2X2_XP(in0, in1, align_in, p_out, off) \
{ \
  ae_int32x2 d_in32_0, d_in32_1; \
  ae_int16x4 d_in16_0, d_in16_1; \
  d_in16_0 = AE_MOVINT16X4_FROMINT32X2(in0); \
  d_in16_1 = AE_MOVINT16X4_FROMINT32X2(in1); \
  d_in16_0 = AE_SEL16_2301(d_in16_0, d_in16_0); \
  d_in16_1 = AE_SEL16_2301(d_in16_1, d_in16_1); \
  AE_SAV16X4X2_XP(d_in16_0, d_in16_1, align_in, (ae_int16x8 *)p_out, off); \
}

WORD32 xa_nn_batch_norm_3D_8_8(WORD8 * __restrict__ p_out,
                               const WORD8 * __restrict__ p_inp,
                               const WORD16 * __restrict__ p_alpha,
                               const WORD32 * __restrict__ p_beta,
                               WORD32 io_height,
                               WORD32 io_width,
                               WORD32 io_depth,
                               WORD32 out_shift,
                               WORD32 out_activation_min,
                               WORD32 out_activation_max,
                               WORD32 inp_data_format,
                               WORD32 out_data_format)
{	
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_alpha, -1);
  XA_NNLIB_ARG_CHK_PTR(p_beta, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_alpha, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_beta, sizeof(WORD32), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((io_height <= 0 || io_width <= 0 || io_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(((out_shift < -31) || (out_shift > 0)), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_min < -128 || out_activation_min > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min || out_activation_min > 127), -1);
  XA_NNLIB_ARG_CHK_COND((inp_data_format != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

  ae_int16x4 out_act_max = out_activation_max, out_act_min = out_activation_min;
  WORD32 i, k;
  if(io_depth <= 4)
  {
    ae_int16x8 *alpha_ptr = (ae_int16x8 *) p_alpha;
    ae_int32x4 *beta_ptr = (ae_int32x4 *) p_beta;
    
    ae_valignx2 alpha_aligner, beta_aligner, inp_aligner;
    alpha_aligner = AE_LA128_PP(alpha_ptr);
    beta_aligner = AE_LA128_PP(beta_ptr);
     
    ae_int16x4 alpha_val, extra_val16, inp_val16;
    ae_int32x2 beta_val1, beta_val2=0;
    AE_LAV16X4X2_XP(alpha_val, extra_val16, alpha_aligner, alpha_ptr, io_depth*2);
    AE_SW_LAV32X2X2_XP(beta_val1, beta_val2, beta_aligner, beta_ptr, io_depth*4);

    ae_int8x16 *inp_ptr = (ae_int8x16 *)p_inp;
    inp_aligner = AE_LA128_PP(inp_ptr);
    ae_int8x8 inp_val, extra_val8;
    ae_int8x16 *out_ptr = (ae_int8x16 *)p_out;
    ae_valignx2 out_aligner = AE_ZALIGN128();
    ae_int8x8 out_val8;
    ae_int16x4 out_val16;

    for(i = 0; i < io_height * io_width; i++)
    {
      ae_int32x2 out1, out2;
      AE_MOVD32X4(out1, out2, beta_val1, beta_val2);
      AE_LAV8X8X2_XP(inp_val, extra_val8, inp_aligner, inp_ptr, io_depth);
     
      AE_CVTA16X4X2F8(inp_val16, extra_val16, inp_val, 0);
      AE_MULA16X4S(out1, out2, inp_val16, alpha_val);

      out1 = AE_SRAA32RS(out1, -out_shift);
      out2 = AE_SRAA32RS(out2, -out_shift);
      
      out_val16=AE_SAT16X4(out1, out2);
      AE_MINMAX16(out_val16, out_act_min, out_act_max);
    
      out_val8 = AE_SAT8X8X16(out_val16, extra_val16);
      AE_SAV8X8X2_XP(out_val8, extra_val8, out_aligner, out_ptr, io_depth); 
    }

    AE_SA128POS_FP(out_aligner, (void *)out_ptr);
  }
  else if(io_depth%4==0 && (((unsigned)p_inp) & 3) == 0 && (((unsigned)p_alpha) & 7) == 0 && (((unsigned)p_beta) & 15) == 0 && (((unsigned)p_out) & 3) == 0)
  {
    const WORD8 *inp_ptr = p_inp;
    ae_int32 *out_ptr = (ae_int32 *) p_out;
    for(i = 0; i < io_height * io_width; i++)
    {
      ae_int16x4 *alpha_ptr = (ae_int16x4 *) p_alpha;
      ae_int32x4 *beta_ptr = (ae_int32x4 *)p_beta;
      for(k = 0; k < io_depth>>2; k++)
      {
        ae_int16x4 inp_val, out_val, alpha_val;
        ae_int32x2 out1,out2;
        
        //Load input,alpha and beta
        AE_L8X4S_IP(inp_val, inp_ptr, 4);
        AE_L16X4_IP(alpha_val, alpha_ptr, 8);
        AE_L32X2X2_IP(out1,out2, beta_ptr, 16);

        //out=input*alpha+beta
        AE_MULA16X4S(out1, out2, inp_val, alpha_val);

        //shift right
        out1 = AE_SRAA32RS(out1, -out_shift);
        out2 = AE_SRAA32RS(out2, -out_shift);

        //ranging and store output
        out_val=AE_SAT16X4(out1, out2);
        AE_MINMAX16(out_val, out_act_min, out_act_max);
        STORE_16x4_8x4(out_val, out_ptr, 4);
      }
    }
  }
  else{
    ae_valignx2 alpha_aligner, beta_aligner, inp_aligner;
    i = 0;
    if(io_depth >=56)
    {
      ae_valign inp_align,out_align;
      for(i = 0; i < io_height * io_width; i++)
      {
        //initialize pointers
        ae_int16x8 *alpha_ptr = (ae_int16x8 *) p_alpha;
        ae_int32x4 *beta_ptr = (ae_int32x4 *)p_beta;
        const WORD8 *inp_ptr = p_inp + i * io_depth;
        ae_int8x8 *out_ptr = (ae_int8x8 *)(p_out + i * io_depth);

        //initialize valign variables
        alpha_aligner=AE_LA128_PP(p_alpha);
        beta_aligner=AE_LA128_PP(p_beta);
        inp_align=AE_LA64_PP(inp_ptr);
        out_align=AE_ZALIGN64();

        for(k = 0; k < (io_depth>>3); k++)
        {
          ae_int8x8 out_res;
          ae_int16x4 inp_val1, inp_val2, out_val1, out_val2, alpha_val1, alpha_val2;
          ae_int32x2 out11, out12, out21, out22;

          //Load input,alpha and beta
          AE_LA8X4S_IP(inp_val1, inp_align, inp_ptr);
          AE_LA8X4S_IP(inp_val2, inp_align, inp_ptr);
          AE_LA16X4X2_IP(alpha_val1, alpha_val2, alpha_aligner, alpha_ptr);
          AE_LA32X2X2_IP(out11, out12, beta_aligner, beta_ptr);
          AE_LA32X2X2_IP(out21, out22, beta_aligner, beta_ptr);


          //out=input*alpha+beta
          AE_MULA16X4S(out11, out12, inp_val1, alpha_val1);
          AE_MULA16X4S(out21, out22, inp_val2, alpha_val2);

          //shift right
          out11 = AE_SRAA32RS(out11, -out_shift);
          out12 = AE_SRAA32RS(out12, -out_shift);
          out21 = AE_SRAA32RS(out21, -out_shift);
          out22 = AE_SRAA32RS(out22, -out_shift);

          //ranging ang store output
          out_val1 = AE_SAT16X4(out11, out12);
          out_val2 = AE_SAT16X4(out21, out22);

          AE_MINMAX16(out_val1, out_act_min, out_act_max);
          AE_MINMAX16(out_val2, out_act_min, out_act_max);

          out_res = AE_SAT8X8X16(out_val1,out_val2);
          AE_SA8X8_IP(out_res, out_align, out_ptr);
          AE_SA64POS_FP(out_align, (void *)out_ptr);
        }
      }
      i = (io_depth & (~7));
    }
    ae_int16x8 *alpha_ptr = (ae_int16x8 *) (p_alpha + i);
    ae_int32x4 *beta_ptr = (ae_int32x4 *) (p_beta + i);
    
    
    alpha_aligner = AE_LA128_PP(alpha_ptr);
    beta_aligner = AE_LA128_PP(beta_ptr);

    
    for(; i<io_depth; i+=8)
    {
      int count=((io_depth-i>=8)?8:io_depth-i);
      ae_int16x4 alpha_val1, alpha_val2, inp1_val16, inp2_val16;
      ae_int32x2 beta_val11, beta_val12, beta_val21, beta_val22=0;
      AE_LAV16X4X2_XP(alpha_val1, alpha_val2, alpha_aligner, alpha_ptr, (count*2));
      AE_SW_LAV32X2X2_XP(beta_val11, beta_val12, beta_aligner, beta_ptr, (count>4 ? 16 : count*4));
      if(count>4){
        AE_SW_LAV32X2X2_XP(beta_val21, beta_val22, beta_aligner, beta_ptr, count*4-16);
      }
      for(k = 0; k < io_height * io_width; k++)
      {
        
        ae_int8x16 *inp_ptr = (ae_int8x16 *)(p_inp + k*io_depth + i);
        inp_aligner = AE_LA128_PP(inp_ptr);
        ae_int8x8 inp_val, extra_val8;
        
        ae_int8x16 *out_ptr = (ae_int8x16 *)(p_out + k*io_depth +i);
        ae_valignx2 out_aligner = AE_ZALIGN128();
        ae_int8x8 out_val8;
        ae_int16x4 out1_val16, out2_val16;
        ae_int32x2 out11, out12, out21, out22;

        AE_MOVD32X4(out11, out12, beta_val11, beta_val12);
        AE_MOVD32X4(out21, out22, beta_val21, beta_val22);
        AE_LAV8X8X2_XP(inp_val, extra_val8, inp_aligner, inp_ptr, count);

        AE_CVTA16X4X2F8(inp1_val16, inp2_val16, inp_val, 0);
        AE_MULA16X4S(out11, out12, inp1_val16, alpha_val1);
        AE_MULA16X4S(out21, out22, inp2_val16, alpha_val2);

        out11 = AE_SRAA32RS(out11, -out_shift);
        out12 = AE_SRAA32RS(out12, -out_shift);
    
        out21 = AE_SRAA32RS(out21, -out_shift);
        out22 = AE_SRAA32RS(out22, -out_shift);      

        out1_val16=AE_SAT16X4(out11, out12);
        out2_val16=AE_SAT16X4(out21, out22);

        AE_MINMAX16(out1_val16, out_act_min, out_act_max);
        AE_MINMAX16(out2_val16, out_act_min, out_act_max);

        out_val8 = AE_SAT8X8X16(out1_val16, out2_val16);
        AE_SAV8X8X2_XP(out_val8, extra_val8, out_aligner, out_ptr, count);
        AE_SA128POS_FP(out_aligner, (void *)out_ptr);
      }
    }
    
  }
  //printf("complete\n");
  return 0;
}


