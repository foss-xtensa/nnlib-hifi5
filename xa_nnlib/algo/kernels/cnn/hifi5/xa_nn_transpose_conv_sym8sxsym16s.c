/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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
#include "common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nn_conv2d_std_state.h"

WORD32 xa_nn_transpose_conv_getsize
(
  WORD32 input_height 
 ,WORD32 input_width 
 ,WORD32 input_channels
 ,WORD32 kernel_height 
 ,WORD32 kernel_width 
 ,WORD32 x_stride
 ,WORD32 y_stride
 ,WORD32 output_height
 ,WORD32 output_width
 ,WORD32 output_channels
 ,WORD32 kernel_precision
 ,WORD32 output_precision
 )
{
    XA_NNLIB_CHK_COND((kernel_height <= 0), -1);
    XA_NNLIB_CHK_COND((kernel_width <= 0), -1);
    XA_NNLIB_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_CHK_COND((x_stride <= 0), -1);
    XA_NNLIB_CHK_COND((y_stride <= 0), -1);
    XA_NNLIB_CHK_COND((output_height <= 0), -1);
    XA_NNLIB_CHK_COND((output_width <= 0), -1);
    XA_NNLIB_CHK_COND((output_channels <= 0), -1);

    WORD32 scratch_bytewidth = 0;
    WORD32 input_size;
    WORD32 kernel_size;
    WORD32 total_size = 0;

    xtbool unit_stride = (x_stride == 1 && y_stride == 1);
    xtbool filt_grt_inp = (kernel_width > input_width || kernel_height > input_height);

    switch (output_precision)
    {
        case -8: /* For sym16s */
            input_size = sizeof(WORD16);
            scratch_bytewidth = 8; /* 64b scratch */
            break;
        default:
            return -1; /* Returning due to invalid input */
            break;
    }

    switch (kernel_precision)
    {
        case -5: /* For sym8s */
            kernel_size = sizeof(WORD8);
            break;
        default:
            return -1; /* Returning due to invalid prec */
            break;
    }

    if(unit_stride && !filt_grt_inp)
    {
      total_size += ALIGNED_SIZE(sizeof(xa_nn_conv_state_t), ALIGNMENT_16);
      WORD32 kernel_bytes = PADDED_SIZE(kernel_height * kernel_width * input_channels * output_channels * kernel_size, ALIGNMENT_16);
      WORD32 cir_buf_size_bytes = (2*(kernel_height-1) + input_height) * kernel_width * input_channels * input_size;
      while(cir_buf_size_bytes%16 !=0)
      {
          cir_buf_size_bytes+= kernel_width*input_channels*input_size;
      }
      total_size += kernel_bytes + cir_buf_size_bytes; 
      total_size += BUS_WIDTH;
      total_size = PADDED_SIZE(total_size, ALIGNMENT_16);
    }
    else
    {
      total_size = (output_height) * (output_width) * (output_channels) * (scratch_bytewidth);
    }
    return total_size;
}

#define MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(out0, inp0, inp1, mult01, l_shift01) \
{ \
  ae_int32x2 d_red_mult01 = AE_SEXT32X2D16_10(AE_ROUND16X4F32SASYM(mult01, mult01)); \
  ae_int32x2 d_red_mult01_l16 = AE_CVT32X2F16_10(AE_ROUND16X4F32SASYM(mult01, mult01)); \
  ae_int32x2 d_inp01_h = AE_ROUND32X2F64SASYM(inp0, inp1); \
  ae_int64 q0_l, q1_l; \
  AE_MUL32X2S_HH_LL(q0_l, q1_l, d_red_mult01, AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(inp0), AE_MOVINT32X2_FROMINT64(inp1))); \
  AE_MULAFP32X2S_HH_LL(q0_l, q1_l, d_red_mult01_l16, AE_SLAI32(d_inp01_h, 15)); \
  q0_l = AE_SLAA64(q0_l, (AE_MOVAD32_H(l_shift01)+17)); \
  q1_l = AE_SLAA64(q1_l, (AE_MOVAD32_L(l_shift01)+17)); \
  out0 = AE_ROUND32X2F64SASYM(q0_l, q1_l); \
}

#define MPY_BY_QUANT_MULT_ACC64_X2_OUT32(out0, inp0, inp1, mult, l_shift) \
{ \
  ae_int32x2 d_red_mult = AE_SEXT32X2D16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult))); \
  ae_int32x2 d_red_mult_l16 = AE_CVT32X2F16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult)));  \
  ae_int32x2 d_inp01_h = AE_ROUND32X2F64SASYM(inp0, inp1); \
  ae_int64 q0_l, q1_l; \
  AE_MUL32X2S_HH_LL(q0_l, q1_l, d_red_mult, AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(inp0), AE_MOVINT32X2_FROMINT64(inp1))); \
  AE_MULAFP32X2S_HH_LL(q0_l, q1_l, d_red_mult_l16, AE_SLAI32(d_inp01_h, 15)); \
  q0_l = AE_SLAA64(q0_l, (l_shift + 17)); \
  q1_l = AE_SLAA64(q1_l, (l_shift + 17)); \
  out0 = AE_ROUND32X2F64SASYM(q0_l, q1_l); \
}

#define MPY_BY_QUANT_MULT_ACC64_OUT32(out0, inp0, mult, l_shift) \
{ \
  ae_int32x2 d_red_mult = AE_SEXT32X2D16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult))); \
  ae_int32x2 d_red_mult_l16 = AE_CVT32X2F16_10(AE_ROUND16X4F32SASYM(AE_MOVDA32(mult), AE_MOVDA32(mult)));  \
  ae_int32x2 d_inp0_h = AE_ROUND32F64SASYM(inp0); \
  ae_int64 q0_l; \
  q0_l = AE_MUL32S_HH(d_red_mult, AE_SEL32_LL(AE_MOVINT32X2_FROMINT64(inp0), AE_MOVINT32X2_FROMINT64(inp0))); \
  AE_MULAF32S_HH(q0_l, d_red_mult_l16, AE_SLAI32(d_inp0_h, 15)); \
  q0_l = AE_SLAA64(q0_l, (l_shift + 17)); \
  out0 = AE_ROUND32F64SASYM(q0_l); \
}

static inline void tconv2d_sym8sxsym16s(WORD16* output_data,
		const WORD16* input_data,
		const WORD8* filter_data,
		const WORD64* bias_data,
		int stride_width, int stride_height,
		int pad_width, int pad_height,
		int input_depth, int output_depth,
		int input_height, int input_width,
		int filter_height, int filter_width,
		int output_height, int output_width,
		int num_elements,
		int *output_shift, int *output_multiplier,
		int64_t* scratch_buffer)
{
	ae_int64 *pscratch = (ae_int64*)scratch_buffer;
	ae_int64 dzero = AE_ZERO64();
	for(int i=0; i<num_elements; i++)
		AE_S64_IP(dzero, pscratch, 8);

	int stride1 = filter_height*filter_width*input_depth;
	WORD16 *pinp;

	/*
	 * SEANet: special case for input_depth multiple of 16
	 */
  if(input_data && filter_data && output_data && scratch_buffer &&
			(((unsigned int)input_data&0xF)==0) && (((unsigned int)filter_data&0xF)==0) && (((unsigned int)output_data&0x7) == 0) &&
			(((unsigned int)scratch_buffer&0xF) == 0) && ((input_depth&0xF)==0) && ((output_depth&0x1)==0))
	{
		{
			//tbd : batch = 1, need to handle other values and in_x_min/max= 0 .. need toc heck for other values
			for (int in_y = 0; in_y < input_height; ++in_y)
			{
				for (int in_x = 0; in_x < input_width; ++in_x)
				{
					const int out_x_orig = in_x*stride_width - pad_width;
					const int out_y_orig = in_y*stride_height - pad_height;
					int filt_x_min = -out_x_orig; 
					int filt_x_max = output_width - out_x_orig; 
					int filt_y_min = -out_y_orig; 
					int filt_y_max = output_height - out_y_orig; 
					filt_x_min = (filt_x_min < filter_width) ? filt_x_min : filter_width;
					filt_x_min = (filt_x_min < 0) ? 0 : filt_x_min;
					filt_x_max = (filt_x_max < filter_width) ? filt_x_max : filter_width;
					filt_x_max = (filt_x_max < 0) ? 0 : filt_x_max;
					filt_y_min = (filt_y_min < filter_height) ? filt_y_min : filter_height;
					filt_y_min = (filt_y_min < 0) ? 0 : filt_y_min;
					filt_y_max = (filt_y_max < filter_height) ? filt_y_max : filter_height;
					filt_y_max = (filt_y_max < 0) ? 0 : filt_y_max;
					pinp =  (WORD16*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
					for (int in_channel = 0; in_channel < input_depth; in_channel+=16)
					{
						ae_int16x4 d_inp0, d_inp1, d_inp2, d_inp3;
						AE_L16X4X2_IP(d_inp0, d_inp1, (ae_int16x8*)pinp, 2*sizeof(WORD64));
						AE_L16X4X2_IP(d_inp2, d_inp3, (ae_int16x8*)pinp, 2*sizeof(WORD64));

            for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
						{
              for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
							{
								// Compute output element location.
								int out_x = out_x_orig + filter_x;//out_x_origin + filter_x;
								int out_y = out_y_orig + filter_y;//out_y_origin + filter_y;
								ae_int64 *pscratch_src = (ae_int64*)&scratch_buffer[out_y*output_width*output_depth+out_x*output_depth];
                ae_int64 *pscratch_dst = pscratch_src;
                ae_int64 d_scr0, d_scr1, d_scr2, d_scr3;
								WORD8* pfilt = (WORD8*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + in_channel];
								ae_int8x8 d_fil0, d_fil1, d_fil2, d_fil3;
								ae_int8x8 d_fil4, d_fil5, d_fil6, d_fil7;
								AE_L8X8X2_XP(d_fil0, d_fil1, (ae_int8x16 *)pfilt, stride1);
								AE_L8X8X2_XP(d_fil2, d_fil3, (ae_int8x16 *)pfilt, stride1);
								AE_L8X8X2_XP(d_fil4, d_fil5, (ae_int8x16 *)pfilt, stride1);
								AE_L8X8X2_XP(d_fil6, d_fil7, (ae_int8x16 *)pfilt, stride1);
                
                int loop_cnt = (output_depth & (~3));
								for (int out_channel = 0; out_channel < loop_cnt; out_channel+=4)
								{
									AE_L64X2_I(d_scr2, d_scr3, (ae_int64x2 *)pscratch_src, 16);
									AE_L64X2_IP(d_scr0, d_scr1, (ae_int64x2 *)pscratch_src, 32);
                  AE_MULA8QW8X16(d_scr0, d_scr1, d_scr2, d_scr3, d_fil0, d_fil2, d_fil4, d_fil6, d_inp0, d_inp1);
                  AE_MULA8QW8X16(d_scr0, d_scr1, d_scr2, d_scr3, d_fil1, d_fil3, d_fil5, d_fil7, d_inp2, d_inp3);
                  AE_L8X8X2_XP(d_fil0, d_fil1, (ae_int8x16 *)pfilt, stride1);
                  AE_L8X8X2_XP(d_fil2, d_fil3, (ae_int8x16 *)pfilt, stride1);
                  AE_L8X8X2_XP(d_fil4, d_fil5, (ae_int8x16 *)pfilt, stride1);
                  AE_L8X8X2_XP(d_fil6, d_fil7, (ae_int8x16 *)pfilt, stride1);
									AE_S64X2_I(d_scr2, d_scr3, (ae_int64x2 *)pscratch_dst, 16);
									AE_S64X2_IP(d_scr0, d_scr1, (ae_int64x2 *)pscratch_dst, 32);
								}
                // tail loop
                if (output_depth & 3)
								{
									ae_int64 d_tmp0 = AE_ZERO64();
									ae_int64 d_tmp1 = AE_ZERO64();
                  AE_L64X2_I(d_scr0, d_scr1, (ae_int64x2 *)pscratch_src, 0);
                  AE_MULA8QW8X16(d_scr0, d_scr1, d_tmp0, d_tmp1, d_fil0, d_fil2, d_fil4, d_fil6, d_inp0, d_inp1);
                  AE_MULA8QW8X16(d_scr0, d_scr1, d_tmp0, d_tmp1, d_fil1, d_fil3, d_fil5, d_fil7, d_inp2, d_inp3);
									AE_S64X2_I(d_scr0, d_scr1, (ae_int64x2 *)pscratch_src, 0);
								}
							}
						}
					}
				}
			}
		}
	}
	else if(input_data && filter_data && output_data && scratch_buffer &&
			(((unsigned int)input_data&0x7)==0) && (((unsigned int)filter_data&0x3)==0) && (((unsigned int)output_data&0x7) == 0) && 
      (((unsigned int)scratch_buffer&0x7) == 0) && ((input_depth&0x3)==0) && ((output_depth&0x1)==0))
	{
		{
			//tbd : batch = 1, need to handle other values and in_x_min/max= 0 .. need to check for other values
			for (int in_y = 0; in_y < input_height; ++in_y)
			{
				for (int in_x = 0; in_x < input_width; ++in_x)
				{
					const int out_x_orig = in_x*stride_width - pad_width;
					const int out_y_orig = in_y*stride_height - pad_height;
					int filt_x_min = -out_x_orig; 
					int filt_x_max = output_width - out_x_orig; 
					int filt_y_min = -out_y_orig; 
					int filt_y_max = output_height - out_y_orig; 
					filt_x_min = (filt_x_min < filter_width) ? filt_x_min : filter_width;
					filt_x_min = (filt_x_min < 0) ? 0 : filt_x_min;
					filt_x_max = (filt_x_max < filter_width) ? filt_x_max : filter_width;
					filt_x_max = (filt_x_max < 0) ? 0 : filt_x_max;
					filt_y_min = (filt_y_min < filter_height) ? filt_y_min : filter_height;
					filt_y_min = (filt_y_min < 0) ? 0 : filt_y_min;
					filt_y_max = (filt_y_max < filter_height) ? filt_y_max : filter_height;
					filt_y_max = (filt_y_max < 0) ? 0 : filt_y_max;
					pinp =  (WORD16*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
					for (int in_channel = 0; in_channel < input_depth; in_channel+=4)
					{
						ae_int16x4 d_inp;
						AE_L16X4_IP(d_inp, (ae_int16x4*)pinp, sizeof(WORD64));

						for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
						{
							for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
							{
								// Compute output element location.
								int out_x = out_x_orig + filter_x;//out_x_origin + filter_x;
								int out_y = out_y_orig + filter_y;//out_y_origin + filter_y;
								ae_int64 *pscratch_src = (ae_int64*)&scratch_buffer[out_y*output_width*output_depth+out_x*output_depth];
								ae_int64 *pscratch_dst = pscratch_src;
								ae_int64 d_scr0, d_scr1;
								WORD8* pfilt0 = (WORD8*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + in_channel];
								WORD8* pfilt1 = pfilt0 + stride1;
                ae_int16x4 d_fil0, d_fil1;
								AE_L8X4S_XP(d_fil0, pfilt0, 2*stride1);
								AE_L8X4S_XP(d_fil1, pfilt1, 2*stride1);
#pragma concurrent
								for (int out_channel = 0; out_channel < (output_depth >> 1); ++out_channel)
								{
									AE_L64X2_IP(d_scr0, d_scr1, (ae_int64x2 *)pscratch_src, 16);
									AE_MULAAAA2Q16(d_scr0, d_scr1, d_inp, d_inp, d_fil0, d_fil1);
                  AE_L8X4S_XP(d_fil0, pfilt0, 2*stride1);
                  AE_L8X4S_XP(d_fil1, pfilt1, 2*stride1);
									AE_S64X2_IP(d_scr0, d_scr1, (ae_int64x2 *)pscratch_dst, 16);
								}
							}
						}
					}
				}
			}
		}
	}
	else
	{
		{
			for (int in_y = 0; in_y < input_height; ++in_y)
			{
				for (int in_x = 0; in_x < input_width; ++in_x)
				{
          const int out_x_origin = (in_x * stride_width) - pad_width;
          const int out_y_origin = (in_y * stride_height) - pad_height;
          int filt_x_min = -out_x_origin; 
          int filt_x_max = output_width - out_x_origin; 
          int filt_y_min = -out_y_origin; 
          int filt_y_max = output_height - out_y_origin; 
          filt_x_min = (filt_x_min < filter_width) ? filt_x_min : filter_width;
          filt_x_min = (filt_x_min < 0) ? 0 : filt_x_min;
          filt_x_max = (filt_x_max < filter_width) ? filt_x_max : filter_width;
          filt_x_max = (filt_x_max < 0) ? 0 : filt_x_max;
          filt_y_min = (filt_y_min < filter_height) ? filt_y_min : filter_height;
          filt_y_min = (filt_y_min < 0) ? 0 : filt_y_min;
          filt_y_max = (filt_y_max < filter_height) ? filt_y_max : filter_height;
          filt_y_max = (filt_y_max < 0) ? 0 : filt_y_max;
					pinp =  (WORD16*)&input_data[in_y*input_width*input_depth+in_x*input_depth];
					for (int in_channel = 0; in_channel < input_depth; in_channel+=8)
					{
            ae_valignx2 align_pinp = AE_LA128_PP(pinp);

						ae_int16x4 d_inp0, d_inp1;
            int offset = XT_MIN(input_depth - in_channel, 8) << 1;
						AE_LAV16X4X2_XP(d_inp0, d_inp1, align_pinp, (ae_int16x8*)pinp, offset);

						for (int filter_y = filt_y_min; filter_y < filt_y_max; ++filter_y)
						{
							for (int filter_x = filt_x_min; filter_x < filt_x_max; ++filter_x)
							{
								const int out_x = out_x_origin + filter_x;
								const int out_y = out_y_origin + filter_y;
								ae_int64 *pscratch_src = (ae_int64*)&scratch_buffer[out_y*output_width*output_depth+out_x*output_depth];
                ae_int64 d_scr0;

								WORD8* pfilt = (WORD8*)&filter_data[filter_y*filter_width*input_depth + filter_x*input_depth + in_channel];
                ae_valignx2 align_pfilt = AE_LA128_PP(pfilt);

								ae_int8x8 d_fil0, d_fil1;
                int offset_8 = offset >> 1;
								AE_LAV8X8X2_XP(d_fil0, d_fil1, align_pfilt, (ae_int8x16 *)pfilt, offset_8);
                pfilt = pfilt + stride1 - offset_8; 

                for (int out_channel = 0; out_channel < output_depth; ++out_channel)
                {
									d_scr0 = AE_L64_I(pscratch_src, 0);
                  ae_int64 d_tmp0 = AE_ZERO64();
                  AE_MULAAAA2Q16X8 (d_scr0, d_tmp0, d_inp0, d_inp1, d_fil0);
                  d_scr0 = AE_ADD64S(d_scr0, d_tmp0);
                  ae_valignx2 align_pfilt = AE_LA128_PP(pfilt);
                  AE_LAV8X8X2_XP(d_fil0, d_fil1, align_pfilt, (ae_int8x16 *)pfilt, offset_8);
                  pfilt = pfilt + stride1 - offset_8; 
									AE_S64_IP(d_scr0, pscratch_src, sizeof(WORD64));
                }
							}
						}
					}
				}
			}
		}
	}

  // Add bias and store output
  ae_int64 acc0, acc1, acc2, acc3; 
  ae_int64 dbias0, dbias1;
  ae_int64 *pbias;
  ae_int64 zero_bias = AE_ZERO64();
  int bias_offset;
  if (bias_data)
  {
    pbias = (ae_int64 *)bias_data;
    bias_offset = sizeof(WORD64);
  }
  else
  {
    pbias = &zero_bias;
    bias_offset = 0;
  }

  int out_channel = 0;

  int loop_cnt = (output_depth & 1) ? 0 : output_depth;
  for (out_channel = 0; out_channel < loop_cnt; out_channel+=2)
  {
    int shift0 = output_shift[out_channel];
    int shift1 = output_shift[out_channel+1];
    AE_L64_XP(dbias0, pbias, bias_offset);
    AE_L64_XP(dbias1, pbias, bias_offset);
    
    ae_int64 *pscratch0 = (ae_int64*)&scratch_buffer[out_channel];
    ae_int64 *pscratch1 = pscratch0 + output_depth; 
    ae_int16 *pout0 = (ae_int16*)&output_data[out_channel];
    ae_int16 *pout1 = pout0 + output_depth;
   
    AE_L64X2_XP(acc0, acc2, (ae_int64x2 *)pscratch0, 2*output_depth*sizeof(WORD64));
    AE_L64X2_XP(acc1, acc3, (ae_int64x2 *)pscratch1, 2*output_depth*sizeof(WORD64));
    for (int i = 0; i < ((output_height*output_width)>>1); i++)
    {
      acc0 = AE_ADD64(acc0, dbias0);
      acc1 = AE_ADD64(acc1, dbias0);
      acc2 = AE_ADD64(acc2, dbias1);
      acc3 = AE_ADD64(acc3, dbias1);
      ae_int32x2 scaled_acc0, scaled_acc1;
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(scaled_acc0, acc0, acc1, output_multiplier[out_channel], shift0);
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(scaled_acc1, acc2, acc3, output_multiplier[out_channel+1], shift1);
      ae_int16x4 d1 = AE_SAT16X4(scaled_acc1, scaled_acc0); 
      AE_L64X2_XP(acc0, acc2, (ae_int64x2 *)pscratch0, 2*output_depth*sizeof(WORD64));
      AE_L64X2_XP(acc1, acc3, (ae_int64x2 *)pscratch1, 2*output_depth*sizeof(WORD64));
      AE_S32_H_XP(AE_MOVINT32X2_FROMINT16X4(AE_SEL16_7531(d1, d1)), (ae_int32 *)pout0, 2*output_depth*sizeof(WORD16));
      AE_S32_H_XP(AE_MOVINT32X2_FROMINT16X4(AE_SEL16_6420(d1, d1)), (ae_int32 *)pout1, 2*output_depth*sizeof(WORD16));
    }
    if((output_height*output_width) & 1)
    {
      acc0 = AE_ADD64(acc0, dbias0);
      acc2 = AE_ADD64(acc2, dbias1);
      ae_int32x2 scaled_acc0;
      ae_int32x2 out_mul10 = AE_MOVDA32X2(output_multiplier[out_channel+1], output_multiplier[out_channel]);
      MPY_BY_QUANT_MULT_ACC64_PER_CHAN_X2_OUT32(scaled_acc0, acc2, acc0, out_mul10, AE_MOVDA32X2(shift1, shift0));
      ae_int16x4 d1 = AE_SAT16X4(scaled_acc0, scaled_acc0);
      AE_S32_H_XP(AE_MOVINT32X2_FROMINT16X4(d1), (ae_int32 *)pout0, 2*output_depth*sizeof(WORD16));
    }
  }
  //  Loop for output_depth not a multiple of 2
  for (out_channel = loop_cnt; out_channel < output_depth; ++out_channel)
  {
    int shift0 = output_shift[out_channel];
    ae_int64 *pscratch0 = (ae_int64*)&scratch_buffer[out_channel];
    ae_int16 *pout = (ae_int16*)&output_data[out_channel];
    ae_int64 *pscratch1 = (ae_int64*)&scratch_buffer[((output_height*output_width)>>1)*output_depth+out_channel];
    ae_int16 *pout1 = (ae_int16*)&output_data[((output_height*output_width)>>1)*output_depth+out_channel];
    AE_L64_XP(dbias0, pbias, bias_offset);
    AE_L64_XP(acc0, pscratch0, output_depth*sizeof(WORD64));
    AE_L64_XP(acc1, pscratch1, output_depth*sizeof(WORD64));
    for (int i = 0; i < ((output_height*output_width)>>1); i++)
    {
      acc0 = AE_ADD64(acc0, dbias0);
      acc1 = AE_ADD64(acc1, dbias0);
      ae_int32x2 scaled_acc;
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(scaled_acc, acc0, acc1, output_multiplier[out_channel], shift0);
      ae_int16x4 d1 = AE_SAT16X4(scaled_acc, scaled_acc);
      AE_L64_XP(acc0, pscratch0, output_depth*sizeof(WORD64));
      AE_L64_XP(acc1, pscratch1, output_depth*sizeof(WORD64));
      AE_S16_0_XP(AE_SEL16_4321(d1, d1), pout, output_depth*sizeof(WORD16));
      AE_S16_0_XP(d1, pout1, output_depth*sizeof(WORD16));
    }
    if((output_height*output_width) & 1)
    {
      acc1 = AE_ADD64(acc1, dbias0);
      ae_int32x2 scaled_acc;
      MPY_BY_QUANT_MULT_ACC64_X2_OUT32(scaled_acc, acc1, acc1, output_multiplier[out_channel], shift0);
      ae_int16x4 d1 = AE_SAT16X4(scaled_acc, scaled_acc);
      AE_S16_0_I(d1, pout1, 0);
    }
  } 
}

/* Handle transpose. Sub-kernels = 1 for unit stride */
static inline void tconv2d_std_reorder_kernel_sym8s
    (pVOID p_scratch
     ,const WORD8* p_kernel
     ,WORD32 kernel_height
     ,WORD32 kernel_width
     ,WORD32 input_channels
     ,WORD32 output_channels
     ,WORD32 x_stride
     ,WORD32 y_stride
    )
{
  /* Presently only adding this for special case when strides = 1*/

  int kx, ky, inCh, outCh, inIdx, outIdx = 0;
  int kxStart, kyStart;

  int pitch_d = input_channels;
  int pitch_w = kernel_width * input_channels;
  int pitch_h = kernel_height * kernel_width * input_channels;

	WORD8 *p_ker;
  WORD8 *p_dst = (WORD8 *)p_scratch;

  /* Conversion from NDWH -> DNWH,                       */
  /* transposing of kernels and formation of sub-kernels */
  outIdx  = 0;
  kyStart = kernel_height - 1 - (kernel_height % y_stride);
  kxStart = kernel_width - 1 - (kernel_width % x_stride);

  for (outCh = 0; outCh < output_channels; outCh++)      /* N */
  {
    for (ky = kyStart; ky >= 0; ky -= y_stride)          /* H */
    {
      for (kx = kxStart; kx >= 0; kx -= x_stride)        /* W */
      {
        p_ker = (WORD8*)&p_kernel[inIdx = outCh * pitch_h + ky * pitch_w + kx * pitch_d];
        ae_valignx2 align_p_ker = AE_LA128_PP(p_ker);
        ae_valignx2 align_p_dst = AE_ZALIGN128();
        ae_int8x8 d_ker0, d_ker1;
        for (inCh = 0; inCh < input_channels >> 4; inCh++)        /* D */
        {
          AE_LA8X8X2_IP(d_ker0, d_ker1, align_p_ker, (ae_int8x16*)p_ker);
          AE_SA8X8X2_IP(d_ker0, d_ker1, align_p_dst, (ae_int8x16*)p_dst);
        }
        AE_LAV8X8X2_XP(d_ker0, d_ker1, align_p_ker, (ae_int8x16*)p_ker, (input_channels & 15));
        AE_SAV8X8X2_XP(d_ker0, d_ker1, align_p_dst, (ae_int8x16*)p_dst, (input_channels & 15));
        AE_SA128POS_FP(align_p_dst, p_dst);
      }
    }
  }
}

static inline void tconv_pad(
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const WORD64* __restrict__ p_bias,
    WORD16 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 idx_width,
    WORD32 idx_height)
{
  WORD32 i, j, k;
  ae_int16x4 d1;

  /* When kernel has no valid input for convolution, output is just bias */
  for(i = idx_height; i < out_height; i++)
  {
    for(j = idx_width; j < out_width; j++)
    {
      ae_int16 *ptrout = (ae_int16*)&p_out[i * out_height_offset + j * out_width_offset];
      ae_int64 *pbias = (ae_int64*)p_bias;
      ae_int64 q1;
      for(k = 0; k < out_channels; k++)
      {
        AE_L64_IP(q1, pbias, 8);
        ae_int32x2 acc;
        MPY_BY_QUANT_MULT_ACC64_OUT32(acc, q1, p_out_multiplier[k], p_out_shift[k]);
        d1 = AE_SAT16X4(acc, acc);
        AE_S16_0_XP(d1, ptrout, out_channels_offset*sizeof(WORD16));
      }
    }
  }
}

static inline void transpose_conv2d_std_sym8sxsym16s(WORD16* output_data,
		const WORD16* input_data,
		const WORD8* filter_data,
		const WORD64* bias_data,
		int stride_width, int stride_height,
		int pad_width, int pad_height,
		int input_depth, int output_depth,
		int input_height, int input_width,
		int filter_height, int filter_width,
		int output_height, int output_width,
		int num_elements,
		int *output_shift, int *output_multiplier,
		pVOID scratch_buffer)
{
  /* Transpose and Reorder the kernel into sub-kernels */
  tconv2d_std_reorder_kernel_sym8s(scratch_buffer, filter_data, filter_height, filter_width, input_depth, output_depth, stride_width, stride_height);

  int kernel_size = PADDED_SIZE(filter_height * filter_width * input_depth * output_depth, ALIGNMENT_16);
  WORD8 *p_trp_ker = (WORD8 *)scratch_buffer; 
  WORD16 *p_scr_cnv = (WORD16 *)((WORD8 *)scratch_buffer + kernel_size);

  /* Conv 2D Standard code init */
  /* Here the x-pad and y-pad values are controlled by the filter dimensions
   * x-r-pad = filter_width - 1 and y-b-pad = filter_height - 1
   * x_pad and y_pad depend on kernel dimension and the padding.
   * Additionally if there's padding specified, less input is required for convolution 
  */

  WORD32 x_pad = (filter_width - 1) - pad_width;
  WORD32 y_pad = (filter_height - 1) - pad_height;
  WORD32 ext_inp_w = -x_pad;
  WORD32 ext_inp_h = -y_pad;
  x_pad = (x_pad < 0) ? 0 : x_pad;
  y_pad = (y_pad < 0) ? 0 : y_pad;
  ext_inp_w = (ext_inp_w < 0) ? 0 : ext_inp_w;
  ext_inp_h = (ext_inp_h < 0) ? 0 : ext_inp_h;
  WORD32 valid_out_h = XT_MIN(output_height, (y_pad + input_height - ext_inp_h));
  WORD32 valid_out_w = XT_MIN(output_width, (x_pad + input_width - ext_inp_w));

  // Determine y-bottom padding
  WORD32 y_b_pad = filter_height - 1;

  WORD32 out_data_format = 0; // Assuming NHWC
  WORD32 out_channels_offset = out_data_format ? output_height * output_width : 1;
  WORD32 out_height_offset = out_data_format ? output_width : output_width * output_depth;
  WORD32 out_width_offset = out_data_format ? 1 : output_depth;

  /* Handle cases that have no valid input for convolution */
  if (ext_inp_w >= input_width || ext_inp_h >= input_height)
  {
    tconv_pad(output_width, output_height, output_depth, out_channels_offset, out_width_offset, out_height_offset, bias_data, output_data, output_multiplier, output_shift, 0, 0);
  }
  else
  {
    /* Handle cases that have no valid input for convolution */
    if(valid_out_h < output_height)
      tconv_pad(output_width, output_height, output_depth, out_channels_offset, out_width_offset, out_height_offset, bias_data, output_data, output_multiplier, output_shift, 0, valid_out_h);

    if(valid_out_w < output_width)
      tconv_pad(output_width, output_height, output_depth, out_channels_offset, out_width_offset, out_height_offset, bias_data, output_data, output_multiplier, output_shift, valid_out_w, 0);

    WORD32 j;
    WORD32 input_bytewidth = 2;
    VOID *pp_inp = (VOID *)((WORD16 *)input_data + (ext_inp_h * input_width * input_depth) + (ext_inp_w * input_depth));

    xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scr_cnv;
    xa_nn_trans_conv2d_std_init_state((void*)p_state
        ,(void*)p_trp_ker
        ,input_height - ext_inp_h
        ,input_depth
        ,filter_height
        ,filter_width
        ,stride_width,stride_height
        ,x_pad
        ,y_pad
        ,output_height
        ,output_width
        ,PREC_SYM16S);

    /* When kernel convolves over input region */
    // Initialize circular buffer
    trans_conv2d_std_init_cir_buf(input_depth, input_depth, input_bytewidth, input_width, input_height - ext_inp_h, y_pad, y_b_pad, x_pad, filter_width, stride_width, ext_inp_w, (VOID**)&pp_inp, p_state);

    // Index to padded input width
    WORD32 idx_beg_inp_width_pad = filter_width - stride_width;
    idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;

    // Process Loop to compute one output plane [out_height x out_channels] per iteration
    for(j = 0; j < valid_out_w; j++)
    {
      // Add x_stride x (input_height x input_channels) new planes to circular buffer
      trans_conv2d_std_update_cir_buf(input_depth, input_depth, input_bytewidth, input_width, input_height - ext_inp_h, y_pad, y_b_pad, x_pad, filter_width, stride_width, ext_inp_w, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state);

      // Update index to input width padded
      idx_beg_inp_width_pad += stride_width;

      // Convolution using matXvec with matrix as circular buffer
      xa_nn_matXvec_sym8sxsym16s_sym16s_circ
        (output_data /* output */
         ,p_state->cir_buf.p_curr/* matrix: rows x cols */
         ,p_trp_ker /* vec: cols */
         ,bias_data /* bias */
         ,valid_out_h /* rows */
         ,input_depth * filter_width * filter_height /* cols */
         ,input_depth * filter_width * stride_height/* row_offset */
         ,output_depth /* vec_count */
         ,input_depth * filter_width * filter_height /* vec_stride */
         ,out_channels_offset /* out_col_offset */
         ,out_height_offset /* out_row_offset */
         ,0
         ,output_multiplier
         ,output_shift
         ,0
        );
      output_data += out_width_offset;
    }
  }
}

int xa_nn_transpose_conv_sym8sxsym16s(WORD16* output_data,
		const WORD16* input_data,
		const WORD8* filter_data,
		const WORD64* bias_data,
		int stride_width, int stride_height,
		int pad_width, int pad_height,
		int input_depth, int output_depth,
		int input_height, int input_width,
		int filter_height, int filter_width,
		int output_height, int output_width,
		int num_elements,
		int *output_shift, int *output_multiplier,
		int64_t* scratch_buffer)
{
	/* NULL pointer checks */
	XA_NNLIB_ARG_CHK_PTR(output_data, -1);
	XA_NNLIB_ARG_CHK_PTR(filter_data, -1);
	XA_NNLIB_ARG_CHK_PTR(input_data, -1);
	XA_NNLIB_ARG_CHK_PTR(scratch_buffer, -1);
	/* Pointer alignment checks */
	XA_NNLIB_ARG_CHK_ALIGN(output_data, sizeof(WORD16), -1);
	XA_NNLIB_ARG_CHK_ALIGN(filter_data, sizeof(WORD8), -1);
	XA_NNLIB_ARG_CHK_ALIGN(input_data, sizeof(WORD16), -1);
	XA_NNLIB_ARG_CHK_ALIGN(bias_data, sizeof(WORD64), -1);
	XA_NNLIB_ARG_CHK_ALIGN(scratch_buffer, 2*sizeof(WORD64), -1);
	/* Basic Parameter checks */
	XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((input_depth <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((filter_height <= 0 || filter_width <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((output_depth <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((stride_height <= 0 || stride_width <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((pad_height < 0 || pad_width < 0), -1);
	XA_NNLIB_ARG_CHK_COND((output_height <= 0 || output_width <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((num_elements <= 0), -1);

  xtbool unit_stride = (stride_width == 1 && stride_height == 1);
  xtbool filt_grt_inp = (filter_width > input_width || filter_height > input_height);

  if(unit_stride && !filt_grt_inp)
  {
    transpose_conv2d_std_sym8sxsym16s(output_data, input_data, filter_data, bias_data,
    stride_width, stride_height, pad_width, pad_height, input_depth, output_depth,
		input_height, input_width, filter_height, filter_width,	output_height, output_width,
		num_elements, output_shift, output_multiplier, scratch_buffer);
  }
  else
  {
    tconv2d_sym8sxsym16s(output_data, input_data, filter_data, bias_data,
    stride_width, stride_height, pad_width, pad_height, input_depth, output_depth,
		input_height, input_width, filter_height, filter_width,	output_height, output_width,
		num_elements, output_shift, output_multiplier, scratch_buffer);
  }

	return 0;
}

