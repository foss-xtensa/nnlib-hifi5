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
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_err_chk.h"
#if XCHAL_HAVE_HIFI5S
static inline ae_int32x2 MultiplyByQuantizedMultiplier_x2_opt(ae_int64 d_x1, ae_int64 d_x2,
                                             int32_t qm1, int32_t qm2, int shift) {                                        
    ae_int32x2 d_q_mulx2 = AE_MOVDA32X2(qm2, qm1);
    ae_int16x4 d_q_mulx4 =  AE_MOVINT16X4_FROMF16X4(AE_ROUND16X4F32SASYM(AE_MOVF32X2_FROMINT32X2(d_q_mulx2), AE_MOVF32X2_FROMINT32X2(d_q_mulx2)));
    ae_int64 q, q2;
    AE_MULP48X16X2_L(q, q2, d_x1, d_x2, d_q_mulx4);  
    ae_int32x2 result = AE_ROUNDAV32X2F64SASYM(q, q2, shift);
    return result;
}
#endif // XCHAL_HAVE_HIFI5S

static inline ae_int32x2 MultiplyByQuantizedMultiplier_ref(ae_int64 d_x,
                                             int32_t quantized_multiplier,
                                             int shift){
  ae_int32x2 d_q_mul = SW_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_MOVINT16X4_FROMF16X4(AE_ROUND16X4F32SASYM(AE_MOVF32X2_FROMINT32X2(d_q_mul), AE_MOVF32X2_FROMINT32X2(d_q_mul)));
  ae_int32x2 d_red_mul32 = AE_SEXT32X2D16_32(d_red_mul16); //upper 32
  ae_int64 qL = AE_MUL32U_LL(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x));
  ae_int64 qH = AE_SLAI64(AE_MUL32_LH(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x)), 32);
  ae_int64 q = AE_ADD64(qL, qH);
  q = SW_SLAA64S_INT64_INT64(q, (shift+17));
  ae_int32x2 result = AE_MOVINT32X2_FROMF32X2(AE_ROUND32F64SASYM(AE_MOVF64_FROMINT64(q)));
  return result;
}

#if 0
static inline ae_int32x2 MultiplyByQuantizedMultiplier_x2_opt(ae_int64 d_x1, ae_int64 d_x2,
                                             int32_t quantized_multiplier,
                                             int shift) {
  ae_int32x2 d_q_mul = AE_MOVDA32(quantized_multiplier);
  ae_int16x4 d_red_mul16 = AE_ROUND16X4F32SASYM(d_q_mul, d_q_mul);
  ae_int32x2 d_red_mul32 = AE_SEXT32X2D16_32(d_red_mul16);
  ae_int64 qL1 = AE_MUL32U_LL(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x1));
  ae_int64 qL2 = AE_MUL32U_LL(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x2));
  ae_int64 qH1 = AE_SLAI64(AE_MUL32_LH(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x1)), 32);
  ae_int64 qH2 = AE_SLAI64(AE_MUL32_LH(d_red_mul32, AE_MOVINT32X2_FROMINT64(d_x2)), 32);
  ae_int64 q1 = AE_ADD64(qL1, qH1);
  ae_int64 q2 = AE_ADD64(qL2, qH2);
  q1 = AE_SLAA64S(q1, (shift+17));
  q2 = AE_SLAA64S(q2, (shift+17));
  ae_int32x2 result = AE_ROUND32X2F64SASYM(q1, q2);
  return result;
}
#endif

/* 2D Convolution implementation */
static inline void conv2d_nchw_sym8sxsym16s_hf5_convmul
(pWORD16 __restrict__ p_out  /* Output:  [Stream] [(out_stride): (actual_out_height): (actual_out_width)] */
 ,const WORD8 *__restrict__ p_ker  /* Kernel:  [Block] [1:             kernel_height:       kernel_width_pad] */
 ,const WORD16 *__restrict__ p_inp  /* Input:   [Block] [1:             input_height:        input_width] */
 ,WORD64 bias_t
 ,int input_height
 ,int input_width
 ,int kernel_height
 ,int kernel_width
 ,int actual_out_height      /* This is the actual output height, processing should be limited to it. */
 ,int actual_out_width       /* This is the actual output width, processing should be limited to it. */
 ,int out_stride
 ,int x_stride
 ,int y_stride
 ,WORD32  out_multiplier
 ,WORD32  out_shift
 ,pWORD64 __restrict__ p_scratch /* Scratch: [Block] [1:             (actual_out_height): (out_width)] */
 )
{
    /* Importance of actual_out_width, since we are appending zeros input left
     * and right side. No problem with left padding, but for right padding that
     * is done to make sure that input_width is multiple of 4. Here
     * 'output_width_for_x_stride_1' value is calculated based on this padded value. But
     * actually expected output width to pick correct values from 'output_width_for_x_stride_1' on
     * jumps of 'x_stride'. */

    int kernel_width_pad = (kernel_width+3)&(~3);
    //int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);

    /* Generic case */
    int i, j, k, l;
    int output_height = input_height - kernel_height + 1;
    int output_width_for_x_stride_1;

    /* Here input_width is nothing but circ_buf_width, which is taken care to be
     * multiple of 4. */
    output_width_for_x_stride_1 = (1 + ((input_width - kernel_width)/1));
    /* output_width_for_x_stride_1 loop is unrolled by 4 so keeping this dimension to multiple of 4 */
    output_width_for_x_stride_1 = ALIGNED_SIZE(output_width_for_x_stride_1, (ALIGNMENT/2));

    /* Please note that below addition of 1 is done to adjust in C style indices
     * */
    if ((actual_out_height - 1) > ((output_height + 1) / (y_stride)))
    {
        return;
    }
    if ((actual_out_width - 1) > ((output_width_for_x_stride_1 + 1) / (x_stride)))
    {
        return;
    }

  ae_int64 accu_int64_0, accu_int64_1, accu_int64_2, accu_int64_3;
  ae_int64 accu_int64_4, accu_int64_5, accu_int64_6, accu_int64_7;
  ae_int32x2 accu_int32x2_0, accu_int32x2_1, accu_int32x2_2, accu_int32x2_3;
  ae_int64 *scratch_ptr = (ae_int64 *)p_scratch;
  ae_int64 *bias = (ae_int64*)(&bias_t);
  ae_int64 _ae_int64_sat_bias;
  _ae_int64_sat_bias = *bias;

  if(kernel_width_pad==12)
  {
      ae_int16x4 d_inp00, d_inp01, d_inp02, d_inp03, d_inp04;
      ae_int16x4 d_inp10, d_inp11, d_inp12, d_inp13, d_inp14;
      ae_int8x8 d_ker0, d_ker1, d_ker2;
      int kernel_height_acc, kh;
      /* Convolution mul instructions have only 32-bit accumulator so for big kernels (size > 256),
      we accumulate outputs to 64-bit accumulator after every kernel_height_acc*12
      (which should be <= 256) multiplications */
      kernel_height_acc = 20; // Largest even value less than 256/12 (21.3333)
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
        int temp = output_width_for_x_stride_1&(~7);
        for(j = 0; j < temp; j+=8)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          accu_int64_4 = AE_ZERO64();
          accu_int64_5 = AE_ZERO64();
          accu_int64_6 = AE_ZERO64();
          accu_int64_7 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            accu_int32x2_2 = AE_ZERO32();
            accu_int32x2_3 = AE_ZERO32();
            ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)p_inp;
            ae_int16x4 *pt16x4_inp1 = (ae_int16x4 *)p_inp;
            AE_ADDCIRC16X4_XC(pt16x4_inp0, (sizeof(WORD16) * (((i * y_stride + kh) * input_width) + j)));
            AE_ADDCIRC16X4_XC(pt16x4_inp1, (sizeof(WORD16) * (((i * y_stride + kh + 1) * input_width) + j)));
            ae_int16x8 *pt_inp0 = (ae_int16x8 *)(pt16x4_inp0);
            ae_int16x8 *pt_inp1 = (ae_int16x8 *)(pt16x4_inp1);
            ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[kh*kernel_width_pad]);
            ae_valign ker_a = AE_LA64_PP(pt_ker);
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
            for(k = 0; k < (loop_count>>1); k++)
            {
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
              AE_LA8X8_IP(d_ker1, ker_a, pt_ker);
              AE_LA8X8_IP(d_ker2, ker_a, pt_ker);

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              AE_L16X4X2_XC(d_inp02, d_inp03, pt_inp0, 16);
              pt16x4_inp0 = (ae_int16x4 *)pt_inp0;
              AE_L16X4_XC(d_inp04, pt16x4_inp0, sizeof(WORD16)*(2*input_width-16));
              pt_inp0 = (ae_int16x8 *)pt16x4_inp0;

              AE_L16X4X2_XC(d_inp10, d_inp11, pt_inp1, 16);
              AE_L16X4X2_XC(d_inp12, d_inp13, pt_inp1, 16);
              pt16x4_inp1 = (ae_int16x4 *)pt_inp1;
              AE_L16X4_XC(d_inp14, pt16x4_inp1, sizeof(WORD16)*(2*input_width-16));
              pt_inp1 = (ae_int16x8 *)pt16x4_inp1;

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              AE_MULA2X4Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp02, d_inp03, d_inp10, d_inp11);
              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker2, d_inp11, d_inp12, d_inp13);

              AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp01, d_inp02, d_inp03);
              AE_MULA2X4Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker1, d_inp03, d_inp04, d_inp11, d_inp12);
              AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker2, d_inp12, d_inp13, d_inp14);
            }
            if((loop_count)&1)
            {
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
              /* 4 values outside kernel are loaded but it is safe because second
              row of input is 0, i.e. d_inp10, d_inp11 are initialized to 0 */
              AE_LA8X8_IP(d_ker1, ker_a, pt_ker);

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              AE_L16X4X2_XC(d_inp02, d_inp03, pt_inp0, 16);
              pt16x4_inp0 = (ae_int16x4 *)pt_inp0;
              AE_L16X4_XC(d_inp04, pt16x4_inp0, sizeof(WORD16)*(2*input_width-16));
              pt_inp0 = (ae_int16x8 *)pt16x4_inp0;

              d_inp10 = d_inp11 = d_inp12 = AE_ZERO16();

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              AE_MULA2X4Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp02, d_inp03, d_inp10, d_inp11);

              AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp01, d_inp02, d_inp03);
              AE_MULA2X4Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker1, d_inp03, d_inp04, d_inp11, d_inp12);
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
            AE_ACCW32(accu_int64_4, accu_int64_5, accu_int32x2_2, AE_ZERO32());
            AE_ACCW32(accu_int64_6, accu_int64_7, accu_int32x2_3, AE_ZERO32());
          }
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
          AE_S64X2_I(accu_int64_4, accu_int64_5, p_sc, 32);
          AE_S64X2_I(accu_int64_6, accu_int64_7, p_sc, 48);
        }
        if(j < output_width_for_x_stride_1)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            ae_int16x4 *pt16x4_inp0 =(ae_int16x4 *)p_inp;
            ae_int16x4 *pt16x4_inp1 =(ae_int16x4 *)p_inp;
            AE_ADDCIRC16X4_XC(pt16x4_inp0, (sizeof(WORD16) * (((i * y_stride + kh) * input_width) + j)));
            AE_ADDCIRC16X4_XC(pt16x4_inp1, (sizeof(WORD16) * (((i * y_stride + kh + 1) * input_width) + j)));
            ae_int16x8 *pt_inp0 = (ae_int16x8 *)(pt16x4_inp0);
            ae_int16x8 *pt_inp1 = (ae_int16x8 *)(pt16x4_inp1);
            ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[kh*kernel_width_pad]);
            ae_valign ker_a = AE_LA64_PP(pt_ker);
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
            for(k = 0; k < (loop_count>>1); k++)
            {
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
              AE_LA8X8_IP(d_ker1, ker_a, pt_ker);
              AE_LA8X8_IP(d_ker2, ker_a, pt_ker);

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              AE_L16X4X2_XC(d_inp02, d_inp03, pt_inp0, sizeof(WORD16)*(2*input_width-8));

              AE_L16X4X2_XC(d_inp10, d_inp11, pt_inp1, 16);
              AE_L16X4X2_XC(d_inp12, d_inp13, pt_inp1, sizeof(WORD16)*(2*input_width-8));

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              AE_MULA2X4Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp02, d_inp03, d_inp10, d_inp11);
              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker2, d_inp11, d_inp12, d_inp13);
            }
            if((loop_count)&1)
            {
              AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
              /* 4 values outside kernel are loaded but it is safe because second
              row of input is 0, i.e. d_inp10, d_inp11 are initialized to 0 */
              AE_LA8X8_IP(d_ker1, ker_a, pt_ker);

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              AE_L16X4X2_XC(d_inp02, d_inp03, pt_inp0, sizeof(WORD16)*(2*input_width-8));

              d_inp10 = d_inp11 = AE_ZERO16();

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              AE_MULA2X4Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker1, d_inp02, d_inp03, d_inp10, d_inp11);
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          }
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
        }
      }
  }
  else if(kernel_width_pad==8)
  {
      /* Regression is yet to be tested, but runperf.sh case is not matching with
      ref output, it is most probably due to kernel not being properly padded,
      need to fix this in testbench */
      ae_int16x4 d_inp00, d_inp01, d_inp02, d_inp03;
      ae_int8x8 d_ker0;
      int kernel_height_acc, kh;
      /* Convolution mul instructions have only 32-bit accumulator so for big kernels (size > 256),
      we accumulate outputs to 64-bit accumulator after every kernel_height_acc*8
      (which should be <= 256) multiplications */
      kernel_height_acc = 32; //256/8
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
        int temp = output_width_for_x_stride_1&(~7);
        for(j = 0; j < temp; j+=8)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          accu_int64_4 = AE_ZERO64();
          accu_int64_5 = AE_ZERO64();
          accu_int64_6 = AE_ZERO64();
          accu_int64_7 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            accu_int32x2_2 = AE_ZERO32();
            accu_int32x2_3 = AE_ZERO32();
            ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)p_inp;
            AE_ADDCIRC16X4_XC(pt16x4_inp0, (sizeof(WORD16) * (((i * y_stride + kh) * input_width) + j)));
            ae_int16x8 *pt_inp0 = (ae_int16x8 *)(pt16x4_inp0);
            ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[kh*kernel_width_pad]);
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
            for(k = 0; k < loop_count; k++)
            {
              d_ker0 = *pt_ker++;

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              AE_L16X4X2_XC(d_inp02, d_inp03, pt_inp0, sizeof(WORD16)*(input_width-8));

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp01, d_inp02, d_inp03);
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
            AE_ACCW32(accu_int64_4, accu_int64_5, accu_int32x2_2, AE_ZERO32());
            AE_ACCW32(accu_int64_6, accu_int64_7, accu_int32x2_3, AE_ZERO32());
          }
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
          AE_S64X2_I(accu_int64_4, accu_int64_5, p_sc, 32);
          AE_S64X2_I(accu_int64_6, accu_int64_7, p_sc, 48);
        }
        if(j < output_width_for_x_stride_1)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)p_inp;
            AE_ADDCIRC16X4_XC(pt16x4_inp0, (sizeof(WORD16) * (((i * y_stride + kh) * input_width) + j)));
            ae_int16x8 *pt_inp0 = (ae_int16x8 *)(pt16x4_inp0);
            ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[kh*kernel_width_pad]);
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
            for(k = 0; k < loop_count; k++)
            {
              d_ker0 = *pt_ker++;

              AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
              pt16x4_inp0 = (ae_int16x4 *)pt_inp0;
              AE_L16X4_XC(d_inp02, pt16x4_inp0, sizeof(WORD16)*(input_width-8));
              pt_inp0 = (ae_int16x8 *)pt16x4_inp0;

              AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          }
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
        }
      }
  }
  else
  {
      ae_int16x4 d_inp00, d_inp01, d_inp02, d_inp03;
      ae_int8x8 d_ker0;
      int kernel_height_acc, kh;
      /* Convolution mul instructions have only 32-bit accumulator so for big kernels (size > 256),
      we accumulate outputs to 64-bit accumulator after every kernel_height_acc*kernel_width_pad
      (which should be <= 256) multiplications */
      kernel_height_acc = 256/kernel_width_pad;
      for(i = 0; i < actual_out_height; i++)
      {
        scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
        int temp = output_width_for_x_stride_1&(~7);
        for(j = 0; j < temp; j+=8)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
          accu_int64_4 = AE_ZERO64();
          accu_int64_5 = AE_ZERO64();
          accu_int64_6 = AE_ZERO64();
          accu_int64_7 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            accu_int32x2_2 = AE_ZERO32();
            accu_int32x2_3 = AE_ZERO32();
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
#pragma loop_count min=1
            for(k = 0; k < loop_count; k++)
            {
              ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)p_inp;
              AE_ADDCIRC16X4_XC(pt16x4_inp0, (sizeof(WORD16) * (((i * y_stride + kh + k) * input_width) + j)));
              ae_int16x8 *pt_inp0 = (ae_int16x8 *)(pt16x4_inp0);
              ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[(kh+k)*kernel_width_pad]);
              ae_valign ker_a = AE_LA64_PP(pt_ker);
#pragma no_unroll
              for(l = 0; l < (kernel_width_pad>>3); l++)
              {
                AE_LA8X8_IP(d_ker0, ker_a, pt_ker);

                AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
                AE_L16X4X2_I(d_inp02, d_inp03, pt_inp0, 0);

                AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
                AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp01, d_inp02, d_inp03);
              }
              if(kernel_width_pad&4)
              {
                AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
                /* Last 4 value are not from kernel so making them 0 */
                d_ker0 = AE_MOVINT8X8_FROMINT32X2(AE_AND32(AE_MOVINT32X2_FROMINT8X8(d_ker0), AE_MOVDA32X2(0xffffffff, 0)));

                AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
                AE_L16X4X2_I(d_inp02, d_inp03, pt_inp0, 0);

                AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
                AE_MULA8Q8X16CNV(accu_int32x2_2, accu_int32x2_3, d_ker0, d_inp01, d_inp02, d_inp03);
              }
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
            AE_ACCW32(accu_int64_4, accu_int64_5, accu_int32x2_2, AE_ZERO32());
            AE_ACCW32(accu_int64_6, accu_int64_7, accu_int32x2_3, AE_ZERO32());
          }
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
          AE_S64X2_I(accu_int64_4, accu_int64_5, p_sc, 32);
          AE_S64X2_I(accu_int64_6, accu_int64_7, p_sc, 48);
        }
        if(j < output_width_for_x_stride_1)
        {
          accu_int64_0 = AE_ZERO64();
          accu_int64_1 = AE_ZERO64();
          accu_int64_2 = AE_ZERO64();
          accu_int64_3 = AE_ZERO64();
#pragma loop_count min=1
          for(kh = 0; kh < kernel_height; kh += kernel_height_acc)
          {
            accu_int32x2_0 = AE_ZERO32();
            accu_int32x2_1 = AE_ZERO32();
            /* For last iteration loop_count should be (kernel_height - kh) */
            int loop_count = kernel_height_acc > (kernel_height - kh) ? kernel_height - kh : kernel_height_acc;
#pragma loop_count min=1
            for(k = 0; k < loop_count; k++)
            {
              ae_int16x4 *pt16x4_inp0 = (ae_int16x4 *)p_inp;
              AE_ADDCIRC16X4_XC(pt16x4_inp0, (sizeof(WORD16) * (((i * y_stride + kh + k) * input_width) + j)));
              ae_int16x8 *pt_inp0 = (ae_int16x8 *)(pt16x4_inp0);
              ae_int8x8 *pt_ker = (ae_int8x8 *)(&p_ker[(kh+k)*kernel_width_pad]);
              ae_valign ker_a = AE_LA64_PP(pt_ker);
#pragma no_unroll
              for(l = 0; l < (kernel_width_pad>>3); l++)
              {
                AE_LA8X8_IP(d_ker0, ker_a, pt_ker);

                AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
                pt16x4_inp0 = (ae_int16x4 *)pt_inp0;
                d_inp02 = AE_L16X4_I(pt16x4_inp0, 0);
                pt_inp0 = (ae_int16x8 *)pt16x4_inp0;

                AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              }
              if(kernel_width_pad&4)
              {
                AE_LA8X8_IP(d_ker0, ker_a, pt_ker);
                /* Last 4 value are not from kernel so making them 0 */
                d_ker0 = AE_MOVINT8X8_FROMINT32X2(AE_AND32(AE_MOVINT32X2_FROMINT8X8(d_ker0), AE_MOVDA32X2(0xffffffff, 0)));

                AE_L16X4X2_XC(d_inp00, d_inp01, pt_inp0, 16);
                pt16x4_inp0 = (ae_int16x4 *)pt_inp0;
                d_inp02 = AE_L16X4_I(pt16x4_inp0, 0);
                pt_inp0 = (ae_int16x8 *)pt16x4_inp0;

                AE_MULA8Q8X16CNV(accu_int32x2_0, accu_int32x2_1, d_ker0, d_inp00, d_inp01, d_inp02);
              }
            }
            AE_ACCW32(accu_int64_0, accu_int64_1, accu_int32x2_0, AE_ZERO32());
            AE_ACCW32(accu_int64_2, accu_int64_3, accu_int32x2_1, AE_ZERO32());
          }
          ae_int64x2 *p_sc = (ae_int64x2 *)(scratch_ptr + j);
          AE_S64X2_I(accu_int64_0, accu_int64_1, p_sc,  0);
          AE_S64X2_I(accu_int64_2, accu_int64_3, p_sc, 16);
        }
      }
  }

  /* Here we store output based on strides. For values in a row, values
   * will be picked from it as per 'x_stride'. No need to worry about
   * height dimension, since we took care of it by efficient row
   * accesses. */
  scratch_ptr = (ae_int64 *) p_scratch;

  for(i = 0; i < actual_out_height; i++)
  {
    scratch_ptr = (ae_int64 *) p_scratch + (i * output_width_for_x_stride_1);
    ae_int16 *out_ptr  = (ae_int16 *) p_out + (i * out_stride * actual_out_width);

    for(j = 0; j < actual_out_width; j++)
    {
      accu_int64_0 = scratch_ptr[(j * x_stride)];
      accu_int64_0 =  SW_ADD64S_INT64_INT64(accu_int64_0, _ae_int64_sat_bias);
      ae_int32x2 outval32 = MultiplyByQuantizedMultiplier_ref(accu_int64_0, out_multiplier, out_shift);
      ae_int16x4 outval16 = AE_SAT16X4(outval32, outval32);
      out_ptr[(j * out_stride)] = AE_MOVINT16_FROMINT16X4(outval16);
    }
  }
}

#define COPY_KERNEL_TO_SCRATCH(p_out, p_in, kh, kw, kw_pad) \
{ \
  int itr_kh, itr_kw; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    ae_int8x8 *pae_in = (ae_int8x8 *)(&p_in[itr_kh * kw]); \
    ae_int8x8 *pae_out = (ae_int8x8 *)(&p_out[itr_kh * kw_pad]); \
    ae_int8x8 d_tmp; \
    ae_valign in_a = AE_LA64_PP(pae_in); \
    ae_valign out_a = AE_ZALIGN64(); \
_Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < (kw >> 3); itr_kw++) \
    { \
      AE_LA8X8_IP(d_tmp, in_a, pae_in); \
      AE_SA8X8_IP(d_tmp, out_a, pae_out); \
    } \
    if(kw & 7) \
    { \
      AE_LA8X8_IP(d_tmp, in_a, pae_in); \
      ae_int64 d_tmp64 = AE_MOVINT64_FROMINT8X8(d_tmp); \
      d_tmp64 = AE_SRAA64(d_tmp64, 8 * (8 - (kw & 7))); \
      d_tmp64 = AE_SLAA64(d_tmp64, 8 * (8 - (kw & 7))); \
      d_tmp = AE_MOVINT8X8_FROMINT64(d_tmp64); \
      AE_SA8X8_IP(d_tmp, out_a, pae_out); \
    } \
    AE_SA64POS_FP(out_a, pae_out); \
  } \
}

static WORD32 xa_nn_conv2d_depthwise_per_chan_nchw_sym8sxsym16s
(pWORD16 __restrict__ p_out
 ,const WORD8 *__restrict__ p_kernel
 ,const WORD16 *__restrict__ p_inp
 ,const WORD64 *__restrict__ p_bias
 ,WORD32  input_height
 ,WORD32  input_width
 ,WORD32  input_channels
 ,WORD32  kernel_height
 ,WORD32  kernel_width
 ,WORD32  channels_multiplier
 ,WORD32  x_stride
 ,WORD32  y_stride
 ,WORD32  x_padding
 ,WORD32  y_padding
 ,WORD32  out_height
 ,WORD32  out_width
 ,WORD32  input_zero_bias
 ,const WORD32  *p_out_multiplier
 ,const WORD32  *p_out_shift
 ,WORD32 out_zero_bias
 ,WORD32 inp_data_format
,WORD32  out_data_format
,pVOID p_scratch
)
{
    int i;
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(input_zero_bias != 0, -1);
    for(i = 0; i < input_channels*channels_multiplier; i++)
      XA_NNLIB_ARG_CHK_COND((p_out_shift[i] < -31 || p_out_shift[i] > 15), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 1), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

    (VOID) out_data_format;
    int input_zero_bias_neg = 0;
    xa_nn_dilated_conv2d_depthwise_init
        (p_scratch
         ,input_height
         ,input_width
         ,input_channels
         ,kernel_height
         ,kernel_width
         ,channels_multiplier
         ,1
         ,1
         ,x_stride
         ,y_stride
         ,x_padding
         ,y_padding
         ,out_height
         ,out_width
         ,16
         ,1
         ,(pVOID)(&input_zero_bias_neg)
    );

  xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
  xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
  int itr_ic, itr_cm, itr_oh;
  int circ_out_height = (p_circ_buf->rows - kernel_height)/y_stride + 1;
  int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
  int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
  int rows_to_add, top_pad, bottom_pad, rows_added;
  int input_row;
  const WORD8 *pt_ker;
  const WORD16 *pt_inp;
  pWORD16 p_inp_circ;

    WORD8 *p_kernel_padded = (WORD8 *)(p_state->p_scratch);
    p_kernel_padded = (WORD8 *)ALIGN_PTR(p_kernel_padded, 8);
    pWORD64 p_tmp_out = (pWORD64)(p_kernel_padded + kernel_height_pad * kernel_width_pad);
    p_tmp_out = (pWORD64)ALIGN_PTR(p_tmp_out, 16);

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    WORD64 bias = 0;

    /* Initialize whole scratch for padded kernel to padding value, after this
       we only have to copy actual kernel values, padding area should remain
       untouched */

    WORD8 *pae_ker_pad8 = (WORD8 *)p_kernel_padded;
    for(i = 0; i < (kernel_height_pad * kernel_width_pad); i++) {
      pae_ker_pad8[i] = 0;
    }

    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = (const WORD16 *)&p_inp[itr_ic*input_height*input_width];

        CIRC_BUF_ADD_ROWS_INIT(rows_added
                ,rows_to_add
                ,top_pad
                ,bottom_pad
                ,input_row
                ,input_height
                ,input_width
                ,kernel_height
                ,y_stride
                ,x_padding
                ,y_padding
                ,p_circ_buf
                ,pt_inp
                );

        for(itr_oh = 0; itr_oh < out_height - (circ_out_height - 1); itr_oh += circ_out_height)
        {
            CIRC_BUF_ADD_ROWS(rows_added
                    ,rows_to_add
                    ,top_pad
                    ,bottom_pad
                    ,input_row
                    ,input_height
                    ,input_width
                    ,circ_out_height
                    ,y_stride
                    ,x_padding
                    ,y_padding
                    ,p_circ_buf
                    ,pt_inp
                    );

            p_inp_circ = p_circ_buf->p_curr;

            for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
            {
                pt_ker = (const WORD8 *)&p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
                COPY_KERNEL_TO_SCRATCH(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
                bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

                conv2d_nchw_sym8sxsym16s_hf5_convmul
                    ((pWORD16)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
                     ,p_kernel_padded
                     ,p_inp_circ
                     ,bias
                     ,p_circ_buf->rows
                     ,p_circ_buf->row_offset
                     ,kernel_height
                     ,kernel_width
                     ,circ_out_height
                     ,out_width
                     ,(input_channels * channels_multiplier)
                     ,x_stride
                     ,y_stride
                     ,p_out_multiplier[itr_ic * channels_multiplier + itr_cm]
                     ,p_out_shift[itr_ic * channels_multiplier + itr_cm]
                     ,p_tmp_out
                    );
            }
        }

        CIRC_BUF_ADD_ROWS(rows_added
                ,rows_to_add
                ,top_pad
                ,bottom_pad
                ,input_row
                ,input_height
                ,input_width
                ,circ_out_height
                ,y_stride
                ,x_padding
                ,y_padding
                ,p_circ_buf
                ,pt_inp
                );

        p_inp_circ = p_circ_buf->p_curr;


        for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
        {
            pt_ker = (const WORD8 *)&p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
            COPY_KERNEL_TO_SCRATCH(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
            bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

            conv2d_nchw_sym8sxsym16s_hf5_convmul
                ((pWORD16)(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
                 ,p_kernel_padded
                 ,p_inp_circ
                 ,bias
                 ,p_circ_buf->rows
                 ,p_circ_buf->row_offset
                 ,kernel_height
                 ,kernel_width
                 ,(out_height - itr_oh)
                 ,out_width
                 ,(input_channels * channels_multiplier)
                 ,x_stride
                 ,y_stride
                 ,p_out_multiplier[itr_ic * channels_multiplier + itr_cm]
                 ,p_out_shift[itr_ic * channels_multiplier + itr_cm]
                 ,p_tmp_out
                );
        }
    }
    return 0;
}

/* 2D Convolution implementation for nhwc input */
static inline void conv2d_v2_per_chan_nhwc_sym8sxsym16s
(pWORD16 __restrict__ p_out
 ,const WORD8 *__restrict__ p_ker
 ,const WORD16 *__restrict__ p_inp
 ,const WORD64 *p_bias
 ,int kernel_height
 ,int kernel_width
 ,int out_height
 ,int out_width
 ,int out_channels
 ,int x_stride
 ,int y_stride
 ,const WORD32 *p_out_multiplier
 ,const WORD32 *p_out_shift
 ,pWORD32 __restrict__ p_scratch
 ,int out_activation_min
 ,int out_activation_max
 )
{
    (VOID) x_stride;
    (VOID) p_scratch;
    WORD32 ker_channels_pad, inp_channels_pad;
    WORD32 i, itr_oh, itr_ch, itr_kh;//, itr_kw;
    ae_int16x4 *pt_inp0, *pt_inp1;
    const WORD8 *pt_ker;
    pWORD16 out_ptr0, out_ptr1;
    ae_int16x4 d_inp0, d_ker;
    ae_int16x4 d_inp00, d_inp01, d_ker0;
    ae_int16x4 d_inp10, d_inp11, d_ker1;
    ae_int32x2 d32_ker0, d32_ker1;
    const ae_int64 *pt_bias;
    ae_valign ker_a;
    ae_int64 d64_bias0, d64_bias1, d64_bias2, d64_bias3;
    ae_int32x2 d32_acc01_0, d32_acc23_0, d32_acc45_0, d32_acc67_0;
    ae_int32x2 d32_acc01_1, d32_acc23_1, d32_acc45_1, d32_acc67_1;
    ae_int32x2 d32_zero= SW_MOVDA32(0);
    ae_int64 d64_acc0, d64_acc1, d64_acc2, d64_acc3;
    ae_int64 d64_acc4, d64_acc5, d64_acc6, d64_acc7;
    ae_int16x4 d_acc16x4;

    ker_channels_pad = out_channels;
    inp_channels_pad = ALIGNED_SIZE(out_channels, 4);

    for(itr_oh = 0; itr_oh < (out_height-1); itr_oh+=2)
    {
        out_ptr0 = (WORD16 *)(&p_out[itr_oh*out_channels*out_width]);
        out_ptr1 = (WORD16 *)(&p_out[(itr_oh+1)*out_channels*out_width]);
        pt_bias = (const ae_int64 *)p_bias;
        for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
        {
            int r_shift[4];
            r_shift[0] = p_out_shift[itr_ch+0];
            r_shift[1] = p_out_shift[itr_ch+1];
            r_shift[2] = p_out_shift[itr_ch+2];
            r_shift[3] = p_out_shift[itr_ch+3];

#if XCHAL_HAVE_HIFI5S
            int out_shift0 = (15 - r_shift[0]) << 16 | (0x0000FFFF & (15 - r_shift[1]));
            int out_shift1 = (15 - r_shift[2]) << 16 | (0x0000FFFF & (15 - r_shift[3]));
#endif
            pt_inp0 = (ae_int16x4 *)p_inp;
            pt_inp1 = (ae_int16x4 *)p_inp;
            AE_ADDCIRC16X4_XC(pt_inp0, (itr_ch + itr_oh*y_stride*kernel_width*inp_channels_pad)*sizeof(WORD16));
            AE_ADDCIRC16X4_XC(pt_inp1, (itr_ch + (itr_oh+1)*y_stride*kernel_width*inp_channels_pad)*sizeof(WORD16));
            pt_ker = (const WORD8 *)(&p_ker[itr_ch]);
            d64_acc0 = AE_ZERO64();
            d64_acc1 = AE_ZERO64();
            d64_acc2 = AE_ZERO64();
            d64_acc3 = AE_ZERO64();
            d64_acc4 = AE_ZERO64();
            d64_acc5 = AE_ZERO64();
            d64_acc6 = AE_ZERO64();
            d64_acc7 = AE_ZERO64();
            ker_a = AE_LA64_PP(pt_ker);
            ae_int16x4 *ptt_inp0, *ptt_inp1;
            ptt_inp0 = pt_inp0;
            ptt_inp1 = pt_inp1;
            
            for(itr_kh = 0; itr_kh < (kernel_height*kernel_width-1); itr_kh+=2)
            {
                AE_L16X4_XC(d_inp00,ptt_inp0,inp_channels_pad*sizeof(WORD16));
                AE_L16X4_XC(d_inp10,ptt_inp1,inp_channels_pad*sizeof(WORD16));
                AE_LA8X4S_IP(d_ker0, ker_a, pt_ker);
                pt_ker += ker_channels_pad - 4;
                ker_a = AE_LA64_PP(pt_ker);
                AE_L16X4_XC(d_inp01,ptt_inp0,inp_channels_pad*sizeof(WORD16));
                AE_L16X4_XC(d_inp11,ptt_inp1,inp_channels_pad*sizeof(WORD16));
                AE_LA8X4S_IP(d_ker1, ker_a, pt_ker);
                pt_ker += ker_channels_pad - 4;
                ker_a = AE_LA64_PP(pt_ker);
                /* Need to accumulate in 64 bit accumulator so converting
                muls 32x16 muls, so that mul and accumulation can be combined */
                AE_MUL16X4S(d32_acc01_0,d32_acc23_0,d_inp00,d_ker0);
                AE_MUL16X4S(d32_acc45_0,d32_acc67_0,d_inp10,d_ker0);
                AE_MUL16X4S(d32_acc01_1,d32_acc23_1,d_inp01,d_ker1);
                AE_MUL16X4S(d32_acc45_1,d32_acc67_1,d_inp11,d_ker1);
                AE_ACCW32(d64_acc0,d64_acc1,d32_acc01_0,d32_acc01_1);
                AE_ACCW32(d64_acc2,d64_acc3,d32_acc23_0,d32_acc23_1);
                AE_ACCW32(d64_acc4,d64_acc5,d32_acc45_0,d32_acc45_1);
                AE_ACCW32(d64_acc6,d64_acc7,d32_acc67_0,d32_acc67_1);
            }
            if(((kernel_height*kernel_width)&1)==1)
            {
              AE_L16X4_XC(d_inp00,ptt_inp0,inp_channels_pad*sizeof(WORD16));
              AE_L16X4_XC(d_inp10,ptt_inp1,inp_channels_pad*sizeof(WORD16));
              AE_LA8X4S_IP(d_ker0, ker_a, pt_ker);
              AE_MUL16X4S(d32_acc01_0,d32_acc23_0,d_inp00,d_ker0);
              AE_MUL16X4S(d32_acc45_0,d32_acc67_0,d_inp10,d_ker0);
              AE_ACCW32(d64_acc0,d64_acc1,d32_acc01_0,d32_zero);
              AE_ACCW32(d64_acc2,d64_acc3,d32_acc23_0,d32_zero);
              AE_ACCW32(d64_acc4,d64_acc5,d32_acc45_0,d32_zero);
              AE_ACCW32(d64_acc6,d64_acc7,d32_acc67_0,d32_zero);
            }
            d64_bias0 = pt_bias[itr_ch];
            d64_bias1 = pt_bias[itr_ch+1];
            d64_bias2 = pt_bias[itr_ch+2];
            d64_bias3 = pt_bias[itr_ch+3];

            d64_acc0 = SW_ADD64S_INT64_INT64(d64_acc0, d64_bias0);
            d64_acc1 = SW_ADD64S_INT64_INT64(d64_acc1, d64_bias1);
            d64_acc2 = SW_ADD64S_INT64_INT64(d64_acc2, d64_bias2);
            d64_acc3 = SW_ADD64S_INT64_INT64(d64_acc3, d64_bias3);

#if XCHAL_HAVE_HIFI5S
            ae_int32x2 d32_acc0 = MultiplyByQuantizedMultiplier_x2_opt(d64_acc0, d64_acc1, p_out_multiplier[itr_ch + 0], p_out_multiplier[itr_ch + 1], out_shift0);
            ae_int32x2 d32_acc1 = MultiplyByQuantizedMultiplier_x2_opt(d64_acc2, d64_acc3, p_out_multiplier[itr_ch + 2], p_out_multiplier[itr_ch + 3], out_shift1);
#else
            ae_int32x2 tmp32_0 = MultiplyByQuantizedMultiplier_ref(d64_acc0, p_out_multiplier[itr_ch + 0], r_shift[0]);
            ae_int32x2 tmp32_1 = MultiplyByQuantizedMultiplier_ref(d64_acc1, p_out_multiplier[itr_ch + 1], r_shift[1]);
            ae_int32x2 tmp32_2 = MultiplyByQuantizedMultiplier_ref(d64_acc2, p_out_multiplier[itr_ch + 2], r_shift[2]); 
            ae_int32x2 tmp32_3 = MultiplyByQuantizedMultiplier_ref(d64_acc3, p_out_multiplier[itr_ch + 3], r_shift[3]); 

            ae_int32x2 d32_acc0 = AE_SEL32_LL(tmp32_0, tmp32_1);
            ae_int32x2 d32_acc1 = AE_SEL32_LL(tmp32_2, tmp32_3);
#endif
            d_acc16x4 = AE_SAT16X4(d32_acc0, d32_acc1);
            d_acc16x4 = AE_MAX16(d_acc16x4,AE_MOVDA16(out_activation_min));
            d_acc16x4 = AE_MIN16(d_acc16x4,AE_MOVDA16(out_activation_max));
#pragma no_unroll
            for(i = 0; i < XT_MIN(out_channels-itr_ch, 4); i++)
            {
                d_acc16x4 = AE_SEL16_6543(d_acc16x4, d_acc16x4);
                *(ae_int16 *)(&out_ptr0[itr_ch+i]) = AE_MOVINT16_FROMINT16X4(d_acc16x4);
            }

            d64_acc4 = SW_ADD64S_INT64_INT64(d64_acc4, d64_bias0);
            d64_acc5 = SW_ADD64S_INT64_INT64(d64_acc5, d64_bias1);
            d64_acc6 = SW_ADD64S_INT64_INT64(d64_acc6, d64_bias2);
            d64_acc7 = SW_ADD64S_INT64_INT64(d64_acc7, d64_bias3);

#if XCHAL_HAVE_HIFI5S
            ae_int32x2 d32_acc2 = MultiplyByQuantizedMultiplier_x2_opt(d64_acc4, d64_acc5, p_out_multiplier[itr_ch + 0], p_out_multiplier[itr_ch + 1], out_shift0);
            ae_int32x2 d32_acc3 = MultiplyByQuantizedMultiplier_x2_opt(d64_acc6, d64_acc7, p_out_multiplier[itr_ch + 2], p_out_multiplier[itr_ch + 3], out_shift1);
#else
            ae_int32x2 tmp32_4 = MultiplyByQuantizedMultiplier_ref(d64_acc4, p_out_multiplier[itr_ch + 0], r_shift[0]);
            ae_int32x2 tmp32_5 = MultiplyByQuantizedMultiplier_ref(d64_acc5, p_out_multiplier[itr_ch + 1], r_shift[1]);
            ae_int32x2 tmp32_6 = MultiplyByQuantizedMultiplier_ref(d64_acc6, p_out_multiplier[itr_ch + 2], r_shift[2]); 
            ae_int32x2 tmp32_7 = MultiplyByQuantizedMultiplier_ref(d64_acc7, p_out_multiplier[itr_ch + 3], r_shift[3]); 

            ae_int32x2 d32_acc2 = AE_SEL32_LL(tmp32_4, tmp32_5);
            ae_int32x2 d32_acc3 = AE_SEL32_LL(tmp32_6, tmp32_7);
#endif
            d_acc16x4 = AE_SAT16X4(d32_acc2, d32_acc3);
            d_acc16x4 = AE_MAX16(d_acc16x4,AE_MOVDA16(out_activation_min));
            d_acc16x4 = AE_MIN16(d_acc16x4,AE_MOVDA16(out_activation_max));            
#pragma no_unroll
            for(i = 0; i < XT_MIN(out_channels-itr_ch, 4); i++)
            {
                d_acc16x4 = AE_SEL16_6543(d_acc16x4, d_acc16x4);
                *(ae_int16 *)(&out_ptr1[itr_ch+i]) = AE_MOVINT16_FROMINT16X4(d_acc16x4);
            }
        }
    }
    if(itr_oh < out_height)
    {
        out_ptr0 = (WORD16 *)(&p_out[itr_oh*out_channels*out_width]);
        pt_bias = (const ae_int64 *)p_bias;
        for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
        {
            int r_shift[4];
            r_shift[0] = p_out_shift[itr_ch+0];
            r_shift[1] = p_out_shift[itr_ch+1];
            r_shift[2] = p_out_shift[itr_ch+2];
            r_shift[3] = p_out_shift[itr_ch+3];

#if XCHAL_HAVE_HIFI5S
            int out_shift0 = (15 - r_shift[0]) << 16 | (0x0000FFFF & (15 - r_shift[1]));
            int out_shift1 = (15 - r_shift[2]) << 16 | (0x0000FFFF & (15 - r_shift[3])); 
#endif
            pt_inp0 = (ae_int16x4 *)p_inp;
            AE_ADDCIRC16X4_XC(pt_inp0, (itr_ch + itr_oh*y_stride*kernel_width*inp_channels_pad)*sizeof(WORD16));
            pt_ker = (const WORD8 *)(&p_ker[itr_ch]);
            d64_acc0 = AE_ZERO64();
            d64_acc1 = AE_ZERO64();
            d64_acc2 = AE_ZERO64();
            d64_acc3 = AE_ZERO64();
            ker_a = AE_LA64_PP(pt_ker);
            ae_int16x4 *ptt_inp0 = pt_inp0;
            for(itr_kh = 0; itr_kh < kernel_height*kernel_width; itr_kh++)
            {
                    d_inp0 = AE_L16X4_I(ptt_inp0, 0);
                    AE_LA8X4S_IP(d_ker, ker_a, pt_ker);
                    pt_ker += ker_channels_pad - 4;
                    ker_a = AE_LA64_PP(pt_ker);
                    /* Need to accumulate in 64 bit accumulator so converting
                    muls 32x16 muls, so that mul and accumulation can be combined */
                    d32_ker0 = AE_SEXT32X2D16_32(d_ker);
                    d32_ker1 = AE_SEXT32X2D16_10(d_ker);
                    AE_MULA32X16_H3(d64_acc0, d32_ker0, d_inp0);
                    AE_MULA32X16_L2(d64_acc1, d32_ker0, d_inp0);
                    AE_MULA32X16_H1(d64_acc2, d32_ker1, d_inp0);
                    AE_MULA32X16_L0(d64_acc3, d32_ker1, d_inp0);
                    AE_ADDCIRC16X4_XC(ptt_inp0, inp_channels_pad*sizeof(WORD16));
            }
            d64_bias0 = pt_bias[itr_ch];
            d64_bias1 = pt_bias[itr_ch+1];
            d64_bias2 = pt_bias[itr_ch+2];
            d64_bias3 = pt_bias[itr_ch+3];

            d64_acc0 = SW_ADD64S_INT64_INT64(d64_acc0, d64_bias0);
            d64_acc1 = SW_ADD64S_INT64_INT64(d64_acc1, d64_bias1);
            d64_acc2 = SW_ADD64S_INT64_INT64(d64_acc2, d64_bias2);
            d64_acc3 = SW_ADD64S_INT64_INT64(d64_acc3, d64_bias3);

#if XCHAL_HAVE_HIFI5S
            ae_int32x2 d32_acc0 = MultiplyByQuantizedMultiplier_x2_opt(d64_acc0, d64_acc1, p_out_multiplier[itr_ch + 0], p_out_multiplier[itr_ch + 1], out_shift0);
            ae_int32x2 d32_acc1 = MultiplyByQuantizedMultiplier_x2_opt(d64_acc2, d64_acc3, p_out_multiplier[itr_ch + 2], p_out_multiplier[itr_ch + 3], out_shift1);
#else
            ae_int32x2 tmp32_0 = MultiplyByQuantizedMultiplier_ref(d64_acc0, p_out_multiplier[itr_ch + 0], r_shift[0]);
            ae_int32x2 tmp32_1 = MultiplyByQuantizedMultiplier_ref(d64_acc1, p_out_multiplier[itr_ch + 1], r_shift[1]);
            ae_int32x2 tmp32_2 = MultiplyByQuantizedMultiplier_ref(d64_acc2, p_out_multiplier[itr_ch + 2], r_shift[2]); 
            ae_int32x2 tmp32_3 = MultiplyByQuantizedMultiplier_ref(d64_acc3, p_out_multiplier[itr_ch + 3], r_shift[3]); 

            ae_int32x2 d32_acc0 = AE_SEL32_LL(tmp32_0, tmp32_1);
            ae_int32x2 d32_acc1 = AE_SEL32_LL(tmp32_2, tmp32_3);
#endif
            d_acc16x4 = AE_SAT16X4(d32_acc0, d32_acc1);
            
            d_acc16x4 = AE_MAX16(d_acc16x4,AE_MOVDA16(out_activation_min));
            d_acc16x4 = AE_MIN16(d_acc16x4,AE_MOVDA16(out_activation_max));
#pragma no_unroll
            for(i = 0; i < XT_MIN(out_channels-itr_ch, 4); i++)
            {
                d_acc16x4 = AE_SEL16_6543(d_acc16x4, d_acc16x4);
                *(ae_int16 *)(&out_ptr0[itr_ch+i]) = AE_MOVINT16_FROMINT16X4(d_acc16x4);
            }
        }
    }
}

WORD32 xa_nn_conv2d_depthwise_v2_per_chan_sym8sxsym16s
(pWORD16 __restrict__ p_out
 ,const WORD8 *__restrict__ p_kernel
 ,const WORD16 *__restrict__ p_inp
 ,const WORD64 *__restrict__ p_bias
 ,WORD32  input_height
 ,WORD32  input_width
 ,WORD32  input_channels
 ,WORD32  kernel_height
 ,WORD32  kernel_width
 ,WORD32  channels_multiplier
 ,WORD32  x_stride
 ,WORD32  y_stride
 ,WORD32  x_padding
 ,WORD32  y_padding
 ,WORD32  out_height
 ,WORD32  out_width
 ,WORD32  input_zero_bias
 ,const WORD32  *p_out_multiplier
 ,const WORD32  *p_out_shift
 ,WORD32  output_zero_bias
 ,WORD32  inp_data_format
 ,WORD32  out_data_format
 ,pVOID p_scratch
 ,WORD32  out_activation_min
 ,WORD32  out_activation_max
 ,xa_dma_cfg_t *p_dma_cfg
)
{
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(input_zero_bias != 0, -1);
    for(int i = 0; i < input_channels*channels_multiplier; i++)
      XA_NNLIB_ARG_CHK_COND((p_out_shift[i] < -31 || p_out_shift[i] > 15), -1);
    XA_NNLIB_ARG_CHK_COND((output_zero_bias != 0 ), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);
    XA_NNLIB_CHK_COND((inp_data_format != 0), -1);
    
    (VOID) out_data_format;
    WORD16 pad_val = 0;
    xa_nn_dilated_conv2d_depthwise_init
        (p_scratch
         ,input_height
         ,input_width
         ,input_channels
         ,kernel_height
         ,kernel_width
         ,channels_multiplier
         ,1
         ,1
         ,x_stride
         ,y_stride
         ,x_padding
         ,y_padding
         ,out_height
         ,out_width
         ,16
         ,0
         ,(pVOID)(&pad_val)
        );

    xa_nn_circ_buf_t *p_state = (xa_nn_circ_buf_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = p_state;
    int itr_ow;
    int cols_to_add, left_pad, right_pad, cols_added;
    int input_col;
    const WORD16 *pt_inp;
    pWORD16 p_inp_circ;

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    pt_inp = (const WORD16 *)p_inp;

    CIRC_BUF_ADD_COLS_INIT(cols_added
            ,cols_to_add
            ,left_pad
            ,right_pad
            ,input_col
            ,input_height
            ,input_width
            ,input_channels
            ,kernel_height
            ,kernel_width
            ,channels_multiplier
            ,x_stride
            ,x_padding
            ,y_padding
            ,out_height
            ,p_circ_buf
            ,pt_inp
            );

    for(itr_ow = 0; itr_ow < out_width; itr_ow++)
    {
        CIRC_BUF_ADD_COLS(cols_added
                ,cols_to_add
                ,left_pad
                ,right_pad
                ,input_col
                ,input_height
                ,input_width
                ,input_channels
                ,kernel_height
                ,kernel_width
                ,channels_multiplier
                ,x_stride
                ,x_padding
                ,y_padding
                ,out_height
                ,p_circ_buf
                ,pt_inp
                );

        p_inp_circ = (WORD16 *)p_circ_buf->p_curr;

        conv2d_v2_per_chan_nhwc_sym8sxsym16s
            ((pWORD16)(&p_out[itr_ow*input_channels*channels_multiplier])
             ,p_kernel
             ,p_inp_circ
             ,p_bias
             ,kernel_height
             ,kernel_width
             ,out_height
             ,out_width
             ,(input_channels * channels_multiplier)
             ,x_stride
             ,y_stride
             ,p_out_multiplier
             ,p_out_shift
             ,p_scratch
             ,out_activation_min
             ,out_activation_max
            );
    }
    return 0;
}

WORD32 xa_nn_conv2d_depthwise_per_chan_sym8sxsym16s
  (pWORD16 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD16 *__restrict__ p_inp
  ,const WORD64 *__restrict__ p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  x_stride
  ,WORD32  y_stride
  ,WORD32  x_padding
  ,WORD32  y_padding
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  input_zero_bias
  ,const WORD32 *p_out_multiplier
  ,const WORD32 *p_out_shift
  ,WORD32  out_zero_bias
  ,WORD32  inp_data_format
  ,WORD32  out_data_format
  ,pVOID p_scratch
  )
{
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
    if(inp_data_format == 0)
    {
        return xa_nn_conv2d_depthwise_v2_per_chan_sym8sxsym16s
            (p_out
             ,p_kernel
             ,p_inp
             ,p_bias
             ,input_height
             ,input_width
             ,input_channels
             ,kernel_height
             ,kernel_width
             ,channels_multiplier
             ,x_stride
             ,y_stride
             ,x_padding
             ,y_padding
             ,out_height
             ,out_width
             ,input_zero_bias
             ,p_out_multiplier
             ,p_out_shift
             ,out_zero_bias
             ,inp_data_format
             ,out_data_format
             ,p_scratch
             ,-32768 
             ,32767
             ,NULL);
    }
    else if(inp_data_format == 1)
    {
        return xa_nn_conv2d_depthwise_per_chan_nchw_sym8sxsym16s
            (p_out
             ,p_kernel
             ,p_inp
             ,p_bias
             ,input_height
             ,input_width
             ,input_channels
             ,kernel_height
             ,kernel_width
             ,channels_multiplier
             ,x_stride
             ,y_stride
             ,x_padding
             ,y_padding
             ,out_height
             ,out_width
             ,input_zero_bias
             ,p_out_multiplier
             ,p_out_shift
             ,out_zero_bias
             ,inp_data_format
             ,out_data_format
             ,p_scratch);
    }

    return 0;
}


/* Dilated 2D Convolution implementation */
static WORD32 gcd(WORD32 a, WORD32 b)
{
  while (a != b)
  {
    if (a > b)
    {
      return gcd(a - b, b);
    }
    else
    {
      return gcd(a, b - a);
    }
  }
  return a;
}

WORD32 xa_nn_dilated_conv2d_depthwise_v2_per_chan_sym8sxsym16s
  (pWORD16 __restrict__ p_out
  ,const WORD8 *__restrict__ p_kernel
  ,const WORD16 *__restrict__ p_inp
  ,const WORD64 *__restrict__ p_bias
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  kernel_height
  ,WORD32  kernel_width
  ,WORD32  channels_multiplier
  ,WORD32  dilation_height
  ,WORD32  dilation_width
  ,WORD32  x_stride
  ,WORD32  y_stride
  ,WORD32  x_padding
  ,WORD32  y_padding
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  input_zero_bias
  ,const WORD32 *p_out_multiplier
  ,const WORD32 *p_out_shift
  ,WORD32  out_zero_bias
  ,WORD32  inp_data_format
  ,WORD32  out_data_format
  ,pVOID p_scratch
  ,WORD32  out_activation_min
  ,WORD32  out_activation_max
  ,xa_dma_cfg_t *p_dma_cfg
  )
{
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_height <= 0 || dilation_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND(input_zero_bias != 0, -1);
  for(int i = 0; i < input_channels*channels_multiplier; i++)
    XA_NNLIB_ARG_CHK_COND((p_out_shift[i] < -31 || p_out_shift[i] > 15), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);
  XA_NNLIB_CHK_COND((inp_data_format != 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_min < -32768 || out_activation_min > 32767), -1);
  XA_NNLIB_ARG_CHK_COND((out_activation_max < -32768 || out_activation_max > 32767), -1);
  (VOID) out_data_format;
  WORD16 pad_val = 0;
  xa_nn_dilated_conv2d_depthwise_init
  (p_scratch
   ,input_height
   ,input_width
   ,input_channels
   ,kernel_height
   ,kernel_width
   ,channels_multiplier
   ,dilation_height
   ,dilation_width
   ,x_stride
   ,y_stride
   ,x_padding
   ,y_padding
   ,out_height
   ,out_width
   ,16
   ,0
   ,(pVOID)(&pad_val)
  );

  xa_nn_circ_buf_t *p_state = (xa_nn_circ_buf_t *)p_scratch;
  xa_nn_circ_buf_t *p_circ_buf = p_state;
  
  AE_SETCBEGIN0(p_circ_buf->p_begin);
  AE_SETCEND0(p_circ_buf->p_end);
  
  
  int itr_ow;
  int itr_dh, itr_dw;
  int cols_to_add, left_pad, right_pad, cols_added;
  int input_col;
  const WORD16 *pt_inp;
  pWORD16 p_inp_circ;
  
  pt_inp = (const WORD16 *)p_inp;
  
  WORD32 dh_count, dw_count;
  WORD32 y_padding_dh, x_padding_dw;
  WORD32 x_stride_dw;
  WORD32 out_height_dh, out_width_dw;
  WORD32 rem_dh, rem_dw;
  WORD32 gcd_h, gcd_w;
  WORD32 y_stride_circ_buf;
  
  gcd_h = gcd(dilation_height, y_stride);
  gcd_w = gcd(dilation_width, x_stride);
  dh_count = dilation_height/gcd_h;
  dw_count = dilation_width/gcd_w;
  y_padding_dh = y_padding;
  out_height_dh = out_height / dh_count;
  out_width_dw = out_width / dw_count;
  rem_dh = out_height - out_height_dh * dh_count;
  y_stride_circ_buf = y_stride / gcd_h;
  
  for(itr_dh = 0; itr_dh < dh_count; itr_dh++,rem_dh--)
  {
    x_padding_dw = x_padding;
    x_stride_dw = x_stride * dw_count;
    rem_dw = out_width - out_width_dw * dw_count;
    
    WORD32 out_height_dh_cur = out_height_dh + (rem_dh > 0 ? 1 : 0);
    if(out_height_dh_cur < 1)
      break;
    
    for(itr_dw = 0; itr_dw < dw_count; itr_dw++, rem_dw--)
    {
      WORD32 out_width_dw_cur = out_width_dw + (rem_dw > 0 ? 1 : 0);
      DILATED_CIRC_BUF_ADD_COLS_INIT(
          cols_added,
          cols_to_add,
          left_pad,
          right_pad,
          input_col,
          input_height,
          input_width,
          input_channels,
          kernel_height,
          kernel_width,
          channels_multiplier,
          dilation_height,
          dilation_width,
          x_stride_dw,
          y_stride_circ_buf,
          x_padding_dw,
          y_padding_dh,
          out_height_dh_cur,
          p_circ_buf,
          pt_inp);
      for(itr_ow = 0; itr_ow < out_width_dw_cur; itr_ow++)
      {
        WORD16 *pt_out = (WORD16 *)&p_out[(itr_dh * out_width + itr_dw + itr_ow * dw_count)*input_channels * channels_multiplier];
        DILATED_CIRC_BUF_ADD_COLS(
            cols_added,
            cols_to_add,
            left_pad,
            right_pad,
            input_col,
            input_height,
            input_width,
            input_channels,
            kernel_height,
            kernel_width,
            channels_multiplier,
            dilation_height,
            dilation_width,
            x_stride_dw,
            y_stride_circ_buf,
            x_padding_dw,
            y_padding_dh,
            out_height_dh_cur,
            p_circ_buf,
            pt_inp);

        p_inp_circ = (WORD16 *)p_circ_buf->p_curr;
        
        conv2d_v2_per_chan_nhwc_sym8sxsym16s
            (pt_out
             ,p_kernel
             ,p_inp_circ
             ,p_bias
             ,kernel_height
             ,kernel_width
             ,out_height_dh_cur
             ,out_width * dh_count
             ,(input_channels * channels_multiplier)
             ,x_stride
             ,y_stride_circ_buf
             ,p_out_multiplier
             ,p_out_shift
             ,p_scratch
             ,out_activation_min
             ,out_activation_max
            );
      }
      x_padding_dw -= x_stride;
    }
    y_padding_dh -= y_stride;
  }
  return 0;
}