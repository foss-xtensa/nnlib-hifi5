/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
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
#include "xa_nnlib_common_fpu.h"
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_err_chk.h"

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_conv2d_depthwise_f16,(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  channels_multiplier,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
    WORD32  inp_data_format,
    WORD32  out_data_format,
    pVOID p_scratch))
#else /* #if !HAVE_HP_VFPU */

#define DSELHX4(out0, out1, inp0, inp1, dsel){\
  ae_int16x4 out0_tmp, out1_tmp, inp0_tmp, inp1_tmp;\
  inp0_tmp = AE_MOVF16X4_FROMHALFX4(inp0);\
  inp1_tmp = AE_MOVF16X4_FROMHALFX4(inp1);\
  AE_DSEL16X4(out0_tmp, out1_tmp, inp0_tmp, inp1_tmp, dsel);\
  out0 = AE_MOVHALFX4_FROMF16X4(out0_tmp);\
  out1 = AE_MOVHALFX4_FROMF16X4(out1_tmp);\
}

static void convolve_f16
  (pWORD16 __restrict__ p_out  
  ,const WORD16* __restrict__ p_ker  
  ,const WORD16* __restrict__ p_inp  
  ,WORD16 bias
  ,int input_height
  ,int input_width
  ,int kernel_height
  ,int kernel_width
  ,int actual_out_height      
  ,int actual_out_width  
  ,int out_stride
  ,int x_stride
  ,int y_stride
  ,pWORD64 __restrict__ p_scratch 
  )
{
  int kernel_width_pad = (kernel_width+3)&(~3);

  int i, j, k ,l;
  int output_height = input_height - kernel_height + 1;
  int output_width_for_x_stride_1;

  output_width_for_x_stride_1 = (1 + ((input_width - kernel_width)/1));
  output_width_for_x_stride_1 = ALIGNED_SIZE(output_width_for_x_stride_1, (ALIGNMENT/2));
  if ((actual_out_height - 1) > ((output_height + 1) / (y_stride)))
  {
    return;
  }
  if ((actual_out_width - 1) > ((output_width_for_x_stride_1 + 1) / (x_stride)))
  {
    return;
  }

  xthalfx4 acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;

  xthalf *scratch_ptr = (xthalf *)p_scratch;

  xthalfx4 d_inp0, d_inp1, d_inp2, d_inp3, d_inp4;
  xthalfx4 d_inp6543, d_inp5432, d_inp4321;
  xthalfx4 d_ker0, d_ker1, d_ker2;

  xthalfx4 y0, y1, y01, y2, y3, y23, y02, y13;
  
  ae_int16x4 dsel0 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07060504, 0x03020100));
  ae_int16x4 dsel1 = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x06050504, 0x04030302));

  if(kernel_width_pad == 12)
  {
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (xthalf *) p_scratch + (i * output_width_for_x_stride_1);
      int temp = output_width_for_x_stride_1 -output_width_for_x_stride_1%8;
      for(j = 0; j < temp; j+=8)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);
        CONST_HX4X2(acc4, acc5, 0);
        CONST_HX4X2(acc6, acc7, 0);
        for(k=0; k < kernel_height; k++)
        {
          xthalfx8 *pt_inp = (xthalfx8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp,((sizeof(xthalf)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);

          AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
          AE_LHX4X2_XC(d_inp2, d_inp3, pt_inp, 16);
          AE_LHX4XC(d_inp4, (xthalfx4 *)pt_inp, sizeof(xthalf)*(input_width-16));

          AE_LHX4IP(d_ker0, pt_ker, 8);
          
          DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

          MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

          DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);

          MADDQ_H(acc4, acc5, d_inp1, d_inp6543, d_ker0);
          MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker0);

          AE_LHX4IP(d_ker1, pt_ker, 8);

          MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);

          DSELHX4(d_inp6543, d_inp5432, d_inp2, d_inp3, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp2, d_inp3);      

          MADDQ_H(acc4, acc5, d_inp2, d_inp6543, d_ker1);
          MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker1);

          AE_LHX4IP(d_ker2, pt_ker, 8);

          MADDQ_H(acc0, acc1, d_inp2, d_inp6543, d_ker2);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker2);

          DSELHX4(d_inp6543, d_inp5432, d_inp3, d_inp4, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp3, d_inp4); 

          MADDQ_H(acc4, acc5, d_inp3, d_inp6543, d_ker2);
          MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker2);
        }
  
        WORD16 *scratch_j = (WORD16 *)scratch_ptr + j;
        
        xthalfx4 out0, out1;
        CONST_HX4X2(out0, out1, 0);
        
        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);   
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);     
        DSELHX4(y02, y13, y01, y23, dsel0);
        DSELHX4(y0, y1, acc4, acc5, dsel0);     
        DSELHX4(y2, y3, acc6, acc7, dsel0);    
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);  
        xthalfx4 y02_1, y13_1;    
        DSELHX4(y02_1, y13_1, y01, y23, dsel0);

        ADD_HX4X2(out0, out1, out0, out1, y02, y02_1);
        ADD_HX4X2(out0, out1, out0, out1, y13, y13_1);

        AE_SHX4IP(out0, (xthalfx4 *)scratch_j, 4 * sizeof(xthalf));
        AE_SHX4IP(out1, (xthalfx4 *)scratch_j, 4 * sizeof(xthalf));
      }

      for(j=temp; j < output_width_for_x_stride_1; j+=4)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);        
        for(k=0; k < kernel_height; k++)
        {
          xthalfx8 *pt_inp = (xthalfx8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp,((sizeof(WORD16)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);
          AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
          AE_LHX4X2_XC(d_inp2, d_inp3, pt_inp, sizeof(WORD16)*(input_width-8));

          AE_LHX4IP(d_ker0, pt_ker, 8);
          DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

          MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

          AE_LHX4IP(d_ker1, pt_ker, 8);     
          DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);    
          d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);
          
          MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);   

          AE_LHX4IP(d_ker2, pt_ker, 8);     
          DSELHX4(d_inp6543, d_inp5432, d_inp2, d_inp3, dsel1);    
          d_inp4321 = AE_SELH_4321(d_inp2, d_inp3);
          
          MADDQ_H(acc0, acc1, d_inp2, d_inp6543, d_ker1);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);           

        }
        WORD16 *scratch_j = (WORD16 *)scratch_ptr + j;

        xthalfx4 out0 = CONST_HX4(0);

        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);        
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);
        DSELHX4(y02, y13, y01, y23, dsel0);
        out0 = ADD_HX4(out0, y02);
        out0 = ADD_HX4(out0, y13);

        AE_SHX4IP(out0, (xthalfx4 *)scratch_j, 4 * sizeof(xthalf));
      }
    }
  }
  else if(kernel_width_pad == 8)
  {
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (xthalf *) p_scratch + (i * output_width_for_x_stride_1);
      int temp = output_width_for_x_stride_1 -output_width_for_x_stride_1%8;
      for(j = 0; j < temp; j+=8)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);
        CONST_HX4X2(acc4, acc5, 0);
        CONST_HX4X2(acc6, acc7, 0);
        for(k=0; k < kernel_height; k++)
        {
          xthalfx8 *pt_inp = (xthalfx8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp,((sizeof(xthalf)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);

          AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
          AE_LHX4X2_XC(d_inp2, d_inp3, pt_inp, sizeof(xthalf)*(input_width-8));
          AE_LHX4IP(d_ker0, pt_ker, 8);
          
          DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

          MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

          DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);

          MADDQ_H(acc4, acc5, d_inp1, d_inp6543, d_ker0);
          MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker0);

          AE_LHX4IP(d_ker1, pt_ker, 8);

          MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);

          DSELHX4(d_inp6543, d_inp5432, d_inp2, d_inp3, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp2, d_inp3);      

          MADDQ_H(acc4, acc5, d_inp2, d_inp6543, d_ker1);
          MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker1);
        }
  
        WORD16 *scratch_j = (WORD16 *)scratch_ptr + j;
        
        xthalfx4 out0, out1;
        CONST_HX4X2(out0, out1, 0);
        
        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);   
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);     
        DSELHX4(y02, y13, y01, y23, dsel0);
        DSELHX4(y0, y1, acc4, acc5, dsel0);     
        DSELHX4(y2, y3, acc6, acc7, dsel0);    
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);  
        xthalfx4 y02_1, y13_1;    
        DSELHX4(y02_1, y13_1, y01, y23, dsel0);

        ADD_HX4X2(out0, out1, out0, out1, y02, y02_1);
        ADD_HX4X2(out0, out1, out0, out1, y13, y13_1);

        AE_SHX4IP(out0, (xthalfx4 *)scratch_j, 4 * sizeof(xthalf));
        AE_SHX4IP(out1, (xthalfx4 *)scratch_j, 4 * sizeof(xthalf));
      }

      for(j=temp; j < output_width_for_x_stride_1; j+=4)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);        
        for(k=0; k < kernel_height; k++)
        {
          xthalfx8 *pt_inp = (xthalfx8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp,((sizeof(WORD16)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);
          AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
          d_inp2 = AE_LHX4I((xthalfx4 *)pt_inp, 0);
          AE_LHX4IP(d_ker0, pt_ker, 8);

          DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

          MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

          AE_LHX4IP(d_ker1, pt_ker, 8);     
          DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);    
          d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);
          
          MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);         
        }
        WORD16 *scratch_j = (WORD16 *)scratch_ptr + j;

        xthalfx4 out0 = CONST_HX4(0);
        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);        
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);
        DSELHX4(y02, y13, y01, y23, dsel0);
        out0 = ADD_HX4(out0, y02);
        out0 = ADD_HX4(out0, y13);

        AE_SHX4IP(out0, (xthalfx4 *)scratch_j, 4 * sizeof(xthalf));
      }
    }
  }
  else if(kernel_width_pad == 4)
  {
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (xthalf *) p_scratch + (i * output_width_for_x_stride_1);
      for(j=0; j < output_width_for_x_stride_1; j+=4)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);        
        for(k=0; k < kernel_height; k++)
        {
          xthalfx8 *pt_inp = (xthalfx8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp,((sizeof(WORD16)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);
          AE_LHX4XC(d_inp0, (xthalfx4 *)pt_inp, 8);
          AE_LHX4XC(d_inp1, (xthalfx4 *)pt_inp, sizeof(xthalf)*(input_width-4)); 

          AE_LHX4IP(d_ker0, pt_ker, 8);
          DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
          d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

          MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
          MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);       
        }
        WORD16 *scratch_j = (WORD16 *)scratch_ptr + j;

        xthalfx4 out0 = CONST_HX4(0);
        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);        
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);
        DSELHX4(y02, y13, y01, y23, dsel0);
        out0 = ADD_HX4(out0, y02);
        out0 = ADD_HX4(out0, y13);
        AE_SHX4IP(out0, (xthalfx4 *)scratch_j, 4 * sizeof(xthalf));
      }    
    }
  }
  else
  {
    for(i = 0; i < actual_out_height; i++)
    {
      scratch_ptr = (xthalf *) p_scratch + (i * output_width_for_x_stride_1);
      int temp = output_width_for_x_stride_1 -output_width_for_x_stride_1%8;
      for(j = 0; j < temp; j+=8)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);
        CONST_HX4X2(acc4, acc5, 0);
        CONST_HX4X2(acc6, acc7, 0);    
        for(k=0; k < kernel_height; k++)
        {
          xthalfx8 *pt_inp = (xthalfx8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp,((sizeof(xthalf)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);
  #pragma no_unroll
          for(l = 0; l < (kernel_width_pad>>3); l++)
          {
            AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
            AE_LHX4X2_I(d_inp2, d_inp3, pt_inp, 0);
            AE_LHX4IP(d_ker0, pt_ker, 8);
            
            DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

            MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

            DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);

            MADDQ_H(acc4, acc5, d_inp1, d_inp6543, d_ker0);
            MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker0);

            AE_LHX4IP(d_ker1, pt_ker, 8);

            MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);

            DSELHX4(d_inp6543, d_inp5432, d_inp2, d_inp3, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp2, d_inp3);      

            MADDQ_H(acc4, acc5, d_inp2, d_inp6543, d_ker1);
            MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker1);
          }
          if(kernel_width_pad&7)
          {
            AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
            d_inp2 = AE_LHX4I((xthalfx4 *)pt_inp, 0);
            d_ker0 = AE_LHX4I(pt_ker, 0);      

            DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

            MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

            DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);

            MADDQ_H(acc4, acc5, d_inp1, d_inp6543, d_ker0);
            MADDQ_H(acc6, acc7, d_inp5432, d_inp4321, d_ker0);
          }
        }
  
        WORD16 *scratch_j = (WORD16 *)scratch_ptr + j;
        
        xthalfx4 out0, out1;
        CONST_HX4X2(out0, out1, 0);
        
        DSELHX4(y0, y1, acc0, acc1, dsel0);
        DSELHX4(y2, y3, acc2, acc3, dsel0);   
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);     
        DSELHX4(y02, y13, y01, y23, dsel0);
        DSELHX4(y0, y1, acc4, acc5, dsel0);     
        DSELHX4(y2, y3, acc6, acc7, dsel0);    
        ADD_HX4X2(y01, y23, y0, y2, y1, y3);  
        xthalfx4 y02_1, y13_1;    
        DSELHX4(y02_1, y13_1, y01, y23, dsel0);

        ADD_HX4X2(out0, out1, out0, out1, y02, y02_1);
        ADD_HX4X2(out0, out1, out0, out1, y13, y13_1);

        AE_SHX4IP(out0, (xthalfx4 *)scratch_j, 4 * sizeof(xthalf));
        AE_SHX4IP(out1, (xthalfx4 *)scratch_j, 4 * sizeof(xthalf));
      }

      for(j=temp; j < output_width_for_x_stride_1; j+=4)
      {
        CONST_HX4X2(acc0, acc1, 0);
        CONST_HX4X2(acc2, acc3, 0);
        for(k=0; k < kernel_height; k++)
        {
          xthalfx8 *pt_inp = (xthalfx8 *)(p_inp);
          AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp,((sizeof(WORD16)) * ((i * y_stride * input_width) + j + k*input_width)));
          xthalfx4 *pt_ker = (xthalfx4 *)(p_ker + k*kernel_width_pad);
          for(l = 0; l < (kernel_width_pad>>3); l++)
          {
            AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
            d_inp2 = AE_LHX4I((xthalfx4 *)pt_inp, 0);
            AE_LHX4IP(d_ker0, pt_ker, 8);

            DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

            MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);

            AE_LHX4IP(d_ker1, pt_ker, 8);     
            DSELHX4(d_inp6543, d_inp5432, d_inp1, d_inp2, dsel1);    
            d_inp4321 = AE_SELH_4321(d_inp1, d_inp2);
            
            MADDQ_H(acc0, acc1, d_inp1, d_inp6543, d_ker1);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker1);         
          }
          if(kernel_width_pad&7)
          {
            AE_LHX4X2_XC(d_inp0, d_inp1, pt_inp, 16);
            d_ker0 = AE_LHX4I(pt_ker, 0); 

            DSELHX4(d_inp6543, d_inp5432, d_inp0, d_inp1, dsel1);
            d_inp4321 = AE_SELH_4321(d_inp0, d_inp1);

            MADDQ_H(acc0, acc1, d_inp0, d_inp6543, d_ker0);
            MADDQ_H(acc2, acc3, d_inp5432, d_inp4321, d_ker0);
          }
        }
        WORD16 *scratch_j = (WORD16 *)scratch_ptr + j;

        xthalfx4 out0 = CONST_HX4(0);

        DSELHX4(y0, y1, acc0, acc1, dsel0);
        y01  = ADD_HX4(y0, y1); 
        DSELHX4(y2, y3, acc2, acc3, dsel0);       
        y23  = ADD_HX4(y2, y3);   
        DSELHX4(y02, y13, y01, y23, dsel0);
        out0 = ADD_HX4(out0, y02);
        out0 = ADD_HX4(out0, y13);

        AE_SHX4IP(out0, (xthalfx4 *)scratch_j, 4 * sizeof(xthalf));
      }
    }
  }

  /* Here we store output based on strides. For values in a row, values
   * will be picked from it as per 'x_stride'. No need to worry about
   * height dimension, since we took care of it by efficient row
   * accesses. */

  xthalf acc_scratch;
  scratch_ptr = (xthalf *) p_scratch;
  for(i = 0; i < actual_out_height; i++)
  {
    scratch_ptr = (xthalf *) p_scratch + (i * output_width_for_x_stride_1);
    xthalf *out_ptr  = (xthalf *) p_out + (i * out_stride * actual_out_width);
    
    xthalf b0 = AE_LHI(((xthalf *)&bias), 0);

    for(j = 0; j < actual_out_width; j++)
    {
      acc_scratch = AE_LHX(scratch_ptr, (sizeof(xthalf) * (j * x_stride)));
      acc_scratch = ADD_H(acc_scratch, b0);
      AE_SHX(acc_scratch, out_ptr, (sizeof(xthalf) * (j * out_stride)));
    }
  }  
}

#define COPY_KERNEL_TO_SCRATCH(p_out, p_in, kh, kw, kw_pad) \
{ \
  int itr_kh, itr_kw; \
  for(itr_kh = 0; itr_kh < kh; itr_kh++) \
  { \
    xthalfx4 *pae_in = (xthalfx4 *)(&p_in[itr_kh * kw]); \
    xthalfx4 *pae_out = (xthalfx4 *)(&p_out[itr_kh * kw_pad]); \
    xthalfx4 d_tmp0; \
    ae_valign in_a = AE_LAHX4PP(pae_in); \
_Pragma("no_unroll") \
    for(itr_kw = 0; itr_kw < (kw >> 2); itr_kw++) \
    { \
      AE_LAHX4IP(d_tmp0, in_a, pae_in); \
      AE_SHX4IP(d_tmp0, pae_out, 4*sizeof(xthalf)); \
    } \
    if(kw & 3) \
    { \
      AE_LAHX4IP(d_tmp0, in_a, pae_in); \
      ae_int64 d_tmp64 = AE_MOVINT64_FROMINT16X4(AE_MOVINT16X4_FROMXTHALFX4 (d_tmp0)); \
      d_tmp64 = AE_SRAA64(d_tmp64, 16 * (4 - (kw & 3))); \
      d_tmp64 = AE_SLAA64(d_tmp64, 16 * (4 - (kw & 3))); \
      d_tmp0 = AE_MOVHALFX4_FROMF16X4(AE_MOVINT16X4_FROMINT64(d_tmp64)); \
      AE_SHX4IP(d_tmp0, (xthalfx4 *)pae_out, 4*sizeof(xthalf)); \
    } \
  } \
}

WORD32 xa_nn_conv2d_depthwise_nchw_f16(
    WORD16* __restrict__ p_out,
    const WORD16* __restrict__ p_kernel,
    const WORD16* __restrict__ p_inp,
    const WORD16* __restrict__ p_bias,
    WORD32  input_height,
    WORD32  input_width,
    WORD32  input_channels,
    WORD32  kernel_height,
    WORD32  kernel_width,
    WORD32  channels_multiplier,
    WORD32  x_stride,
    WORD32  y_stride,
    WORD32  x_padding,
    WORD32  y_padding,
    WORD32  out_height,
    WORD32  out_width,
    WORD32  out_data_format,
    pVOID p_scratch)
{
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
        ,-2
        ,1
        ,(pVOID)(&pad_val)
        );

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ic, itr_cm, itr_oh;
    int circ_out_height = (p_circ_buf->rows - kernel_height)/y_stride + 1;
    int kernel_height_pad = ALIGNED_SIZE(kernel_height, 2);
    int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
    int rows_to_add, top_pad, bottom_pad, rows_added;
    int input_row;
    const WORD16 *pt_ker;
    const WORD16 *pt_inp;
    pWORD16 p_inp_circ;
    int i;
    WORD16 *p_kernel_padded = (WORD16 *)(p_state->p_scratch);
    p_kernel_padded = (WORD16 *)ALIGN_PTR(p_kernel_padded, 8);
    pWORD64 p_tmp_out = (pWORD64)(p_kernel_padded + kernel_height_pad * kernel_width_pad);
    p_tmp_out = (pWORD64)ALIGN_PTR(p_tmp_out, 16);

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    WORD16 bias = 0;
    /* Initialize whole scratch for padded kernel to padding value, after this
     we only have to copy actual kernel values, padding area should remain
     untouched */
    xthalfx4 *pae_ker_pad = (xthalfx4 *)p_kernel_padded;

    for(i = 0; i < ((kernel_height_pad * kernel_width_pad) >> 2); i++)
    {
        pae_ker_pad[i] = ZERO_HX4();
    }
    
    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = &p_inp[itr_ic*input_height*input_width];
        for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
        {
            pt_ker = &p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width];
            COPY_KERNEL_TO_SCRATCH(p_kernel_padded, pt_ker, kernel_height, kernel_width, kernel_width_pad);
            bias = p_bias[(itr_ic*channels_multiplier+itr_cm)];

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
                              
                p_inp_circ = (WORD16 *)p_circ_buf->p_curr;
                convolve_f16
                ((&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
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
                            ,p_tmp_out
                            );
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

            p_inp_circ = (WORD16 *)p_circ_buf->p_curr;
            convolve_f16
            ((&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)])
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
                        ,p_tmp_out
                        );
        }
    }

    return 0;
}

static inline void conv2d_nhwc_f16
(pWORD16 __restrict__ p_out
 ,const WORD16 *__restrict__ p_ker
 ,const WORD16 *__restrict__ p_inp
 ,const WORD16 *p_bias
 ,int kernel_height
 ,int kernel_width
 ,int out_height
 ,int out_width
 ,int out_channels
 ,int x_stride
 ,int y_stride
 ,pWORD32 __restrict__ p_scratch
 )
{
    (VOID) x_stride;
    (VOID) p_scratch;
    WORD32 ker_channels_pad, inp_channels_pad;
    WORD32 itr_oh, itr_ch, itr_kw;
    xthalfx4 *pt_inp0, *pt_inp1, *pt_ker;
    xthalfx8 *out_ptr0, *out_ptr1;

    xthalfx4 d_inp0, d_inp1, d_ker;

    const xthalfx4 *pt_bias;
    ae_valign ker_a;
    ae_valign bias_a;
    ae_valignx2 out0_a, out1_a;

    xthalfx4 d_acc0123, d_acc4567;
    xthalfx4 d_acc0123_1, d_acc4567_1;
    xthalfx4 d_bias0123, d_bias0123_1;

    ker_channels_pad = out_channels;
    inp_channels_pad = (out_channels + 3) & (~3);

    for(itr_oh = 0; itr_oh < (out_height-1); itr_oh+=2)
    {
      out_ptr0 = (xthalfx8 *)(&p_out[itr_oh*out_channels*out_width]);
      out_ptr1 = (xthalfx8 *)(&p_out[(itr_oh+1)*out_channels*out_width]);
      pt_bias = (xthalfx4 *)p_bias;
      bias_a = AE_LAHX4PP(pt_bias);
      out0_a = AE_ZALIGN128();
      out1_a = AE_ZALIGN128();        

      if ((inp_channels_pad%8) == 0)
      {
        for(itr_ch = 0; itr_ch < out_channels; itr_ch+=8)
        {
            xthalfx4 d_ker1;
            pt_inp0 = (xthalfx4 *)p_inp;
            pt_inp1 = (xthalfx4 *)p_inp;
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, (itr_ch + itr_oh*y_stride*kernel_width*inp_channels_pad)*sizeof(xthalf));
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, (itr_ch + (itr_oh+1)*y_stride*kernel_width*inp_channels_pad)*sizeof(xthalf));
            xthalfx8 *pt_kerx2 = (xthalfx8 *)(&p_ker[itr_ch]);
            ae_valignx2 ker_ax2 = AE_LA128_PP(pt_kerx2);
            
            AE_LAHX4IP(d_bias0123, bias_a, pt_bias);            
            AE_LAHX4IP(d_bias0123_1, bias_a, pt_bias);            
            d_acc0123 = d_bias0123;
            d_acc4567 = d_bias0123;
            d_acc0123_1 = d_bias0123_1;
            d_acc4567_1 = d_bias0123_1;

            for(itr_kw = 0; itr_kw < kernel_width * kernel_height; itr_kw++)
            {
                xthalfx4 d_inp01, d_inp11;
                AE_LHX4X2_XC(d_inp0, d_inp01, (xthalfx8 *)pt_inp0, inp_channels_pad*sizeof(xthalf));
                AE_LHX4X2_XC(d_inp1, d_inp11, (xthalfx8 *)pt_inp1, inp_channels_pad*sizeof(xthalf));

                AE_LAHX4X2_IP(d_ker, d_ker1, ker_ax2, pt_kerx2);
            
                pt_kerx2 = (xthalfx8 *)((WORD8 *)pt_kerx2 + sizeof(xthalf) * (ker_channels_pad - 8));
                ker_ax2 = AE_LA128_PP(pt_kerx2);
                MADDQ_H(d_acc0123, d_acc4567, d_inp0, d_inp1, d_ker);
                MADDQ_H(d_acc0123_1, d_acc4567_1, d_inp01, d_inp11, d_ker1);
            }
            AE_SAVHX4X2_XP(d_acc0123, d_acc0123_1, out0_a, out_ptr0, (XT_MIN(out_channels-itr_ch, 8) << 1));
            AE_SAVHX4X2_XP(d_acc4567, d_acc4567_1, out1_a, out_ptr1, (XT_MIN(out_channels-itr_ch, 8) << 1));   
        }
      }
      else 
      {
        for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
        {
            pt_inp0 = (xthalfx4 *)p_inp;
            pt_inp1 = (xthalfx4 *)p_inp;
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, (itr_ch + itr_oh*y_stride*kernel_width*inp_channels_pad)*sizeof(xthalf));
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp1, (itr_ch + (itr_oh+1)*y_stride*kernel_width*inp_channels_pad)*sizeof(xthalf));
            pt_ker = (xthalfx4 *)(&p_ker[itr_ch]);
            ker_a = AE_LAHX4PP(pt_ker);
            
            AE_LAHX4IP(d_bias0123, bias_a, pt_bias);            
            d_acc0123 = d_bias0123;
            d_acc4567 = d_bias0123;

            for(itr_kw = 0; itr_kw < kernel_width * kernel_height; itr_kw++)
            {
                AE_LHX4XC(d_inp0, pt_inp0, inp_channels_pad*sizeof(xthalf));
                AE_LHX4XC(d_inp1, pt_inp1, inp_channels_pad*sizeof(xthalf));

                AE_LAHX4IP(d_ker, ker_a, pt_ker);
            
                pt_ker = (xthalfx4 *)((WORD8 *)pt_ker + sizeof(xthalf) * (ker_channels_pad - 4));
                ker_a = AE_LAHX4PP(pt_ker);
                MADDQ_H(d_acc0123, d_acc4567, d_inp0, d_inp1, d_ker);
            }
            AE_SAVHX4X2_XP(d_acc0123, ZERO_HX4(), out0_a, out_ptr0, (XT_MIN(out_channels-itr_ch, 4) << 1));
            AE_SAVHX4X2_XP(d_acc4567, ZERO_HX4(), out1_a, out_ptr1, (XT_MIN(out_channels-itr_ch, 4) << 1));   
        }
      }

      AE_SA128POS_FP(out0_a, out_ptr0);
      AE_SA128POS_FP(out1_a, out_ptr1);
    }
    if(itr_oh < out_height)
    {
        out_ptr0 = (xthalfx8 *)(&p_out[itr_oh*out_channels*out_width]);
        pt_bias = (const xthalfx4 *)p_bias;
        bias_a = AE_LAHX4PP(pt_bias);   
        out0_a = AE_ZALIGN128();     
        for(itr_ch = 0; itr_ch < out_channels; itr_ch+=4)
        {
            pt_inp0 = (xthalfx4 *)p_inp;
            AE_ADDCIRC16X4_XC((ae_int16x4 *)pt_inp0, (itr_ch + itr_oh*y_stride*kernel_width*inp_channels_pad)*sizeof(xthalf));
            pt_ker = (xthalfx4 *)(&p_ker[itr_ch]);
            ker_a = AE_LAHX4PP(pt_ker);
            AE_LAHX4IP(d_bias0123, bias_a, pt_bias);
            d_acc0123 = d_bias0123;

            for(itr_kw = 0; itr_kw < kernel_width * kernel_height; itr_kw++)
            {
                AE_LHX4XC(d_inp0, pt_inp0, inp_channels_pad*sizeof(xthalf));
                AE_LAHX4IP(d_ker, ker_a, pt_ker);                 

                pt_ker = (xthalfx4 *)((WORD8 *)pt_ker + sizeof(xthalf) * (ker_channels_pad - 4));
                ker_a = AE_LAHX4PP(pt_ker);
                MADD_HX4(d_acc0123, d_ker, d_inp0);
            }
            AE_SAVHX4X2_XP(d_acc0123, ZERO_HX4(), out0_a, out_ptr0, (XT_MIN(out_channels-itr_ch, 4) << 1));
        }
        AE_SA128POS_FP(out0_a, out_ptr0);
    }
}

static void xa_nn_conv2d_depthwise_nhwc_f16
(pWORD16 __restrict__ p_out
 ,const WORD16 *__restrict__ p_kernel
 ,const WORD16 *__restrict__ p_inp
 ,const WORD16 *__restrict__ p_bias
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
 ,WORD32  out_data_format
 ,pVOID p_scratch
)
{
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
         ,-2
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

        conv2d_nhwc_f16
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
             ,p_scratch
            );
    }
}


WORD32 xa_nn_conv2d_depthwise_f16(
        WORD16* __restrict__ p_out,
        const WORD16* __restrict__ p_kernel,
        const WORD16* __restrict__ p_inp,
        const WORD16* __restrict__ p_bias,
        WORD32  input_height,
        WORD32  input_width,
        WORD32  input_channels,
        WORD32  kernel_height,
        WORD32  kernel_width,
        WORD32  channels_multiplier,
        WORD32  x_stride,
        WORD32  y_stride,
        WORD32  x_padding,
        WORD32  y_padding,
        WORD32  out_height,
        WORD32  out_width,
        WORD32  inp_data_format,
        WORD32  out_data_format,
        pVOID p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0 && inp_data_format != 1), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);

    if(inp_data_format == 0)
    {
        xa_nn_conv2d_depthwise_nhwc_f16
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
             ,out_data_format
             ,p_scratch);      
    }
    else if(inp_data_format == 1)
    {
        xa_nn_conv2d_depthwise_nchw_f16(
                p_out,
                p_kernel,
                p_inp,
                p_bias,
                input_height,
                input_width,
                input_channels,
                kernel_height,
                kernel_width,
                channels_multiplier,
                x_stride,
                y_stride,
                x_padding,
                y_padding,
                out_height,
                out_width,
                out_data_format,
                p_scratch);
    }
    return 0;
}
#endif /* #if !HAVE_HP_VFPU */
