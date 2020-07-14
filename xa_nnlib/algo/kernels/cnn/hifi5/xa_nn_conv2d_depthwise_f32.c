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
#include "common_fpu.h"
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_conv2d_depthwise_state.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_err_chk.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_conv2d_depthwise_f32,(
    FLOAT32* __restrict__ p_out,
    FLOAT32* __restrict__ p_kernel,
    FLOAT32* __restrict__ p_inp,
    FLOAT32* __restrict__ p_bias,
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
    pVOID p_scratch))
#else /* #if !HAVE_VFPU */
static void convolve_f32(
    FLOAT32*  __restrict__ p_out,
    FLOAT32* __restrict__ p_ker,
    FLOAT32* __restrict__ p_inp,
    FLOAT32* __restrict__ p_bias,
    WORD32   input_width,
    WORD32   kernel_height,
    WORD32   kernel_width,
    WORD32   x_stride,
    WORD32   y_stride,
    WORD32   out_height,
    WORD32   out_width,
    WORD32   out_stride,
    pVOID    p_scratch)
{
    int itr_oh, itr_ow, itr_kh, itr_kw;
    int total_out_width = (input_width - kernel_width) + 1;
    int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
    xtfloatx4 *ptr_inp;
    xtfloatx4 *ptr_ker;
    xtfloatx2 *ptr_out;

    xtfloatx2 ker0, ker1, ker2, ker3, ker4, ker5;
    xtfloatx2 accu_x2_0, accu_x2_1;
    xtfloatx2 accu_x2_0_a, accu_x2_1_a;
    xtfloatx2 accu_x2_0_b, accu_x2_1_b;
    xtfloatx2 accu_x2_0_c, accu_x2_1_c;
    xtfloatx2 id4, id8, id12, id16, id20, id24, id28, id32;
    xtfloatx2 id5, id6, id7, id13, id14, id15, id21, id22, id23;

  if(kernel_width_pad == 12)
  {
    for(itr_oh=0; itr_oh<out_height; itr_oh++)
    {
        ptr_out = (xtfloatx2 *)p_scratch;
        for(itr_ow=0; itr_ow<((total_out_width+3)>>2); itr_ow++)
        {
            accu_x2_0 = CONST_S(0);
            accu_x2_1 = CONST_S(0);
            accu_x2_0_a = CONST_S(0);
            accu_x2_1_a = CONST_S(0);
            accu_x2_0_b = CONST_S(0);
            accu_x2_1_b = CONST_S(0);
            accu_x2_0_c = CONST_S(0);
            accu_x2_1_c = CONST_S(0);

            ptr_ker = (xtfloatx4 *)p_ker;
            ptr_inp = (xtfloatx4 *)p_inp;
            AE_ADDCIRC16X4_XC((ae_int16x4 *)ptr_inp, ((itr_oh*y_stride)*input_width+4*itr_ow)*sizeof(FLOAT32));
#pragma loop_count min=1
#pragma no_unroll
            for(itr_kh=0; itr_kh<kernel_height; itr_kh++)
            {
               
                //Input loads
                AE_LSX2X2_XC(id4,id8,(xtfloatx4 *)ptr_inp,16); 
                AE_LSX2X2_XC(id12,id16,(xtfloatx4 *)ptr_inp,16); 
                AE_LSX2X2_XC(id20,id24,(xtfloatx4 *)ptr_inp,16); 
                AE_LSX2X2_XC(id28,id32,(xtfloatx4 *)ptr_inp,sizeof(FLOAT32)*(input_width - 12)); 
                
                //Kernel Loads
                AE_LSX2X2_I(ker2,ker3, ptr_ker,16);
                AE_LSX2X2_I(ker4,ker5, ptr_ker,32);
                AE_LSX2X2_IP(ker0,ker1, ptr_ker,3*sizeof(xtfloatx4));
                    
                id5 = XT_SEL32_HL_SX2(id8, id4);
                id6 = XT_SEL32_HL_SX2(id12, id8);
                id13 = id7 = XT_SEL32_HL_SX2(id16, id12);
                id14 = XT_SEL32_HL_SX2(id20, id16);
                id21 = id15 = XT_SEL32_HL_SX2(id24, id20);
                id22 = XT_SEL32_HL_SX2(id28, id24);
                id23 = XT_SEL32_HL_SX2(id32, id28);
               
                MADDMUX_SX2X2(accu_x2_0,accu_x2_1,ker0,ker0,id4,id8,0);
                MADDMUX_SX2X2(accu_x2_0_a,accu_x2_1_a,ker0,ker0,id5,id6,5);  
                MADDMUX_SX2X2(accu_x2_0_b,accu_x2_1_b,ker1,ker1,id8,id12,0);
                MADDMUX_SX2X2(accu_x2_0_c,accu_x2_1_c,ker1,ker1,id6,id7,5);
                
                MADDMUX_SX2X2(accu_x2_0,accu_x2_1,ker2,ker2,id12,id16,0);
                MADDMUX_SX2X2(accu_x2_0_a,accu_x2_1_a,ker2,ker2,id13,id14,5);  
                MADDMUX_SX2X2(accu_x2_0_b,accu_x2_1_b,ker3,ker3,id16,id20,0);
                MADDMUX_SX2X2(accu_x2_0_c,accu_x2_1_c,ker3,ker3,id14,id15,5);
                
                MADDMUX_SX2X2(accu_x2_0,accu_x2_1,ker4,ker4,id20,id24,0);
                MADDMUX_SX2X2(accu_x2_0_a,accu_x2_1_a,ker4,ker4,id21,id22,5);  
                MADDMUX_SX2X2(accu_x2_0_b,accu_x2_1_b,ker5,ker5,id24,id28,0);
                MADDMUX_SX2X2(accu_x2_0_c,accu_x2_1_c,ker5,ker5,id22,id23,5);
                
            }
            accu_x2_0 += accu_x2_0_a;
            accu_x2_0_b += accu_x2_0_c;
            accu_x2_1 += accu_x2_1_a;
            accu_x2_1_b += accu_x2_1_c;
            accu_x2_0 += accu_x2_0_b;
            accu_x2_1 += accu_x2_1_b;

            *ptr_out++ = accu_x2_0;
            *ptr_out++ = accu_x2_1;
        }

        float *ptr_out1 = (float *)p_scratch;
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            p_out[itr_oh*out_width*out_stride+itr_ow*out_stride] = ptr_out1[itr_ow*x_stride] + p_bias[0];
        }
    }    
  }
  else if(kernel_width_pad == 8)
  {
    for(itr_oh=0; itr_oh<out_height; itr_oh++)
    {
        ptr_out = (xtfloatx2 *)p_scratch;
        for(itr_ow=0; itr_ow<((total_out_width+3)>>2); itr_ow++)
        {
            accu_x2_0 = CONST_S(0);
            accu_x2_1 = CONST_S(0);
            accu_x2_0_a = CONST_S(0);
            accu_x2_1_a = CONST_S(0);
            accu_x2_0_b = CONST_S(0);
            accu_x2_1_b = CONST_S(0);
            accu_x2_0_c = CONST_S(0);
            accu_x2_1_c = CONST_S(0);

            ptr_ker = (xtfloatx4 *)p_ker;
            ptr_inp = (xtfloatx4 *)p_inp;
            AE_ADDCIRC16X4_XC((ae_int16x4 *)ptr_inp, ((itr_oh*y_stride)*input_width+4*itr_ow)*sizeof(FLOAT32));
#pragma loop_count min=1
#pragma no_unroll
            for(itr_kh=0; itr_kh<kernel_height; itr_kh++)
            {
               
                //Input loads
                AE_LSX2X2_XC(id4,id8,(xtfloatx4 *)ptr_inp,16); 
                AE_LSX2X2_XC(id12,id16,(xtfloatx4 *)ptr_inp,16); 
                AE_LSX2X2_XC(id20,id24,(xtfloatx4 *)ptr_inp,sizeof(FLOAT32)*(input_width - 8)); 
                
                //Kernel Loads
                AE_LSX2X2_I(ker2,ker3, ptr_ker,16);
                AE_LSX2X2_IP(ker0,ker1, ptr_ker,2*sizeof(xtfloatx4));
                    
                id5 = XT_SEL32_HL_SX2(id8, id4);
                id6 = XT_SEL32_HL_SX2(id12, id8);
                id13 = id7 = XT_SEL32_HL_SX2(id16, id12);
                id14 = XT_SEL32_HL_SX2(id20, id16);
                id15 = XT_SEL32_HL_SX2(id24, id20);
               
                MADDMUX_SX2X2(accu_x2_0,accu_x2_1,ker0,ker0,id4,id8,0);
                MADDMUX_SX2X2(accu_x2_0_a,accu_x2_1_a,ker0,ker0,id5,id6,5);  
                MADDMUX_SX2X2(accu_x2_0_b,accu_x2_1_b,ker1,ker1,id8,id12,0);
                MADDMUX_SX2X2(accu_x2_0_c,accu_x2_1_c,ker1,ker1,id6,id7,5);
                
                MADDMUX_SX2X2(accu_x2_0,accu_x2_1,ker2,ker2,id12,id16,0);
                MADDMUX_SX2X2(accu_x2_0_a,accu_x2_1_a,ker2,ker2,id13,id14,5);  
                MADDMUX_SX2X2(accu_x2_0_b,accu_x2_1_b,ker3,ker3,id16,id20,0);
                MADDMUX_SX2X2(accu_x2_0_c,accu_x2_1_c,ker3,ker3,id14,id15,5);
                
            }
            accu_x2_0 += accu_x2_0_a;
            accu_x2_0_b += accu_x2_0_c;
            accu_x2_1 += accu_x2_1_a;
            accu_x2_1_b += accu_x2_1_c;
            accu_x2_0 += accu_x2_0_b;
            accu_x2_1 += accu_x2_1_b;

            *ptr_out++ = accu_x2_0;
            *ptr_out++ = accu_x2_1;
        }

        float *ptr_out1 = (float *)p_scratch;
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            p_out[itr_oh*out_width*out_stride+itr_ow*out_stride] = ptr_out1[itr_ow*x_stride] + p_bias[0];
        }
    }    
  }
  else
  {
    /* No reminder loop, run extra iteration, extra output will be thrown away
    when we pick correct outputs using x_stride */
    for(itr_oh=0; itr_oh<out_height; itr_oh++)
    {
        ptr_out = (xtfloatx2 *)p_scratch;
        for(itr_ow=0; itr_ow<((total_out_width+3)>>2); itr_ow++)
        {
            accu_x2_0 = CONST_S(0);
            accu_x2_1 = CONST_S(0);
            accu_x2_0_a = CONST_S(0);
            accu_x2_1_a = CONST_S(0);
            accu_x2_0_b = CONST_S(0);
            accu_x2_1_b = CONST_S(0);
            accu_x2_0_c = CONST_S(0);
            accu_x2_1_c = CONST_S(0);

            ptr_ker = (xtfloatx4 *)p_ker;
#pragma loop_count min=1
            for(itr_kh=0; itr_kh<kernel_height; itr_kh++)
            {
                ptr_inp = (xtfloatx4 *)p_inp;
                AE_ADDCIRC16X4_XC((ae_int16x4 *)ptr_inp, ((itr_kh+itr_oh*y_stride)*input_width+4*itr_ow)*sizeof(FLOAT32));
#pragma loop_count min=1
#pragma no_unroll
                
                AE_LSX2X2_XC(id4,id8,(xtfloatx4 *)ptr_inp,16); 
                for(itr_kw=0; itr_kw<(kernel_width_pad>>2); itr_kw++)
                {
                    AE_LSX2X2_IP(ker0,ker1, ptr_ker,sizeof(xtfloatx4));
                  //  AE_LSX2XC(id4,ptr_inp,8);
                  //  AE_LSX2XC(id8,ptr_inp,8);
                  //  AE_LSX2XC(id12,ptr_inp,8);
                  //  AE_LSX2XC(id16,ptr_inp,-8);
                    AE_LSX2X2_XC(id12,id16,(xtfloatx4 *)ptr_inp,16); 
                    id5 = XT_SEL32_HL_SX2(id8, id4);
                    id6 = XT_SEL32_HL_SX2(id12, id8);
                    id7 = XT_SEL32_HL_SX2(id16, id12);
                    MADDMUX_SX2X2(accu_x2_0,accu_x2_1,ker0,ker0,id4,id8,0);
                    MADDMUX_SX2X2(accu_x2_0_a,accu_x2_1_a,ker0,ker0,id5,id6,5);  
                    MADDMUX_SX2X2(accu_x2_0_b,accu_x2_1_b,ker1,ker1,id8,id12,0);
                    MADDMUX_SX2X2(accu_x2_0_c,accu_x2_1_c,ker1,ker1,id6,id7,5);
                    id4=id12;
                    id8=id16;

                }
            }
            accu_x2_0 += accu_x2_0_a;
            accu_x2_0_b += accu_x2_0_c;
            accu_x2_1 += accu_x2_1_a;
            accu_x2_1_b += accu_x2_1_c;
            accu_x2_0 += accu_x2_0_b;
            accu_x2_1 += accu_x2_1_b;

            *ptr_out++ = accu_x2_0;
            *ptr_out++ = accu_x2_1;
        }

        float *ptr_out1 = (float *)p_scratch;
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            p_out[itr_oh*out_width*out_stride+itr_ow*out_stride] = ptr_out1[itr_ow*x_stride] + p_bias[0];
        }
    }
  }
}

WORD32 xa_nn_conv2d_depthwise_f32(
    FLOAT32* __restrict__ p_out,
    FLOAT32* __restrict__ p_kernel,
    FLOAT32* __restrict__ p_inp,
    FLOAT32* __restrict__ p_bias,
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
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, ALIGNMENT, -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_kernel, ALIGNMENT, -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, ALIGNMENT, -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, ALIGNMENT, -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
    XA_NNLIB_ARG_CHK_COND((channels_multiplier <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND((y_stride > kernel_height), -1);
    XA_NNLIB_ARG_CHK_COND((x_stride > kernel_width), -1);

    xa_nn_conv2d_depthwise_init
        (p_scratch
        ,input_width
        ,kernel_height
        ,kernel_width
        ,x_stride
        ,y_stride
        ,x_padding
        ,out_width
        ,-1
        );

    xa_nn_conv2d_dw_state_t *p_state = (xa_nn_conv2d_dw_state_t *)p_scratch;
    xa_nn_circ_buf_t *p_circ_buf = &(p_state->circ_buf);
    int itr_ic, itr_cm, itr_oh;
    int circ_out_height = (p_circ_buf->rows - kernel_height)/y_stride + 1;
    int kernel_width_pad = ALIGNED_SIZE(kernel_width, 4);
    int rows_to_add, top_pad, bottom_pad, rows_added;
    int input_row;
    FLOAT32 *pt_inp, *pt_ker;
    FLOAT32 *p_inp_circ;
    p_scratch = (FLOAT32 *)(p_state->p_scratch);

    AE_SETCBEGIN0(p_circ_buf->p_begin);
    AE_SETCEND0(p_circ_buf->p_end);

    for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
    {
        pt_inp = &p_inp[itr_ic*input_height*input_width];
        for(itr_cm = 0; itr_cm < channels_multiplier; itr_cm++)
        {
            pt_ker = &p_kernel[(itr_ic*channels_multiplier+itr_cm)*kernel_height*kernel_width_pad];

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
                                   )
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
                              )
                p_inp_circ = (FLOAT32 *)p_circ_buf->p_curr;
                convolve_f32(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)]
                            ,pt_ker
                            ,p_inp_circ
                            ,&p_bias[itr_ic*channels_multiplier+itr_cm]
                            ,p_circ_buf->row_offset
                            ,kernel_height
                            ,kernel_width
                            ,x_stride
                            ,y_stride
                            ,circ_out_height
                            ,out_width
                            ,input_channels*channels_multiplier
                            ,p_scratch
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
                              )
            p_inp_circ = (FLOAT32 *)p_circ_buf->p_curr;
            convolve_f32(&p_out[(itr_ic*channels_multiplier+itr_cm)+itr_oh*out_width*(input_channels*channels_multiplier)]
                        ,pt_ker
                        ,p_inp_circ
                        ,&p_bias[itr_ic*channels_multiplier+itr_cm]
                        ,p_circ_buf->row_offset
                        ,kernel_height
                        ,kernel_width
                        ,x_stride
                        ,y_stride
                        ,(out_height-itr_oh)
                        ,out_width
                        ,input_channels*channels_multiplier
                        ,p_scratch
                        );
        }
    }
    return 0;
}
#endif /* #if !HAVE_VFPU */
