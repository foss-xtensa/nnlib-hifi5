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
#include "xa_nnlib_common.h"
#include <string.h>
#include "xa_nnlib_common_macros_hifi5.h"

#define ALIGNMENT_16   16   /* 16 bytes alignment */

#define ALIGNED_SIZE(x, bytes)  (((x)+(bytes-1))&(~(bytes-1)))

#define ALIGN_PTR(x, bytes)     ((((unsigned)(x))+(bytes-1))&(~(bytes-1)))

#define BUS_WIDTH_16 15

WORD32 xa_nn_reduce_getsize_nhwc(WORD32 inp_precision
                                 ,const WORD32 *const p_inp_shape
                                 ,WORD32 num_inp_dims
                                 ,const WORD32 *p_axis
                                 ,WORD32 num_axis_dims
                                 ,WORD32 reduce_ops)
{
    int scratch_bytewidth;
    /*Optimizing scratch requirement by calculating largest input dims in axis*/
    if(num_axis_dims)
    {
      int inp_shape_max = p_inp_shape[p_axis[0]];
      int axis_itr = 1;
      for(axis_itr = 1; axis_itr < num_axis_dims; axis_itr++)
      {
        inp_shape_max = (p_inp_shape[p_axis[axis_itr]] > inp_shape_max) ? p_inp_shape[p_axis[axis_itr]] : inp_shape_max;
      }

      int inp_length = 1, inp_itr = 0;
      for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
      {
        inp_length *= p_inp_shape[inp_itr];
      }

      if((inp_precision == -4) && (reduce_ops == REDUCE_MAX)) /*For Reduce Max*/
      {
        scratch_bytewidth = sizeof(WORD8);
        if(inp_shape_max)
        {
          return (ALIGNED_SIZE(((inp_length / inp_shape_max) * scratch_bytewidth), ALIGNMENT_16) + (BUS_WIDTH_16));
        }
      }
      else if((inp_precision == -4) && (reduce_ops == REDUCE_MEAN)) /*For Reduce Mean*/
      {
        scratch_bytewidth = sizeof(WORD32);
        if(inp_shape_max)
        {
          return (ALIGNED_SIZE(((inp_length / inp_shape_max) * scratch_bytewidth), ALIGNMENT_16) + (BUS_WIDTH_16));
        }
      }
    }

    return 0;
}

/*
 * Currently only supports upto 4D input tensors.
 * 1/2/3 D input tensors will be scaled up to 4D.
 * For example, 2x3 -> 1x1x2x3.
 * Currently TFLM reduce max operator requires input and output
 * quantization to be same. Therefore, the kernel does not involve
 * quantization.
 */

WORD32 xa_nn_reduce_max_4D_asym8s_asym8s(WORD8 * __restrict__ p_out
                                        ,const WORD32 *const p_out_shape
                                        ,const WORD8 * __restrict__ p_inp
                                        ,const WORD32 *const p_inp_shape
                                        ,const WORD32 * __restrict__ p_axis
                                        ,WORD32 num_out_dims
                                        ,WORD32 num_inp_dims
                                        ,WORD32 num_axis_dims
                                        ,pVOID p_scratch_in)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_axis, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);

  /* Invalid input checks */
  XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND(((num_out_dims <= 0) || (num_out_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND(((num_axis_dims < 0) || (num_axis_dims > 4)), -1);

  int axis_itr = 0, inp_itr = 0, out_itr = 0;
  for(axis_itr=0; axis_itr < num_axis_dims; axis_itr++)
  {
    XA_NNLIB_ARG_CHK_COND(((p_axis[axis_itr] < 0) || (p_axis[axis_itr] > (num_inp_dims - 1))), -1);
  }

  for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[inp_itr] <= 0), -1);
  }

  int out_length = 1;
  for(out_itr=0; out_itr < num_out_dims; out_itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shape[out_itr] <= 0), -1);
    out_length *= p_out_shape[out_itr];
  }

  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_axis, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);

  WORD8 *p_in = (WORD8 *)(p_inp);
  WORD8 *p_scratch = (WORD8 *)(p_scratch_in);

  // Changing order of axis data so that reduce max will be first computed
  // across largest inp shape dim in axis. This is required to
  // minimize the scratch usage.
  int inp_length = 1, p_axis_data[4], inp_shape_max;
  if(num_axis_dims)
  {
    inp_shape_max = p_inp_shape[p_axis[0]];
    int axis_itr = 1, max_axis_itr = 0;
    int temp_p_axis_0 = p_axis[0];
    for(axis_itr = 0; axis_itr < num_axis_dims; axis_itr++)
    {
      p_axis_data[axis_itr] = p_axis[axis_itr];
    }
    for(axis_itr = 1; axis_itr < num_axis_dims; axis_itr++)
    {
      if(p_inp_shape[p_axis[axis_itr]] > inp_shape_max)
      {
        inp_shape_max = p_inp_shape[p_axis[axis_itr]];
        max_axis_itr = axis_itr;
      }
    }
    p_axis_data[0] = p_axis_data[max_axis_itr];
    p_axis_data[max_axis_itr] = temp_p_axis_0;

    int inp_itr = 0;
    for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
    {
      inp_length *= p_inp_shape[inp_itr];
    }

    memset(p_scratch, -128, (inp_length / inp_shape_max)); //TODO: Alternate approach for memset?
  }

  // Promoting lesser dim tensors to 4D tensors. Also modifying axis
  // data accordingly.
  int p_4D_inp_shape[4] = {1, 1, 1, 1};
  int itr = num_inp_dims - 1;
  int count = 3;
  while(itr >= 0)
  {
    p_4D_inp_shape[count] = p_inp_shape[itr];
    itr--;
    count--;
  }
  for(itr = 0; itr < num_axis_dims; itr++)
  {
    p_axis_data[itr] = p_axis_data[itr] + (4 - num_inp_dims);
  }

  int temp_inp_n = p_4D_inp_shape[0]; 
  int temp_inp_h = p_4D_inp_shape[1]; 
  int temp_inp_w = p_4D_inp_shape[2]; 
  int temp_inp_c = p_4D_inp_shape[3];

  int flag = 0;
  int itr_axis, itr_n, itr_h, itr_w, itr_c;
  ae_int8x16 *p_src1, *p_src2, *p_src3;
  ae_int8x16 * p_dst;
  ae_valignx2 align_src1, align_src2, align_src3, align_dst;

  align_dst = AE_ZALIGN128();
  for(itr_axis=0; itr_axis < num_axis_dims; itr_axis++)
  {
    switch(p_axis_data[itr_axis])
    {
      case 0: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int rem_hwc = (plane_size & 15); 
        for(itr_n=0; itr_n < (temp_inp_n & ~(2 - 1)); itr_n += 2)
        {
          p_src1 = (ae_int8x16 *)(p_scratch);
          p_src2 = (ae_int8x16 *)(p_in + itr_n * plane_size);
          p_src3 = (ae_int8x16 *)(p_in + (itr_n + 1) * plane_size);
          p_dst  = (ae_int8x16 *)(p_scratch);
          align_src1 = AE_LA128_PP(p_src1);
          align_src2 = AE_LA128_PP(p_src2);
          align_src3 = AE_LA128_PP(p_src3);

          int itr_hwc = 0;
          for(itr_hwc=0; itr_hwc < (plane_size >> 4); itr_hwc++)
          {
            ae_int8x8 i1, i2, j1, j2, k1, k2, out1, out2, out3, out4;
            AE_LA8X8X2_IP(i1, i2, align_src1, p_src1);
            AE_LA8X8X2_IP(j1, j2, align_src2, p_src2);
            AE_LA8X8X2_IP(k1, k2, align_src3, p_src3);
            out1 = AE_MAX8(i1, j1);
            out2 = AE_MAX8(i2, j2);
            out3 = AE_MAX8(out1, k1);
            out4 = AE_MAX8(out2, k2);
            AE_SA8X8X2_IP(out3, out4, align_dst, p_dst);
          }

          //Remainder Loop
          if(rem_hwc)
          {
            ae_int8x8 i1, i2, j1, j2, k1, k2, out1, out2, out3, out4;
            AE_LAV8X8X2_XP(i1, i2, align_src1, p_src1, rem_hwc);
            AE_LAV8X8X2_XP(j1, j2, align_src2, p_src2, rem_hwc);
            AE_LAV8X8X2_XP(k1, k2, align_src3, p_src3, rem_hwc);
            out1 = AE_MAX8(i1, j1);
            out2 = AE_MAX8(i2, j2);
            out3 = AE_MAX8(out1, k1);
            out4 = AE_MAX8(out2, k2);
            AE_SAV8X8X2_XP(out3, out4, align_dst, p_dst, rem_hwc);
          }
          AE_SA128POS_FP(align_dst, p_dst); // finalize the stream
        }

        if(temp_inp_n & 1)
        {
          p_src1 = (ae_int8x16 *)(p_scratch);
          p_src2 = (ae_int8x16 *)(p_in + itr_n * plane_size);
          p_dst  = (ae_int8x16 *)(p_scratch);
          align_src1 = AE_LA128_PP(p_src1);
          align_src2 = AE_LA128_PP(p_src2);

          int itr_hwc = 0;
          for(itr_hwc=0; itr_hwc < (plane_size >> 4); itr_hwc++)
          {
            ae_int8x8 i1, i2, j1, j2, out1, out2;
            AE_LA8X8X2_IP(i1, i2, align_src1, p_src1);
            AE_LA8X8X2_IP(j1, j2, align_src2, p_src2);
            out1 = AE_MAX8(i1, j1);
            out2 = AE_MAX8(i2, j2);
            AE_SA8X8X2_IP(out1, out2, align_dst, p_dst);
          }

          //Remainder Loop
          if(rem_hwc)
          {
            ae_int8x8 i1, i2, j1, j2, out1, out2;
            AE_LAV8X8X2_XP(i1, i2, align_src1, p_src1, rem_hwc);
            AE_LAV8X8X2_XP(j1, j2, align_src2, p_src2, rem_hwc);
            out1 = AE_MAX8(i1, j1);
            out2 = AE_MAX8(i2, j2);
            AE_SAV8X8X2_XP(out1, out2, align_dst, p_dst, rem_hwc);
          }
          AE_SA128POS_FP(align_dst, p_dst); // finalize the stream
        }
        temp_inp_n = 1;  
        }break;
      case 1: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int rem_wc = (wc_plane_size & 15); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          p_src1 = (ae_int8x16 *)(p_scratch + (itr_n * wc_plane_size * (!flag)) + (flag * itr_n * plane_size));
          for(itr_h=0; itr_h < (temp_inp_h & ~(2 - 1)); itr_h += 2)
          {
            p_src2 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
            p_src3 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size));
            p_dst = (ae_int8x16 *)(p_scratch + (itr_n * wc_plane_size));
            align_src1 = AE_LA128_PP(p_src1);
            align_src2 = AE_LA128_PP(p_src2);
            align_src3 = AE_LA128_PP(p_src3);

            int itr_wc = 0;
            for(itr_wc=0; itr_wc < (wc_plane_size >> 4); itr_wc++)
            {
              ae_int8x8 i1, i2, j1, j2, k1, k2, out1, out2, out3, out4;
              AE_LA8X8X2_IP(i1, i2, align_src1, p_src1);
              AE_LA8X8X2_IP(j1, j2, align_src2, p_src2);
              AE_LA8X8X2_IP(k1, k2, align_src3, p_src3);
              out1 = AE_MAX8(i1, j1);
              out2 = AE_MAX8(i2, j2);
              out3 = AE_MAX8(out1, k1);
              out4 = AE_MAX8(out2, k2);
              AE_SA8X8X2_IP(out3, out4, align_dst, p_dst);
            }

            //Remainder Loop
            if(rem_wc)
            {
              ae_int8x8 i1, i2, j1, j2, k1, k2, out1, out2, out3, out4;
              AE_LAV8X8X2_XP(i1, i2, align_src1, p_src1, rem_wc);
              AE_LAV8X8X2_XP(j1, j2, align_src2, p_src2, rem_wc);
              AE_LAV8X8X2_XP(k1, k2, align_src3, p_src3, rem_wc);
              out1 = AE_MAX8(i1, j1);
              out2 = AE_MAX8(i2, j2);
              out3 = AE_MAX8(out1, k1);
              out4 = AE_MAX8(out2, k2);
              AE_SAV8X8X2_XP(out3, out4, align_dst, p_dst, rem_wc);
            }
            AE_SA128POS_FP(align_dst, p_dst); // finalize the stream
            p_src1 = (ae_int8x16 *)(p_scratch + (itr_n * wc_plane_size));
          }

          if(temp_inp_h & 1)
          {
            p_src2 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
            p_dst = (ae_int8x16 *)(p_scratch + (itr_n * wc_plane_size));
            align_src1 = AE_LA128_PP(p_src1);
            align_src2 = AE_LA128_PP(p_src2);

            int itr_wc = 0;
            for(itr_wc=0; itr_wc < (wc_plane_size >> 4); itr_wc++)
            {
              ae_int8x8 i1, i2, j1, j2, out1, out2;
              AE_LA8X8X2_IP(i1, i2, align_src1, p_src1);
              AE_LA8X8X2_IP(j1, j2, align_src2, p_src2);
              out1 = AE_MAX8(i1, j1);
              out2 = AE_MAX8(i2, j2);
              AE_SA8X8X2_IP(out1, out2, align_dst, p_dst);
            }

            //Remainder Loop
            if(rem_wc)
            {
              ae_int8x8 i1, i2, j1, j2, out1, out2;
              AE_LAV8X8X2_XP(i1, i2, align_src1, p_src1, rem_wc);
              AE_LAV8X8X2_XP(j1, j2, align_src2, p_src2, rem_wc);
              out1 = AE_MAX8(i1, j1);
              out2 = AE_MAX8(i2, j2);
              AE_SAV8X8X2_XP(out1, out2, align_dst, p_dst, rem_wc);
            }
            AE_SA128POS_FP(align_dst, p_dst); // finalize the stream
            p_src1 = (ae_int8x16 *)(p_scratch + (itr_n * wc_plane_size));
          }
        }
        temp_inp_h = 1;
        }break;
      case 2:{
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hc_plane_size = temp_inp_h * temp_inp_c;
        int rem_c = (temp_inp_c & 15); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            p_src1 = (ae_int8x16 *)(p_scratch + (((itr_n * hc_plane_size) + itr_h * temp_inp_c) * (!flag)) + (flag)*((itr_n * plane_size) + (itr_h * wc_plane_size)));
            for(itr_w=0; itr_w < (temp_inp_w & ~(2 - 1)); itr_w += 2)
            {
              p_src2 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_src3 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + ((itr_w + 1) * temp_inp_c));
              p_dst = (ae_int8x16 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              align_src1 = AE_LA128_PP(p_src1);
              align_src2 = AE_LA128_PP(p_src2);
              align_src3 = AE_LA128_PP(p_src3);

              for(itr_c=0; itr_c < (temp_inp_c >> 4); itr_c++)
              {
                ae_int8x8 i1, i2, j1, j2, k1, k2, out1, out2, out3, out4;
                AE_LA8X8X2_IP(i1, i2, align_src1, p_src1);
                AE_LA8X8X2_IP(j1, j2, align_src2, p_src2);
                AE_LA8X8X2_IP(k1, k2, align_src3, p_src3);
                out1 = AE_MAX8(i1, j1);
                out2 = AE_MAX8(i2, j2);
                out3 = AE_MAX8(out1, k1);
                out4 = AE_MAX8(out2, k2);
                AE_SA8X8X2_IP(out3, out4, align_dst, p_dst);
              }

              //Remainder Loop
              if(rem_c)
              {
                ae_int8x8 i1, i2, j1, j2, k1, k2, out1, out2, out3, out4;
                AE_LAV8X8X2_XP(i1, i2, align_src1, p_src1, rem_c);
                AE_LAV8X8X2_XP(j1, j2, align_src2, p_src2, rem_c);
                AE_LAV8X8X2_XP(k1, k2, align_src3, p_src3, rem_c);
                out1 = AE_MAX8(i1, j1);
                out2 = AE_MAX8(i2, j2);
                out3 = AE_MAX8(out1, k1);
                out4 = AE_MAX8(out2, k2);
                AE_SAV8X8X2_XP(out3, out4, align_dst, p_dst, rem_c);
              }
              AE_SA128POS_FP(align_dst, p_dst); // finalize the stream
              p_src1 = (ae_int8x16 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
            }

            if(temp_inp_w & 1)
            {
              p_src2 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_dst = (ae_int8x16 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              align_src1 = AE_LA128_PP(p_src1);
              align_src2 = AE_LA128_PP(p_src2);

              for(itr_c=0; itr_c < (temp_inp_c >> 4); itr_c++)
              {
                ae_int8x8 i1, i2, j1, j2, out1, out2;
                AE_LA8X8X2_IP(i1, i2, align_src1, p_src1);
                AE_LA8X8X2_IP(j1, j2, align_src2, p_src2);
                out1 = AE_MAX8(i1, j1);
                out2 = AE_MAX8(i2, j2);
                AE_SA8X8X2_IP(out1, out2, align_dst, p_dst);
              }

              //Remainder Loop
              if(rem_c)
              {
                ae_int8x8 i1, i2, j1, j2, out1, out2;
                AE_LAV8X8X2_XP(i1, i2, align_src1, p_src1, rem_c);
                AE_LAV8X8X2_XP(j1, j2, align_src2, p_src2, rem_c);
                out1 = AE_MAX8(i1, j1);
                out2 = AE_MAX8(i2, j2);
                AE_SAV8X8X2_XP(out1, out2, align_dst, p_dst, rem_c);
              }
              AE_SA128POS_FP(align_dst, p_dst); // finalize the stream
              p_src1 = (ae_int8x16 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
            }
          }
        }
        temp_inp_w = 1;
        }break;
      case 3: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hw_plane_size = temp_inp_h * temp_inp_w;
        int rem_c = (temp_inp_c & 15); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
            {
              p_src1 = (ae_int8x16 *)(p_scratch + (((itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w) * (!flag)) + ((flag) * ((itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w *temp_inp_c))));
              p_src2 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_dst = (ae_int8x16 *)(p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w);
              align_src2 = AE_LA128_PP(p_src2);

              for(itr_c=0; itr_c < (temp_inp_c >> 4); itr_c++)
              {
                ae_int8x8 i1, j1, j2, out1, out2;
                ae_int8 out;
                i1 = AE_L8_I((ae_int8 *)p_src1, 0);
                AE_LA8X8X2_IP(j1, j2, align_src2, p_src2);
                out1 = AE_MAX8(j1, j2);
                out = AE_RMAX8X8(out1);
                out2 = AE_MAX8(AE_MOVINT8X8_FROMINT8(out), i1);
                AE_S8_0_I(out2, (ae_int8 *)p_dst, 0);
                p_src1 = p_dst;
              }

              //Remainder Loop
              for(itr_c=0; itr_c < rem_c; itr_c++)
              {
                ae_int8x8 i1, j1, out1;
                i1 = AE_L8_I((ae_int8 *)p_src1, 0);
                AE_L8_IP(j1, (ae_int8 *)p_src2, 1);
                out1 = AE_MAX8(i1, j1);
                AE_S8_0_I(out1, (ae_int8 *)p_dst, 0);
                p_src1 = p_dst;
              }
            }
          }
        }
        temp_inp_c = 1;
        }break;
      default:
        return -1;
        break;
    }

    p_in = p_scratch;
    flag = 1;
  }
  if(num_axis_dims)
  {
    memcpy(p_out, p_scratch, out_length); //TODO: Alternate approach?
  }
  else
  {
    memcpy(p_out, p_inp, inp_length); //TODO: Alternate approach?
  }

  return 0;
}

static inline void xa_nn_reduce_sum_4D_asym8s_asym8s(const WORD8 * __restrict__ p_inp
                                                    ,const WORD32 *const p_4D_inp_shape
                                                    ,const WORD32 * __restrict__ p_axis_data
                                                    ,WORD32 num_inp_dims
                                                    ,WORD32 num_axis_dims
                                                    ,pVOID p_scratch_in)
{
  WORD8 *p_in = (WORD8 *)(p_inp);
  WORD32 *p_scratch = (WORD32 *)(p_scratch_in);

  int temp_inp_n = p_4D_inp_shape[0]; 
  int temp_inp_h = p_4D_inp_shape[1]; 
  int temp_inp_w = p_4D_inp_shape[2]; 
  int temp_inp_c = p_4D_inp_shape[3];

  int itr_axis = 0, itr_n = 0, itr_h = 0, itr_w = 0, itr_c = 0;
  ae_int8x16 *p_src2, *p_src3;
  ae_int32x4 *p_src1;
  ae_int32x4 * p_dst;
  ae_valignx2 align_src1, align_src2, align_src3, align_dst;
  align_dst = AE_ZALIGN128();

  ae_int16x4 one16 = AE_MOVDA16(1);
  ae_int8x8 zero8 = AE_MOVDA8(0);
  ae_int16x4 zero16 = AE_MOVDA16(0);

  int axis_dims_count = num_axis_dims;
  if(axis_dims_count)
  {
    switch(p_axis_data[itr_axis])
    {
      case 0: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int rem_hwc = (plane_size & 15); 
        for(itr_n=0; itr_n < (temp_inp_n & ~(2 - 1)); itr_n += 2)
        {
          p_src1 = (ae_int32x4 *)(p_scratch);
          p_src2 = (ae_int8x16 *)(p_in + itr_n * plane_size);
          p_src3 = (ae_int8x16 *)(p_in + (itr_n + 1) * plane_size);
          p_dst  = (ae_int32x4 *)(p_scratch);
          align_src2 = AE_LA128_PP(p_src2);
          align_src3 = AE_LA128_PP(p_src3);

          int itr_hwc = 0;
          for(itr_hwc=0; itr_hwc < (plane_size >> 4); itr_hwc++)
          {
            ae_int8x8 j1, j2, k1, k2;
            ae_int16x4 wj1, wj2, wk1, wk2;
            ae_int32x2 wout1, wout2, wout3, wout4;
            ae_int32x2 wout5, wout6, wout7, wout8;
            ae_int16x4 one = one16;
            AE_L32X2X2_I(wout3, wout4, p_src1, 16);
            AE_L32X2X2_I(wout5, wout6, p_src1, 32);
            AE_L32X2X2_I(wout7, wout8, p_src1, 48);
            AE_L32X2X2_IP(wout1, wout2, p_src1, 64);
            AE_LA8X8X2_IP(j1, k1, align_src2, p_src2);
            AE_LA8X8X2_IP(j2, k2, align_src3, p_src3);
            AE_ADDW8(wj1, wj2, j1, j2);
            AE_ADDW8(wk1, wk2, k1, k2);
            AE_MULA16X4(wout1, wout2, wj1, one);
            AE_MULA16X4(wout3, wout4, wj2, one);
            AE_MULA16X4(wout5, wout6, wk1, one);
            AE_MULA16X4(wout7, wout8, wk2, one);
            AE_S32X2X2_I(wout3, wout4, p_dst, 16);
            AE_S32X2X2_I(wout5, wout6, p_dst, 32);
            AE_S32X2X2_I(wout7, wout8, p_dst, 48);
            AE_S32X2X2_IP(wout1, wout2, p_dst, 64);
          }

          //Remainder Loop
          for(itr_hwc=0; itr_hwc < rem_hwc; itr_hwc++)
          {
            ae_int8x8 j1, j2;
            ae_int16x4 wj1, wj2;
            ae_int32x2 wout1, wout2;
            AE_L8_IP(j1, (ae_int8 *)p_src2, sizeof(ae_int8));
            AE_L8_IP(j2, (ae_int8 *)p_src3, sizeof(ae_int8));
            AE_ADDW8(wj1, wj2, j1, j2);
            AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
            AE_ACCW16(wout1, wout2, wj1, zero16);
            AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
          }
        }

        if(temp_inp_n & 1)
        {
          p_src1 = (ae_int32x4 *)(p_scratch);
          p_src2 = (ae_int8x16 *)(p_in + itr_n * plane_size);
          p_dst  = (ae_int32x4 *)(p_scratch);
          align_src2 = AE_LA128_PP(p_src2);

          int itr_hwc = 0;
          for(itr_hwc=0; itr_hwc < (plane_size >> 4); itr_hwc++)
          {
            ae_int8x8 j1, k1;
            ae_int16x4 wj1, wj2, wk1, wk2;
            ae_int32x2 wout1, wout2, wout3, wout4;
            ae_int32x2 wout5, wout6, wout7, wout8;
            ae_int16x4 one = one16;
            AE_L32X2X2_I(wout3, wout4, p_src1, 16);
            AE_L32X2X2_I(wout5, wout6, p_src1, 32);
            AE_L32X2X2_I(wout7, wout8, p_src1, 48);
            AE_L32X2X2_IP(wout1, wout2, p_src1, 64);
            AE_LA8X8X2_IP(j1, k1, align_src2, p_src2);
            AE_ADDW8(wj1, wj2, j1, zero8);
            AE_ADDW8(wk1, wk2, k1, zero8);
            AE_MULA16X4(wout1, wout2, wj1, one);
            AE_MULA16X4(wout3, wout4, wj2, one);
            AE_MULA16X4(wout5, wout6, wk1, one);
            AE_MULA16X4(wout7, wout8, wk2, one);
            AE_S32X2X2_I(wout3, wout4, p_dst, 16);
            AE_S32X2X2_I(wout5, wout6, p_dst, 32);
            AE_S32X2X2_I(wout7, wout8, p_dst, 48);
            AE_S32X2X2_IP(wout1, wout2, p_dst, 64);
          }

          //Remainder Loop
          for(itr_hwc=0; itr_hwc < rem_hwc; itr_hwc++)
          {
            ae_int8x8 j1;
            ae_int16x4 wj1, wj2;
            ae_int32x2 wout1, wout2;
            AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
            AE_L8_IP(j1, (ae_int8 *)p_src2, sizeof(ae_int8));
            AE_ADDW8(wj1, wj2, j1, zero8);
            AE_ACCW16(wout1, wout2, wj1, zero16);
            AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
          }
        }
        temp_inp_n = 1;  
        }break;
      case 1: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int rem_wc = (wc_plane_size & 15); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          p_src1 = (ae_int32x4 *)(p_scratch + (itr_n * wc_plane_size)); 
          for(itr_h=0; itr_h < (temp_inp_h & ~(2 - 1)); itr_h += 2)
          {
            p_src2 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
            p_src3 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size));
            p_dst = (ae_int32x4 *)(p_scratch + (itr_n * wc_plane_size));
            align_src1 = AE_LA128_PP(p_src1);
            align_src2 = AE_LA128_PP(p_src2);
            align_src3 = AE_LA128_PP(p_src3);

            int itr_wc = 0;
            for(itr_wc=0; itr_wc < (wc_plane_size >> 4); itr_wc++)
            {
              ae_int8x8 j1, j2, k1, k2;
              ae_int16x4 wj1, wj2, wk1, wk2;
              ae_int32x2 wout1, wout2, wout3, wout4;
              ae_int32x2 wout5, wout6, wout7, wout8;
              AE_LA32X2X2_IP(wout1, wout2, align_src1, p_src1);
              AE_LA32X2X2_IP(wout3, wout4, align_src1, p_src1);
              AE_LA32X2X2_IP(wout5, wout6, align_src1, p_src1);
              AE_LA32X2X2_IP(wout7, wout8, align_src1, p_src1);
              AE_LA8X8X2_IP(j1, k1, align_src2, p_src2);
              AE_LA8X8X2_IP(j2, k2, align_src3, p_src3);
              AE_ADDW8(wj1, wj2, j1, j2);
              AE_ADDW8(wk1, wk2, k1, k2);
              AE_MULA16X4(wout1, wout2, wj1, one16);
              AE_MULA16X4(wout3, wout4, wj2, one16);
              AE_MULA16X4(wout5, wout6, wk1, one16);
              AE_MULA16X4(wout7, wout8, wk2, one16);
              AE_SA32X2X2_IP(wout1, wout2, align_dst, p_dst);
              AE_SA32X2X2_IP(wout3, wout4, align_dst, p_dst);
              AE_SA32X2X2_IP(wout5, wout6, align_dst, p_dst);
              AE_SA32X2X2_IP(wout7, wout8, align_dst, p_dst);
            }
            AE_SA128POS_FP(align_dst, p_dst); // finalize the stream

            //Remainder Loop
            for(itr_wc=0; itr_wc < rem_wc; itr_wc++)
            {
              ae_int8x8 j1, j2;
              ae_int16x4 wj1, wj2;
              ae_int32x2 wout1, wout2;
              AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
              AE_L8_IP(j1, (ae_int8 *)p_src2, sizeof(ae_int8));
              AE_L8_IP(j2, (ae_int8 *)p_src3, sizeof(ae_int8));
              AE_ADDW8(wj1, wj2, j1, j2);
              AE_ACCW16(wout1, wout2, wj1, zero16);
              AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
            }
            p_src1 = (ae_int32x4 *)(p_scratch + (itr_n * wc_plane_size));
          }

          if(temp_inp_h & 1)
          {
            p_src2 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
            p_dst = (ae_int32x4 *)(p_scratch + (itr_n * wc_plane_size));
            align_src1 = AE_LA128_PP(p_src1);
            align_src2 = AE_LA128_PP(p_src2);

            int itr_wc = 0;
            for(itr_wc=0; itr_wc < (wc_plane_size >> 4); itr_wc++)
            {
              ae_int8x8 j1, k1;
              ae_int16x4 wj1, wj2, wk1, wk2;
              ae_int32x2 wout1, wout2, wout3, wout4;
              ae_int32x2 wout5, wout6, wout7, wout8;
              AE_LA32X2X2_IP(wout1, wout2, align_src1, p_src1);
              AE_LA32X2X2_IP(wout3, wout4, align_src1, p_src1);
              AE_LA32X2X2_IP(wout5, wout6, align_src1, p_src1);
              AE_LA32X2X2_IP(wout7, wout8, align_src1, p_src1);
              AE_LA8X8X2_IP(j1, k1, align_src2, p_src2);
              AE_ADDW8(wj1, wj2, j1, zero8);
              AE_ADDW8(wk1, wk2, k1, zero8);
              AE_MULA16X4(wout1, wout2, wj1, one16);
              AE_MULA16X4(wout3, wout4, wj2, one16);
              AE_MULA16X4(wout5, wout6, wk1, one16);
              AE_MULA16X4(wout7, wout8, wk2, one16);
              AE_SA32X2X2_IP(wout1, wout2, align_dst, p_dst);
              AE_SA32X2X2_IP(wout3, wout4, align_dst, p_dst);
              AE_SA32X2X2_IP(wout5, wout6, align_dst, p_dst);
              AE_SA32X2X2_IP(wout7, wout8, align_dst, p_dst);
            }
            AE_SA128POS_FP(align_dst, p_dst); // finalize the stream

            //Remainder Loop
            for(itr_wc=0; itr_wc < rem_wc; itr_wc++)
            {
              ae_int8x8 j1;
              ae_int16x4 wj1, wj2;
              ae_int32x2 wout1, wout2;
              AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
              AE_L8_IP(j1, (ae_int8 *)p_src2, sizeof(ae_int8));
              AE_ADDW8(wj1, wj2, j1, zero8);
              AE_ACCW16(wout1, wout2, wj1, zero16);
              AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
            }
          }
        }
        temp_inp_h = 1;
        }break;
      case 2:{
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hc_plane_size = temp_inp_h * temp_inp_c;
        int rem_c = (temp_inp_c & 15); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            p_src1 = (ae_int32x4 *)(p_scratch + (((itr_n * hc_plane_size) + itr_h * temp_inp_c))); 
            for(itr_w=0; itr_w < (temp_inp_w & ~(2 - 1)); itr_w += 2)
            {
              p_src2 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_src3 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + ((itr_w + 1) * temp_inp_c));
              p_dst = (ae_int32x4 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              align_src1 = AE_LA128_PP(p_src1);
              align_src2 = AE_LA128_PP(p_src2);
              align_src3 = AE_LA128_PP(p_src3);

              for(itr_c=0; itr_c < (temp_inp_c >> 4); itr_c++)
              {
                ae_int8x8 j1, j2, k1, k2;
                ae_int16x4 wj1, wj2, wk1, wk2;
                ae_int32x2 wout1, wout2, wout3, wout4;
                ae_int32x2 wout5, wout6, wout7, wout8;
                AE_LA32X2X2_IP(wout1, wout2, align_src1, p_src1);
                AE_LA32X2X2_IP(wout3, wout4, align_src1, p_src1);
                AE_LA32X2X2_IP(wout5, wout6, align_src1, p_src1);
                AE_LA32X2X2_IP(wout7, wout8, align_src1, p_src1);
                AE_LA8X8X2_IP(j1, k1, align_src2, p_src2);
                AE_LA8X8X2_IP(j2, k2, align_src3, p_src3);
                AE_ADDW8(wj1, wj2, j1, j2);
                AE_ADDW8(wk1, wk2, k1, k2);
                AE_MULA16X4(wout1, wout2, wj1, one16);
                AE_MULA16X4(wout3, wout4, wj2, one16);
                AE_MULA16X4(wout5, wout6, wk1, one16);
                AE_MULA16X4(wout7, wout8, wk2, one16);
                AE_SA32X2X2_IP(wout1, wout2, align_dst, p_dst);
                AE_SA32X2X2_IP(wout3, wout4, align_dst, p_dst);
                AE_SA32X2X2_IP(wout5, wout6, align_dst, p_dst);
                AE_SA32X2X2_IP(wout7, wout8, align_dst, p_dst);
              }
              AE_SA128POS_FP(align_dst, p_dst); // finalize the stream

              //Remainder Loop
              for(itr_c=0; itr_c < rem_c; itr_c++)
              {
                ae_int8x8 j1, j2;
                ae_int16x4 wj1, wj2;
                ae_int32x2 wout1, wout2;
                AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
                AE_L8_IP(j1, (ae_int8 *)p_src2, sizeof(ae_int8));
                AE_L8_IP(j2, (ae_int8 *)p_src3, sizeof(ae_int8));
                AE_ADDW8(wj1, wj2, j1, j2);
                AE_ACCW16(wout1, wout2, wj1, zero16);
                AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
              }
              p_src1 = (ae_int32x4 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
            }

            if(temp_inp_w & 1)
            {
              p_src2 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_dst = (ae_int32x4 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              align_src1 = AE_LA128_PP(p_src1);
              align_src2 = AE_LA128_PP(p_src2);

              for(itr_c=0; itr_c < (temp_inp_c >> 4); itr_c++)
              {
                ae_int8x8 j1, k1;
                ae_int16x4 wj1, wj2, wk1, wk2;
                ae_int32x2 wout1, wout2, wout3, wout4;
                ae_int32x2 wout5, wout6, wout7, wout8;
                AE_LA32X2X2_IP(wout1, wout2, align_src1, p_src1);
                AE_LA32X2X2_IP(wout3, wout4, align_src1, p_src1);
                AE_LA32X2X2_IP(wout5, wout6, align_src1, p_src1);
                AE_LA32X2X2_IP(wout7, wout8, align_src1, p_src1);
                AE_LA8X8X2_IP(j1, k1, align_src2, p_src2);
                AE_ADDW8(wj1, wj2, j1, zero8);
                AE_ADDW8(wk1, wk2, k1, zero8);
                AE_MULA16X4(wout1, wout2, wj1, one16);
                AE_MULA16X4(wout3, wout4, wj2, one16);
                AE_MULA16X4(wout5, wout6, wk1, one16);
                AE_MULA16X4(wout7, wout8, wk2, one16);
                AE_SA32X2X2_IP(wout1, wout2, align_dst, p_dst);
                AE_SA32X2X2_IP(wout3, wout4, align_dst, p_dst);
                AE_SA32X2X2_IP(wout5, wout6, align_dst, p_dst);
                AE_SA32X2X2_IP(wout7, wout8, align_dst, p_dst);
              }
              AE_SA128POS_FP(align_dst, p_dst); // finalize the stream

              //Remainder Loop
              for(itr_c=0; itr_c < rem_c; itr_c++)
              {
                ae_int8x8 j1;
                ae_int16x4 wj1, wj2;
                ae_int32x2 wout1, wout2;
                AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
                AE_L8_IP(j1, (ae_int8 *)p_src2, sizeof(ae_int8));
                AE_ADDW8(wj1, wj2, j1, zero8);
                AE_ACCW16(wout1, wout2, wj1, zero16);
                AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
              }
            }
          }
        }
        temp_inp_w = 1;
        }break;
      case 3: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hw_plane_size = temp_inp_h * temp_inp_w;
        int rem_c = (temp_inp_c & 15); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
            {
              p_src1 = (ae_int32x4 *)(p_scratch + (((itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w)));
              p_src2 = (ae_int8x16 *)(p_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_dst = (ae_int32x4 *)(p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w);
              align_src2 = AE_LA128_PP(p_src2);

              for(itr_c=0; itr_c < (temp_inp_c >> 4); itr_c++)
              {
                ae_int8x8 j1, j2;
                ae_int32x2 i1, i2;
                ae_int32 out1, out2;
                i1 = AE_L32_I((ae_int32 *)p_src1, 0);
                AE_LA8X8X2_IP(j1, j2, align_src2, p_src2);
                out1 = AE_RADD8X8_H(j1);
                AE_RADDA8X8_L(out1, j1);
                out2 = AE_RADD8X8_H(j2);
                AE_RADDA8X8_L(out2, j2);
                i2 = AE_ADD32S(AE_MOVDA32(out1), AE_MOVDA32(out2));
                i1 = AE_ADD32S(i1, i2);
                AE_S32_L_I(i1, (ae_int32 *)p_dst, 0);
                p_src1 = p_dst;
              }

              //Remainder Loop
              for(itr_c=0; itr_c < rem_c; itr_c++)
              {
                ae_int8x8 j1;
                ae_int32x2 i1, temp0, temp1;
                i1 = AE_L32_I((ae_int32 *)p_src1, 0);
                AE_L8_IP(j1, (ae_int8 *)p_src2, 1);
                AE_CVTI32X4F8_H(temp0, temp1, j1, 0);
                i1 = AE_ADD32S(i1, temp0);
                AE_S32_L_I(i1, (ae_int32 *)p_dst, 0);
                p_src1 = p_dst;
              }
            }
          }
        }
        temp_inp_c = 1;
        }break;
      default:
        break;
    }

    axis_dims_count--;
    itr_axis++;
  }

  while(axis_dims_count)
  {
    WORD32 *p_scr_in =(WORD32 *)p_scratch;
    ae_int32x4 *p_wsrc2, *p_wsrc3;
    switch(p_axis_data[itr_axis])
    {
      case 0: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int rem_hwc = (plane_size & 3); 
        for(itr_n=1; itr_n < ((temp_inp_n -1) & ~(2 - 1)); itr_n += 2)
        {
          p_src1 = (ae_int32x4 *)(p_scratch);
          p_wsrc2 = (ae_int32x4 *)(p_scr_in + itr_n * plane_size);
          p_wsrc3 = (ae_int32x4 *)(p_scr_in + (itr_n + 1) * plane_size);
          p_dst  = (ae_int32x4 *)(p_scratch);
          align_src2 = AE_LA128_PP(p_wsrc2);
          align_src3 = AE_LA128_PP(p_wsrc3);

          int itr_hwc = 0;
          for(itr_hwc=0; itr_hwc < (plane_size >> 2); itr_hwc++)
          {
            ae_int32x2 j1, j2, k1, k2;
            ae_int32x2 wj1, wk1;
            ae_int32x2 wout1, wout2;
            AE_L32X2X2_IP(wout1, wout2, p_src1, 16);
            AE_LA32X2X2_IP(j1, k1, align_src2, p_wsrc2);
            AE_LA32X2X2_IP(j2, k2, align_src3, p_wsrc3);
            wj1 = AE_ADD32S(j1, j2);
            wk1 = AE_ADD32S(k1, k2);
            wout1 = AE_ADD32S(wout1, wj1);
            wout2 = AE_ADD32S(wout2, wk1);
            AE_S32X2X2_IP(wout1, wout2, p_dst, 16);
          }

          //Remainder Loop
          for(itr_hwc=0; itr_hwc < rem_hwc; itr_hwc++)
          {
            ae_int32x2 j1, j2;
            ae_int32x2 wj1;
            ae_int32x2 wout1;
            AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
            AE_L32_IP(j1, (ae_int32 *)p_wsrc2, 4);
            AE_L32_IP(j2, (ae_int32 *)p_wsrc3, 4);
            wj1 = AE_ADD32S(j1, j2);
            wout1 = AE_ADD32S(wout1, wj1);
            AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
          }
        }

        if((temp_inp_n - 1) & 1)
        {
          p_src1 = (ae_int32x4 *)(p_scratch);
          p_wsrc2 = (ae_int32x4 *)(p_scr_in + itr_n * plane_size);
          p_dst  = (ae_int32x4 *)(p_scratch);
          align_src2 = AE_LA128_PP(p_wsrc2);

          int itr_hwc = 0;
          for(itr_hwc=0; itr_hwc < (plane_size >> 2); itr_hwc++)
          {
            ae_int32x2 j1, k1;
            ae_int32x2 wout1, wout2;
            AE_L32X2X2_IP(wout1, wout2, p_src1, 16);
            AE_LA32X2X2_IP(j1, k1, align_src2, p_wsrc2);
            wout1 = AE_ADD32S(wout1, j1);
            wout2 = AE_ADD32S(wout2, k1);
            AE_S32X2X2_IP(wout1, wout2, p_dst, 16);
          }

          //Remainder Loop
          for(itr_hwc=0; itr_hwc < rem_hwc; itr_hwc++)
          {
            ae_int32x2 j1;
            ae_int32x2 wout1;
            AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
            AE_L32_IP(j1, (ae_int32 *)p_wsrc2, 4);
            wout1 = AE_ADD32S(wout1, j1);
            AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
          }
        }
        temp_inp_n = 1;
        }break;
      case 1: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int rem_wc = (wc_plane_size & 3); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          p_src1 = (ae_int32x4 *)(p_scratch + + (itr_n * plane_size));
          for(itr_h = 1; itr_h < ((temp_inp_h - 1) & ~(2 - 1)); itr_h += 2)
          {
            p_wsrc2 = (ae_int32x4 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
            p_wsrc3 = (ae_int32x4 *)(p_scr_in + (itr_n * plane_size) + ((itr_h + 1) * wc_plane_size));
            p_dst = (ae_int32x4 *)(p_scratch + (itr_n * wc_plane_size));
            align_src1 = AE_LA128_PP(p_src1);
            align_src2 = AE_LA128_PP(p_wsrc2);
            align_src3 = AE_LA128_PP(p_wsrc3);

            int itr_wc = 0;
            for(itr_wc=0; itr_wc < (wc_plane_size >> 2); itr_wc++)
            {
              ae_int32x2 j1, j2, k1, k2;
              ae_int32x2 wj1, wk1;
              ae_int32x2 wout1, wout2;
              AE_LA32X2X2_IP(wout1, wout2, align_src1, p_src1);
              AE_LA32X2X2_IP(j1, k1, align_src2, p_wsrc2);
              AE_LA32X2X2_IP(j2, k2, align_src3, p_wsrc3);
              wj1 = AE_ADD32S(j1, j2);
              wk1 = AE_ADD32S(k1, k2);
              wout1 = AE_ADD32S(wout1, wj1);
              wout2 = AE_ADD32S(wout2, wk1);
              AE_SA32X2X2_IP(wout1, wout2, align_dst, p_dst);
            }
            AE_SA128POS_FP(align_dst, p_dst); // finalize the stream

            //Remainder Loop
            for(itr_wc=0; itr_wc < rem_wc; itr_wc++)
            {
              ae_int32x2 j1, j2;
              ae_int32x2 wj1;
              ae_int32x2 wout1;
              AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
              AE_L32_IP(j1, (ae_int32 *)p_wsrc2, 4);
              AE_L32_IP(j2, (ae_int32 *)p_wsrc3, 4);
              wj1 = AE_ADD32S(j1, j2);
              wout1 = AE_ADD32S(wout1, wj1);
              AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
            }
            p_src1 = (ae_int32x4 *)(p_scratch + (itr_n * wc_plane_size));
          }

          if((temp_inp_h - 1) & 1)
          {
            p_wsrc2 = (ae_int32x4 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size));
            p_dst = (ae_int32x4 *)(p_scratch + (itr_n * wc_plane_size));
            align_src1 = AE_LA128_PP(p_src1);
            align_src2 = AE_LA128_PP(p_wsrc2);

            int itr_wc = 0;
            for(itr_wc=0; itr_wc < (wc_plane_size >> 2); itr_wc++)
            {
              ae_int32x2 j1, k1;
              ae_int32x2 wout1, wout2;
              AE_LA32X2X2_IP(wout1, wout2, align_src1, p_src1);
              AE_LA32X2X2_IP(j1, k1, align_src2, p_wsrc2);
              wout1 = AE_ADD32S(wout1, j1);
              wout2 = AE_ADD32S(wout2, k1);
              AE_SA32X2X2_IP(wout1, wout2, align_dst, p_dst);
            }
            AE_SA128POS_FP(align_dst, p_dst); // finalize the stream

            //Remainder Loop
            for(itr_wc=0; itr_wc < rem_wc; itr_wc++)
            {
              ae_int32x2 j1;
              ae_int32x2 wout1;
              AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
              AE_L32_IP(j1, (ae_int32 *)p_wsrc2, 4);
              wout1 = AE_ADD32S(wout1, j1);
              AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
            }
          }
        }
        temp_inp_h = 1;
        }break;
      case 2:{
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hc_plane_size = temp_inp_h * temp_inp_c;
        int rem_c = (temp_inp_c & 3); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            p_src1 = (ae_int32x4 *)(p_scratch + ((itr_n * plane_size) + (itr_h * wc_plane_size)));
            for(itr_w = 1; itr_w < ((temp_inp_w - 1) & ~(2 - 1)); itr_w += 2)
            {
              p_wsrc2 = (ae_int32x4 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_wsrc3 = (ae_int32x4 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + ((itr_w + 1) * temp_inp_c));
              p_dst = (ae_int32x4 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              align_src1 = AE_LA128_PP(p_src1);
              align_src2 = AE_LA128_PP(p_wsrc2);
              align_src3 = AE_LA128_PP(p_wsrc3);

              for(itr_c=0; itr_c < (temp_inp_c >> 2); itr_c++)
              {
                ae_int32x2 j1, j2, k1, k2;
                ae_int32x2 wj1, wk1;
                ae_int32x2 wout1, wout2;
                AE_LA32X2X2_IP(wout1, wout2, align_src1, p_src1);
                AE_LA32X2X2_IP(j1, k1, align_src2, p_wsrc2);
                AE_LA32X2X2_IP(j2, k2, align_src3, p_wsrc3);
                wj1 = AE_ADD32S(j1, j2);
                wk1 = AE_ADD32S(k1, k2);
                wout1 = AE_ADD32S(wout1, wj1);
                wout2 = AE_ADD32S(wout2, wk1);
                AE_SA32X2X2_IP(wout1, wout2, align_dst, p_dst);
              }
              AE_SA128POS_FP(align_dst, p_dst); // finalize the stream

              //Remainder Loop
              for(itr_c=0; itr_c < rem_c; itr_c++)
              {
                ae_int32x2 j1, j2;
                ae_int32x2 wj1;
                ae_int32x2 wout1;
                AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
                AE_L32_IP(j1, (ae_int32 *)p_wsrc2, 4);
                AE_L32_IP(j2, (ae_int32 *)p_wsrc3, 4);
                wj1 = AE_ADD32S(j1, j2);
                wout1 = AE_ADD32S(wout1, wj1);
                AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
              }
              p_src1 = (ae_int32x4 *)(p_scratch + (itr_n * hc_plane_size) + (itr_h * temp_inp_c));
            }

            if((temp_inp_w - 1) & 1)
            {
              p_wsrc2 = (ae_int32x4 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c));
              p_dst = (ae_int32x4 *)(p_scratch + (itr_n * hc_plane_size) + itr_h * temp_inp_c);
              align_src1 = AE_LA128_PP(p_src1);
              align_src2 = AE_LA128_PP(p_wsrc2);

              for(itr_c=0; itr_c < (temp_inp_c >> 2); itr_c++)
              {
                ae_int32x2 j1, k1;
                ae_int32x2 wout1, wout2;
                AE_LA32X2X2_IP(wout1, wout2, align_src1, p_src1);
                AE_LA32X2X2_IP(j1, k1, align_src2, p_wsrc2);
                wout1 = AE_ADD32S(wout1, j1);
                wout2 = AE_ADD32S(wout2, k1);
                AE_SA32X2X2_IP(wout1, wout2, align_dst, p_dst);
              }
              AE_SA128POS_FP(align_dst, p_dst); // finalize the stream

              //Remainder Loop
              for(itr_c=0; itr_c < rem_c; itr_c++)
              {
                ae_int32x2 j1;
                ae_int32x2 wout1;
                AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
                AE_L32_IP(j1, (ae_int32 *)p_wsrc2, 4);
                wout1 = AE_ADD32S(wout1, j1);
                AE_S32_L_IP(wout1, (ae_int32 *)p_dst, sizeof(WORD32));
              }
            }
          }
        }
        temp_inp_w = 1;
        }break;
      case 3: {
        int plane_size = temp_inp_h * temp_inp_w * temp_inp_c;
        int wc_plane_size = temp_inp_w * temp_inp_c;
        int hw_plane_size = temp_inp_h * temp_inp_w;
        int rem_c = ((temp_inp_c - 1) & 3); 
        for(itr_n=0; itr_n < (temp_inp_n); itr_n++)
        {
          for(itr_h=0; itr_h < (temp_inp_h); itr_h++)
          {
            for(itr_w=0; itr_w < (temp_inp_w); itr_w++)
            {
              p_src1 = (ae_int32x4 *)(p_scratch + ((itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w *temp_inp_c)));
              p_wsrc2 = (ae_int32x4 *)(p_scr_in + (itr_n * plane_size) + (itr_h * wc_plane_size) + (itr_w * temp_inp_c) + 1);
              p_dst = (ae_int32x4 *)(p_scratch + (itr_n * hw_plane_size) + (itr_h * temp_inp_w) + itr_w);
              align_src2 = AE_LA128_PP(p_wsrc2);

              for(itr_c = 0; itr_c < ((temp_inp_c - 1) >> 2); itr_c++)
              {
                ae_int32x2 j1, j2;
                ae_int32x2 i1, i2;
                ae_int32 out1, out2;
                i1 = AE_L32_I((ae_int32 *)p_src1, 0);
                AE_LA32X2X2_IP(j1, j2, align_src2, p_wsrc2);
                out1 = AE_INT32X2_RADD(j1);
                out2 = AE_INT32X2_RADD(j2);
                i2 = AE_ADD32S(AE_MOVDA32(out1), AE_MOVDA32(out2));
                i1 = AE_ADD32S(i1, i2);
                AE_S32_L_I(i1, (ae_int32 *)p_dst, 0);
                p_src1 = p_dst;
              }

              //Remainder Loop
              for(itr_c=0; itr_c < rem_c; itr_c++)
              {
                ae_int32x2 i1, j1;
                i1 = AE_L32_I((ae_int32 *)p_src1, 0);
                AE_L32_IP(j1, (ae_int32 *)p_wsrc2, 4);
                i1 = AE_ADD32S(i1, j1);
                AE_S32_L_I(i1, (ae_int32 *)p_dst, 0);
                p_src1 = p_dst;
              }
            }
          }
        }
        temp_inp_c = 1;
        }break;
      default:
        break;
    }
    axis_dims_count--;
    itr_axis++;
  }
}

WORD32 xa_nn_reduce_mean_4D_asym8s_asym8s(WORD8 * __restrict__ p_out
                                        ,const WORD32 *const p_out_shape
                                        ,const WORD8 * __restrict__ p_inp
                                        ,const WORD32 *const p_inp_shape
                                        ,const WORD32 * __restrict__ p_axis
                                        ,WORD32 num_out_dims
                                        ,WORD32 num_inp_dims
                                        ,WORD32 num_axis_dims
                                        ,WORD32 inp_zero_bias
                                        ,WORD32 out_multiplier
                                        ,WORD32 out_shift
                                        ,WORD32 out_zero_bias
                                        ,void * __restrict__ p_scratch_in)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_axis, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);

  /* Invalid input checks */
  XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND(((num_out_dims <= 0) || (num_out_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND(((num_axis_dims < 0) || (num_axis_dims > 4)), -1);
  XA_NNLIB_ARG_CHK_COND((inp_zero_bias < -128 || inp_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_multiplier < 0), -1);

  extern const unsigned int inv_256_tbl[1025];
  ae_int32x2 inv_mult = AE_MOVDA32(inv_256_tbl[1]);
  int axis_itr = 0, inp_itr = 0, out_itr = 0;
  int num_elm_in_axis = 1;
  for(axis_itr=0; axis_itr < num_axis_dims; axis_itr++)
  {
    XA_NNLIB_ARG_CHK_COND(((p_axis[axis_itr] < 0) || (p_axis[axis_itr] > (num_inp_dims - 1))), -1);
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[p_axis[axis_itr]] > 1024), -1);
   
    int current = p_inp_shape[p_axis[axis_itr]];
    ae_int32x2 current_inv_mult = AE_MOVDA32(inv_256_tbl[current]);
   
    /* Avoid calculation in case of repeated axis dims*/
    if((!axis_itr) || (p_axis[axis_itr] != p_axis[axis_itr-1]))
    {
      ae_int64 d_tmp = AE_MUL32U_LL(inv_mult, current_inv_mult);
      inv_mult = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);
       
      num_elm_in_axis *= p_inp_shape[p_axis[axis_itr]];
    }
  }

  for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[inp_itr] <= 0), -1);
  }

  int out_length = 1;
  for(out_itr=0; out_itr < num_out_dims; out_itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shape[out_itr] <= 0), -1);
    out_length *= p_out_shape[out_itr];
  }

  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_axis, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);

  int left_shift, right_shift;
#if TFLITE_SINGLE_ROUNDING
  left_shift = out_shift;
  (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
  left_shift = out_shift < 0 ? 0 : out_shift;
  right_shift = out_shift > 0 ? 0 : -out_shift;
#endif /* #if TFLITE_SINGLE_ROUNDING */

  WORD8 *p_in = (WORD8 *)(p_inp);
  WORD32 *p_scratch = (WORD32 *)(ALIGN_PTR(p_scratch_in, ALIGNMENT_16));

  // Changing order of axis data so that reduce max will be first computed
  // across largest inp shape dim in axis. This is required to
  // minimize the scratch usage.
  int inp_length = 1, p_axis_data[4], inp_shape_max;
  if(num_axis_dims)
  {
    inp_shape_max = p_inp_shape[p_axis[0]];
    int axis_itr = 1, max_axis_itr = 0;
    int temp_p_axis_0 = p_axis[0];
    for(axis_itr = 0; axis_itr < num_axis_dims; axis_itr++)
    {
      p_axis_data[axis_itr] = p_axis[axis_itr];
    }
    for(axis_itr = 1; axis_itr < num_axis_dims; axis_itr++)
    {
      if(p_inp_shape[p_axis[axis_itr]] > inp_shape_max)
      {
        inp_shape_max = p_inp_shape[p_axis[axis_itr]];
        max_axis_itr = axis_itr;
      }
    }
    p_axis_data[0] = p_axis_data[max_axis_itr];
    p_axis_data[max_axis_itr] = temp_p_axis_0;

    int inp_itr = 0;
    for(inp_itr=0; inp_itr < num_inp_dims; inp_itr++)
    {
      inp_length *= p_inp_shape[inp_itr];
    }

    memset(p_scratch, 0, ((inp_length / inp_shape_max) * sizeof(WORD32))); //TODO: Alternate approach for memset?
  }

  // Promoting lesser dim tensors to 4D tensors. Also modifying axis
  // data accordingly.
  int p_4D_inp_shape[4] = {1, 1, 1, 1};
  int itr = num_inp_dims - 1;
  int count = 3;
  while(itr >= 0)
  {
    p_4D_inp_shape[count] = p_inp_shape[itr];
    itr--;
    count--;
  }
  for(itr = 0; itr < num_axis_dims; itr++)
  {
    p_axis_data[itr] = p_axis_data[itr] + (4 - num_inp_dims);
  }

  if(num_axis_dims)
  {
    if(num_elm_in_axis > 1)
    { 
      xa_nn_reduce_sum_4D_asym8s_asym8s(p_in,
                                        p_4D_inp_shape,
                                        p_axis_data,
                                        4,
                                        num_axis_dims,
                                        p_scratch);

      xtbool same_quant = (inp_zero_bias == out_zero_bias) && (out_multiplier == 0x40000000) && (out_shift == 1);

      int itr = 0;
      ae_int32x4 *p_src1 = (ae_int32x4 *)(p_scratch);
      ae_valign align_dst = AE_ZALIGN64();

      if(same_quant)
      {
        for(itr = 0; itr < (out_length >> 3); itr++)
        {
          ae_int32x2 wout1, wout2, wout3, wout4;
          ae_int16x4 d0_out16, d1_out16;
          ae_int8x8 d0_out8;
          ae_int32x2 temp1, temp2, temp3, temp4;

          AE_L32X2X2_I(wout3, wout4, p_src1, 16);
          AE_L32X2X2_IP(wout1, wout2, p_src1, 32);
          AE_MULF2P32X4RS(temp1, temp2, inv_mult, inv_mult, wout1, wout2);
          AE_MULF2P32X4RS(temp3, temp4, inv_mult, inv_mult, wout3, wout4);
          d0_out16 = AE_SEL16I(AE_MOVINT16X4_FROMINT32X2(temp1), AE_MOVINT16X4_FROMINT32X2(temp2), 8);
          d1_out16 = AE_SEL16I(AE_MOVINT16X4_FROMINT32X2(temp3), AE_MOVINT16X4_FROMINT32X2(temp4), 8);
          d0_out8 = AE_SAT8X8X16(d0_out16, d1_out16);
          AE_SA8X8_IP(d0_out8, align_dst, (ae_int8x8 *)p_out);
        }
        AE_SA64POS_FP(align_dst, p_out); // finalize the stream
        for(itr = 0; itr < (out_length & 7); itr++)
        {
          ae_int32x2 wout1;
          ae_int16x4 d0_out16;
          ae_int8x8 d0_out8;
          ae_int32x2 temp1;

          AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
          temp1 = AE_MULFP32X2RS(inv_mult, wout1);
          d0_out16 = AE_SAT16X4(temp1, temp1);
          d0_out8 = AE_SAT8X8X16(d0_out16, d0_out16);
          AE_S8_0_IP(d0_out8, (ae_int8 *)p_out, 1);
        }
      }
      else
      {
        ae_int32x2 total_bias = AE_MOVDA32(-inp_zero_bias*num_elm_in_axis);
        for(itr = 0; itr < (out_length >> 3); itr++)
        {
          ae_int32x2 wout1, wout2, wout3, wout4;
          ae_int16x4 d0_out16, d1_out16;
          ae_int8x8 d0_out8;

          AE_L32X2X2_I(wout3, wout4, p_src1, 16);
          AE_L32X2X2_IP(wout1, wout2, p_src1, 32);
          wout1 = AE_ADD32S(wout1, total_bias);
          wout2 = AE_ADD32S(wout2, total_bias);
          wout3 = AE_ADD32S(wout3, total_bias);
          wout4 = AE_ADD32S(wout4, total_bias);

          MPY_BY_QUANT_MULT_SLS_X2X2_OUT32(wout1, wout2, wout1, wout2, out_multiplier, left_shift, right_shift);
          AE_MULF2P32X4RS(wout1, wout2, inv_mult, inv_mult, wout1, wout2);
          d0_out16 = AE_SAT16X4(wout1, wout2);
          d0_out16 = AE_ADD16S(AE_MOVDA16(out_zero_bias), d0_out16);
          MPY_BY_QUANT_MULT_SLS_X2X2_OUT32(wout3, wout4, wout3, wout4, out_multiplier, left_shift, right_shift);
          AE_MULF2P32X4RS(wout3, wout4, inv_mult, inv_mult, wout3, wout4);
          d1_out16 = AE_SAT16X4(wout3, wout4);
          d1_out16 = AE_ADD16S(AE_MOVDA16(out_zero_bias), d1_out16);

          d0_out8 = AE_SAT8X8X16(d0_out16, d1_out16);
          AE_SA8X8_IP(d0_out8, align_dst, (ae_int8x8 *)p_out);
        }
        AE_SA64POS_FP(align_dst, p_out); // finalize the stream
        for(itr = 0; itr < (out_length & 7); itr++)
        {
          ae_int32x2 wout1;
          ae_int16x4 d0_out16;
          ae_int8x8 d0_out8;

          AE_L32_IP(wout1, (ae_int32 *)p_src1, 4);
          wout1 = AE_ADD32S(wout1, total_bias);
          
          MPY_BY_QUANT_MULT_SLS_X2_OUT32(wout1, wout1, out_multiplier, left_shift, right_shift);
          wout1 = AE_MULFP32X2RS(inv_mult, wout1);
          d0_out16 = AE_SAT16X4(wout1, wout1);
          d0_out16 = AE_ADD16S(AE_MOVDA16(out_zero_bias), d0_out16);

          d0_out8 = AE_SAT8X8X16(d0_out16, d0_out16);
          AE_S8_0_IP(d0_out8, (ae_int8 *)p_out, 1);
        }
      }
    }
    else
    {
      xtbool same_quant = (inp_zero_bias == out_zero_bias) && (out_multiplier == 0x40000000) && (out_shift == 1);

      int itr = 0;
      ae_valignx2 align_inp = AE_LA128_PP(p_in);
      ae_valignx2 align_dst = AE_ZALIGN128();

      if(same_quant)
      {
        memcpy(p_out, p_inp, inp_length); //TODO: Alternate approach?
      }
      else
      {
        ae_int8x8 total_bias = AE_MOVDA8(inp_zero_bias);
        int rem_out = out_length & 15;

        for(itr = 0; itr < (out_length >> 4); itr++)
        {
          ae_int8x8 wout1, wout2;
          ae_int16x4 d0_out16, d1_out16, d2_out16, d3_out16;
          ae_int8x8 d0_out8, d1_out8;
          ae_int32x2 temp1, temp2, temp3, temp4;
          ae_int32x2 temp5, temp6, temp7, temp8;

          AE_LA8X8X2_IP(wout1, wout2, align_inp, (ae_int8x16 *)p_in);
          AE_SUBW8(d0_out16, d1_out16, wout1, total_bias);
          AE_SUBW8(d2_out16, d3_out16, wout2, total_bias);
         
          AE_MUL16X4(temp1, temp2, d0_out16, AE_MOVDA16(1));
          AE_MUL16X4(temp3, temp4, d1_out16, AE_MOVDA16(1));
          AE_MUL16X4(temp5, temp6, d2_out16, AE_MOVDA16(1));
          AE_MUL16X4(temp7, temp8, d3_out16, AE_MOVDA16(1));

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d0_out16, temp1, temp2, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d1_out16, temp3, temp4, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d2_out16, temp5, temp6, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d3_out16, temp7, temp8, out_multiplier, left_shift, right_shift, out_zero_bias);

          d0_out8 = AE_SAT8X8X16(d0_out16, d1_out16);
          d1_out8 = AE_SAT8X8X16(d2_out16, d3_out16);
          AE_SA8X8X2_IP(d0_out8, d1_out8, align_dst, (ae_int8x16 *)p_out);
        }
        if(out_length & 15)
        {
          ae_int8x8 wout1, wout2;
          ae_int16x4 d0_out16, d1_out16, d2_out16, d3_out16;
          ae_int8x8 d0_out8, d1_out8;
          ae_int32x2 temp1, temp2, temp3, temp4;
          ae_int32x2 temp5, temp6, temp7, temp8;

          AE_LAV8X8X2_XP(wout1, wout2, align_inp, (ae_int8x16 *)p_in, rem_out);
          AE_SUBW8(d0_out16, d1_out16, wout1, total_bias);
          AE_SUBW8(d2_out16, d3_out16, wout2, total_bias);
         
          AE_MUL16X4(temp1, temp2, d0_out16, AE_MOVDA16(1));
          AE_MUL16X4(temp3, temp4, d1_out16, AE_MOVDA16(1));
          AE_MUL16X4(temp5, temp6, d2_out16, AE_MOVDA16(1));
          AE_MUL16X4(temp7, temp8, d3_out16, AE_MOVDA16(1));

          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d0_out16, temp1, temp2, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d1_out16, temp3, temp4, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d2_out16, temp5, temp6, out_multiplier, left_shift, right_shift, out_zero_bias);
          MPY_BY_QUANT_MULT_X2X2_OUT16_ZB(d3_out16, temp7, temp8, out_multiplier, left_shift, right_shift, out_zero_bias);
          
          d0_out8 = AE_SAT8X8X16(d0_out16, d1_out16);
          d1_out8 = AE_SAT8X8X16(d2_out16, d3_out16);


          AE_SAV8X8X2_XP(d0_out8, d1_out8, align_dst, (ae_int8x16 *)p_out, rem_out);
        }
        AE_SA128POS_FP(align_dst, p_out); // finalize the stream
      }

    }
  }
  else
  {
    memcpy(p_out, p_inp, inp_length); //TODO: Alternate approach?
  }

  return 0;
}
