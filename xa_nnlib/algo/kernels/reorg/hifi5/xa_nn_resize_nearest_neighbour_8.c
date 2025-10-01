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
#include "xa_nnlib_common_fpu.h"
#include "xa_nn_common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_err_chk.h"

#include "xa_nnlib_common.h"

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_resize_nearest_neighbour_8_8,
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_inp
  ,WORD32  input_batch
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  out_batch
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  out_channels
  ,FLOAT32 height_scale
  ,FLOAT32 width_scale
  ,FLOAT32 height_offset
  ,FLOAT32 width_offset
  ,WORD32  align_corners
  ))
#else
WORD32 xa_nn_resize_nearest_neighbour_8_8
  (pWORD8 __restrict__ p_out
  ,const WORD8 *__restrict__ p_inp
  ,WORD32  input_batch
  ,WORD32  input_height
  ,WORD32  input_width
  ,WORD32  input_channels
  ,WORD32  out_batch
  ,WORD32  out_height
  ,WORD32  out_width
  ,WORD32  out_channels
  ,FLOAT32 height_scale
  ,FLOAT32 width_scale
  ,FLOAT32 height_offset
  ,FLOAT32 width_offset
  ,WORD32  align_corners
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_batch <= 0 || input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_batch != input_batch || out_channels != input_channels), -1);

  int itr_n, itr_h, itr_w;

  int width_off  = input_channels;
  int height_off = input_width * width_off;
  int batch_off  = input_height * height_off;

  WORD8 *ptmp_inp = (WORD8 *)p_inp, *ptmp_out = (WORD8 *)p_out;
  WORD8 *ptmp_inp_h, *ptmp_inp_w;

  xtfloat heightf_offset = *(xtfloat *)&height_offset;
  xtfloat heightf_scale = *(xtfloat *)&height_scale;
  xtfloat widthf_offset = *(xtfloat *)&width_offset;
  xtfloat widthf_scale = *(xtfloat *)&width_scale;
  for(itr_n = 0; itr_n < out_batch; itr_n++)
  {
    for(itr_h = 0; itr_h < out_height; itr_h++)
    {
      xtfloat outh_idx; 
      outh_idx = ADD_S(FLOAT_S(itr_h, 0), heightf_offset); 
      outh_idx = MUL_S(outh_idx, heightf_scale); 
      outh_idx = align_corners ? FIROUND_S(outh_idx) : FIFLOOR_S(outh_idx); 
      outh_idx = MIN_S(outh_idx, FLOAT_S(input_height - 1, 0));
      outh_idx = MAX_S(FLOAT_S(0,0), outh_idx);
      int outh = xtfloatx2_rtor_int32(AE_MOVXTFLOATX2_FROMXTFLOAT(outh_idx));
      ptmp_inp_h = ptmp_inp + (outh * height_off);

      for(itr_w = 0; itr_w < out_width; itr_w++)
      {
        xtfloat outw_idx; 
        outw_idx = ADD_S(FLOAT_S(itr_w,0), widthf_offset); 
        outw_idx = MUL_S(outw_idx, widthf_scale); 
        outw_idx = align_corners ? FIROUND_S(outw_idx) : FIFLOOR_S(outw_idx); 
        outw_idx = MIN_S(outw_idx, FLOAT_S(input_width - 1,0));
        outw_idx = MAX_S(FLOAT_S(0,0), outw_idx);
        int outw = xtfloatx2_rtor_int32(AE_MOVXTFLOATX2_FROMXTFLOAT(outw_idx));
        ptmp_inp_w = ptmp_inp_h + (outw * width_off);

        MEMCPY_8b(ptmp_out, ptmp_inp_w, input_channels);
        ptmp_out += input_channels;
      }
    }
    ptmp_inp += batch_off;
  }

  return 0;
}
#endif

