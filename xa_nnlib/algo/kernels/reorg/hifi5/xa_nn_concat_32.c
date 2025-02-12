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
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common.h"

WORD32 xa_nn_concat_32_32(WORD32 * __restrict__ p_out
                        ,const WORD32 *const p_out_shape
                        ,const WORD32 **pp_inps
                        ,const WORD32 *const *pp_inps_shape
                        ,WORD32 num_out_dims
                        ,WORD32 num_inp
                        ,WORD32 num_inp_dims
                        ,WORD32 axis)
{
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(pp_inps, -1);
  XA_NNLIB_ARG_CHK_PTR(pp_inps_shape, -1);

  int i = 0, j = 0;
 
  WORD32 p_out_shape_8[6];
  WORD32 p_inp_shape_8[6*10];
  WORD32* pp_inps_shape_8[10];
  WORD8* p_out_8 = (WORD8*) p_out;
  
  for(j=0;j<num_inp;j++)
  {
    pp_inps_shape_8[j]=p_inp_shape_8 + j*6;
    for(i=0;i<num_inp_dims-1;i++)
    {
      pp_inps_shape_8[j][i]=pp_inps_shape[j][i];
    }
    if(num_inp_dims>0)
      pp_inps_shape_8[j][num_inp_dims-1]=pp_inps_shape[j][num_inp_dims-1]*4;
  }
  
  for(i=0;i<num_out_dims-1;i++)
  {
    p_out_shape_8[i]=p_out_shape[i];
  }
  if(num_out_dims>0)
    p_out_shape_8[num_out_dims-1]=p_out_shape[num_inp_dims-1]*4;
  
  return xa_nn_concat_8_8(p_out_8,
                          p_out_shape_8,
                          (const WORD8**)pp_inps,
                          (const WORD32 *const *)pp_inps_shape_8,
                          num_out_dims,
                          num_inp,
                          num_inp_dims,
                          axis);
  
}
