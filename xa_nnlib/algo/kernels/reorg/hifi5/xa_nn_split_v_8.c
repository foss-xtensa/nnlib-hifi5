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
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_macros_hifi5.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_common.h"

WORD32 xa_nn_split_v_8_8(WORD8 ** __restrict__ pp_outs
                        ,const WORD32 *const *pp_outs_shape
                        ,const WORD8 *p_inp
                        ,const WORD32 *const p_inp_shape
                        ,WORD32 num_out
                        ,WORD32 num_out_dims
                        ,WORD32 num_inp_dims
                        ,WORD32 axis)
{
  XA_NNLIB_ARG_CHK_PTR(pp_outs, -1);
  XA_NNLIB_ARG_CHK_PTR(pp_outs_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(pp_outs, sizeof(WORD8 *), -1);
  XA_NNLIB_ARG_CHK_ALIGN(pp_outs_shape, sizeof(WORD32 *), -1);
  //Validate Arguments
  XA_NNLIB_ARG_CHK_COND((num_out <= 0 || num_out > 10), -1);
  XA_NNLIB_ARG_CHK_COND((num_inp_dims <= 0 || num_inp_dims > 6), -1);
  XA_NNLIB_ARG_CHK_COND((num_inp_dims != num_out_dims), -1);
  XA_NNLIB_ARG_CHK_COND((axis < -num_out_dims || axis >= num_out_dims), -1);

  int i = 0, j = 0;
  for(i = 0; i < num_inp_dims; i++)
  { 
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[i] <= 0), -1);
  }

  if(axis < 0)
    axis = num_inp_dims + axis;

  WORD32 concat_size = 0;
  for (i = 0; i < num_out; i++)
  {
    XA_NNLIB_ARG_CHK_PTR(pp_outs[i], -1);
    XA_NNLIB_ARG_CHK_PTR(pp_outs_shape[i], -1);
    XA_NNLIB_ARG_CHK_ALIGN(pp_outs_shape[i], sizeof(WORD32), -1);
#pragma loop_count min=1
    for(j = 0; j < num_inp_dims; j++)
    {
      XA_NNLIB_ARG_CHK_COND((p_inp_shape[j] != pp_outs_shape[i][j] && j != axis), -1);
    }
    XA_NNLIB_ARG_CHK_COND((pp_outs_shape[i][axis] <= 0), -1);
    concat_size += pp_outs_shape[i][axis];
  }

  XA_NNLIB_ARG_CHK_COND((p_inp_shape[axis] != concat_size), -1);
  
  /*Calculate outer and inner size for axis*/
  WORD32 outer_size = 1;
#pragma no_simd
  for(i = 0; i < axis; i++)
  {
    outer_size *= p_inp_shape[i];
  }

  WORD32 base_inner_size = 1;
#pragma no_simd
  for(i = axis + 1; i < num_inp_dims; i++)
  {
    base_inner_size *= p_inp_shape[i];
  }
  if(outer_size == 1)
  {
    const WORD8 *ptmp_inp = p_inp;
    for(int i = 0; i < num_out; i++)
    {
      const WORD32 copy_size = pp_outs_shape[i][axis] * base_inner_size;
      
      {      
        const WORD8 *input_ptr = ptmp_inp;
        WORD8* output_ptr = pp_outs[i];
        
        {
          MEMCPY_8b(output_ptr, input_ptr, copy_size);
        }
        ptmp_inp += copy_size;
      }
    }
  }
  else
  {
    WORD32 inp_axis = p_inp_shape[axis];
    WORD32 next_inp_idx = 0;
    for(j = 0; j < num_out; j++)
    {
      WORD8 *output_ptr = pp_outs[j];
      WORD32 copy_size = pp_outs_shape[j][axis] * base_inner_size;
      if(copy_size <= 16)
      {
        ae_int8x8 d_inp1, d_inp2;
        ae_valignx2 input_valign, output_valign;
        output_valign = AE_ZALIGN128();
#pragma loop_count min=1
#pragma concurrent        
        for(i = 0; i < outer_size; i++)
        {
          ae_int8x16 *input_ptr = (ae_int8x16*) (p_inp + next_inp_idx + i* inp_axis * base_inner_size);
          input_valign = AE_LA128_PP((ae_int8x16 *)input_ptr);
          AE_LAV8X8X2_XP(d_inp1, d_inp2, input_valign, input_ptr, copy_size);
          AE_SAV8X8X2_XP(d_inp1, d_inp2, output_valign, (ae_int8x16*)output_ptr, copy_size);
        }
        AE_SA128POS_FP(output_valign, (void *)output_ptr);
      }
      else
      {
#pragma loop_count min=1        
        for(i = 0; i < outer_size; i++)
        {
          const WORD8* input_ptr = p_inp + next_inp_idx + i * inp_axis * base_inner_size;
          MEMCPY_8b(output_ptr, input_ptr, (int)(copy_size * sizeof(WORD8)));
          output_ptr += copy_size;
        }
      }
      next_inp_idx += copy_size;
    }
  }
  return 0;
}
