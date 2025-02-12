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
#include "xa_nnlib_common_fpu.h"
#include "xa_type_def.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_err_chk.h"

WORD32 xa_nn_fully_connected_v2_asym8sxasym8s_asym8s
  (WORD8 *__restrict__ p_out
   ,const WORD8 *__restrict__ p_weight
   ,const WORD8 *__restrict__ p_inp
   ,const WORD32 *__restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
   ,WORD32  input_zero_bias
   ,WORD32  weight_zero_bias
   ,WORD32  out_multiplier
   ,WORD32  out_shift
   ,WORD32  out_zero_bias
   ,WORD32  out_activation_min
   ,WORD32  out_activation_max
   ,xa_dma_cfg_t *p_dma_cfg
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD64), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(WORD64), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD64), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);

  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((weight_zero_bias < -127 || weight_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_v2_asym8sxasym8s_asym8s
    (p_out
     ,p_weight
     ,p_inp
     ,p_bias
     ,out_depth
     ,weight_depth
     ,weight_depth
     ,weight_zero_bias
     ,input_zero_bias
     ,out_multiplier
     ,out_shift
     ,out_zero_bias
     ,out_activation_min
     ,out_activation_max
     ,p_dma_cfg
    );
  return ret;
}

WORD32 xa_nn_fully_connected_v2_sym8sxsym16s_sym16s
  (WORD16 *__restrict__ p_out
   ,const WORD8 *__restrict__ p_weight
   ,const WORD16 *__restrict__ p_inp
   ,const WORD64 *__restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
   ,WORD32  out_multiplier
   ,WORD32  out_shift
   ,WORD32  out_activation_min
   ,WORD32  out_activation_max
   ,xa_dma_cfg_t *p_dma_cfg
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD64), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 15), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_v2_sym8sxsym16s_sym16s
    (p_out
     ,p_weight
     ,p_inp
     ,p_bias
     ,out_depth
     ,weight_depth
     ,weight_depth
     ,out_multiplier
     ,out_shift
     ,out_activation_min
     ,out_activation_max
     ,p_dma_cfg
    );
  return ret;
}
