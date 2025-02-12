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

#include "xa_nn_fully_connected_common.h"

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32, xa_nn_fully_connected_f16,
    (WORD16 *__restrict__ p_out
     ,const WORD16 *__restrict__ p_weight
     ,const WORD16 *__restrict__ p_inp
     ,const WORD16 *__restrict__ p_bias
     ,WORD32  weight_depth
     ,WORD32  out_depth
    )
    )
#else /* #if !HAVE_HP_VFPU */
WORD32 xa_nn_fully_connected_f16
  (WORD16 *__restrict__ p_out
   ,const WORD16 *__restrict__ p_weight
   ,const WORD16 *__restrict__ p_inp
   ,const WORD16 *__restrict__ p_bias
   ,WORD32  weight_depth
   ,WORD32  out_depth
  )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_weight, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((out_depth <= 0), -1);

  WORD32 ret = 0;
  ret = xa_nn_matXvec_f16xf16_f16
    (p_out
     ,(WORD16 *)p_weight
     ,0
     ,(WORD16 *)p_inp
     ,0
     ,(WORD16 *)p_bias
     ,out_depth
     ,weight_depth
     ,0
     ,weight_depth
     ,0
    );
  return ret;
}
#endif /* #if !HAVE_HP_VFPU */

