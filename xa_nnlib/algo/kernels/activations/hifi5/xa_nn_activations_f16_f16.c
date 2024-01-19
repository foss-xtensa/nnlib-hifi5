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
/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"
#include "xa_type_def.h"
#include "../../../ndsp/hifi5/include/NatureDSP_Signal_math.h"
#include "xa_nnlib_err_chk.h"
#include <math.h>

#if HAVE_VFPU

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_sigmoid_f16_f16,(
    WORD16       *  p_out,
    const WORD16 *  p_vec,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_sigmoid_f16_f16(
    WORD16       * __restrict__ p_out,        /* result, floating point */
    const WORD16 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  xa_nnlib_vec_sigmoid_fp16(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_HP_VFPU */

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_tanh_f16_f16,(
    WORD16        *  p_out,
    const WORD16  *  p_vec,
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_tanh_f16_f16(
    WORD16       * __restrict__ p_out,        /* result, floating point */
    const WORD16 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  xa_nnlib_vec_tanh_fp16(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_HP_VFPU */
#endif

