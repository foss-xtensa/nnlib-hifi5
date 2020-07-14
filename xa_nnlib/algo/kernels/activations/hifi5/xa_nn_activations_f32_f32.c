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
/* Common helper macros. */
#include "common_fpu.h"
#include "xa_type_def.h"
#include "NatureDSP_Signal_math.h"

#if HAVE_VFPU

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_sigmoid_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_sigmoid_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_sigmoidf(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_tanh_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_tanh_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_tanhf(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_relu_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    FLOAT32       threshold,                   
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_relu_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    FLOAT32       threshold,                   /* threshold, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_reluf(p_out, p_vec, threshold, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_relu1_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_relu1_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_reluf(p_out, p_vec, 1.0f, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_relu6_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_relu6_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_reluf(p_out, p_vec, 6.0f, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_vec_softmax_f32_f32,(               
    FLOAT32       *  p_out,        
    const FLOAT32 *  p_vec,        
    WORD32        vec_length)                  )
#else
WORD32 xa_nn_vec_softmax_f32_f32(               
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    const FLOAT32 * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length)                  /* length of vectors */
{
  vec_softmaxf(p_out, p_vec, vec_length);
  return 0;
}
#endif /* !HAVE_VFPU */
#endif

