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

/*
 * xa_nn_broadcast_32_32.c
 */

#include "xa_nnlib_common.h"

#define NUMDIMS_MAX 8

#ifdef AE_LAV32X2X2_XP
  #define AE_SW_LAV32X2X2_XP(d1, d2, va, ptr, off)  AE_LAV32X2X2_XP(d1, d2, va, ptr, off)
#else
  #define AE_SW_LAV32X2X2_XP(d1, d2, va, ptr, off) \
  { \
    ae_int16x4 d_out16_0, d_out16_1; \
    ae_int16x8 *ptr_temp = (ae_int16x8 *)ptr; \
    AE_LAV16X4X2_XP(d_out16_0, d_out16_1, va, ptr_temp, off); \
    d_out16_0 = AE_SEL16_2301(d_out16_0, d_out16_0); \
    d_out16_1 = AE_SEL16_2301(d_out16_1, d_out16_1); \
    d1 = AE_MOVINT32X2_FROMINT16X4(d_out16_0); \
    d2 = AE_MOVINT32X2_FROMINT16X4(d_out16_1); \
    ptr = (ae_int32x4*)ptr_temp; \
  }
#endif
#ifdef AE_SAV32X2X2_XP
  #define AE_SW_SAV32X2X2_XP(d1, d2, va, ptr, off)  AE_SAV32X2X2_XP(d1, d2, va, ptr, off)
#else
  #define AE_SW_SAV32X2X2_XP(d1, d2, va, ptr, off) \
  { \
    ae_int16x4 d_in16_0, d_in16_1; \
    d_in16_0 = AE_MOVINT16X4_FROMINT32X2(d1); \
    d_in16_1 = AE_MOVINT16X4_FROMINT32X2(d2); \
    d_in16_0 = AE_SEL16_2301(d_in16_0, d_in16_0); \
    d_in16_1 = AE_SEL16_2301(d_in16_1, d_in16_1); \
    ae_int16x8 *ptr_temp = (ae_int16x8 *)ptr; \
    AE_SAV16X4X2_XP(d_in16_0, d_in16_1, va, ptr_temp, off); \
    ptr = (ae_int32x4*)ptr_temp; \
  }
#endif

typedef struct bcast_expansion_struct_{
    size_t load_num_elem;
    int    replicate_loadedElm_times;
    int    repeat_operation;
} bcast_expansion_rule ;

static const WORD32* broadcast_node_32(bcast_expansion_rule *steps, unsigned int step_id,
        WORD32 *dst, const WORD32 *src);

static void internal_elm_broadcast_2D_32_32(WORD32 * __restrict__ p_out,
                    const    WORD32 * __restrict__ p_inp,
                             WORD32  out_lc,
                             WORD32  in_lc,
                             WORD32 repeat_count)
{
  int i, j, k;

  ae_int32x4 * __restrict__ p_i = (ae_int32x4 *)p_inp;
  ae_int32x4 * __restrict__ p_o = (ae_int32x4 *)p_out;

  int num_simd4_ops;
  int num_scalar_ops;

  num_simd4_ops = in_lc >> 2;
  num_scalar_ops = in_lc & 3;

  ae_int32x2 x1, x2;
  for(k=0; k < repeat_count; k++)
  {
    for(i = 0; i < out_lc; i++)
    {
        p_i = (ae_int32x4 *)&p_inp[k*in_lc];
        if(((((unsigned)p_i)&0xF) == 0) && ((((unsigned)p_o)&0xF) == 0))
        {
            for(j = 0; j < num_simd4_ops; j++)
            {
                AE_L32X2X2_IP(x1, x2, p_i, 4 * sizeof(WORD32));
                AE_S32X2X2_IP(x1, x2, p_o, 4 * sizeof(WORD32)); 
            }
        }
        else
        {
            ae_valignx2  vinp, out_a = AE_ZALIGN128();
            vinp = AE_LA128_PP(p_i);
            for(j = 0; j < num_simd4_ops; j++)
            {
                AE_LA32X2X2_IP(x1, x2, vinp, p_i);
                AE_SA32X2X2_IP(x1, x2, out_a, p_o); 
            }
            AE_SA128POS_FP(out_a, p_o);
        }
        if(num_scalar_ops !=0)
        {
            ae_valignx2  vinp, out_a = AE_ZALIGN128();
            vinp = AE_LA128_PP(p_i);
            AE_SW_LAV32X2X2_XP(x1, x2, vinp, p_i, num_scalar_ops* sizeof(WORD32));
            AE_SW_SAV32X2X2_XP(x1, x2, out_a, p_o,num_scalar_ops* sizeof(WORD32));
            AE_SA128POS_FP(out_a, p_o);
        }
    }
  }
}

static void internal_elm_broadcast_32_32(WORD32 * __restrict__ p_out,
                    const    WORD32 * __restrict__ p_inp,
                             WORD32  num_elm,
                             WORD32 repeat_count)
{
  int i, j;
  ae_int32   * __restrict__ p_i = (ae_int32  *)p_inp;
  ae_int32x4   *__restrict__  p_o =  (ae_int32x4 *)p_out;

  const int num_simd4_ops = num_elm >> 2;
  const int num_scalar_ops = num_elm & 3;

  ae_int32x2 x1, x2;
  for (j =0; j< repeat_count; j++)
  {
    p_i = (ae_int32 *)&p_inp[j];
    x1 = AE_L32_I((ae_int32 *)p_i, 0);
    x2 = AE_L32_I((ae_int32 *)p_i, 0);

    ae_valignx2  out_a = AE_ZALIGN128();
    for(i = 0; i < num_simd4_ops; i++)
    {
        AE_SA32X2X2_IP(x1, x2, out_a, p_o); 
    }
    if(num_scalar_ops !=0)
    {
        AE_SW_SAV32X2X2_XP(x1, x2, out_a, p_o,num_scalar_ops* sizeof(WORD32));
    }
    AE_SA128POS_FP(out_a, p_o);
  }
}

WORD32 xa_nn_broadcast_32_32( WORD32* __restrict__ p_out,   /* pointer to write broadcasted output data to */
        const int *const out_shape,                         /* output shape resulting after broadcast */
        const WORD32* __restrict__ p_in,                    /* pointer to unextended input data */
        const int * const in_shape,                         /* input shape */
        int num_dims)
{

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(out_shape, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in, -1);
    XA_NNLIB_ARG_CHK_PTR(in_shape, -1);

    /* IO pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in, sizeof(WORD32), -1);

    /* IO shape pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(in_shape, sizeof(WORD32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(out_shape, sizeof(WORD32), -1);

    /* Check if number of dims is valid */
    XA_NNLIB_ARG_CHK_COND(num_dims<=0 || num_dims>8, -1);

    int i = 0;

    /* Check for valid IO shapes */
    for(i=0; i<num_dims; i++){
        XA_NNLIB_CHK_COND(in_shape[i]<=0, -1);
        XA_NNLIB_CHK_COND(out_shape[i]<=0, -1);
    }

    /* Check if input shape can be broadcasted to requested output shape */
    for(i=0; i<num_dims; i++){
        if(in_shape[i] != out_shape[i]){
            /* in_shape is either same as out_shape or 1 */
            XA_NNLIB_CHK_COND( in_shape[i] != 1, -1);
        }
    }

    /* bcast_expansion_steps contains a sequence to steps execute for a broadcast op */
    bcast_expansion_rule bcast_expansion_steps[NUMDIMS_MAX] = {{0}};

    int k=0;
    int dim=0;
    const void *res=0;
    int num_elem_load = 1;
    int num_copy_times = 1;
    int num_repeat = 1;

    dim = num_dims-1;
    while(dim>=0){

        /* Find the sub-matrix size */
        while(in_shape[dim] != 1 && dim>=0){
            num_elem_load *= out_shape[dim];
            dim--;
        }

        /* Find the number of times this sub-matrix needs to be copied */
        num_copy_times = 1;
        while(in_shape[dim] == 1 && dim>=0){
            num_copy_times *= out_shape[dim];
            dim--;
        }

        /* Find the number of times the above copy needs to be repeated */
        num_repeat = 1;
        while(in_shape[dim] != 1 && dim>=0){
            num_repeat *= 1 * out_shape[dim];
            dim--;
        }

        bcast_expansion_steps[k].load_num_elem  = num_elem_load;
        bcast_expansion_steps[k].replicate_loadedElm_times = num_copy_times;
        bcast_expansion_steps[k].repeat_operation = num_repeat;
        k++;

        num_elem_load = num_elem_load * num_copy_times * num_repeat;
    }

    res = broadcast_node_32(bcast_expansion_steps, num_dims-1, p_out, p_in);
    (void)res; /* Unused return value */

    return 0;
}

static const WORD32* broadcast_node_32(bcast_expansion_rule *steps, 
                                       unsigned int step_id,
                                       WORD32 *dst, const WORD32 *src) {
    int step_itr=0;
    bcast_expansion_rule *step = NULL;

    // ignore steps that are null
    while(steps[step_id].repeat_operation == 0 && step_id>0){
        step_id--;
    }

    // step is now the parent node for this iteration
    step = &steps[step_id];
    size_t numLoadedElm = step->load_num_elem;

    WORD32 *cp_dst = dst;
    const WORD32 *cp_src = src;

    if(step_id > 0){
        for(step_itr=0; step_itr<step->repeat_operation; step_itr++){
            src = broadcast_node_32(steps, step_id-1, dst, src);
            cp_src = dst;
            cp_dst = dst + numLoadedElm;
            internal_elm_broadcast_2D_32_32(cp_dst, cp_src, step->replicate_loadedElm_times - 1, numLoadedElm, 1);
            dst = cp_dst + numLoadedElm*(step->replicate_loadedElm_times - 1);
        }
        return src;
    } else {
        if(numLoadedElm == 1){
            internal_elm_broadcast_32_32(cp_dst, cp_src, step->replicate_loadedElm_times, step->repeat_operation);
            cp_src += step->repeat_operation;
        } else {
            internal_elm_broadcast_2D_32_32(cp_dst, cp_src, step->replicate_loadedElm_times, numLoadedElm, step->repeat_operation);
            cp_src += numLoadedElm*step->repeat_operation;
        }
        return cp_src;
    }
}
