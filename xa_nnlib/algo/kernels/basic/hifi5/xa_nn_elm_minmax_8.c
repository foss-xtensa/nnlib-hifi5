/*******************************************************************************
* Copyright (c) 2018-2021 Cadence Design Systems, Inc.
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
 * xa_nn_elm_minmax_asym8s.c
 */

#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"

// out = (in1 > in2 ) ? in1 : in2 ;
WORD32 xa_nn_elm_max_8x8_8( WORD8* __restrict__ p_out,
                      const WORD8* __restrict__ p_in1,
                      const WORD8* __restrict__ p_in2,
                            WORD32              num_element)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in2, -1);

    /* Invalid input checks */
    XA_NNLIB_ARG_CHK_COND((num_element <= 0), -1);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in2, sizeof(WORD8), -1);

    const  UWORD8 num_elm_per_simd  = 8;
    const  UWORD8 num_simd_per_iter = 2;

    const UWORD16 num_elm_per_iter = num_elm_per_simd * num_simd_per_iter ;

    xtbool io_pointers_aligned =    ((uintptr_t)p_in1 % num_elm_per_iter == 0) &&
                                    ((uintptr_t)p_in2 % num_elm_per_iter == 0) &&
                                    ((uintptr_t)p_out % num_elm_per_iter == 0);

    UWORD32 num_simd_iter   = num_element / num_elm_per_iter ;

    WORD8 *p_a = (WORD8 *)p_in1;
    WORD8 *p_b = (WORD8 *)p_in2;
    WORD8 *p_c = (WORD8 *)p_out;

    ae_int8x8 a0_7, a8_15, b0_7, b8_15, c0_7, c8_15;

    UWORD32 i = 0;

    // iterate over the simd elements
    if(io_pointers_aligned){

        for(i = 0; i<num_simd_iter; i++){

            AE_L8X8X2_IP(a0_7, a8_15, (ae_int8x16 *)p_a, 16*sizeof(WORD8));
            AE_L8X8X2_IP(b0_7, b8_15, (ae_int8x16 *)p_b, 16*sizeof(WORD8));

            c0_7  = AE_MAX8(a0_7,  b0_7);
            c8_15 = AE_MAX8(a8_15, b8_15);

            AE_S8X8X2_IP(c0_7, c8_15, (ae_int8x16 *)p_c, 16*sizeof(WORD8));

        }

    } else {

        ae_valignx2 va_a = AE_LA128_PP(p_a);
        ae_valignx2 va_b = AE_LA128_PP(p_b);
        ae_valignx2 va_c = AE_ZALIGN128();

        for(i = 0; i<num_simd_iter; i++){

            AE_LA8X8X2_IP(a0_7, a8_15, va_a, (ae_int8x16 *)p_a);
            AE_LA8X8X2_IP(b0_7, b8_15, va_b, (ae_int8x16 *)p_b);

            c0_7  = AE_MAX8(a0_7,  b0_7);
            c8_15 = AE_MAX8(a8_15, b8_15);

            AE_SA8X8X2_IP(c0_7, c8_15, va_c, (ae_int8x16 *)p_c);

        }

        AE_SA128POS_FP(va_c, p_c);

    }

    // remaining scalar elements
    i *= num_elm_per_iter;

    for(; i<num_element; i++){

        p_out[i] = (p_in1[i] > p_in2[i]) ? p_in1[i] : p_in2[i];

    }

    return 0;

}


// out = (in1 < in2 ) ? in1 : in2 ;
WORD32 xa_nn_elm_min_8x8_8( WORD8* __restrict__ p_out,
                      const WORD8* __restrict__ p_in1,
                      const WORD8* __restrict__ p_in2,
                            WORD32              num_element)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in2, -1);

    /* Invalid input checks */
    XA_NNLIB_ARG_CHK_COND((num_element <= 0), -1);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in2, sizeof(WORD8), -1);

    const  UWORD8 num_elm_per_simd  = 8;
    const  UWORD8 num_simd_per_iter = 2;

    const UWORD16 num_elm_per_iter = num_elm_per_simd * num_simd_per_iter ;

    xtbool io_pointers_aligned =    ((uintptr_t)p_in1 % num_elm_per_iter == 0) &&
                                    ((uintptr_t)p_in2 % num_elm_per_iter == 0) &&
                                    ((uintptr_t)p_out % num_elm_per_iter == 0);

    UWORD32 num_simd_iter   = num_element / num_elm_per_iter ;

    WORD8 *p_a = (WORD8 *)p_in1;
    WORD8 *p_b = (WORD8 *)p_in2;
    WORD8 *p_c = (WORD8 *)p_out;

    ae_int8x8 a0_7, a8_15, b0_7, b8_15, c0_7, c8_15;

    UWORD32 i = 0;

    // iterate over the simd elements
    if(io_pointers_aligned){

        for(i = 0; i<num_simd_iter; i++){

            AE_L8X8X2_IP(a0_7, a8_15, (ae_int8x16 *)p_a, 16*sizeof(WORD8));
            AE_L8X8X2_IP(b0_7, b8_15, (ae_int8x16 *)p_b, 16*sizeof(WORD8));

            c0_7  = AE_MIN8(a0_7,  b0_7);
            c8_15 = AE_MIN8(a8_15, b8_15);

            AE_S8X8X2_IP(c0_7, c8_15, (ae_int8x16 *)p_c, 16*sizeof(WORD8));

        }

    } else {

        ae_valignx2 va_a = AE_LA128_PP(p_a);
        ae_valignx2 va_b = AE_LA128_PP(p_b);
        ae_valignx2 va_c = AE_ZALIGN128();

        for(i = 0; i<num_simd_iter; i++){

            AE_LA8X8X2_IP(a0_7, a8_15, va_a, (ae_int8x16 *)p_a);
            AE_LA8X8X2_IP(b0_7, b8_15, va_b, (ae_int8x16 *)p_b);

            c0_7  = AE_MIN8(a0_7,  b0_7);
            c8_15 = AE_MIN8(a8_15, b8_15);

            AE_SA8X8X2_IP(c0_7, c8_15, va_c, (ae_int8x16 *)p_c);

        }

        AE_SA128POS_FP(va_c, p_c);

    }

    // remaining scalar elements
    i *= num_elm_per_iter;

    for(; i<num_element; i++){

        p_out[i] = (p_in1[i] < p_in2[i]) ? p_in1[i] : p_in2[i];

    }

    return 0;

}
