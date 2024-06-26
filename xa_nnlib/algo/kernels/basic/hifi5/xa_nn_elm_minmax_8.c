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
/*
 * xa_nn_elm_minmax_8.c
 */

#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"

typedef enum {
    in1 = 0,
    in2 = 1,
} in_selector;

#define NUMDIMS_4D (4)
#define NUMDIMS_8D (8)

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

    const  WORD8 num_elm_per_simd  = 8;
    const  WORD8 num_simd_per_iter = 2;

    const WORD16 num_elm_per_iter = num_elm_per_simd * num_simd_per_iter ;

    xtbool io_pointers_aligned =    ((uintptr_t)p_in1 % num_elm_per_iter == 0) &&
                                    ((uintptr_t)p_in2 % num_elm_per_iter == 0) &&
                                    ((uintptr_t)p_out % num_elm_per_iter == 0);

    WORD32 num_simd_iter   = num_element / num_elm_per_iter ;

    WORD8 *p_a = (WORD8 *)p_in1;
    WORD8 *p_b = (WORD8 *)p_in2;
    WORD8 *p_c = (WORD8 *)p_out;

    ae_int8x8 a0_7, a8_15, b0_7, b8_15, c0_7, c8_15;

    WORD32 i = 0;

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

    const  WORD8 num_elm_per_simd  = 8;
    const  WORD8 num_simd_per_iter = 2;

    const WORD16 num_elm_per_iter = num_elm_per_simd * num_simd_per_iter ;

    xtbool io_pointers_aligned =    ((uintptr_t)p_in1 % num_elm_per_iter == 0) &&
                                    ((uintptr_t)p_in2 % num_elm_per_iter == 0) &&
                                    ((uintptr_t)p_out % num_elm_per_iter == 0);

    WORD32 num_simd_iter   = num_element / num_elm_per_iter ;

    WORD8 *p_a = (WORD8 *)p_in1;
    WORD8 *p_b = (WORD8 *)p_in2;
    WORD8 *p_c = (WORD8 *)p_out;

    ae_int8x8 a0_7, a8_15, b0_7, b8_15, c0_7, c8_15;

    WORD32 i = 0;

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

/*
 * The following four functions implement minumim/maximum operation with broadcast.
 * xa_nn_elm_min_4D_Bcast_8x8_8(), xa_nn_elm_max_4D_Bcast_8x8_8(),
 * xa_nn_elm_min_8D_Bcast_8x8_8(), xa_nn_elm_max_8D_Bcast_8x8_8().
 *
 * Although the number of dimensions is theoretically unrestricted, TFLM, as of v2.4.1, implements broadcast for 4/5/8D tensors only.
 * (Look for SubscriptToIndex() in xa_nn_common.h)
 *
 * So, HiFi5 NNLib, as of v1.5.0, contains two sets of functions, each implementing broadcast for 4D and 8D tensors.
 * 2/3 D tensors must be scaled up to 4D and use the 4D set of functions. Similary 5/6/7D must promote to 8D and use the 8D set.
 *
 * The cost of promotion is low as it requires prepending the array dimensions with '1' and will iterate the outer for-loops only once.
 * For example, 2x3 -> 1x1x2x3, and the two outer most loops iterate only once.
 *
 * TODO :
 * P1. Add negative check for extents and strides. DONE.
 *
 * 1. If the dimensions are large enough and the cost of promotion is _negligibly_ low, kill the 4D and use 8D.
 *
 */

WORD32 xa_nn_elm_min_4D_Bcast_8x8_8(
                            WORD8* __restrict__ p_out,          /* pointer to write output data to */
                            const int *const out_extents,       /* shape of output. This is the shape resulting after broadcast */

                            const  WORD8* __restrict__ p_in1,   /* pointer to unextended input data for tensor 1 */
                            const int * const in1_strides,      /* member 'strides' as defined in struct 'NdArrayDesc' for tensor 1*/

                            const  WORD8* __restrict__ p_in2,   /* pointer to unextended input data for tensor 2 */
                            const int * const in2_strides) {    /* member 'strides' as defined in struct 'NdArrayDesc' for tensor 2*/

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in2, -1);
    XA_NNLIB_ARG_CHK_PTR(out_extents, -1);
    XA_NNLIB_ARG_CHK_PTR(in1_strides, -1);
    XA_NNLIB_ARG_CHK_PTR(in2_strides, -1);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in2, sizeof(WORD8), -1);
    
    /* Invalid input checks */
    int i;
    for(i=0; i<NUMDIMS_4D; i++){
        XA_NNLIB_ARG_CHK_COND(out_extents[i] < 1, -1);
        XA_NNLIB_ARG_CHK_COND(in1_strides[i] < 0, -1);
        XA_NNLIB_ARG_CHK_COND(in2_strides[i] < 0, -1);
    }
    
    int linear_index = 0;
    int dim[NUMDIMS_4D] = {0};
    size_t index[NUMDIMS_4D][2];

    ae_int8x8 a, b, c;

    index[0][in1] = 0;
    index[0][in2] = 0;
    for(dim[0]=0; dim[0]<out_extents[0]; dim[0]++){
        index[1][in1] = index[0][in1];
        index[1][in2] = index[0][in2];
        for(dim[1]=0; dim[1]<out_extents[1]; dim[1]++){
            index[2][in1] = index[1][in1];
            index[2][in2] = index[1][in2];
            for(dim[2]=0; dim[2]<out_extents[2]; dim[2]++){
                index[3][in1] = index[2][in1];
                index[3][in2] = index[2][in2];
#pragma nounroll
                for(dim[3]=0; dim[3]<out_extents[3]; dim[3]++){
                
                    /*
                    if(__some_condition__){
                        printf("[%d][%d][%d][%d] %10d %5d %5d val = %5d %5d\n", dim[0], dim[1], dim[2], dim[3],
                            linear_index,
                            index[3][in1], index[3][in2],
                            p_in1[index[3][in1]], p_in2[index[3][in2]]);
                    }
                    */

                    a = AE_L8_X((ae_int8 *)p_in1, index[3][in1]);
                    b = AE_L8_X((ae_int8 *)p_in2, index[3][in2]);

                    c = AE_MIN8(a, b);

                    AE_S8_0_X(c, (ae_int8 *)p_out, linear_index);

                    linear_index++;

                    index[3][in1] += in1_strides[3];
                    index[3][in2] += in2_strides[3];
                }
                index[2][in1] += in1_strides[2];
                index[2][in2] += in2_strides[2];
            }
            index[1][in1] += in1_strides[1];
            index[1][in2] += in2_strides[1];
        }
        index[0][in1] += in1_strides[0];
        index[0][in2] += in2_strides[0];
    }

    return 0;

} /* xa_nn_elm_min_4D_Bcast_8x8_8 */

WORD32 xa_nn_elm_max_4D_Bcast_8x8_8(
                            WORD8* __restrict__ p_out,          /* pointer to write output data to */
                            const int *const out_extents,       /* shape of output. This is the resulting shape after broadcast */

                            const  WORD8* __restrict__ p_in1,   /* pointer to unextended input data for tensor 1 */
                            const int * const in1_strides,      /* member 'strides' as defined in struct 'NdArrayDesc' for tensor 1*/

                            const  WORD8* __restrict__ p_in2,   /* pointer to unextended input data for tensor 2 */
                            const int * const in2_strides) {    /* member 'strides' as defined in struct 'NdArrayDesc' for tensor 2*/

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in2, -1);
    XA_NNLIB_ARG_CHK_PTR(out_extents, -1);
    XA_NNLIB_ARG_CHK_PTR(in1_strides, -1);
    XA_NNLIB_ARG_CHK_PTR(in2_strides, -1);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in2, sizeof(WORD8), -1);
    
    /* Invalid input checks */
    int i;
    for(i=0; i<NUMDIMS_4D; i++){
        XA_NNLIB_ARG_CHK_COND(out_extents[i] < 1, -1);
        XA_NNLIB_ARG_CHK_COND(in1_strides[i] < 0, -1);
        XA_NNLIB_ARG_CHK_COND(in2_strides[i] < 0, -1);
    }
    
    int linear_index = 0;
    int dim[NUMDIMS_4D] = {0};
    size_t index[NUMDIMS_4D][2];

    ae_int8x8 a, b, c;

    index[0][in1] = 0;
    index[0][in2] = 0;
    for(dim[0]=0; dim[0]<out_extents[0]; dim[0]++){
        index[1][in1] = index[0][in1];
        index[1][in2] = index[0][in2];
        for(dim[1]=0; dim[1]<out_extents[1]; dim[1]++){
            index[2][in1] = index[1][in1];
            index[2][in2] = index[1][in2];
            for(dim[2]=0; dim[2]<out_extents[2]; dim[2]++){
                index[3][in1] = index[2][in1];
                index[3][in2] = index[2][in2];
#pragma nounroll
                for(dim[3]=0; dim[3]<out_extents[3]; dim[3]++){

                    /*
                    if(__some_condition__){
                        printf("[%d][%d][%d][%d] %10d %5d %5d val = %5d %5d\n", dim[0], dim[1], dim[2], dim[3],
                            linear_index,
                            index[3][in1], index[3][in2],
                            p_in1[index[3][in1]], p_in2[index[3][in2]]);
                    }
                    */

                    a = AE_L8_X((ae_int8 *)p_in1, index[3][in1]);
                    b = AE_L8_X((ae_int8 *)p_in2, index[3][in2]);

                    c = AE_MAX8(a, b);

                    AE_S8_0_X(c, (ae_int8 *)p_out, linear_index);

                    linear_index++;

                    index[3][in1] += in1_strides[3];
                    index[3][in2] += in2_strides[3];
                }
                index[2][in1] += in1_strides[2];
                index[2][in2] += in2_strides[2];
            }
            index[1][in1] += in1_strides[1];
            index[1][in2] += in2_strides[1];
        }
        index[0][in1] += in1_strides[0];
        index[0][in2] += in2_strides[0];
    }

    return 0;

} /* xa_nn_elm_max_4D_Bcast_8x8_8 */


WORD32 xa_nn_elm_min_8D_Bcast_8x8_8(
                            WORD8* __restrict__ p_out,          /* pointer to write output data to */
                            const int *const out_extents,       /* shape of output. This is the resulting shape after broadcast */

                            const  WORD8* __restrict__ p_in1,   /* pointer to unextended input data for tensor 1 */
                            const int * const in1_strides,      /* member 'strides' as defined in struct 'NdArrayDesc' for tensor 1*/

                            const  WORD8* __restrict__ p_in2,   /* pointer to unextended input data for tensor 2 */
                            const int * const in2_strides) {    /* member 'strides' as defined in struct 'NdArrayDesc' for tensor 2*/

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in2, -1);
    XA_NNLIB_ARG_CHK_PTR(out_extents, -1);
    XA_NNLIB_ARG_CHK_PTR(in1_strides, -1);
    XA_NNLIB_ARG_CHK_PTR(in2_strides, -1);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in2, sizeof(WORD8), -1);

    /* Invalid input checks */
    int i;
    for(i=0; i<NUMDIMS_8D; i++){
        XA_NNLIB_ARG_CHK_COND(out_extents[i] < 1, -1);
        XA_NNLIB_ARG_CHK_COND(in1_strides[i] < 0, -1);
        XA_NNLIB_ARG_CHK_COND(in2_strides[i] < 0, -1);
    }
                            
//    int linear_index = 0;
    int dim[NUMDIMS_8D] = {0};
    size_t index[2][NUMDIMS_8D];

    int num_scalars_loaded = 0;
    const int num_scalar_per_simd = 16;

    WORD8 ALIGN(16) a[16];
    WORD8 ALIGN(16) b[16];

    ae_int8x16 *out = (ae_int8x16 *)p_out;

    ae_int8x8 a0_7, a8_15, b0_7, b8_15, c0_7, c8_15;
    
    ae_valignx2 out_align = AE_ZALIGN128();

    for(dim[0]=0; dim[0]<out_extents[0]; dim[0]++){
        index[in1][0] = dim[0]*in1_strides[0];
        index[in2][0] = dim[0]*in2_strides[0];
        
        for(dim[1]=0; dim[1]<out_extents[1]; dim[1]++){
            index[in1][1] = index[in1][0] + dim[1]*in1_strides[1];
            index[in2][1] = index[in2][0] + dim[1]*in2_strides[1];
            
            for(dim[2]=0; dim[2]<out_extents[2]; dim[2]++){
                index[in1][2] = index[in1][1] + dim[2]*in1_strides[2];
                index[in2][2] = index[in2][1] + dim[2]*in2_strides[2];
                
                for(dim[3]=0; dim[3]<out_extents[3]; dim[3]++){
                    index[in1][3] = index[in1][2] + dim[3]*in1_strides[3];
                    index[in2][3] = index[in2][2] + dim[3]*in2_strides[3];
                    
                    for(dim[4]=0; dim[4]<out_extents[4]; dim[4]++){
                        index[in1][4] = index[in1][3] + dim[4]*in1_strides[4];
                        index[in2][4] = index[in2][3] + dim[4]*in2_strides[4];
                        
                        for(dim[5]=0; dim[5]<out_extents[5]; dim[5]++){
                            index[in1][5] = index[in1][4] + dim[5]*in1_strides[5];
                            index[in2][5] = index[in2][4] + dim[5]*in2_strides[5];
                            
                            for(dim[6]=0; dim[6]<out_extents[6]; dim[6]++){
                                index[in1][6] = index[in1][5] + dim[6]*in1_strides[6];
                                index[in2][6] = index[in2][5] + dim[6]*in2_strides[6];
                                
                                for(dim[7]=0; dim[7]<out_extents[7]; dim[7]++){
                                    index[in1][7] = index[in1][6] + dim[7]*in1_strides[7];
                                    index[in2][7] = index[in2][6] + dim[7]*in2_strides[7];

                                    /*
                                    if(__some_condition__){ 
                                        printf("[%3d][%3d][%3d][%3d][%3d][%3d][%3d][%3d] %5d || p_in1[%4d]=%4d p_in2[%5d]=%4d\n", dim[0], dim[1], dim[2], dim[3], dim[4], dim[5], dim[6], dim[7],
                                            linear_index,
                                            index[in1][7], p_in1[index[in1][7]],
                                            index[in2][7], p_in2[index[in2][7]]);
                                    }
                                    */

                                    a[num_scalars_loaded] = p_in1[index[in1][7]];
                                    b[num_scalars_loaded] = p_in2[index[in2][7]];

                                    num_scalars_loaded++;
//                                    linear_index++;

                                    if(num_scalars_loaded == num_scalar_per_simd){
                                       AE_L8X8X2_I(a0_7, a8_15, (ae_int8x16 *)a, 0);
                                       AE_L8X8X2_I(b0_7, b8_15, (ae_int8x16 *)b, 0);

                                       c0_7  = AE_MIN8(a0_7,  b0_7);   c8_15 = AE_MIN8(a8_15, b8_15);

                                       AE_SA8X8X2_IP(c0_7, c8_15, out_align, (ae_int8x16 *)out);
                                       num_scalars_loaded = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* There are some remaining elements */
    if(num_scalars_loaded != 0 ){
        AE_L8X8X2_I(a0_7, a8_15, (ae_int8x16 *)a, 0);
        AE_L8X8X2_I(b0_7, b8_15, (ae_int8x16 *)b, 0);

        c0_7  = AE_MIN8(a0_7,  b0_7);   c8_15 = AE_MIN8(a8_15, b8_15);
        
        AE_SAV8X8X2_XP(c0_7, c8_15, out_align, (ae_int8x16 *)out, num_scalars_loaded);
    }

    AE_SA128POS_FP(out_align, out);

    return 0;

} /* xa_nn_elm_min_8D_Bcast_8x8_8 */

WORD32 xa_nn_elm_max_8D_Bcast_8x8_8(
                            WORD8* __restrict__ p_out,          /* pointer to write output data to */
                            const int *const out_extents,       /* shape of output. This is the resulting shape after broadcast */

                            const  WORD8* __restrict__ p_in1,   /* pointer to unextended input data for tensor 1 */
                            const int * const in1_strides,      /* member 'strides' as defined in struct 'NdArrayDesc' for tensor 1*/

                            const  WORD8* __restrict__ p_in2,   /* pointer to unextended input data for tensor 2 */
                            const int * const in2_strides) {    /* member 'strides' as defined in struct 'NdArrayDesc' for tensor 2*/

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_in2, -1);
    XA_NNLIB_ARG_CHK_PTR(out_extents, -1);
    XA_NNLIB_ARG_CHK_PTR(in1_strides, -1);
    XA_NNLIB_ARG_CHK_PTR(in2_strides, -1);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_in2, sizeof(WORD8), -1);
    
    /* Invalid input checks */
    int i;
    for(i=0; i<NUMDIMS_8D; i++){
        XA_NNLIB_ARG_CHK_COND(out_extents[i] < 1, -1);
        XA_NNLIB_ARG_CHK_COND(in1_strides[i] < 0, -1);
        XA_NNLIB_ARG_CHK_COND(in2_strides[i] < 0, -1);
    }

//    int linear_index = 0;
    int dim[NUMDIMS_8D] = {0};
    size_t index[2][NUMDIMS_8D];

    int num_scalars_loaded = 0;
    const int num_scalar_per_simd = 16;

    WORD8 ALIGN(16) a[16];
    WORD8 ALIGN(16) b[16];

    ae_int8x16 *out = (ae_int8x16 *)p_out;

    ae_int8x8 a0_7, a8_15, b0_7, b8_15, c0_7, c8_15;
    
    ae_valignx2 out_align = AE_ZALIGN128();

    for(dim[0]=0; dim[0]<out_extents[0]; dim[0]++){
        index[in1][0] = dim[0]*in1_strides[0];
        index[in2][0] = dim[0]*in2_strides[0];
        
        for(dim[1]=0; dim[1]<out_extents[1]; dim[1]++){
            index[in1][1] = index[in1][0] + dim[1]*in1_strides[1];
            index[in2][1] = index[in2][0] + dim[1]*in2_strides[1];
            
            for(dim[2]=0; dim[2]<out_extents[2]; dim[2]++){
                index[in1][2] = index[in1][1] + dim[2]*in1_strides[2];
                index[in2][2] = index[in2][1] + dim[2]*in2_strides[2];
                
                for(dim[3]=0; dim[3]<out_extents[3]; dim[3]++){
                    index[in1][3] = index[in1][2] + dim[3]*in1_strides[3];
                    index[in2][3] = index[in2][2] + dim[3]*in2_strides[3];
                    
                    for(dim[4]=0; dim[4]<out_extents[4]; dim[4]++){
                        index[in1][4] = index[in1][3] + dim[4]*in1_strides[4];
                        index[in2][4] = index[in2][3] + dim[4]*in2_strides[4];
                        
                        for(dim[5]=0; dim[5]<out_extents[5]; dim[5]++){
                            index[in1][5] = index[in1][4] + dim[5]*in1_strides[5];
                            index[in2][5] = index[in2][4] + dim[5]*in2_strides[5];
                            
                            for(dim[6]=0; dim[6]<out_extents[6]; dim[6]++){
                                index[in1][6] = index[in1][5] + dim[6]*in1_strides[6];
                                index[in2][6] = index[in2][5] + dim[6]*in2_strides[6];
                                
                                for(dim[7]=0; dim[7]<out_extents[7]; dim[7]++){
                                    index[in1][7] = index[in1][6] + dim[7]*in1_strides[7];
                                    index[in2][7] = index[in2][6] + dim[7]*in2_strides[7];

                                    /*
                                    printf("%4d.%4d.%4d.%4d.%4d.%4d.%4d.%4d %5d=%5d.%5d\n", dim[0], dim[1], dim[2], dim[3], dim[4], dim[5], dim[6], dim[7],
                                    linear_index,
                                    index[in1][7], index[in2][7]);
                                    */

                                    a[num_scalars_loaded] = p_in1[index[in1][7]];
                                    b[num_scalars_loaded] = p_in2[index[in2][7]];

                                    num_scalars_loaded++;
//                                    linear_index++;

                                    if(num_scalars_loaded == num_scalar_per_simd){
                                       AE_L8X8X2_I(a0_7, a8_15, (ae_int8x16 *)a, 0);
                                       AE_L8X8X2_I(b0_7, b8_15, (ae_int8x16 *)b, 0);

                                       c0_7  = AE_MAX8(a0_7,  b0_7);   c8_15 = AE_MAX8(a8_15, b8_15);

                                       AE_SA8X8X2_IP(c0_7, c8_15, out_align, (ae_int8x16 *)out);
                                       num_scalars_loaded = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* There are some remaining elements */
    if(num_scalars_loaded != 0 ){
        AE_L8X8X2_I(a0_7, a8_15, (ae_int8x16 *)a, 0);
        AE_L8X8X2_I(b0_7, b8_15, (ae_int8x16 *)b, 0);

        c0_7  = AE_MAX8(a0_7,  b0_7);   c8_15 = AE_MAX8(a8_15, b8_15);
        
        AE_SAV8X8X2_XP(c0_7, c8_15, out_align, (ae_int8x16 *)out, num_scalars_loaded);
    }

    AE_SA128POS_FP(out_align, out);
    
    return 0;
} /* xa_nn_elm_max_8D_Bcast_8x8_8 */


