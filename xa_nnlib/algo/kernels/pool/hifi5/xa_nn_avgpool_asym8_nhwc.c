/*******************************************************************************
* Copyright (c) 2018-2023 Cadence Design Systems, Inc.
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
#include "common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nn_avgpool_state.h"
#include <math.h>

#define INCR_N_PLANE(ptr, n, plane_size) \
    ptr = (ptr) + ((n) * (plane_size));

#define INCR_PLANE_IF_HEIGHT(ptr, height, plane_size) \
        if(height) \
        { \
            INCR_N_PLANE(ptr, 1, plane_size); \
            height--; \
        }\
        else\
        {\
            ptr = (WORD8 *)p_zeros_mem;\
        }

#define INCR_N_ROW(ptr, n, row_size) \
    ptr = (ptr) + ((n) * (row_size));

#define INCR_ROW_IF_WIDTH_32(ptr, width, row_size) \
        if(width)\
        { \
            INCR_N_ROW(ptr, 1, row_size);\
            width--;\
        }\
        else\
        {\
            ptr = (WORD32 *)p_zeros_mem;\
        }

#define INCR_ROW_IF_WIDTH_16(ptr, width, row_size) \
        if(width)\
        { \
            INCR_N_ROW(ptr, 1, row_size);\
            width--;\
        }\
        else\
        {\
            ptr = (WORD16 *)p_zeros_mem;\
        }


/* Average pooling without using extra copy of input data
 * Works with unaligned input, output.
 */

void xa_nn_avgpool_asym8_hwc_16(
      UWORD8* __restrict__ p_out,
const UWORD8* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in,
      pVOID    p_zeros_mem,
      WORD32   *p_den_height,
      WORD32   *p_den_width)
{
    WORD16 *p_scratch = (WORD16 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int plane_size;
    WORD8 * p_src1, * p_src2, * p_src3;
    WORD16 * p_src1_w, * p_src2_w, * p_src3_w;
    WORD8 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    ae_int16x8 * __restrict p_src1_temp_w, * __restrict p_src2_temp_w, * __restrict p_src3_temp_w;
    ae_int16x8 * p_dst, *p_dst_temp;
    ae_int32x4 * p_dst_temp_w, *p_src1_32x4;
    UWORD8 *p_out_temp;
    ae_int16x8 * p_src1_scratch;
    ae_valignx2 align_src1, align_src2, align_src3, align_dst;

    int i;
    WORD16 *p_dst_pad;
    ae_int8x8 ZERO8 = AE_MOVDA8(0);
    ae_int16x4 ONE16 = AE_MOVDA16(1);

    plane_size = input_width * input_channels;

    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row, end_row;
        int start_plane, end_plane;

        /* Pool height processing */
        /* Processing width-channel planes for pool_height no. of planes  */
        /* Calculating avg of k_h w-c planes and saving into the scratch memory*/

        start_plane  = itr_oh * y_stride - y_padding;
        end_plane = start_plane + kernel_height;
        LIMIT(start_plane , 0, input_height);
        LIMIT(end_plane , 0, input_height);
        pool_height = end_plane - start_plane;
        p_dst = (ae_int16x8 *)p_scratch ;

        if(pool_height)
        {
            p_src1 = (WORD8 *)p_inp;
            INCR_N_PLANE(p_src1, start_plane, plane_size);
            pool_height--;

            p_src2 = p_src1;
            INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

            p_src3 = p_src2;
            INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

            /* 1st instance: Compare three rows per iteration */
            {
                p_dst_temp = p_dst;
                p_src1_temp = p_src1;
                p_src2_temp = p_src2;
                p_src3_temp = p_src3;

                align_src1 = AE_LA128_PP(p_src1_temp);
                align_src2 = AE_LA128_PP(p_src2_temp);
                align_src3 = AE_LA128_PP(p_src3_temp);

                for(i = 0; i < (plane_size >> 4); i++)
                {
                    ae_int16x4 out1, out2, out3, out4;
                    ae_int8x8 i1, i2, i3, i4, i5, i6;
                    AE_LA8X8X2_IP(i1, i2, align_src1, (ae_int8x16 *)p_src1_temp);
                    AE_LA8X8X2_IP(i3, i4, align_src2, (ae_int8x16 *)p_src2_temp);
                    AE_LA8X8X2_IP(i5, i6, align_src3, (ae_int8x16 *)p_src3_temp);
                    AE_ADDW8U(out1, out2, i1, i3);
                    AE_ADDW8U(out3, out4, i2, i4);
                    AE_ACCW8U(out1, out2, i5, ZERO8);
                    AE_ACCW8U(out3, out4, i6, ZERO8);
                    AE_S16X4X2_IP(out1, out2, p_dst_temp, 16);
                    AE_S16X4X2_IP(out3, out4, p_dst_temp, 16);
                }

                /* remainder part for input_width */
                int rem_itr = (plane_size & 15);
                {
                    int rem_off0, rem_off1;
                    ae_int16x4 out1, out2, out3, out4;
                    ae_int8x8 i1, i2, i3, i4, i5, i6;
                    align_dst = AE_ZALIGN128();

                    rem_off0 = (rem_itr > 8 ? 8 : rem_itr) << 1;
                    rem_off1 = (rem_itr - 8 < 0 ? 0 : rem_itr - 8) << 1;

                    AE_LAV8X8X2_XP(i1, i2, align_src1, (ae_int8x16 *)p_src1_temp, rem_itr);
                    AE_LAV8X8X2_XP(i3, i4, align_src2, (ae_int8x16 *)p_src2_temp, rem_itr);
                    AE_LAV8X8X2_XP(i5, i6, align_src3, (ae_int8x16 *)p_src3_temp, rem_itr);
                    AE_ADDW8U(out1, out2, i1, i3);
                    AE_ADDW8U(out3, out4, i2, i4);
                    AE_ACCW8U(out1, out2, i5, ZERO8);
                    AE_ACCW8U(out3, out4, i6, ZERO8);
                    AE_SAV16X4X2_XP(out1, out2, align_dst, p_dst_temp, rem_off0);
                    AE_SAV16X4X2_XP(out3, out4, align_dst, p_dst_temp, rem_off1);
                    AE_SA128POS_FP(align_dst, p_dst_temp);
                }
            }

            if(pool_height)
            {
                p_src2 = p_src3;
                INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

                p_src3 = p_src2;
                INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

                do
                {
                    p_dst_temp = p_dst;
                    p_src1_scratch = p_dst;
                    p_src2_temp = p_src2;
                    p_src3_temp = p_src3;

                    align_src2 = AE_LA128_PP(p_src2_temp);
                    align_src3 = AE_LA128_PP(p_src3_temp);

                    for(i = 0; i < (plane_size >> 4); i++)
                    {
                        ae_int16x4 j1, j2, j3, j4;
                        ae_int8x8 i1, i2, i3, i4;
                        AE_L16X4X2_IP(j1, j2, p_src1_scratch, 16);
                        AE_L16X4X2_IP(j3, j4, p_src1_scratch, 16);
                        AE_LA8X8X2_IP(i1, i2, align_src2, (ae_int8x16 *)p_src2_temp);
                        AE_LA8X8X2_IP(i3, i4, align_src3, (ae_int8x16 *)p_src3_temp);
                        AE_ACCW8U(j1, j2, i1, i3);
                        AE_ACCW8U(j3, j4, i2, i4);
                        AE_S16X4X2_IP(j1, j2, p_dst_temp, 16);
                        AE_S16X4X2_IP(j3, j4, p_dst_temp, 16);
                    }

                    /* remainder part */
                    int rem_itr = (plane_size & 15);
                    {
                        int rem_off0, rem_off1;
                        ae_int16x4 j1, j2, j3, j4;
                        ae_int8x8 i1, i2, i3, i4;
                        align_src1 = AE_LA128_PP(p_src1_scratch);
                        align_dst = AE_ZALIGN128();

                        rem_off0 = (rem_itr > 8 ? 8 : rem_itr) << 1;
                        rem_off1 = (rem_itr - 8 < 0 ? 0 : rem_itr - 1) << 1;

                        AE_LAV16X4X2_XP(j1, j2, align_src1, p_src1_scratch, rem_off0);
                        AE_LAV16X4X2_XP(j3, j4, align_src1, p_src1_scratch, rem_off1);
                        AE_LAV8X8X2_XP(i1, i2, align_src2, (ae_int8x16 *)p_src2_temp, rem_itr);
                        AE_LAV8X8X2_XP(i3, i4, align_src3, (ae_int8x16 *)p_src3_temp, rem_itr);
                        AE_ACCW8U(j1, j2, i1, i3);
                        AE_ACCW8U(j3, j4, i2, i4);
                        AE_SAV16X4X2_XP(j1, j2, align_dst, p_dst_temp, rem_off0);
                        AE_SAV16X4X2_XP(j3, j4, align_dst, p_dst_temp, rem_off1);
                        AE_SA128POS_FP(align_dst, p_dst_temp);
                    }

                    if(!pool_height)
                        break;

                    p_src2 = p_src3;
                    INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

                    p_src3 = p_src2;
                    INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

                }while(1);
            }
        }
        else
        {
            /* If there is no valid input present, fill the output with zeros */
            p_dst_pad = (WORD16 *)p_scratch;
            for(i = 0; i < plane_size; i++)
            {
                p_dst_pad[i] =  0; //-INFINITY;
            }
        }

        /* Pool width processing */
        /* Processing the output of the height processing block (which is a w-c plane); along width */
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            start_row  = itr_ow * x_stride - x_padding;
            end_row = start_row + kernel_width;
            LIMIT(start_row , 0, input_width);
            LIMIT(end_row , 0, input_width);
            pool_width = end_row - start_row;
            p_out_temp = p_out + (itr_oh*out_width*input_channels) + (itr_ow*input_channels);
            p_dst = (ae_int16x8 *)((WORD16 *)p_scratch + ALIGNED_SIZE(plane_size, ALIGNMENT/sizeof(WORD16)));

            if(pool_width)
            {
                p_src1_w = (WORD16 *)p_scratch;
                INCR_N_ROW(p_src1_w, start_row, input_channels);
                pool_width--;

                p_src2_w = p_src1_w;
                INCR_ROW_IF_WIDTH_16(p_src2_w, pool_width, input_channels);

                p_src3_w = p_src2_w;
                INCR_ROW_IF_WIDTH_16(p_src3_w, pool_width, input_channels);

                // 1st instance
                {
                    p_dst_temp_w = (ae_int32x4 *)p_dst;
                    p_src1_temp_w = (ae_int16x8 *)p_src1_w;
                    p_src2_temp_w = (ae_int16x8 *)p_src2_w;
                    p_src3_temp_w = (ae_int16x8 *)p_src3_w;

                    /* prime */
                    align_src1 = AE_LA128_PP(p_src1_temp_w);
                    align_src2 = AE_LA128_PP(p_src2_temp_w);
                    align_src3 = AE_LA128_PP(p_src3_temp_w);

                    for(i = 0; i < (input_channels >> 3); i++)
                    {
                        ae_int16x4 i1, i2, i3, i4, i5, i6;
                        ae_int16x4 one = ONE16;
                        ae_int32x2 wout1, wout2, wout3, wout4;
                        AE_LA16X4X2_IP(i1, i2, align_src1, p_src1_temp_w);
                        AE_LA16X4X2_IP(i3, i4, align_src2, p_src2_temp_w);
                        AE_LA16X4X2_IP(i5, i6, align_src3, p_src3_temp_w);
                        AE_MUL16X4 (wout1, wout2, i1, one);
                        AE_MULA16X4(wout1, wout2, i3, one);
                        AE_MULA16X4(wout1, wout2, i5, one);
                        AE_MUL16X4 (wout3, wout4, i2, one);
                        AE_MULA16X4(wout3, wout4, i4, one);
                        AE_MULA16X4(wout3, wout4, i6, one);
                        AE_S32X2X2_IP(wout1, wout2, p_dst_temp_w, 16);
                        AE_S32X2X2_IP(wout3, wout4, p_dst_temp_w, 16);
                    }

#if !defined(AE_SAV32X2X2_XP)
                    /* remainder loop */
#pragma loop_count max=7
                    for(i = 0; i < (input_channels & 7); i++)
                    {
                        ae_int16x4 i1, i2, i3;
                        ae_int32x2 wout1, wout2;
                        ae_int16x4 one = ONE16;
                        AE_L16_IP(i1, (ae_int16 *)p_src1_temp_w, 2);
                        AE_L16_IP(i2, (ae_int16 *)p_src2_temp_w, 2);
                        AE_L16_IP(i3, (ae_int16 *)p_src3_temp_w, 2);
                        AE_MUL16X4 (wout1, wout2, i1, one);
                        AE_MULA16X4(wout1, wout2, i2, one);
                        AE_MULA16X4(wout1, wout2, i3, one);
                        AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp_w, 4);
                    }
#else /* #if !defined(AE_SAV32X2X2_XP) */
                    /* remainder part */
                    int rem_itr = (input_channels & 7);
                    {
                        int rem_off0, rem_off1;
                        ae_int16x4 i1, i2, i3, i4, i5, i6;
                        ae_int16x4 one = ONE16;
                        ae_int32x2 wout1, wout2, wout3, wout4;
                        align_dst = AE_ZALIGN128();

                        rem_off0 = (rem_itr > 4 ? 4 : rem_itr) << 2;
                        rem_off1 = (rem_itr - 4 < 0 ? 0 : rem_itr - 4) << 2;

                        AE_LAV16X4X2_XP(i1, i2, align_src1, p_src1_temp_w, (rem_itr << 1));
                        AE_LAV16X4X2_XP(i3, i4, align_src2, p_src2_temp_w, (rem_itr << 1));
                        AE_LAV16X4X2_XP(i5, i6, align_src3, p_src3_temp_w, (rem_itr << 1));
                        AE_MUL16X4 (wout1, wout2, i1, one);
                        AE_MULA16X4(wout1, wout2, i3, one);
                        AE_MULA16X4(wout1, wout2, i5, one);
                        AE_MUL16X4 (wout3, wout4, i2, one);
                        AE_MULA16X4(wout3, wout4, i4, one);
                        AE_MULA16X4(wout3, wout4, i6, one);
                        AE_SAV32X2X2_XP(wout1, wout2, align_dst, p_dst_temp_w, rem_off0);
                        AE_SAV32X2X2_XP(wout3, wout4, align_dst, p_dst_temp_w, rem_off1);
                        AE_SA128POS_FP(align_dst, p_dst_temp_w);
                    }
#endif /* #if !defined(AE_SAV32X2X2_XP) */
                }

                if(pool_width)
                {
                    p_src2_w = p_src3_w;
                    INCR_ROW_IF_WIDTH_16(p_src2_w, pool_width, input_channels);

                    p_src3_w = p_src2_w;
                    INCR_ROW_IF_WIDTH_16(p_src3_w, pool_width, input_channels);

                    /* Compare three rows per iteration */
                    do
                    {
                        p_dst_temp_w = (ae_int32x4 *)p_dst;
                        p_src1_32x4 = (ae_int32x4 *)p_dst;
                        p_src2_temp_w = (ae_int16x8 *)p_src2_w;
                        p_src3_temp_w = (ae_int16x8 *)p_src3_w;

                        /* prime */
                        align_src2 = AE_LA128_PP(p_src2_temp_w);
                        align_src3 = AE_LA128_PP(p_src3_temp_w);

                        for(i = 0; i < (input_channels >> 3); i++)
                        {
                            ae_int16x4 i2, i3, i4, i5;
                            ae_int32x2 wout1, wout2, wout3, wout4;
                            AE_L32X2X2_IP(wout1, wout2, p_src1_32x4, 16);
                            AE_L32X2X2_IP(wout3, wout4, p_src1_32x4, 16);
                            AE_LA16X4X2_IP(i2, i3, align_src2, p_src2_temp_w);
                            AE_LA16X4X2_IP(i4, i5, align_src3, p_src3_temp_w);
                            AE_ACCW16(wout1, wout2, i2, i4);
                            AE_ACCW16(wout3, wout4, i3, i5);
                            AE_S32X2X2_IP(wout1, wout2, p_dst_temp_w, 16);
                            AE_S32X2X2_IP(wout3, wout4, p_dst_temp_w, 16);
                        }

#if !(defined(AE_LAV32X2X2_XP) && defined(AE_SAV32X2X2_XP))
                        /* remainder loop */
#pragma loop_count max=7
                        for(i = 0; i < (input_channels & 7); i++)
                        {
                            ae_int16x4 i2, i3;
                            ae_int32x2 wout1, wout2 = AE_ZERO32();
                            AE_L32_IP(wout1, (ae_int32 *)p_src1_32x4, 4);
                            AE_L16_IP(i2, (ae_int16 *)p_src2_temp_w, 2);
                            AE_L16_IP(i3, (ae_int16 *)p_src3_temp_w, 2);
                            AE_ACCW16(wout1, wout2, i2, i3);
                            AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp_w, 4);
                        }
#else /* #if !(defined(AE_LAV32X2X2_XP) && defined(AE_SAV32X2X2_XP)) */
                        int rem_itr = (input_channels & 7);
                        {
                            int rem_off0, rem_off1;
                            ae_int16x4 i2, i3, i4, i5;
                            ae_int32x2 wout1, wout2, wout3, wout4;
                            align_src1 = AE_LA128_PP(p_src1_32x4);
                            align_dst = AE_ZALIGN128();

                            rem_off0 = (rem_itr > 4 ? 4 : rem_itr) << 2;
                            rem_off1 = (rem_itr - 4 < 0 ? 0 : rem_itr - 4) << 2;

                            AE_LAV32X2X2_XP(wout1, wout2, align_src1, p_src1_32x4, rem_off0);
                            AE_LAV32X2X2_XP(wout3, wout4, align_src1, p_src1_32x4, rem_off1);
                            AE_LAV16X4X2_XP(i2, i3, align_src2, p_src2_temp_w, (rem_itr << 1));
                            AE_LAV16X4X2_XP(i4, i5, align_src3, p_src3_temp_w, (rem_itr << 1));
                            AE_ACCW16(wout1, wout2, i2, i4);
                            AE_ACCW16(wout3, wout4, i3, i5);
                            AE_SAV32X2X2_XP(wout1, wout2, align_dst, p_dst_temp_w, rem_off0);
                            AE_SAV32X2X2_XP(wout3, wout4, align_dst, p_dst_temp_w, rem_off1);
                            AE_SA128POS_FP(align_dst, p_dst_temp_w);
                        }
#endif /* #if !(defined(AE_LAV32X2X2_XP) && defined(AE_SAV32X2X2_XP)) */

                        if(!pool_width)
                            break;

                        p_src2_w = p_src3_w;
                        INCR_ROW_IF_WIDTH_16(p_src2_w, pool_width, input_channels);

                        p_src3_w = p_src2_w;
                        INCR_ROW_IF_WIDTH_16(p_src3_w, pool_width, input_channels);

                    }while(1);
                }

                // Saving Output
                ae_int32x2 den_h, den_w, d_tmp76, d_tmp54, d_tmp32, d_tmp10, d_tmp1_76, d_tmp1_54, d_tmp1_32, d_tmp1_10;
                ae_int32x2 d_out1, d_out2, d_out3, d_out4, d_out5, d_out6, d_out7, d_out8, d_tmp32hw;
                ae_int64 d_tmp;
                WORD32 *p_out1;
                ae_int8x8 out1, out2, out3;

                p_out1 = (WORD32 *)p_dst;

                /* prime */
                align_dst = AE_ZALIGN128(); // zero alignment reg

                if(kernel_height * kernel_width <= 1024)
                {
                    d_tmp32hw = AE_MOVDA32(inv_256_tbl[p_den_height[itr_oh] * p_den_width[itr_ow]]);
                }
                else
                {
                    den_h = AE_MOVDA32(inv_256_tbl[p_den_height[itr_oh]]);
                    den_w = AE_MOVDA32(inv_256_tbl[p_den_width[itr_ow]]);
                    d_tmp = AE_MUL32U_LL(den_h, den_w);

                    /* Max value of den_h or den_w is 0x80000000
                       so 1 left shift is possible without overflow */
                    d_tmp32hw = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);
                }

                for(i=0; i<(input_channels>>4); i++)
                {
                    AE_L32X2X2_IP(d_out1, d_out2, (ae_int32x4 *)p_out1, 16);
                    AE_L32X2X2_IP(d_out3, d_out4, (ae_int32x4 *)p_out1, 16);
                    AE_L32X2X2_IP(d_out5, d_out6, (ae_int32x4 *)p_out1, 16);
                    AE_L32X2X2_IP(d_out7, d_out8, (ae_int32x4 *)p_out1, 16);
                    AE_MULF2P32X4RS(d_tmp76, d_tmp54, d_out1, d_out2, d_tmp32hw, d_tmp32hw);
                    AE_MULF2P32X4RS(d_tmp32, d_tmp10, d_out3, d_out4, d_tmp32hw, d_tmp32hw);
                    AE_MULF2P32X4RS(d_tmp1_76, d_tmp1_54, d_out5, d_out6, d_tmp32hw, d_tmp32hw);
                    AE_MULF2P32X4RS(d_tmp1_32, d_tmp1_10, d_out7, d_out8, d_tmp32hw, d_tmp32hw);
                    out1 = AE_SATU8X4X32_L(d_tmp76, d_tmp54);
                    out2 = AE_SATU8X4X32_L(d_tmp32, d_tmp10);
                    out1 = AE_SEL8X8I(out1, out2, 3);
                    out2 = AE_SATU8X4X32_L(d_tmp1_76, d_tmp1_54);
                    out3 = AE_SATU8X4X32_L(d_tmp1_32, d_tmp1_10);
                    out2 = AE_SEL8X8I(out2, out3, 3);
                    AE_SA8X8X2_IP(out1, out2, align_dst, (ae_int8x16 *)p_out_temp);
                }
                AE_SA128POS_FP(align_dst, p_out_temp);

#pragma loop_count max=15
                for(i=0; i < (input_channels & 15); i++)
                {
                  AE_L32_IP(d_out1, (ae_int32 *)p_out1, 4);
                  d_tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32hw);
                  out1 = AE_SATU8X4X32_L(d_tmp32, d_tmp32);
                  AE_S8_0_IP(out1, (ae_int8 *)p_out_temp, sizeof(UWORD8));
                }
            }
            else
            {
                /* If there is no valid input present, fill the output with zeros*/
                for(i = 0; i < input_channels; i++)
                {
                    p_out_temp[i] = (UWORD8)0x0;
                }
            }
        }
    }
}

void xa_nn_avgpool_asym8_hwc_32(
      UWORD8* __restrict__ p_out,
const UWORD8* __restrict__ p_inp,
      WORD32   input_height,
      WORD32   input_width,
      WORD32   input_channels,
      WORD32   kernel_height,
      WORD32   kernel_width,
      WORD32   x_stride,
      WORD32   y_stride,
      WORD32   x_padding,
      WORD32   y_padding,
      WORD32   out_height,
      WORD32   out_width,
      pVOID    p_scratch_in,
      pVOID    p_zeros_mem,
      WORD32   *p_den_height,
      WORD32   *p_den_width)
{
    WORD16 *p_scratch = (WORD16 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int plane_size;
    int i;
    WORD8 * p_src1, * p_src2, * p_src3;
    WORD8 * __restrict p_src1_temp, * __restrict p_src2_temp, * __restrict p_src3_temp;
    ae_int32x4 * p_src1_scratch;
    WORD32 * p_src1_w, * p_src2_w, * p_src3_w;
    ae_int32x4 * __restrict p_src1_temp_w, * __restrict p_src2_temp_w, * __restrict p_src3_temp_w;
    ae_int32x4 * p_dst, *p_dst_temp;
    UWORD8 *p_out_temp;
    ae_valignx2 align_src1, align_src2, align_src3, align_dst;

    WORD32 *p_dst_pad;
    ae_int8x8 ZERO8 = AE_MOVDA8(0);
    ae_int16x4 ZERO16 = AE_ZERO16();
    ae_int16x4 ONE16 = AE_MOVDA16(1);
    plane_size = input_width * input_channels;

    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row, end_row;
        int start_plane, end_plane;

        /* Pool height processing */
        /* Processing width-channel planes for pool_height no. of planes  */
        /* Calculating avg of k_h w-c planes and saving into the scratch memory*/
        start_plane  = itr_oh * y_stride - y_padding;
        end_plane = start_plane + kernel_height;
        LIMIT(start_plane , 0, input_height);
        LIMIT(end_plane , 0, input_height);
        pool_height = end_plane - start_plane;
        p_dst = (ae_int32x4 *)p_scratch ;

        if(pool_height)
        {
            p_src1 = (WORD8 *)p_inp;
            INCR_N_PLANE(p_src1, start_plane, plane_size);
            pool_height--;

            p_src2 = p_src1;
            INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

            p_src3 = p_src2;
            INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

            /* 1st instance: Add three rows per iteration */
            {
                p_dst_temp = p_dst;
                p_src1_temp = p_src1;
                p_src2_temp = p_src2;
                p_src3_temp = p_src3;

                align_src1 = AE_LA128_PP(p_src1_temp);
                align_src2 = AE_LA128_PP(p_src2_temp);
                align_src3 = AE_LA128_PP(p_src3_temp);

                for(i = 0; i < (plane_size >> 4); i++)
                {
                    ae_int8x8 i1, i2, i3, i4, i5, i6;
                    ae_int16x4 out1, out2, out3, out4;
                    ae_int32x2 wout1, wout2, wout3, wout4, wout5, wout6, wout7, wout8;
                    ae_int16x4 one = ONE16;
                    AE_LA8X8X2_IP(i1, i2, align_src1, (ae_int8x16 *)p_src1_temp);
                    AE_LA8X8X2_IP(i3, i4, align_src2, (ae_int8x16 *)p_src2_temp);
                    AE_LA8X8X2_IP(i5, i6, align_src3, (ae_int8x16 *)p_src3_temp);
                    AE_ADDW8U(out1, out2, i1, i3);
                    AE_ADDW8U(out3, out4, i2, i4);
                    AE_ACCW8U(out1, out2, i5, ZERO8);
                    AE_ACCW8U(out3, out4, i6, ZERO8);
                    AE_ADDW16(wout1, wout2, ZERO16, out1);
                    AE_MUL16X4 (wout3, wout4, out2, one);
                    AE_MUL16X4 (wout5, wout6, out3, one);
                    AE_MUL16X4 (wout7, wout8, out4, one);
                    AE_S32X2X2_IP(wout1, wout2, p_dst_temp, 16);
                    AE_S32X2X2_IP(wout3, wout4, p_dst_temp, 16);
                    AE_S32X2X2_IP(wout5, wout6, p_dst_temp, 16);
                    AE_S32X2X2_IP(wout7, wout8, p_dst_temp, 16);
                }

#if !defined(AE_SAV32X2X2_XP)
                /* remainder loop */
#pragma loop_count max=15
                for(i = 0; i < (plane_size & 15); i++)
                {
                    ae_int8x8 i1, i2, i3;
                    ae_int16x4 out1, out2;
                    ae_int32x2 wout1, wout2;
                    ae_int16x4 one = ONE16;
                    AE_L8_IP(i1, (ae_int8 *)p_src1_temp, 1);
                    AE_L8_IP(i2, (ae_int8 *)p_src2_temp, 1);
                    AE_L8_IP(i3, (ae_int8 *)p_src3_temp, 1);
                    AE_ADDW8U(out1, out2, i1, i2);
                    AE_ACCW8U(out1, out2, i3, ZERO8);
                    AE_MUL16X4(wout1, wout2, out1, one);
                    AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp, 4);
                }
#else /* #if !defined(AE_SAV32X2X2_XP) */
                /* remainder part */
                int rem_itr = (plane_size & 15);
                {
                    int rem_off0, rem_off1, rem_off2, rem_off3;
                    ae_int8x8 i1, i2, i3, i4, i5, i6;
                    ae_int16x4 out1, out2, out3, out4;
                    ae_int32x2 wout1, wout2, wout3, wout4, wout5, wout6, wout7, wout8;
                    ae_int16x4 one = ONE16;
                    align_dst = AE_ZALIGN128();

                    rem_off0 = (rem_itr > 4 ? 4 : rem_itr) << 2;
                    rem_off1 = (rem_itr - 4 < 0 ? 0 : (rem_itr - 4 > 4 ? 4 : rem_itr - 4)) << 2;
                    rem_off2 = (rem_itr - 8 < 0 ? 0 : (rem_itr - 8 > 4 ? 4 : rem_itr - 8)) << 2;
                    rem_off3 = (rem_itr - 12 < 0 ? 0 : (rem_itr - 12 > 4 ? 4 : rem_itr - 12)) << 2;

                    AE_LAV8X8X2_XP(i1, i2, align_src1, (ae_int8x16 *)p_src1_temp, rem_itr);
                    AE_LAV8X8X2_XP(i3, i4, align_src2, (ae_int8x16 *)p_src2_temp, rem_itr);
                    AE_LAV8X8X2_XP(i5, i6, align_src3, (ae_int8x16 *)p_src3_temp, rem_itr);
                    AE_ADDW8U(out1, out2, i1, i3);
                    AE_ADDW8U(out3, out4, i2, i4);
                    AE_ACCW8U(out1, out2, i5, ZERO8);
                    AE_ACCW8U(out3, out4, i6, ZERO8);
                    AE_ADDW16(wout1, wout2, ZERO16, out1);
                    AE_MUL16X4 (wout3, wout4, out2, one);
                    AE_MUL16X4 (wout5, wout6, out3, one);
                    AE_MUL16X4 (wout7, wout8, out4, one);
                    AE_SAV32X2X2_XP(wout1, wout2, align_dst, p_dst_temp, rem_off0);
                    AE_SAV32X2X2_XP(wout3, wout4, align_dst, p_dst_temp, rem_off1);
                    AE_SAV32X2X2_XP(wout5, wout6, align_dst, p_dst_temp, rem_off2);
                    AE_SAV32X2X2_XP(wout7, wout8, align_dst, p_dst_temp, rem_off3);
                    AE_SA128POS_FP(align_dst, p_dst_temp);
                }
#endif /* #if !defined(AE_SAV32X2X2_XP) */
            }

            if(pool_height)
            {
                p_src2 = p_src3;
                INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

                p_src3 = p_src2;
                INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

                do
                {
                    p_dst_temp = p_dst;
                    p_src1_scratch = p_dst;
                    p_src2_temp = p_src2;
                    p_src3_temp = p_src3;

                    align_src2 = AE_LA128_PP(p_src2_temp);
                    align_src3 = AE_LA128_PP(p_src3_temp);

                    for(i = 0; i < (plane_size >> 4); i++)
                    {
                        ae_int8x8 i1, i2, i3, i4;
                        ae_int16x4 out1, out2, out3, out4;
                        ae_int32x2 wout1, wout2, wout3, wout4, wout5, wout6, wout7, wout8;
                        ae_int16x4 one = ONE16;
                        AE_L32X2X2_IP(wout1, wout2, p_src1_scratch, 16);
                        AE_L32X2X2_IP(wout3, wout4, p_src1_scratch, 16);
                        AE_L32X2X2_IP(wout5, wout6, p_src1_scratch, 16);
                        AE_L32X2X2_IP(wout7, wout8, p_src1_scratch, 16);
                        AE_LA8X8X2_IP(i1, i2, align_src2, (ae_int8x16 *)p_src2_temp);
                        AE_LA8X8X2_IP(i3, i4, align_src3, (ae_int8x16 *)p_src3_temp);
                        AE_ADDW8U(out1, out2, i1, i3);
                        AE_ADDW8U(out3, out4, i2, i4);
                        AE_MULA16X4(wout1, wout2, out1, one);
                        AE_MULA16X4(wout3, wout4, out2, one);
                        AE_MULA16X4(wout5, wout6, out3, one);
                        AE_MULA16X4(wout7, wout8, out4, one);
                        AE_S32X2X2_IP(wout1, wout2, p_dst_temp, 16);
                        AE_S32X2X2_IP(wout3, wout4, p_dst_temp, 16);
                        AE_S32X2X2_IP(wout5, wout6, p_dst_temp, 16);
                        AE_S32X2X2_IP(wout7, wout8, p_dst_temp, 16);
                    }

                    /* remainder loop */
#pragma loop_count max=15
                    for(i = 0; i < (plane_size & 15); i++)
                    {
                        ae_int8x8 i2, i3;
                        ae_int16x4 i0, i1;
                        ae_int32x2 wout1, wout2 = AE_ZERO32();
                        ae_int16x4 one = ONE16;
                        AE_L32_IP(wout1, (ae_int32 *)p_src1_scratch, 4);
                        AE_L8_IP(i2, (ae_int8 *)p_src2_temp, 1);
                        AE_L8_IP(i3, (ae_int8 *)p_src3_temp, 1);
                        AE_ADDW8U(i0, i1, i2, i3);
                        AE_MULA16X4(wout1, wout2, i0, one);
                        AE_S32_L_IP(wout1, (ae_int32 *)p_dst_temp, 4);
                    }

                    if(!pool_height)
                        break;

                    p_src2 = p_src3;
                    INCR_PLANE_IF_HEIGHT(p_src2, pool_height, plane_size);

                    p_src3 = p_src2;
                    INCR_PLANE_IF_HEIGHT(p_src3, pool_height, plane_size);

                }while(1);
            }
        }
        else
        {
            /* If there is no valid input present, fill the output with zeros */
            p_dst_pad = (WORD32 *)p_scratch;
            for(i = 0; i < plane_size; i++)
            {
                p_dst_pad[i] = 0;
            }
        }

        /* Pool width processing */
        /* Processing the output of the height processing block (which is a w-c plane); along width */
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            start_row  = itr_ow * x_stride - x_padding;
            end_row = start_row + kernel_width;
            LIMIT(start_row , 0, input_width);
            LIMIT(end_row , 0, input_width);
            pool_width = end_row - start_row;
            p_out_temp = p_out + (itr_oh*out_width*input_channels) + (itr_ow*input_channels);
            p_dst = (ae_int32x4 *)((WORD32 *)p_scratch + ALIGNED_SIZE(plane_size, ALIGNMENT/sizeof(WORD32)));

            if(pool_width)
            {
                p_src1_w = (WORD32 *)p_scratch;
                INCR_N_ROW(p_src1_w, start_row, input_channels);
                pool_width--;

                p_src2_w = p_src1_w;
                INCR_ROW_IF_WIDTH_32(p_src2_w, pool_width, input_channels);

                p_src3_w = p_src2_w;
                INCR_ROW_IF_WIDTH_32(p_src3_w, pool_width, input_channels);

                /* Add three rows per iteration */
                do
                {
                    p_dst_temp = p_dst;
                    p_src1_temp_w = (ae_int32x4 *)p_src1_w;
                    p_src2_temp_w = (ae_int32x4 *)p_src2_w;
                    p_src3_temp_w = (ae_int32x4 *)p_src3_w;

                    /* prime */
                    align_src1 = AE_LA128_PP(p_src1_temp_w);
                    align_src2 = AE_LA128_PP(p_src2_temp_w);
                    align_src3 = AE_LA128_PP(p_src3_temp_w);

                    for(i = 0; i < (input_channels >> 2); i++)
                    {
                        ae_int32x2 i1, i2, i3, i4, i5, i6, out, out1;
                        AE_LA32X2X2_IP(i1, i2, align_src1, p_src1_temp_w);
                        AE_LA32X2X2_IP(i3, i4, align_src2, p_src2_temp_w);
                        AE_LA32X2X2_IP(i5, i6, align_src3, p_src3_temp_w);
                        out = AE_ADD32S(i1, i3);
                        out = AE_ADD32S(out, i5);
                        out1 = AE_ADD32S(i2, i4);
                        out1 = AE_ADD32S(out1, i6);
                        AE_S32X2X2_IP(out, out1, p_dst_temp, 16);
                    }

#if !(defined(AE_LAV32X2X2_XP) && defined(AE_SAV32X2X2_XP))
                    /* remainder loop */
#pragma loop_count max=3
                    for(i = 0; i < (input_channels & 3); i++)
                    {
                        ae_int32x2 i1, i2, i3, out;
                        AE_L32_IP(i1, (ae_int32 *)p_src1_temp_w, 4);
                        AE_L32_IP(i2, (ae_int32 *)p_src2_temp_w, 4);
                        AE_L32_IP(i3, (ae_int32 *)p_src3_temp_w, 4);
                        out = AE_ADD32S(i1, i2);
                        out = AE_ADD32S(out, i3);
                        AE_S32_L_IP(out, (ae_int32 *)p_dst_temp, 4);
                    }
#else /* #if !(defined(AE_LAV32X2X2_XP) && defined(AE_SAV32X2X2_XP)) */
                    int rem_itr = (input_channels & 3);
                    {
                        ae_int32x2 i1, i2, i3, i4, i5, i6, out, out1;
                        align_dst = AE_ZALIGN128();

                        AE_LAV32X2X2_XP(i1, i2, align_src1, p_src1_temp_w, (rem_itr << 2));
                        AE_LAV32X2X2_XP(i3, i4, align_src2, p_src2_temp_w, (rem_itr << 2));
                        AE_LAV32X2X2_XP(i5, i6, align_src3, p_src3_temp_w, (rem_itr << 2));
                        out = AE_ADD32S(i1, i3);
                        out = AE_ADD32S(out, i5);
                        out1 = AE_ADD32S(i2, i4);
                        out1 = AE_ADD32S(out1, i6);
                        AE_SAV32X2X2_XP(out, out1, align_dst, p_dst_temp, (rem_itr << 2));
                        AE_SA128POS_FP(align_dst, p_dst_temp);
                    }
#endif /* #if !(defined(AE_LAV32X2X2_XP) && defined(AE_SAV32X2X2_XP)) */

                    if(!pool_width)
                        break;

                    p_src1_w = (WORD32 *)p_dst;

                    p_src2_w = p_src3_w;
                    INCR_ROW_IF_WIDTH_32(p_src2_w, pool_width, input_channels);

                    p_src3_w = p_src2_w;
                    INCR_ROW_IF_WIDTH_32(p_src3_w, pool_width, input_channels);

                }while(1);

                // Saving Output
                ae_int32x2 den_h, den_w, d_tmp76, d_tmp54, d_tmp32, d_tmp10, d_tmp1_76, d_tmp1_54, d_tmp1_32, d_tmp1_10;
                ae_int32x2 d_out1, d_out2, d_out3, d_out4, d_out5, d_out6, d_out7, d_out8, d_tmp32hw;
                ae_int64 d_tmp;
                WORD32 *p_out1;
                ae_int8x8 out1, out2, out3;

                p_out1 = (WORD32 *)p_dst;

                align_dst = AE_ZALIGN128(); // zero alignment reg

                if(kernel_height * kernel_width <= 1024)
                {
                    d_tmp32hw = AE_MOVDA32(inv_256_tbl[p_den_height[itr_oh] * p_den_width[itr_ow]]);
                }
                else
                {
                    den_h = AE_MOVDA32(inv_256_tbl[p_den_height[itr_oh]]);
                    den_w = AE_MOVDA32(inv_256_tbl[p_den_width[itr_ow]]);
                    d_tmp = AE_MUL32U_LL(den_h, den_w);

                    /* Max value of den_h or den_w is 0x80000000
                       so 1 left shift is possible without overflow */
                    d_tmp32hw = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);
                }

                for(i=0; i<(input_channels>>4); i++)
                {
                    AE_L32X2X2_IP(d_out1, d_out2, (ae_int32x4 *)p_out1, 16);
                    AE_L32X2X2_IP(d_out3, d_out4, (ae_int32x4 *)p_out1, 16);
                    AE_L32X2X2_IP(d_out5, d_out6, (ae_int32x4 *)p_out1, 16);
                    AE_L32X2X2_IP(d_out7, d_out8, (ae_int32x4 *)p_out1, 16);
                    AE_MULF2P32X4RS(d_tmp76, d_tmp54, d_out1, d_out2, d_tmp32hw, d_tmp32hw);
                    AE_MULF2P32X4RS(d_tmp32, d_tmp10, d_out3, d_out4, d_tmp32hw, d_tmp32hw);
                    AE_MULF2P32X4RS(d_tmp1_76, d_tmp1_54, d_out5, d_out6, d_tmp32hw, d_tmp32hw);
                    AE_MULF2P32X4RS(d_tmp1_32, d_tmp1_10, d_out7, d_out8, d_tmp32hw, d_tmp32hw);
                    out1 = AE_SATU8X4X32_L(d_tmp76, d_tmp54);
                    out2 = AE_SATU8X4X32_L(d_tmp32, d_tmp10);
                    out1 = AE_SEL8X8I(out1, out2, 3);
                    out2 = AE_SATU8X4X32_L(d_tmp1_76, d_tmp1_54);
                    out3 = AE_SATU8X4X32_L(d_tmp1_32, d_tmp1_10);
                    out2 = AE_SEL8X8I(out2, out3, 3);
                    AE_SA8X8X2_IP(out1, out2, align_dst, (ae_int8x16 *)p_out_temp);
                }
                AE_SA128POS_FP(align_dst, p_out_temp);

#pragma loop_count max=15
                for(i=0; i < (input_channels & 15); i++)
                {
                  AE_L32_IP(d_out1, (ae_int32 *)p_out1, 4);
                  d_tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32hw);
                  out1 = AE_SATU8X4X32_L(d_tmp32, d_tmp32);
                  AE_S8_0_IP(out1, (ae_int8 *)p_out_temp, sizeof(UWORD8));
                }
            }
            else
            {
                /* If there is no valid input present, fill the output with zeros*/
                for(i = 0; i < input_channels; i++)
                {
                    p_out_temp[i] = (UWORD8)0x0;
                }
            }
        }
    }
}


