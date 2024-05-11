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
#include "xa_type_def.h"
#include "xa_nn_common.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nn_avgpool_state.h"
#include "xa_nnlib_err_chk.h"

static void avgpool_8_16(
      WORD8 *__restrict__ p_out,
const WORD8 *__restrict__ p_inp,
      WORD32 *p_den_height,
      WORD32 *p_den_width,
      WORD32  input_height,
      WORD32  input_width,
      WORD32  kernel_height,
      WORD32  kernel_width,
      WORD32  x_stride,
      WORD32  y_stride,
      WORD32  x_padding,
      WORD32  y_padding,
      WORD32  out_height,
      WORD32  out_width,
      pVOID   p_scratch_in)
{
    WORD16 *p_scratch = (WORD16 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int left_pad_aligned, right_pad, total_out_width, scratch_width;
    ae_int8x8 * p_src1, * p_src2;
    ae_int8x8 * __restrict p_src1_temp, * __restrict p_src2_temp;
    ae_int16x4  * p_wsrc1;
    ae_int16x4 *p_wsrc1_temp;
    ae_int32x2 * p_wsrc2;
    ae_int32x2 *p_wsrc2_temp;
    ae_int32x2 *p_dst, *p_dst_temp;
    ae_int16x4 *p_dst16, *p_dst16_temp;
    ae_valignx2 align_src1, align_src2;
    ae_valignx2 align_wsrc1;
    int i;
    WORD16 *p_dst16_pad;

    left_pad_aligned = ALIGNED_SIZE(x_padding, ALIGNMENT/sizeof(WORD16));

    /* Left padding of temporary output with min_value */
    p_dst16_pad = p_scratch;
    for(i = 0; i < left_pad_aligned; i++)
    {
        p_dst16_pad[i] = 0;
    }

    total_out_width = XT_MAX(input_width + x_padding, (out_width - 1) * x_stride + kernel_width);
    right_pad = total_out_width - (x_padding + input_width);

    /* Right padding of temporary output with min_value,
     * add kernel_width values more for the aligning load operations */

    p_dst16_pad = p_scratch + left_pad_aligned + input_width;
    for(i = 0; i < right_pad + kernel_width; i++)
    {
        p_dst16_pad[i] = 0;
    }

    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row, end_row;

        /* Pool height processing */

        /* Compare the input rows for the required pooling height */
        start_row  = itr_oh * y_stride - y_padding;
        end_row = start_row + kernel_height;
        LIMIT(start_row , 0, input_height);
        LIMIT(end_row , 0, input_height);

        pool_height = end_row - start_row;

        p_dst16 = (ae_int16x4 *)((WORD16 *)p_scratch + left_pad_aligned);

        /*Adding first two rows and storing on the scratch*/
        if(pool_height-1 > 0)
        {
            p_src1 = (ae_int8x8 *)p_inp;
            p_src1 = (ae_int8x8 *)((WORD8 *)p_src1 + start_row*input_width);
            pool_height--;
            p_dst16_temp = p_dst16;
            p_src1_temp = p_src1;

            p_src2 = p_src1;
            p_src2 = (ae_int8x8 *)((WORD8 *)p_src2 + input_width);
            pool_height--;
            p_src2_temp = p_src2;

            /* prime */
            align_src1 = AE_LA128_PP(p_src1_temp);
            align_src2 = AE_LA128_PP(p_src2_temp);

            for(i = 0; i < (input_width >> 4); i++)
            {
                ae_int8x8 i1, i2, j1, j2;
                ae_int16x4 wi1, wi2, wj1, wj2;
                AE_LA8X8X2_IP(i1, j1, align_src1, (ae_int8x16 *)p_src1_temp);
                AE_LA8X8X2_IP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp);
                AE_ADDW8(wi1, wi2, i1, i2);
                AE_ADDW8(wj1, wj2, j1, j2);
                AE_S16X4X2_IP(wi1, wi2, (ae_int16x8 *)p_dst16_temp, 16);
                AE_S16X4X2_IP(wj1, wj2, (ae_int16x8 *)p_dst16_temp, 16);
            }

            /* reminder part for input_width */
            int rem_itr = (input_width & 15);
            {
                int rem_off0, rem_off1;
                ae_int8x8 i1, i2, j1, j2;
                ae_int16x4 wi1, wi2, wj1, wj2;
                ae_valignx2 align_dst;
                align_dst = AE_ZALIGN128();

                rem_off0 = (rem_itr > 8 ? 8 : rem_itr) << 1;
                rem_off1 = (rem_itr - 8 < 0 ? 0 : rem_itr - 8) << 1;

                AE_LAV8X8X2_XP(i1, j1, align_src1, (ae_int8x16 *)p_src1_temp, rem_itr);
                AE_LAV8X8X2_XP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp, rem_itr);
                AE_ADDW8(wi1, wi2, i1, i2);
                AE_ADDW8(wj1, wj2, j1, j2);
                AE_SAV16X4X2_XP(wi1, wi2, align_dst, (ae_int16x8 *)p_dst16_temp, rem_off0);
                AE_SAV16X4X2_XP(wj1, wj2, align_dst, (ae_int16x8 *)p_dst16_temp, rem_off1);
                AE_SA128POS_FP(align_dst, p_dst16_temp);
            }
            /* Add one row per iteration */
            while(pool_height)
            {
                p_src2 = (ae_int8x8 *)((WORD8 *)p_src2 + input_width);
                ae_int16x8 *p_dst16x8_temp = (ae_int16x8 *)p_dst16;
                ae_int16x8 *p_wsrc1_16x8_temp = (ae_int16x8 *)p_dst16;
                p_src2_temp = p_src2;

#pragma ymemory(p_wsrc1_16x8_temp)
#pragma ymemory(p_src2_temp)

                /* prime */
                align_src2 = AE_LA128_PP(p_src2_temp);

#pragma concurrent
                for(i = 0; i < (input_width >> 4); i++)
                {
                    ae_int8x8 i1, j1, zero = AE_MOVDA8(0);
                    ae_int16x4 wi1, wi2, wj1, wj2;

                    AE_LA8X8X2_IP(i1, j1, align_src2, (ae_int8x16 *)p_src2_temp);
                    AE_L16X4X2_I(wi1, wi2, p_wsrc1_16x8_temp, 0);
                    AE_L16X4X2_I(wj1, wj2, p_wsrc1_16x8_temp, 16);
                    p_wsrc1_16x8_temp += 2;
                    AE_ACCW8(wi1, wi2, i1, zero);
                    AE_ACCW8(wj1, wj2, j1, zero);
                    AE_S16X4X2_I(wi1, wi2, p_dst16x8_temp, 0);
                    AE_S16X4X2_I(wj1, wj2, p_dst16x8_temp, 16);
                    p_dst16x8_temp += 2;
                }

                /* reminder part for input_width */
                int rem_itr = (input_width & 15);
                if(rem_itr)
                {
                    int rem_off0, rem_off1;
                    ae_int8x8 i1, j1, zero = AE_MOVINT8X8_FROMINT16X4(AE_ZERO16());
                    ae_int16x4 wi1, wi2, wj1, wj2;
                    ae_valignx2 align_dst = AE_ZALIGN128();
                    align_wsrc1 = AE_LA128_PP(p_wsrc1_16x8_temp);

                    rem_off0 = (rem_itr > 8 ? 8 : rem_itr)<<1;
                    rem_off1 = (rem_itr - 8 < 0 ? 0 : rem_itr - 8)<<1;
                    AE_LAV16X4X2_XP(wi1, wi2, align_wsrc1, p_wsrc1_16x8_temp, rem_off0);
                    AE_LAV16X4X2_XP(wj1, wj2, align_wsrc1, p_wsrc1_16x8_temp, rem_off1);
                    AE_LAV8X8X2_XP(i1, j1, align_src2, (ae_int8x16 *)p_src2_temp, rem_itr);
                    AE_ACCW8(wi1, wi2, i1, zero);
                    AE_ACCW8(wj1, wj2, j1, zero);
                    AE_SAV16X4X2_XP(wi1, wi2, align_dst, p_dst16x8_temp, rem_off0);
                    AE_SAV16X4X2_XP(wj1, wj2, align_dst, p_dst16x8_temp, rem_off1);
                    AE_SA128POS_FP(align_dst, p_dst16x8_temp);
                }
                pool_height--;
            };
        }
        /*Storing the first row on scratch when pool height is one*/
        else if(pool_height==1)
        {
            p_src1 = (ae_int8x8 *)p_inp;
            p_src1 = (ae_int8x8 *)((WORD8 *)p_src1 + start_row*input_width);
            pool_height--;
            p_dst16_temp = p_dst16;
            p_src1_temp = p_src1;
            align_src1 = AE_LA128_PP(p_src1_temp);

            for(i = 0; i < (input_width >> 4); i++)
            {
                ae_int8x8 i1, j1, zero = AE_MOVDA8(0);
                ae_int16x4 wi1, wi2, wj1, wj2;
                AE_LA8X8X2_IP(i1, j1, align_src1, (ae_int8x16 *)p_src1_temp);
                AE_ADDW8(wi1, wi2, i1, zero);
                AE_ADDW8(wj1, wj2, j1, zero);
                AE_S16X4X2_IP(wi1, wi2, (ae_int16x8 *)p_dst16_temp, 16);
                AE_S16X4X2_IP(wj1, wj2, (ae_int16x8 *)p_dst16_temp, 16);
            }

            /* reminder loop for input_width */
            for(i = 0; i < (input_width & 15); i++)
            {
                ae_int8x8 i1, zero = AE_MOVDA8(0);
                ae_int16x4 hi1, hj1;

                AE_L8_IP(i1, (ae_int8 *)p_src1_temp,1);
                AE_ADDW8(hi1, hj1, i1, zero);
                ((ae_int16 *)p_dst16_temp)[i] = hj1;
            }
        }
        else
        {
            // If there is no valid input present, fill the output with min_value
            p_dst16_pad = p_scratch + left_pad_aligned ;
            for(i = 0; i < input_width; i++)
            {
                p_dst16_pad[i] = 0;
            }
        }

        /* Pool width processing */

        /* On scratch, compare width-wise with padding*/
        total_out_width = ALIGNED_SIZE(left_pad_aligned + input_width + right_pad + kernel_width, ALIGNMENT/sizeof(WORD16));
        scratch_width = x_padding + input_width + right_pad;
        p_dst = (ae_int32x2 *)((WORD16 *)p_scratch + total_out_width);
        pool_width = kernel_width;

        p_wsrc1 = (ae_int16x4 *)((WORD16 *)p_scratch + left_pad_aligned - x_padding);
        pool_width--;
        p_dst_temp = p_dst;
        p_wsrc1_temp = p_wsrc1;

        /* prime */
        align_wsrc1 = AE_LA128_PP(p_wsrc1_temp);

#pragma concurrent
        for(i = 0; i < (scratch_width >> 3); i++)
        {
            ae_int16x4 wsrc1, wsrc2;
            ae_int32x2 wi1, wi2, wj1, wj2;
            AE_LA16X4X2_IP(wsrc1, wsrc2, align_wsrc1, (ae_int16x8 *)p_wsrc1_temp);

            wi1 = AE_SEXT32X2D16_32(wsrc1);
            wi2 = AE_SEXT32X2D16_10(wsrc1);
            wj1 = AE_SEXT32X2D16_32(wsrc2);
            wj2 = AE_SEXT32X2D16_10(wsrc2);
            AE_S32X2X2_IP(wi1, wi2, (ae_int32x4 *)p_dst_temp, 16);
            AE_S32X2X2_IP(wj1, wj2, (ae_int32x4 *)p_dst_temp, 16);
        }

        /* reminder loop for scratch_width */
        for(i=0; i<(scratch_width & 7); i++)
        {
           ae_int16x4 wsrc1;
           wsrc1 = ((ae_int16 *)p_wsrc1_temp)[i];
           ((ae_int32 *)p_dst_temp)[i] = AE_SEXT32X2D16_32(wsrc1);
        }

        p_wsrc2 = p_dst;

        while(pool_width > 0)
        {
            p_wsrc1 = (ae_int16x4 *)((WORD16 *)p_wsrc1 + 1);
            p_dst_temp = p_dst;
            p_wsrc1_temp = p_wsrc1;
            p_wsrc2_temp = p_wsrc2;

            /* prime */
            align_wsrc1 = AE_LA128_PP(p_wsrc1_temp);

#pragma concurrent
            for(i = 0; i < (scratch_width >> 3); i++)
            {
                ae_int16x4 wi1, wi2, zero = AE_ZERO16();
                ae_int32x2 wsrc1, wsrc2, wsrc3, wsrc4;

                AE_L32X2X2_IP(wsrc1, wsrc2, (ae_int32x4 *)p_wsrc2_temp, 16);
                AE_L32X2X2_IP(wsrc3, wsrc4, (ae_int32x4 *)p_wsrc2_temp, 16);
                AE_LA16X4X2_IP(wi1, wi2, align_wsrc1, (ae_int16x8 *)p_wsrc1_temp);
                AE_ACCW16(wsrc1, wsrc2, wi1, zero);
                AE_ACCW16(wsrc3, wsrc4, wi2, zero);

                AE_S32X2X2_IP(wsrc1, wsrc2, (ae_int32x4 *)p_dst_temp, 16);
                AE_S32X2X2_IP(wsrc3, wsrc4, (ae_int32x4 *)p_dst_temp, 16);
            }

            /* remainder part for scratch_width */
#pragma concurrent
#pragma loop_count max=7
            for(i=0; i<(scratch_width & 7); i++)
            {
                ae_int16x4 wsrc1;
                ae_int32x2 wsrc2, out;

                wsrc2 = ((ae_int32 *)p_wsrc2_temp)[i];
                wsrc1 = ((ae_int16 *)p_wsrc1_temp)[i];

                out = AE_ADD32(wsrc2, AE_SEXT32X2D16_32(wsrc1));
                ((ae_int32 *)p_dst_temp)[i] = out;
            }
            pool_width--;
        };

        WORD32 *ptr_out1 = (WORD32 *)((WORD16 *)p_scratch + total_out_width);
        ae_int8x8 d_out8, d_1out8;
        ae_int32x2 d_tmp32, d_out1, d_out, d_1tmp32, d_out2;
        ae_int64 d_tmp, d_1tmp;
        if(kernel_height * kernel_width <= 1024)
        {
            WORD32 den_hw, den1_hw;
            for(itr_ow = 0; itr_ow < out_width-1; itr_ow+=2)
            {
                den_hw = inv_256_tbl[p_den_height[itr_oh] * p_den_width[itr_ow]];
                den1_hw = inv_256_tbl[p_den_height[itr_oh] * p_den_width[itr_ow + 1]];
                d_tmp32 = AE_MOVDA32X2(den_hw, den1_hw);
                d_out1 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride]);
                d_out2 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride+x_stride]);

                d_out = AE_SEL32_LL(d_out1, d_out2);

                d_1tmp32 = AE_MULFP32X2RS(d_out, d_tmp32);
                d_out8  = AE_SAT8X4X32_L(d_1tmp32, d_1tmp32);
                d_1out8 = AE_SEL8X8I(d_out8, d_out8, 26);
                AE_S8_0_I(d_1out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow], 0);
                AE_S8_0_I(d_out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow+1], 0);
            }
            if(out_width & 1)
            {
                den_hw = inv_256_tbl[p_den_height[itr_oh] * p_den_width[itr_ow]];
                d_out1 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride]);
                d_tmp32 = AE_MOVDA32(den_hw);
                d_1tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32);
                d_out8 =AE_SAT8X4X32_L(d_1tmp32, d_1tmp32);
                AE_S8_0_I(d_out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow], 0);
            
            }
        }
        else
        {
            ae_int32x2 den_h, den_w, den1_w;
            den_h = AE_MOVDA32(inv_256_tbl[p_den_height[itr_oh]]);
            for(itr_ow = 0; itr_ow < out_width-1; itr_ow+=2)
            {
                den_w = AE_MOVDA32(inv_256_tbl[p_den_width[itr_ow]]);
                den1_w = AE_MOVDA32(inv_256_tbl[p_den_width[itr_ow + 1]]);
                d_out1 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride]);
                d_out2 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride+x_stride]);

                d_tmp = AE_MUL32U_LL(den_h, den_w);
                d_1tmp = AE_MUL32U_LL(den_h, den1_w);

                d_tmp32 = AE_TRUNCI32X2F64S(d_tmp, d_1tmp, 1);

                d_out = AE_SEL32_LL(d_out1, d_out2);

                d_1tmp32 = AE_MULFP32X2RS(d_out, d_tmp32);
                d_out8  = AE_SAT8X4X32_L(d_1tmp32, d_1tmp32);
                d_1out8 = AE_SEL8X8I(d_out8, d_out8, 26);
                AE_S8_0_I(d_1out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow], 0);
                AE_S8_0_I(d_out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow+1], 0);
            }
            if(out_width & 1)
            {
                den_w = AE_MOVDA32(inv_256_tbl[p_den_width[itr_ow]]);
                d_out1 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride]);
                d_tmp = AE_MUL32U_LL(den_h, den_w);
                d_tmp32 = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);
                d_1tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32);
                d_out8 =AE_SAT8X4X32_L(d_1tmp32, d_1tmp32);
                AE_S8_0_I(d_out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow], 0);
            
            }
        }
    }
}

static void avgpool_8_32(
      WORD8 *__restrict__ p_out,
const WORD8 *__restrict__ p_inp,
      WORD32 *p_den_height,
      WORD32 *p_den_width,
      WORD32  input_height,
      WORD32  input_width,
      WORD32  kernel_height,
      WORD32  kernel_width,
      WORD32  x_stride,
      WORD32  y_stride,
      WORD32  x_padding,
      WORD32  y_padding,
      WORD32  out_height,
      WORD32  out_width,
      pVOID   p_scratch_in)
{
    WORD32 *p_scratch = (WORD32 *)(p_scratch_in);

    int itr_oh, itr_ow;
    int left_pad_aligned, right_pad, total_out_width, scratch_width;
    ae_int8x8 * p_src1, * p_src2;
    ae_int8x8 * __restrict p_src1_temp, * __restrict p_src2_temp;
    ae_int32x4  * p_wsrc1;
    ae_int32x4 *p_wsrc1_temp;
    ae_int32x4 * p_wsrc2;
    ae_int32x4 *p_wsrc2_temp;
    ae_int32x4 *p_dst, *p_dst_temp;
    ae_int32x4 *p_dst32, *p_dst32_temp;
    ae_valignx2 align_src1, align_src2;
    ae_valignx2 align_wsrc1;
    int i;
    WORD32 *p_dst32_pad;
    ae_int16x4 one16 = AE_MOVDA16(1);

    left_pad_aligned = ALIGNED_SIZE(x_padding, ALIGNMENT/sizeof(WORD32));

    /* Left padding of temporary output with min_value */
    p_dst32_pad = p_scratch;
    for(i = 0; i < left_pad_aligned; i++)
    {
        p_dst32_pad[i] = 0;
    }

    total_out_width = XT_MAX(input_width + x_padding, (out_width - 1) * x_stride + kernel_width);
    right_pad = total_out_width - (x_padding + input_width);

    /* Right padding of temporary output with min_value,
     * add kernel_width values more for the aligning load operations */

    p_dst32_pad = p_scratch + left_pad_aligned + input_width;
    for(i = 0; i < right_pad + kernel_width; i++)
    {
        p_dst32_pad[i] = 0;
    }

    for(itr_oh = 0; itr_oh < out_height; itr_oh++)
    {
        int pool_height, pool_width;
        int start_row, end_row;

        /* Pool height processing */

        /* Compare the input rows for the required pooling height */
        start_row  = itr_oh * y_stride - y_padding;
        end_row = start_row + kernel_height;
        LIMIT(start_row , 0, input_height);
        LIMIT(end_row , 0, input_height);

        pool_height = end_row - start_row;

        p_dst32 = (ae_int32x4 *)((WORD32 *)p_scratch + left_pad_aligned);

        /*Adding first two rows and storing on the scratch*/
        if(pool_height-1 > 0)
        {
            p_src1 = (ae_int8x8 *)p_inp;
            p_src1 = (ae_int8x8 *)((WORD8 *)p_src1 + start_row*input_width);
            pool_height--;
            p_dst32_temp = p_dst32;
            p_src1_temp = p_src1;

            p_src2 = p_src1;
            p_src2 = (ae_int8x8 *)((WORD8 *)p_src2 + input_width);
            pool_height--;
            p_src2_temp = p_src2;

            /* prime */
            align_src1 = AE_LA128_PP(p_src1_temp);
            align_src2 = AE_LA128_PP(p_src2_temp);

            for(i = 0; i < (input_width >> 4); i++)
            {
                ae_int8x8 i1, i2, j1, j2;
                ae_int16x4 wi1, wi2, wj1, wj2;
                ae_int32x2 wout1, wout2, wout3, wout4;
                ae_int32x2 wout5, wout6, wout7, wout8;

                AE_LA8X8X2_IP(i1, j1, align_src1, (ae_int8x16 *)p_src1_temp);
                AE_LA8X8X2_IP(i2, j2, align_src2, (ae_int8x16 *)p_src2_temp);
                AE_ADDW8(wi1, wi2, i1, i2);
                AE_ADDW8(wj1, wj2, j1, j2);
                AE_MUL16X4(wout1, wout2, wi1, one16);
                AE_MUL16X4(wout3, wout4, wi2, one16);
                AE_MUL16X4(wout5, wout6, wj1, one16);
                AE_MUL16X4(wout7, wout8, wj2, one16);
                AE_S32X2X2_IP(wout1, wout2, p_dst32_temp, 16);
                AE_S32X2X2_IP(wout3, wout4, p_dst32_temp, 16);
                AE_S32X2X2_IP(wout5, wout6, p_dst32_temp, 16);
                AE_S32X2X2_IP(wout7, wout8, p_dst32_temp, 16);
            }

            /* reminder part for input_width */
            for(i = 0; i < (input_width & 15); i++)
            {
                ae_int8x8 i1, i2;
                ae_int16x4 wi1, wi2;
                ae_int32x2 wout1, wout2;
                AE_L8_IP(i1, (ae_int8 *)p_src1_temp, 1);
                AE_L8_IP(i2, (ae_int8 *)p_src2_temp, 1);
                AE_ADDW8(wi1, wi2, i1, i2);
                AE_MUL16X4(wout2, wout1, wi1, one16);
                AE_S32_L_IP(wout1, (ae_int32 *)p_dst32_temp, 4);
            }

            /* Add one row per iteration */
            while(pool_height)
            {
                p_src2 = (ae_int8x8 *)((WORD8 *)p_src2 + input_width);
                ae_int32x4 *p_dst32x4_temp = (ae_int32x4 *)p_dst32;
                ae_int32x4 *p_wsrc1_32x4_temp = (ae_int32x4 *)p_dst32;
                p_src2_temp = p_src2;

#pragma ymemory(p_wsrc1_32x4_temp)
#pragma ymemory(p_src2_temp)

                /* prime */
                align_src2 = AE_LA128_PP(p_src2_temp);

#pragma concurrent
                for(i = 0; i < (input_width >> 4); i++)
                {
                    ae_int8x8 i1, j1, zero = AE_MOVDA8(0);
                    ae_int16x4 wi1, wi2, wj1, wj2;
                    ae_int32x2 wout1, wout2, wout3, wout4;
                    ae_int32x2 wout5, wout6, wout7, wout8;

                    AE_LA8X8X2_IP(i1, j1, align_src2, (ae_int8x16 *)p_src2_temp);
                    AE_L32X2X2_I(wout1, wout2, p_wsrc1_32x4_temp, 0);
                    AE_L32X2X2_I(wout3, wout4, p_wsrc1_32x4_temp, 16);
                    AE_L32X2X2_I(wout5, wout6, p_wsrc1_32x4_temp, 32);
                    AE_L32X2X2_I(wout7, wout8, p_wsrc1_32x4_temp, 48);
                    p_wsrc1_32x4_temp += 4;
                    AE_ADDW8(wi1, wi2, i1, zero);
                    AE_ADDW8(wj1, wj2, j1, zero);
                    AE_MULA16X4(wout1, wout2, wi1, one16);
                    AE_MULA16X4(wout3, wout4, wi2, one16);
                    AE_MULA16X4(wout5, wout6, wj1, one16);
                    AE_MULA16X4(wout7, wout8, wj2, one16);
                    AE_S32X2X2_I(wout1, wout2, p_dst32x4_temp, 0);
                    AE_S32X2X2_I(wout3, wout4, p_dst32x4_temp, 16);
                    AE_S32X2X2_I(wout5, wout6, p_dst32x4_temp, 32);
                    AE_S32X2X2_I(wout7, wout8, p_dst32x4_temp, 48);
                    p_dst32x4_temp += 4;
                }

                /* reminder part for input_width */
                for(i = 0; i < (input_width & 15); i++)
                {
                    ae_int8x8 i1, zero = AE_MOVDA8(0);
                    ae_int16x4 wi1, wi2;
                    ae_int32x2 wout1, wout2;

                    AE_L8_IP(i1, (ae_int8 *)p_src2_temp, 1);
                    AE_L32_IP(wout1, (ae_int32 *)p_wsrc1_32x4_temp, 4);
                    AE_ADDW8(wi2, wi1, i1, zero);
                    AE_MULA16X4(wout2, wout1, wi1, one16);
                    AE_S32_L_IP(wout1, (ae_int32 *)p_dst32x4_temp, 4);
                }
                pool_height--;
            };
        }
        /*Storing the first row on scratch when pool height is one*/
        else if(pool_height==1)
        {
            p_src1 = (ae_int8x8 *)p_inp;
            p_src1 = (ae_int8x8 *)((WORD8 *)p_src1 + start_row*input_width);
            pool_height--;
            p_dst32_temp = p_dst32;
            p_src1_temp = p_src1;
            align_src1 = AE_LA128_PP(p_src1_temp);

            for(i = 0; i < (input_width >> 4); i++)
            {
                ae_int8x8 i1, j1, zero = AE_MOVDA8(0);
                ae_int16x4 wi1, wi2, wj1, wj2;
                ae_int32x2 wout1, wout2, wout3, wout4;
                ae_int32x2 wout5, wout6, wout7, wout8;

                AE_LA8X8X2_IP(i1, j1, align_src1, (ae_int8x16 *)p_src1_temp);
                AE_ADDW8(wi1, wi2, i1, zero);
                AE_ADDW8(wj1, wj2, j1, zero);
                AE_MUL16X4(wout1, wout2, wi1, one16);
                AE_MUL16X4(wout3, wout4, wi2, one16);
                AE_MUL16X4(wout5, wout6, wj1, one16);
                AE_MUL16X4(wout7, wout8, wj2, one16);
                AE_S32X2X2_IP(wout1, wout2, p_dst32_temp, 16);
                AE_S32X2X2_IP(wout3, wout4, p_dst32_temp, 16);
                AE_S32X2X2_IP(wout5, wout6, p_dst32_temp, 16);
                AE_S32X2X2_IP(wout7, wout8, p_dst32_temp, 16);
            }

            /* reminder loop for input_width */
            for(i = 0; i < (input_width & 15); i++)
            {
                ae_int8x8 i1, zero = AE_MOVDA8(0);
                ae_int16x4 hi1, hj1;
                ae_int32x2 hout1, hout2;

                AE_L8_IP(i1, (ae_int8 *)p_src1_temp,1);
                AE_ADDW8(hj1, hi1, i1, zero);
                AE_MUL16X4(hout2, hout1, hi1, one16);
                ((ae_int32*)p_dst32_temp)[i] = hout1;
            }
        }
        else
        {
            // If there is no valid input present, fill the output with min_value
            p_dst32_pad = p_scratch + left_pad_aligned ;
            for(i = 0; i < input_width; i++)
            {
                p_dst32_pad[i] = 0;
            }
        }

        /* Pool width processing */

        /* On scratch, compare width-wise with padding*/
        total_out_width = ALIGNED_SIZE(left_pad_aligned + input_width + right_pad + kernel_width, ALIGNMENT/sizeof(WORD32));
        scratch_width = x_padding + input_width + right_pad;
        p_dst = (ae_int32x4 *)((WORD32 *)p_scratch + total_out_width);
        pool_width = kernel_width;

        p_wsrc1 = (ae_int32x4 *)((WORD32 *)p_scratch + left_pad_aligned - x_padding);
        pool_width--;
        p_dst_temp = p_dst;
        p_wsrc1_temp = p_wsrc1;

        /* prime */
        align_wsrc1 = AE_LA128_PP(p_wsrc1_temp);

#pragma concurrent
        for(i = 0; i < (scratch_width >> 3); i++)
        {
            ae_int32x2 wsrc1, wsrc2, wsrc3, wsrc4;
            AE_LA32X2X2_IP(wsrc1, wsrc2, align_wsrc1, (ae_int32x4 *)p_wsrc1_temp);
            AE_LA32X2X2_IP(wsrc3, wsrc4, align_wsrc1, (ae_int32x4 *)p_wsrc1_temp);

            AE_S32X2X2_IP(wsrc1, wsrc2, (ae_int32x4 *)p_dst_temp, 16);
            AE_S32X2X2_IP(wsrc3, wsrc4, (ae_int32x4 *)p_dst_temp, 16);
        }

        /* reminder loop for scratch_width */
        for(i = 0; i < (scratch_width & 7); i++)
        {
           ((ae_int32 *)p_dst_temp)[i] = ((ae_int32 *)p_wsrc1_temp)[i];
        }

        p_wsrc2 = p_dst;

        while(pool_width > 0)
        {
            p_wsrc1 = (ae_int32x4 *)((WORD32 *)p_wsrc1 + 1);
            p_dst_temp = p_dst;
            p_wsrc1_temp = p_wsrc1;
            p_wsrc2_temp = p_wsrc2;

            /* prime */
            align_wsrc1 = AE_LA128_PP(p_wsrc1_temp);

#pragma concurrent
            for(i = 0; i < (scratch_width >> 2); i++)
            {
                ae_int32x2 wi1, wi2;
                ae_int32x2 wsrc1, wsrc2;

                AE_L32X2X2_IP(wsrc1, wsrc2, (ae_int32x4 *)p_wsrc2_temp, 16);
                AE_LA32X2X2_IP(wi1, wi2, align_wsrc1, (ae_int32x4 *)p_wsrc1_temp);
                wsrc1 = AE_ADD32S(wsrc1, wi1);
                wsrc2 = AE_ADD32S(wsrc2, wi2);

                AE_S32X2X2_IP(wsrc1, wsrc2, (ae_int32x4 *)p_dst_temp, 16);
            }

            /* remainder part for scratch_width */
#pragma concurrent
#pragma loop_count max=3
            for(i=0; i<(scratch_width & 3); i++)
            {
                ae_int32x2 wi1;
                ae_int32x2 wsrc1, out;

                wsrc1 = ((ae_int32 *)p_wsrc2_temp)[i];
                wi1 = ((ae_int32 *)p_wsrc1_temp)[i];

                out = AE_ADD32S(wsrc1, wi1);
                ((ae_int32 *)p_dst_temp)[i] = out;
            }
            pool_width--;
        };

        WORD32 *ptr_out1 = (WORD32 *)((WORD32 *)p_scratch + total_out_width);
        ae_int8x8 d_out8, d_1out8;
        ae_int32x2 d_tmp32, d_out1, d_out, d_1tmp32, d_out2;
        ae_int64 d_tmp, d_1tmp;
        if(kernel_height * kernel_width <= 1024)
        {
            WORD32 den_hw, den1_hw;
            for(itr_ow = 0; itr_ow < out_width-1; itr_ow+=2)
            {
                den_hw = inv_256_tbl[p_den_height[itr_oh] * p_den_width[itr_ow]];
                den1_hw = inv_256_tbl[p_den_height[itr_oh] * p_den_width[itr_ow + 1]];
                d_tmp32 = AE_MOVDA32X2(den_hw, den1_hw);
                d_out1 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride]);
                d_out2 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride+x_stride]);

                d_out = AE_SEL32_LL(d_out1, d_out2);

                d_1tmp32 = AE_MULFP32X2RS(d_out, d_tmp32);
                d_out8  = AE_SAT8X4X32_L(d_1tmp32, d_1tmp32);
                d_1out8 = AE_SEL8X8I(d_out8, d_out8, 26);
                AE_S8_0_I(d_1out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow], 0);
                AE_S8_0_I(d_out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow+1], 0);
            }
            if(out_width & 1)
            {
                den_hw = inv_256_tbl[p_den_height[itr_oh] * p_den_width[itr_ow]];
                d_out1 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride]);
                d_tmp32 = AE_MOVDA32(den_hw);
                d_1tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32);
                d_out8 =AE_SAT8X4X32_L(d_1tmp32, d_1tmp32);
                AE_S8_0_I(d_out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow], 0);
            
            }
        }
        else
        {
            ae_int32x2 den_h, den_w, den1_w;
            den_h = AE_MOVDA32(inv_256_tbl[p_den_height[itr_oh]]);
            for(itr_ow = 0; itr_ow < out_width-1; itr_ow+=2)
            {
                den_w = AE_MOVDA32(inv_256_tbl[p_den_width[itr_ow]]);
                den1_w = AE_MOVDA32(inv_256_tbl[p_den_width[itr_ow + 1]]);
                d_out1 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride]);
                d_out2 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride+x_stride]);

                d_tmp = AE_MUL32U_LL(den_h, den_w);
                d_1tmp = AE_MUL32U_LL(den_h, den1_w);

                d_tmp32 = AE_TRUNCI32X2F64S(d_tmp, d_1tmp, 1);

                d_out = AE_SEL32_LL(d_out1, d_out2);

                d_1tmp32 = AE_MULFP32X2RS(d_out, d_tmp32);
                d_out8  = AE_SAT8X4X32_L(d_1tmp32, d_1tmp32);
                d_1out8 = AE_SEL8X8I(d_out8, d_out8, 26);
                AE_S8_0_I(d_1out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow], 0);
                AE_S8_0_I(d_out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow+1], 0);
            }
            if(out_width & 1)
            {
                den_w = AE_MOVDA32(inv_256_tbl[p_den_width[itr_ow]]);
                d_out1 = *(ae_int32 *)(&ptr_out1[itr_ow*x_stride]);
                d_tmp = AE_MUL32U_LL(den_h, den_w);
                d_tmp32 = AE_TRUNCI32X2F64S(d_tmp, d_tmp, 1);
                d_1tmp32 = AE_MULFP32X2RS(d_out1, d_tmp32);
                d_out8 =AE_SAT8X4X32_L(d_1tmp32, d_1tmp32);
                AE_S8_0_I(d_out8, (ae_int8 *)&p_out[itr_oh*out_width+itr_ow], 0);
            
            }
        }
    }
}

WORD32 xa_nn_avgpool_8(
      WORD8* __restrict__ p_out,
const WORD8* __restrict__ p_inp,
      WORD32  input_height,
      WORD32  input_width,
      WORD32  input_channels,
      WORD32  kernel_height,
      WORD32  kernel_width,
      WORD32  x_stride,
      WORD32  y_stride,
      WORD32  x_padding,
      WORD32  y_padding,
      WORD32  out_height,
      WORD32  out_width,
      WORD32  inp_data_format,
      WORD32  out_data_format,
      VOID *p_scratch)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
    XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((out_data_format != 0) && (out_data_format != 1), -1);
    /* Implementation dependent checks */
    XA_NNLIB_ARG_CHK_COND((kernel_height > 1024), -1);
    XA_NNLIB_ARG_CHK_COND((kernel_width > 1024), -1);
    
    XA_NNLIB_ARG_CHK_COND((inp_data_format != 0) && (inp_data_format != 1), -1);
    // Different I/O data formats (not supported!)
    XA_NNLIB_ARG_CHK_COND((out_data_format != inp_data_format), -1);

    if((input_channels == 1) || (out_data_format == 1))
    {
        xa_nn_avgpool_init(8,
                           p_scratch,
                           input_width,
                           kernel_height,
                           kernel_width,
                           x_stride,
                           y_stride,
                           x_padding,
                           out_height,
                           out_width);

        xa_nn_avgpool_state_t *p_state = (xa_nn_avgpool_state_t *)p_scratch;
        int itr_ic, itr_oh, itr_ow;
        const WORD8 *pt_inp; 
        WORD8 *pt_out;
        WORD32 *p_tmp_out = (WORD32 *)(p_state->p_tmp_out);

        /* Calculate denominators for division */
        int kernel_x_start, kernel_x_end, kernel_y_start, kernel_y_end;
        for(itr_oh = 0; itr_oh < out_height; itr_oh++)
        {
            kernel_y_start = itr_oh*y_stride - y_padding;
            kernel_y_end = kernel_y_start + kernel_height;
            LIMIT(kernel_y_start, 0, input_height)
            LIMIT(kernel_y_end, 0, input_height)
            p_state->p_den_height[itr_oh] = (kernel_y_end - kernel_y_start);
        }
        for(itr_ow = 0; itr_ow < out_width; itr_ow++)
        {
            kernel_x_start = itr_ow*x_stride - x_padding;
            kernel_x_end = kernel_x_start + kernel_width;
            LIMIT(kernel_x_start, 0, input_width)
            LIMIT(kernel_x_end, 0, input_width)
            p_state->p_den_width[itr_ow] = (kernel_x_end - kernel_x_start);
        }

        for(itr_ic = 0; itr_ic < input_channels; itr_ic++)
        {
            pt_inp = &p_inp[itr_ic * input_height * input_width];
            pt_out = &p_out[itr_ic * out_height * out_width];

            if(kernel_height <= MAX_HEIGHT_16_BIT_ACC_S8)
            {
                avgpool_8_16(pt_out
                            ,pt_inp
                            ,p_state->p_den_height
                            ,p_state->p_den_width
                            ,input_height
                            ,input_width
                            ,kernel_height
                            ,kernel_width
                            ,x_stride
                            ,y_stride
                            ,x_padding
                            ,y_padding
                            ,out_height
                            ,out_width
                            ,p_tmp_out
                            );
            }
            else
            {
                avgpool_8_32(pt_out
                            ,pt_inp
                            ,p_state->p_den_height
                            ,p_state->p_den_width
                            ,input_height
                            ,input_width
                            ,kernel_height
                            ,kernel_width
                            ,x_stride
                            ,y_stride
                            ,x_padding
                            ,y_padding
                            ,out_height
                            ,out_width
                            ,p_tmp_out
                            );
            }
        }
    }
    else
    {
        int i;
        void *p_scratch_aligned;
        WORD8 *p_zeros, *p_zeros_mem;
        WORD32 *p_rec_den, *p_den_height, *p_den_width;
        WORD32 *p_s;
        int kernel_x_start, kernel_x_end, kernel_y_start, kernel_y_end;
        int cw_plane_size, zero_mem_bytes;


        cw_plane_size = input_width * input_channels;
        p_scratch_aligned = (void *)ALIGN_PTR(p_scratch, ALIGNMENT);

        p_rec_den = (WORD32 *)p_scratch_aligned;
        p_den_height = p_rec_den;
        for(i = 0; i < out_height; i++)
        {
            kernel_y_start = i*y_stride - y_padding;
            kernel_y_end = kernel_y_start + kernel_height;
            LIMIT(kernel_y_start, 0, input_height)
            LIMIT(kernel_y_end, 0, input_height)
            *p_rec_den++ = (kernel_y_end - kernel_y_start);
        }

        p_den_width = (WORD32 *)((WORD8 *)p_scratch_aligned + ALIGNED_SIZE(sizeof(WORD32)*out_height, ALIGNMENT));
        p_rec_den = (WORD32 *)p_den_width;

        for(i = 0; i < out_width; i++)
        {
            kernel_x_start = i*x_stride - x_padding;
            kernel_x_end = kernel_x_start + kernel_width;
            LIMIT(kernel_x_start, 0, input_width)
            LIMIT(kernel_x_end, 0, input_width)
            *p_rec_den++ = (kernel_x_end - kernel_x_start);
        }

        p_s = (WORD32 *)((WORD8 *)p_den_width + ALIGNED_SIZE(sizeof(WORD32)*out_width, ALIGNMENT));
        p_rec_den = p_s;

        if(kernel_height <= (int)MAX_HEIGHT_16_BIT_ACC)
        {
            p_zeros = (WORD8 *)((WORD8 *)p_s + ALIGNED_SIZE(sizeof(WORD16)*cw_plane_size, ALIGNMENT));
            p_zeros = (WORD8 *)((WORD8 *)p_zeros + ALIGNED_SIZE(sizeof(WORD32)*input_channels, ALIGNMENT));
            p_zeros_mem = p_zeros;
            zero_mem_bytes = XT_MAX(sizeof(WORD8)*cw_plane_size, sizeof(WORD16)*input_channels);
        }
        else
        {
            p_zeros = (WORD8 *)((WORD8 *)p_s + ALIGNED_SIZE(sizeof(WORD32)*cw_plane_size, ALIGNMENT));
            p_zeros = (WORD8 *)((WORD8 *)p_zeros + ALIGNED_SIZE(sizeof(WORD32)*input_channels, ALIGNMENT));
            p_zeros_mem = p_zeros;
            zero_mem_bytes = XT_MAX(sizeof(WORD8)*cw_plane_size, sizeof(WORD32)*input_channels);
        }

        for(i = 0; i < zero_mem_bytes; i++)
        {
            *p_zeros++ = 0;
        }

        if(kernel_height <= (int)MAX_HEIGHT_16_BIT_ACC)
        {
            xa_nn_avgpool_8_hwc_16(p_out
                    ,p_inp
                    ,input_height
                    ,input_width
                    ,input_channels
                    ,kernel_height
                    ,kernel_width
                    ,x_stride
                    ,y_stride
                    ,x_padding
                    ,y_padding
                    ,out_height
                    ,out_width
                    ,p_s
                    ,(void *)p_zeros_mem
                    ,p_den_height
                    ,p_den_width);
        }
        else
        {
            xa_nn_avgpool_8_hwc_32(p_out
                    ,p_inp
                    ,input_height
                    ,input_width
                    ,input_channels
                    ,kernel_height
                    ,kernel_width
                    ,x_stride
                    ,y_stride
                    ,x_padding
                    ,y_padding
                    ,out_height
                    ,out_width
                    ,p_s
                    ,(void *)p_zeros_mem
                    ,p_den_height
                    ,p_den_width);
        }
    }
    return 0;
}
