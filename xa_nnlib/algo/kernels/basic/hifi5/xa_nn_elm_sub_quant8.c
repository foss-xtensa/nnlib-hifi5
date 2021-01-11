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
#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"

WORD32 xa_nn_elm_sub_asym8xasym8_asym8(UWORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const   UWORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const   UWORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(UWORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((out_zero_bias < 0) || (out_zero_bias > 255)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -255) || (inp1_zero_bias > 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -255) || (inp2_zero_bias > 0)), -1);
    XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_multiplier < 0) || (inp1_multiplier < 0) || (inp2_multiplier < 0)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < 0) || (out_activation_min > 255)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < 0) || (out_activation_max > 255)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    int i;
    UWORD8 *out = p_out;
    WORD8 *p_i1 = (WORD8 *)p_inp1;
    WORD8 *p_i2 = (WORD8 *)p_inp2;

    ae_f16x4 x1, x2;
    ae_int32x2 temp;
    ae_f16x4 temp16X4, zero_bias1, zero_bias2;
    ae_f32x2 multiplier1, multiplier2, op_multiplier, op_zero_bias, activation_min, activation_max;
    ae_int32x2 ZERO = AE_ZERO32();

    // Taking zero_bias into 16X4 variable
    temp = AE_MOVDA32(inp1_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias1 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

    temp = AE_MOVDA32(inp2_zero_bias);
    temp16X4 = AE_MOVINT16X4_FROMINT32X2(temp);
    zero_bias2 = (ae_f16x4) AE_SEL16_6420(temp16X4, temp16X4);

    op_zero_bias = AE_MOVDA32(out_zero_bias);

    // Taking multiplier into 32x2 variable
    multiplier1 = AE_MOVDA32(inp1_multiplier);
    multiplier2 = AE_MOVDA32(inp2_multiplier);
    op_multiplier = AE_MOVDA32(out_multiplier);

    activation_min = AE_MOVDA32(out_activation_min);
    activation_max = AE_MOVDA32(out_activation_max);

    if(((((unsigned)p_i1)&3) == 0) && ((((unsigned)p_i2)&3) == 0))
    {
        for(i=0;i < num_elm>>2;i++)
        {
            ae_f16x4 v1, v2;
            ae_f32x2 shifted_v1, shifted_v2;
            ae_f32x2 shifted_v3, shifted_v4;
            ae_f32x2 scaled_v1, scaled_v2;
            ae_f32x2 scaled_v3, scaled_v4;
            ae_f32x2 raw_sum12, raw_sum34;
            ae_f32x2 raw_out12, raw_out34;
            ae_f32x2 clamped_out12, clamped_out34;


            AE_L8X4F_IP(x1, p_i1, 4*sizeof(WORD8));
            AE_L8X4F_IP(x2, p_i2, 4*sizeof(WORD8));

            x1 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x1), 8));
            x2 = AE_MOVINT16X4_FROMINT64(AE_SRLI64(AE_MOVINT64_FROMINT16X4(x2), 8));

            v1 = AE_ADD16(x1, zero_bias1);
            v2 = AE_ADD16(x2, zero_bias2);

            shifted_v1 = AE_SEXT32X2D16_32(v1);
            shifted_v2 = AE_SEXT32X2D16_10(v1);
            shifted_v3 = AE_SEXT32X2D16_32(v2);
            shifted_v4 = AE_SEXT32X2D16_10(v2);

            shifted_v1 = AE_SLAA32S(shifted_v1, left_shift);
            shifted_v2 = AE_SLAA32S(shifted_v2, left_shift);
            shifted_v3 = AE_SLAA32S(shifted_v3, left_shift);
            shifted_v4 = AE_SLAA32S(shifted_v4, left_shift);


            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v1, shifted_v1, multiplier1, inp1_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v2, shifted_v2, multiplier1, inp1_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v3, shifted_v3, multiplier2, inp2_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v4, shifted_v4, multiplier2, inp2_left_shift)

            // Raw Difference
            raw_sum12 = AE_SUB32S(scaled_v1, scaled_v3);
            raw_sum34 = AE_SUB32S(scaled_v2, scaled_v4);

            // Raw Output
            MultiplyByQuantizedMultiplierSmallerThanOneExp(raw_out12, raw_sum12, op_multiplier, out_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(raw_out34, raw_sum34, op_multiplier, out_left_shift)
            raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);
            raw_out34 = AE_ADD32S(raw_out34, op_zero_bias);

            // clamped_out
            CLAMP_VAL(clamped_out12, raw_out12, activation_min, activation_max)
            CLAMP_VAL(clamped_out34, raw_out34, activation_min, activation_max)

            // Store Output
            STORE_8X4_FROM_32X4(out, clamped_out12, clamped_out34)
        }
    }
    else
    {
        ALIGN_REGISTER_TYPE i1_a, i2_a;

        PRIME_8X4U(p_i1, i1_a);
        PRIME_8X4U(p_i2, i2_a);
        for(i=0;i < num_elm>>2;i++)
        {
            ae_f16x4 v1, v2;
            ae_f32x2 shifted_v1, shifted_v2;
            ae_f32x2 shifted_v3, shifted_v4;
            ae_f32x2 scaled_v1, scaled_v2;
            ae_f32x2 scaled_v3, scaled_v4;
            ae_f32x2 raw_sum12, raw_sum34;
            ae_f32x2 raw_out12, raw_out34;
            ae_f32x2 clamped_out12, clamped_out34;


            AE_LA8X4U_IP(x1, i1_a, p_i1);
            AE_LA8X4U_IP(x2, i2_a, p_i2);

            v1 = AE_ADD16(x1, zero_bias1);
            v2 = AE_ADD16(x2, zero_bias2);

            shifted_v1 = AE_SEXT32X2D16_32(v1);
            shifted_v2 = AE_SEXT32X2D16_10(v1);
            shifted_v3 = AE_SEXT32X2D16_32(v2);
            shifted_v4 = AE_SEXT32X2D16_10(v2);

            shifted_v1 = AE_SLAA32S(shifted_v1, left_shift);
            shifted_v2 = AE_SLAA32S(shifted_v2, left_shift);
            shifted_v3 = AE_SLAA32S(shifted_v3, left_shift);
            shifted_v4 = AE_SLAA32S(shifted_v4, left_shift);


            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v1, shifted_v1, multiplier1, inp1_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v2, shifted_v2, multiplier1, inp1_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v3, shifted_v3, multiplier2, inp2_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v4, shifted_v4, multiplier2, inp2_left_shift)

            // Raw Difference
            raw_sum12 = AE_SUB32S(scaled_v1, scaled_v3);
            raw_sum34 = AE_SUB32S(scaled_v2, scaled_v4);

            // Raw Output
            MultiplyByQuantizedMultiplierSmallerThanOneExp(raw_out12, raw_sum12, op_multiplier, out_left_shift)
            MultiplyByQuantizedMultiplierSmallerThanOneExp(raw_out34, raw_sum34, op_multiplier, out_left_shift)
            raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);
            raw_out34 = AE_ADD32S(raw_out34, op_zero_bias);

            // clamped_out
            CLAMP_VAL(clamped_out12, raw_out12, activation_min, activation_max)
            CLAMP_VAL(clamped_out34, raw_out34, activation_min, activation_max)

            // Store Output
            STORE_8X4_FROM_32X4(out, clamped_out12, clamped_out34)
        }
    }
    // Remainder Loop
    for(i=0; i < (num_elm & 3); i++)
    {
        ae_f16x4 v1, v2;
        ae_f32x2 shifted_v1;
        ae_f32x2 shifted_v3;
        ae_f32x2 scaled_v1;
        ae_f32x2 scaled_v3;
        ae_f32x2 raw_sum12;
        ae_f32x2 raw_out12;
        ae_f32x2 clamped_out12;

        WORD16 i1, i2;

        i1 = (WORD16) *((UWORD8 *)p_i1 + i);
        i2 = (WORD16) *((UWORD8 *)p_i2 + i);

        x1 = AE_MOVDA16(i1);
        x2 = AE_MOVDA16(i2);

        v1 = AE_ADD16(x1, zero_bias1);
        v2 = AE_ADD16(x2, zero_bias2);

        shifted_v1 = AE_SEXT32X2D16_32(v1);
        shifted_v3 = AE_SEXT32X2D16_32(v2);

        shifted_v1 = AE_SLAA32S(shifted_v1, left_shift);
        shifted_v3 = AE_SLAA32S(shifted_v3, left_shift);

        MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v1, shifted_v1, multiplier1, inp1_left_shift)
        MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_v3, shifted_v3, multiplier2, inp2_left_shift)

        // Raw Difference
        raw_sum12 = AE_SUB32S(scaled_v1, scaled_v3);

        // Raw Output
        MultiplyByQuantizedMultiplierSmallerThanOneExp(raw_out12, raw_sum12, op_multiplier, out_left_shift)
        raw_out12 = AE_ADD32S(raw_out12, op_zero_bias);

        // clamped_out
        CLAMP_VAL(clamped_out12, raw_out12, activation_min, activation_max)

        // Store Output
        i1 = AE_MOVAD32_H(clamped_out12);
        *out++ = (UWORD8) i1;
    }

    return 0;
}

WORD32 xa_nn_elm_sub_asym8sxasym8s_asym8s(WORD8 * __restrict__ p_out,
                            WORD32  out_zero_bias,
                            WORD32  out_left_shift,
                            WORD32  out_multiplier,
                            WORD32  out_activation_min,
                            WORD32  out_activation_max,
                    const    WORD8 * __restrict__ p_inp1,
                            WORD32  inp1_zero_bias,
                            WORD32  inp1_left_shift,
                            WORD32  inp1_multiplier,
                    const    WORD8 * __restrict__ p_inp2,
                            WORD32  inp2_zero_bias,
                            WORD32  inp2_left_shift,
                            WORD32  inp2_multiplier,
                            WORD32  left_shift,
                            WORD32  num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(WORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(WORD8), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);
    XA_NNLIB_ARG_CHK_COND(((out_zero_bias < -128) || (out_zero_bias > 127)), -1);
    XA_NNLIB_ARG_CHK_COND((( out_left_shift < -31) || ( out_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_zero_bias < -127) || (inp1_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_zero_bias < -127) || (inp2_zero_bias > 128)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp1_left_shift < -31) || (inp1_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((inp2_left_shift < -31) || (inp2_left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((left_shift < 0) || (left_shift > 31)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_min < -128) || (out_activation_min > 127)), -1);
    XA_NNLIB_ARG_CHK_COND(((out_activation_max < -128) || (out_activation_max > 127)), -1);
    XA_NNLIB_ARG_CHK_COND((out_activation_max < out_activation_min), -1);

    int i;
    WORD8 *p_a = (WORD8 *)p_inp1;
    WORD8 *p_b = (WORD8 *)p_inp2;
    WORD8 *p_c =          p_out;

    ae_int8x8 a0_7, b0_7;
    const ae_int32x2 activation_min = AE_MOVDA32(out_activation_min);
    const ae_int32x2 activation_max = AE_MOVDA32(out_activation_max);
    const ae_int32x2 ZERO = AE_ZERO32();

    const ae_int8x8  za = AE_SUB8(AE_MOVDA8(0), AE_MOVDA8(inp1_zero_bias));		// replicate 8-bit LSB input into 8x8 output
    const ae_int8x8  zb = AE_SUB8(AE_MOVDA8(0), AE_MOVDA8(inp2_zero_bias));
    const ae_int32x2 zc = AE_MOVDA32( out_zero_bias);
    
    const ae_int32x2 ma = AE_MOVDA32(inp1_multiplier);
    const ae_int32x2 mb = AE_MOVDA32(inp2_multiplier);
    const ae_int32x2 mc = AE_MOVDA32( out_multiplier);	// Multiplier into 32x2 variable

    xtbool io_pointers_aligned =    ((uintptr_t)p_a%8 == 0) &&
                                    ((uintptr_t)p_b%8 == 0) &&
                                    ((uintptr_t)p_c%8 == 0);


    // intermediate results and scratch registers
    ae_int16x4 a0_3, a4_7, b0_3, b4_7;
    ae_int32x2 c0_3, c4_7;

    ae_int32x2 shifted_a0_1, shifted_a2_3, shifted_a4_5, shifted_a6_7;
    ae_int32x2 shifted_b0_1, shifted_b2_3, shifted_b4_5, shifted_b6_7;

    ae_int32x2 scaled_a0_1, scaled_a2_3, scaled_a4_5, scaled_a6_7;
    ae_int32x2 scaled_b0_1, scaled_b2_3, scaled_b4_5, scaled_b6_7;

    ae_int32x2 raw_sum0_1, raw_sum2_3, raw_sum4_5, raw_sum6_7;
    ae_int32x2 out0_1, out2_3, out4_5, out6_7;


    const unsigned int num_simd8_ops = num_elm/8;
    const unsigned int num_scalar_ops = num_elm%8;

    if(io_pointers_aligned){
        for(i=0; i<num_simd8_ops; i++){

            AE_L8X8_IP(a0_7, (ae_int8x8 *)p_a, 8*sizeof(WORD8));
            AE_L8X8_IP(b0_7, (ae_int8x8 *)p_b, 8*sizeof(WORD8));

            AE_SUBW8(a0_3, a4_7, a0_7, za);					// add zero biases
            AE_SUBW8(b0_3, b4_7, b0_7, zb);

            AE_CVTA32X4F16S(shifted_a0_1, shifted_a2_3, a0_3, left_shift);	// LSH and promote to 32-bit
            AE_CVTA32X4F16S(shifted_a4_5, shifted_a6_7, a4_7, left_shift);

            AE_CVTA32X4F16S(shifted_b0_1, shifted_b2_3, b0_3, left_shift);
            AE_CVTA32X4F16S(shifted_b4_5, shifted_b6_7, b4_7, left_shift);

            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_a0_1, shifted_a0_1, ma, inp1_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_a2_3, shifted_a2_3, ma, inp1_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_a4_5, shifted_a4_5, ma, inp1_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_a6_7, shifted_a6_7, ma, inp1_left_shift);


            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_b0_1, shifted_b0_1, mb, inp2_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_b2_3, shifted_b2_3, mb, inp2_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_b4_5, shifted_b4_5, mb, inp2_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_b6_7, shifted_b6_7, mb, inp2_left_shift);

            // Raw Difference
            raw_sum0_1 = AE_SUB32S(scaled_a0_1, scaled_b0_1);
            raw_sum2_3 = AE_SUB32S(scaled_a2_3, scaled_b2_3);
            raw_sum4_5 = AE_SUB32S(scaled_a4_5, scaled_b4_5);
            raw_sum6_7 = AE_SUB32S(scaled_a6_7, scaled_b6_7);

            // Raw Output
            MultiplyByQuantizedMultiplierSmallerThanOneExp(out0_1, raw_sum0_1, mc, out_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(out2_3, raw_sum2_3, mc, out_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(out4_5, raw_sum4_5, mc, out_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(out6_7, raw_sum6_7, mc, out_left_shift);

            out0_1 = AE_ADD32S(out0_1, zc);
            out2_3 = AE_ADD32S(out2_3, zc);
            out4_5 = AE_ADD32S(out4_5, zc);
            out6_7 = AE_ADD32S(out6_7, zc);

            // clamped_out
            AE_MINMAX32(out0_1, activation_min, activation_max);
            AE_MINMAX32(out2_3, activation_min, activation_max);
            AE_MINMAX32(out4_5, activation_min, activation_max);
            AE_MINMAX32(out6_7, activation_min, activation_max);

            // Store Output
            c0_3 = AE_SEL32I(out0_1, out2_3, 8);
            c4_7 = AE_SEL32I(out4_5, out6_7, 8);

            ae_int8x8 res = AE_SEL8X8I(AE_MOVINT8X8_FROMINT32X2(c0_3), AE_MOVINT8X8_FROMINT32X2(c4_7), 25);

            AE_S8X8_IP(res, (ae_int8x8 *)p_c, 8);
        }
    } else {
        ae_valign va_a = AE_LA64_PP(p_a);
        ae_valign va_b = AE_LA64_PP(p_b);
        ae_valign va_c = AE_ZALIGN64();

        for(i=0; i<num_simd8_ops; i++){
            AE_LA8X8_IP(a0_7, va_a, (ae_int8x8 *)p_a);
            AE_LA8X8_IP(b0_7, va_b, (ae_int8x8 *)p_b);
            
            AE_SUBW8(a0_3, a4_7, a0_7, za);
            AE_SUBW8(b0_3, b4_7, b0_7, zb);
            
            AE_CVTA32X4F16S(shifted_a0_1, shifted_a2_3, a0_3, left_shift);	// LSH and promote to 32-bit
            AE_CVTA32X4F16S(shifted_a4_5, shifted_a6_7, a4_7, left_shift);
            
            AE_CVTA32X4F16S(shifted_b0_1, shifted_b2_3, b0_3, left_shift);
            AE_CVTA32X4F16S(shifted_b4_5, shifted_b6_7, b4_7, left_shift);
            
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_a0_1, shifted_a0_1, ma, inp1_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_a2_3, shifted_a2_3, ma, inp1_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_a4_5, shifted_a4_5, ma, inp1_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_a6_7, shifted_a6_7, ma, inp1_left_shift);
            
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_b0_1, shifted_b0_1, mb, inp2_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_b2_3, shifted_b2_3, mb, inp2_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_b4_5, shifted_b4_5, mb, inp2_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(scaled_b6_7, shifted_b6_7, mb, inp2_left_shift);
            
            // Raw Difference
            raw_sum0_1 = AE_SUB32S(scaled_a0_1, scaled_b0_1);
            raw_sum2_3 = AE_SUB32S(scaled_a2_3, scaled_b2_3);
            raw_sum4_5 = AE_SUB32S(scaled_a4_5, scaled_b4_5);
            raw_sum6_7 = AE_SUB32S(scaled_a6_7, scaled_b6_7);
            
            // Raw Output
            MultiplyByQuantizedMultiplierSmallerThanOneExp(out0_1, raw_sum0_1, mc, out_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(out2_3, raw_sum2_3, mc, out_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(out4_5, raw_sum4_5, mc, out_left_shift);
            MultiplyByQuantizedMultiplierSmallerThanOneExp(out6_7, raw_sum6_7, mc, out_left_shift);
            
            out0_1 = AE_ADD32S(out0_1, zc);
            out2_3 = AE_ADD32S(out2_3, zc);
            out4_5 = AE_ADD32S(out4_5, zc);
            out6_7 = AE_ADD32S(out6_7, zc);
            
            // clamped_out
            AE_MINMAX32(out0_1, activation_min, activation_max);
            AE_MINMAX32(out2_3, activation_min, activation_max);
            AE_MINMAX32(out4_5, activation_min, activation_max);
            AE_MINMAX32(out6_7, activation_min, activation_max);
            
            // Store Output
            c0_3 = AE_SEL32I(out0_1, out2_3, 8);
            c4_7 = AE_SEL32I(out4_5, out6_7, 8);
            ae_int8x8 res = AE_SEL8X8I(AE_MOVINT8X8_FROMINT32X2(c0_3), AE_MOVINT8X8_FROMINT32X2(c4_7), 25);
            
            AE_SA8X8_IP(res, va_c, (ae_int8x8 *)p_c);
        }
        
        AE_SA64POS_FP(va_c, p_c);
    }

    for(i=0; i<num_scalar_ops; i++) {
    ae_int32 a, b;
    ae_int32x2 res, res_ab, multiplier_ab;

        a = (ae_int32)(p_a[i] + inp1_zero_bias);            // add input biases
        b = (ae_int32)(p_b[i] + inp2_zero_bias);

        res_ab = AE_MOVDA32X2(a, b);
        res_ab = AE_SLAA32S(res_ab, left_shift);            // shift both inputs by common left_shift

        multiplier_ab = AE_MOVDA32X2(inp1_multiplier, inp2_multiplier);
        res_ab = AE_MULFP32X2RAS(res_ab, multiplier_ab);    // multiply inputs with respective multipliers

        a = AE_MOVAD32_H(res_ab);	b = AE_MOVAD32_L(res_ab);

        a = AE_SRAA32SYMS(a, -inp1_left_shift);             // shift inputs with respective shifters
        b = AE_SRAA32SYMS(b, -inp2_left_shift);

        res = AE_SUB32S(a, b);                              // add inputs to one 32-bit res

        res = AE_MULFP32X2RAS(res, out_multiplier);         // multiply output multiplier
        res = AE_SRAA32SYMS(res, -out_left_shift);          // shift by output shifter

        res = AE_ADD32S(res, out_zero_bias);                // add out zero bias

        AE_MINMAX32(res, activation_min, activation_max);   // clamp

        p_c[i] = (WORD8)AE_MOVAD32_L(res);
    }

    return 0;
}

