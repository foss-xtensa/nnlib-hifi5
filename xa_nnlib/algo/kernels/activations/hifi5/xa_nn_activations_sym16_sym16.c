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
#include "xa_nnlib_common.h"
#include "xa_nnlib_kernels_api.h"

#if !(defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4X2))
static const uint16_t sigmoid_table_uint16[257] = {
    32768, 33451, 34133, 34813, 35493, 36169, 36843, 37513, 38180, 38841, 39498,
    40149, 40794, 41432, 42064, 42688, 43304, 43912, 44511, 45102, 45683, 46255,
    46817, 47369, 47911, 48443, 48964, 49475, 49975, 50464, 50942, 51409, 51865,
    52311, 52745, 53169, 53581, 53983, 54374, 54755, 55125, 55485, 55834, 56174,
    56503, 56823, 57133, 57433, 57724, 58007, 58280, 58544, 58800, 59048, 59288,
    59519, 59743, 59959, 60168, 60370, 60565, 60753, 60935, 61110, 61279, 61441,
    61599, 61750, 61896, 62036, 62172, 62302, 62428, 62549, 62666, 62778, 62886,
    62990, 63090, 63186, 63279, 63368, 63454, 63536, 63615, 63691, 63765, 63835,
    63903, 63968, 64030, 64090, 64148, 64204, 64257, 64308, 64357, 64405, 64450,
    64494, 64536, 64576, 64614, 64652, 64687, 64721, 64754, 64786, 64816, 64845,
    64873, 64900, 64926, 64950, 64974, 64997, 65019, 65039, 65060, 65079, 65097,
    65115, 65132, 65149, 65164, 65179, 65194, 65208, 65221, 65234, 65246, 65258,
    65269, 65280, 65291, 65301, 65310, 65319, 65328, 65337, 65345, 65352, 65360,
    65367, 65374, 65381, 65387, 65393, 65399, 65404, 65410, 65415, 65420, 65425,
    65429, 65433, 65438, 65442, 65445, 65449, 65453, 65456, 65459, 65462, 65465,
    65468, 65471, 65474, 65476, 65479, 65481, 65483, 65485, 65488, 65489, 65491,
    65493, 65495, 65497, 65498, 65500, 65501, 65503, 65504, 65505, 65507, 65508,
    65509, 65510, 65511, 65512, 65513, 65514, 65515, 65516, 65517, 65517, 65518,
    65519, 65520, 65520, 65521, 65522, 65522, 65523, 65523, 65524, 65524, 65525,
    65525, 65526, 65526, 65526, 65527, 65527, 65528, 65528, 65528, 65529, 65529,
    65529, 65529, 65530, 65530, 65530, 65530, 65531, 65531, 65531, 65531, 65531,
    65532, 65532, 65532, 65532, 65532, 65532, 65533, 65533, 65533, 65533, 65533,
    65533, 65533, 65533, 65534, 65534, 65534, 65534, 65534, 65534, 65534, 65534,
    65534, 65534, 65535, 65535};
#endif

/* The scale of input for TFLM reference is 4096*3, which is maintained in the LUT based implementation 
 * in xa_nn_vec_sigmoid_sym16s_sym16s(). 
 * However, the TIE based implementation uses input scale of 4096. The corresponding scaling by 3 in 
 * TFLM prepare function is removed for this implemenation.
 */
WORD32 xa_nn_vec_sigmoid_sym16s_sym16s(WORD16 *p_out,
                      const WORD16 *p_vec,
                            WORD32 input_multiplier,
                            WORD32 input_left_shift,
                            WORD32 vec_length)
{
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_left_shift < 0), -1);

  ae_int16x8 *in_ptr_align  = (ae_int16x8 *)p_vec;
  ae_int16x8 *out_ptr = (ae_int16x8 *)p_out;

  ae_valignx2 inp_align = AE_LA128_PP(in_ptr_align);
  ae_valignx2 out_align = AE_ZALIGN128();

  ae_int16x4 inp0, inp1;
  int i;

#if (defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4X2))
  int sar_reg_val = AE_MOVASAR();
  int sar_reg_low_half = sar_reg_val & 0x7F;
  sar_reg_val = sar_reg_val >> 7;
  int sar_reg_up_half = sar_reg_val & 0x7F;
  WUR_AE_SAR(4);

  if(input_multiplier == 0 && input_left_shift == 0)
  {
#pragma concurrent
    for (i = 0; i < vec_length >> 3; i++)
    {
      AE_LA16X4X2_IP(inp0, inp1, inp_align, in_ptr_align);

      ae_int16x4 out0, out1;
      AE_SIGMOID16X4X2(out0, out1, inp0, inp1);
      out0 = AE_SRLI16(out0, 1);
      out1 = AE_SRLI16(out1, 1);
      AE_SA16X4X2_IP(out0, out1, out_align, out_ptr);
    }
    if(vec_length & 7)
    {
      AE_LAV16X4X2_XP(inp0, inp1, inp_align, in_ptr_align, (vec_length & 7) << 1);

      ae_int16x4 out0, out1;
      AE_SIGMOID16X4X2(out0, out1, inp0, inp1);
      out0 = AE_SRLI16(out0, 1);
      out1 = AE_SRLI16(out1, 1);
      AE_SAV16X4X2_XP(out0, out1, out_align, out_ptr, (vec_length & 7) << 1);
    }
    AE_SA128POS_FP(out_align, out_ptr);
    AE_MOVSARA7X2(sar_reg_up_half, sar_reg_low_half);
    return 0;
  }
#endif // #if (defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4X2))

  if (input_multiplier == 0)
  {
#if (defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4X2))    
    input_multiplier = 1 << input_left_shift;
#else 
    input_multiplier = 3 << input_left_shift;
#endif
    input_left_shift = 0;
  }

  ae_int16x4 inp_mult = input_multiplier;
  ae_int32x2 inp_x_inp_mul0, inp_x_inp_mul1;
  ae_int32x2 inp_x_inp_mul2, inp_x_inp_mul3;
#if !(defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4X2))
  ae_int32x2 uint8_max_x2  = 508;     // To saturate indices at 254
  ae_int16x4 mask_nine_bit = 511;
  ae_int32x2 sub_val       = (1 << (16 + 9)) - 1;

  ae_int16x4 ut0, ut1;
  ae_int32x2 abs_inp_x_inp_mul0, abs_inp_x_inp_mul1;
  ae_int32x2 abs_inp_x_inp_mul2, abs_inp_x_inp_mul3;
  ae_int32x2 ua0, ub0;
  ae_int32x2 ub_minus_ua0;
  ae_int32x2 res0, res1, res0_sub, res1_sub;
  ae_int32x2 res2, res3, res2_sub, res3_sub;
  ae_int32x2 uh_0;
  xtbool2 x0, x1, x2, x3;
#else
  ae_int16x4 sigmoid_in0, sigmoid_in1;

  ae_int32x2 ALIGN(16) round_val[2];
  round_val[0] = AE_MOVDA32(0x40000000);
  round_val[0] = AE_SRAA32(round_val[0], 31 - (input_left_shift < 31 ? input_left_shift : 0));
  round_val[1] = round_val[0];
  ae_int32x4 *p_round_val = (ae_int32x4 *)round_val;
#endif

#pragma concurrent
  for (i = 0; i < vec_length >> 3; i++)
  {
    AE_LA16X4X2_IP(inp0, inp1, inp_align, in_ptr_align);

#if defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4X2)
    AE_L32X2X2_I(inp_x_inp_mul0, inp_x_inp_mul1, p_round_val, 0);
    AE_L32X2X2_IP(inp_x_inp_mul2, inp_x_inp_mul3, p_round_val, 0);

    AE_MULA16X4S(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);
    AE_MULA16X4S(inp_x_inp_mul2, inp_x_inp_mul3, inp1, inp_mult);

    sigmoid_in0 = AE_TRUNCA16X4F32S(inp_x_inp_mul0, inp_x_inp_mul1, 16 - input_left_shift);
    sigmoid_in1 = AE_TRUNCA16X4F32S(inp_x_inp_mul2, inp_x_inp_mul3, 16 - input_left_shift);
    ae_int16x4 out0, out1;
    AE_SIGMOID16X4X2(out0, out1, sigmoid_in0, sigmoid_in1);
    out0 = AE_SRLI16(out0, 1);
    out1 = AE_SRLI16(out1, 1);
#else
    AE_MUL16X4(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);
    AE_MUL16X4(inp_x_inp_mul2, inp_x_inp_mul3, inp1, inp_mult);

    inp_x_inp_mul0 = AE_SRAA32RS(inp_x_inp_mul0, input_left_shift);
    inp_x_inp_mul1 = AE_SRAA32RS(inp_x_inp_mul1, input_left_shift);
    inp_x_inp_mul2 = AE_SRAA32RS(inp_x_inp_mul2, input_left_shift);
    inp_x_inp_mul3 = AE_SRAA32RS(inp_x_inp_mul3, input_left_shift);

    abs_inp_x_inp_mul0 = AE_ABS32S(inp_x_inp_mul0);
    abs_inp_x_inp_mul1 = AE_ABS32S(inp_x_inp_mul1);
    abs_inp_x_inp_mul2 = AE_ABS32S(inp_x_inp_mul2);
    abs_inp_x_inp_mul3 = AE_ABS32S(inp_x_inp_mul3);

    ut0 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul0), AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul1));
    ut0 = AE_AND16(ut0, mask_nine_bit);
    ut1 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul2), AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul3));
    ut1 = AE_AND16(ut1, mask_nine_bit);

    ae_int16x4 uh0 = AE_TRUNCA16X4F32S(abs_inp_x_inp_mul0, abs_inp_x_inp_mul1, 7);
    /* Extra left shift of 1 for using these values in L16_X instruction, using ADD for better slotting */
    uh0 = AE_ADD16S(uh0, uh0);

    uh0 = AE_MIN16(uh0, AE_SAT16X4(uint8_max_x2, uint8_max_x2));

    int id0, id1, id2, id3;
    id0 = AE_MOVAD16_3(uh0);
    id1 = AE_MOVAD16_2(uh0);
    id2 = AE_MOVAD16_1(uh0);
    id3 = AE_MOVAD16_0(uh0);

    ae_int16 *psigmoid_table_uint16 = (ae_int16 *)sigmoid_table_uint16;

    ae_int16x4 ua0_16 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ua0_16 = AE_SEL16_6543(ua0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1));
    ua0_16 = AE_SEL16_6543(ua0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2));
    ua0_16 = AE_SEL16_6543(ua0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3));

    psigmoid_table_uint16++;

    ae_int16x4 ub0_16 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ub0_16 = AE_SEL16_6543(ub0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1));
    ub0_16 = AE_SEL16_6543(ub0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2));
    ub0_16 = AE_SEL16_6543(ub0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3));

    ae_int16x4 uh1 = AE_TRUNCA16X4F32S(abs_inp_x_inp_mul2, abs_inp_x_inp_mul3, 7);
    /* Extra left shift of 1 for using these values in L16_X instruction */
    uh1 = AE_ADD16S(uh1, uh1);

    uh1 = AE_MIN16(uh1, AE_SAT16X4(uint8_max_x2, uint8_max_x2));

    id0 = AE_MOVAD16_3(uh1);
    id1 = AE_MOVAD16_2(uh1);
    id2 = AE_MOVAD16_1(uh1);
    id3 = AE_MOVAD16_0(uh1);

    psigmoid_table_uint16 = (ae_int16 *)sigmoid_table_uint16;

    ae_int16x4 ua1_16 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ua1_16 = AE_SEL16_6543(ua1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1));
    ua1_16 = AE_SEL16_6543(ua1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2));
    ua1_16 = AE_SEL16_6543(ua1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3));

    psigmoid_table_uint16++;

    ae_int16x4 ub1_16 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ub1_16 = AE_SEL16_6543(ub1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1));
    ub1_16 = AE_SEL16_6543(ub1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2));
    ub1_16 = AE_SEL16_6543(ub1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3));

    AE_CVTA32X4F16U(res0, res1, ua0_16, 9);
    AE_CVTA32X4F16U(res2, res3, ua1_16, 9);

    ub0_16 = AE_SUB16(ub0_16, ua0_16);
    ub1_16 = AE_SUB16(ub1_16, ua1_16);

    AE_MULA16X4(res0, res1, ub0_16, ut0);
    AE_MULA16X4(res2, res3, ub1_16, ut1);

    res0_sub = AE_SUB32S(sub_val, res0);
    res1_sub = AE_SUB32S(sub_val, res1);
    res2_sub = AE_SUB32S(sub_val, res2);
    res3_sub = AE_SUB32S(sub_val, res3);

    x0 = AE_LT32(inp_x_inp_mul0, AE_ZERO32());
    x1 = AE_LT32(inp_x_inp_mul1, AE_ZERO32());
    x2 = AE_LT32(inp_x_inp_mul2, AE_ZERO32());
    x3 = AE_LT32(inp_x_inp_mul3, AE_ZERO32());

    AE_MOVT32X2(res0, res0_sub, x0);
    AE_MOVT32X2(res1, res1_sub, x1);
    AE_MOVT32X2(res2, res2_sub, x2);
    AE_MOVT32X2(res3, res3_sub, x3);

    /* Right shift of 10 with asymmetric rounding */
    AE_MULF2P32X16X4RAS(res0, res1, res0, res1, AE_MOVDA16(1 << 5));
    AE_MULF2P32X16X4RAS(res2, res3, res2, res3, AE_MOVDA16(1 << 5));

    ae_int16x4 out0 = AE_SAT16X4(res0, res1);
    ae_int16x4 out1 = AE_SAT16X4(res2, res3);
#endif
    AE_SA16X4X2_IP(out0, out1, out_align, out_ptr);
  }
#if defined(USE_HIFI_ACT_TIE) && defined(AE_SIGMOID16X4X2)
  if(vec_length & 7)
  {
    AE_LAV16X4X2_XP(inp0, inp1, inp_align, in_ptr_align, (vec_length & 7) << 1);
    AE_MUL16X4(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);
    AE_MUL16X4(inp_x_inp_mul2, inp_x_inp_mul3, inp1, inp_mult);

    inp_x_inp_mul0 = AE_SRAA32RS(inp_x_inp_mul0, input_left_shift);
    inp_x_inp_mul1 = AE_SRAA32RS(inp_x_inp_mul1, input_left_shift);
    inp_x_inp_mul2 = AE_SRAA32RS(inp_x_inp_mul2, input_left_shift);
    inp_x_inp_mul3 = AE_SRAA32RS(inp_x_inp_mul3, input_left_shift);

    sigmoid_in0 = AE_SAT16X4(inp_x_inp_mul0, inp_x_inp_mul1);
    sigmoid_in1 = AE_SAT16X4(inp_x_inp_mul2, inp_x_inp_mul3);
    ae_int16x4 out0, out1;
    AE_SIGMOID16X4X2(out0, out1, sigmoid_in0, sigmoid_in1);
    out0 = AE_SRLI16(out0, 1);
    out1 = AE_SRLI16(out1, 1);
    AE_SAV16X4X2_XP(out0, out1, out_align, out_ptr, (vec_length & 7) << 1);
  }
  AE_SA128POS_FP(out_align, out_ptr);
  AE_MOVSARA7X2(sar_reg_up_half, sar_reg_low_half);
#else
  AE_SA128POS_FP(out_align, out_ptr);

  p_vec = (ae_int16 *)in_ptr_align;
  p_out = (ae_int16 *)out_ptr;

#pragma loop_count max=7
#pragma concurrent
  for (i = 0; i < (vec_length & 7); i++)
  {
    inp0 = *p_vec++;

    AE_MUL16X4(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);

    inp_x_inp_mul0 = AE_SRAA32RS(inp_x_inp_mul0, input_left_shift);

    abs_inp_x_inp_mul0 = AE_ABS32S(inp_x_inp_mul0);

    ut0 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul0), AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul0));
    ut0 = AE_AND16(ut0, mask_nine_bit);

    uh_0 = AE_SRAI32(abs_inp_x_inp_mul0, 9);
    /* 1 left shift for using this value in L16_X instruction */
    uh_0 = AE_SLAI32S(uh_0, 1);
    uh_0 = AE_MIN32(uh_0, uint8_max_x2);

    int id0;
    id0 = AE_MOVAD32_H(uh_0);

    ae_int16 *psigmoid_table_uint16 = (ae_int16 *)sigmoid_table_uint16;

    ae_int16x4 sel0 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ua0 = AE_MOVINT32X2_FROMINT16X4(sel0);
    ua0 = AE_SRLI32(ua0, 16);

    psigmoid_table_uint16++;

    sel0 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ub0 = AE_MOVINT32X2_FROMINT16X4(sel0);
    ub0 = AE_SRLI32(ub0, 16);

    res0 = AE_SLAI32S(ua0, 9);

    ub_minus_ua0 = AE_SUB32S(ub0, ua0);

    AE_MULAP32X16X2_H(res0, ub_minus_ua0, ut0);

    res0_sub = AE_SUB32S(sub_val, res0);

    x0 = AE_LT32(inp_x_inp_mul0, AE_ZERO32());

    AE_MOVT32X2(res0, res0_sub, x0);

    res0 = AE_SRAI32R(res0, 10);

    ae_int16x4 out = AE_SAT16X4(res0, res0);
    *p_out++ = out;
  }
#endif
  return 0;
}

/* The scale of input for TFLM reference is 4096*3, which is maintained in the LUT based implementation 
 * in xa_nn_vec_tanh_sym16s_sym16s(). 
 * However, the TIE based implementation uses input scale of 4096. The corresponding scaling by 3 in 
 * TFLM prepare function is removed for this implemenation.
 */
WORD32 xa_nn_vec_tanh_sym16s_sym16s(WORD16 *p_out,
                      const WORD16 *p_vec,
                            WORD32 input_multiplier,
                            WORD32 input_left_shift,
                            WORD32 vec_length)
{
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec, -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_left_shift < 0), -1);

  ae_int16x8 *in_ptr_align  = (ae_int16x8 *)p_vec;
  ae_int16x8 *out_ptr = (ae_int16x8 *)p_out;

  ae_valignx2 inp_align = AE_LA128_PP(in_ptr_align);
  ae_valignx2 out_align = AE_ZALIGN128();

  ae_int16x4 inp0, inp1;
  int i;

#if (defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4X2))
  int sar_reg_val = AE_MOVASAR();
  int sar_reg_low_half = sar_reg_val & 0x7F;
  sar_reg_val = sar_reg_val >> 7;
  int sar_reg_up_half = sar_reg_val & 0x7F;
  WUR_AE_SAR(4);
  if(input_multiplier == 0 && input_left_shift == 0)
  {
#pragma concurrent
    for (i = 0; i < vec_length >> 3; i++)
    {
      AE_LA16X4X2_IP(inp0, inp1, inp_align, in_ptr_align);

      ae_int16x4 out0, out1;
      AE_TANH16X4X2(out0, out1, inp0, inp1);
      AE_SA16X4X2_IP(out0, out1, out_align, out_ptr);
    }
    if(vec_length & 7)
    {
      AE_LAV16X4X2_XP(inp0, inp1, inp_align, in_ptr_align, (vec_length & 7) << 1);

      ae_int16x4 out0, out1;
      AE_TANH16X4X2(out0, out1, inp0, inp1);
      AE_SAV16X4X2_XP(out0, out1, out_align, out_ptr, (vec_length & 7) << 1);
    } 
    AE_SA128POS_FP(out_align, out_ptr);
    AE_MOVSARA7X2(sar_reg_up_half, sar_reg_low_half);
    return 0;
  }
#endif // #if (defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4X2))

  if (input_multiplier == 0)
  {
#if (defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4X2))    
    input_multiplier = 1 << input_left_shift;
#else 
    input_multiplier = 3 << input_left_shift;
#endif
    input_left_shift = 0;
  }

  ae_int16x4 inp_mult = input_multiplier;
  ae_int32x2 inp_x_inp_mul0, inp_x_inp_mul1;
  ae_int32x2 inp_x_inp_mul2, inp_x_inp_mul3;
#if !(defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4X2))
  ae_int32x2 uint8_max_x2  = 510;     // To saturate indices at 255
  ae_int16x4 mask_eight_bit = 255;
  ae_int32x2 one_q23 = 1 << (14 + 9);
  ae_int32x2 sub_val       = - 1;

  ae_int16x4 ut0, ut1;
  ae_int32x2 abs_inp_x_inp_mul0, abs_inp_x_inp_mul1;
  ae_int32x2 abs_inp_x_inp_mul2, abs_inp_x_inp_mul3;
  ae_int32x2 ua0, ub0;
  ae_int32x2 ub_minus_ua0;
  ae_int32x2 res0, res1, res0_sub, res1_sub;
  ae_int32x2 res2, res3, res2_sub, res3_sub;
  ae_int32x2 uh_0;
  xtbool2 x0, x1, x2, x3;
#else
  ae_int16x4 tanh_in0, tanh_in1;

  ae_int32x2 ALIGN(16) round_val[2];
  round_val[0] = AE_MOVDA32(0x40000000);
  round_val[0] = AE_SRAA32(round_val[0], 31 - (input_left_shift < 31 ? input_left_shift : 0));
  round_val[1] = round_val[0];
  ae_int32x4 *p_round_val = (ae_int32x4 *)round_val;
#endif

#pragma concurrent
  for (i = 0; i < vec_length >> 3; i++)
  {
    AE_LA16X4X2_IP(inp0, inp1, inp_align, in_ptr_align);

#if (defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4X2))
    AE_L32X2X2_I(inp_x_inp_mul0, inp_x_inp_mul1, p_round_val, 0);
    AE_L32X2X2_IP(inp_x_inp_mul2, inp_x_inp_mul3, p_round_val, 0);

    AE_MULA16X4S(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);
    AE_MULA16X4S(inp_x_inp_mul2, inp_x_inp_mul3, inp1, inp_mult);

    tanh_in0 = AE_TRUNCA16X4F32S(inp_x_inp_mul0, inp_x_inp_mul1, 16 - input_left_shift);
    tanh_in1 = AE_TRUNCA16X4F32S(inp_x_inp_mul2, inp_x_inp_mul3, 16 - input_left_shift);

    ae_int16x4 out0, out1;
    AE_TANH16X4X2(out0, out1, tanh_in0, tanh_in1);
#else
    AE_MUL16X4(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);
    AE_MUL16X4(inp_x_inp_mul2, inp_x_inp_mul3, inp1, inp_mult);

    inp_x_inp_mul0 = AE_SRAA32RS(inp_x_inp_mul0, input_left_shift);
    inp_x_inp_mul1 = AE_SRAA32RS(inp_x_inp_mul1, input_left_shift);
    inp_x_inp_mul2 = AE_SRAA32RS(inp_x_inp_mul2, input_left_shift);
    inp_x_inp_mul3 = AE_SRAA32RS(inp_x_inp_mul3, input_left_shift);

    abs_inp_x_inp_mul0 = AE_ABS32S(inp_x_inp_mul0);
    abs_inp_x_inp_mul1 = AE_ABS32S(inp_x_inp_mul1);
    abs_inp_x_inp_mul2 = AE_ABS32S(inp_x_inp_mul2);
    abs_inp_x_inp_mul3 = AE_ABS32S(inp_x_inp_mul3);

    ut0 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul0), AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul1));
    ut0 = AE_AND16(ut0, mask_eight_bit);
    ut1 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul2), AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul3));
    ut1 = AE_AND16(ut1, mask_eight_bit);

    ae_int16x4 uh0 = AE_TRUNCA16X4F32S(abs_inp_x_inp_mul0, abs_inp_x_inp_mul1, 8);
    /* Extra left shift of 1 for using these values in L16_X instruction, using ADD for better slotting */
    uh0 = AE_ADD16S(uh0, uh0);

    uh0 = AE_MIN16(uh0, AE_SAT16X4(uint8_max_x2, uint8_max_x2));

    int id0, id1, id2, id3;
    id0 = AE_MOVAD16_3(uh0);
    id1 = AE_MOVAD16_2(uh0);
    id2 = AE_MOVAD16_1(uh0);
    id3 = AE_MOVAD16_0(uh0);

    ae_int16 *psigmoid_table_uint16 = (ae_int16 *)sigmoid_table_uint16;

    ae_int16x4 ua0_16 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ua0_16 = AE_SEL16_6543(ua0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1));
    ua0_16 = AE_SEL16_6543(ua0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2));
    ua0_16 = AE_SEL16_6543(ua0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3));

    psigmoid_table_uint16++;

    ae_int16x4 ub0_16 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ub0_16 = AE_SEL16_6543(ub0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1));
    ub0_16 = AE_SEL16_6543(ub0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2));
    ub0_16 = AE_SEL16_6543(ub0_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3));

    ae_int16x4 uh1 = AE_TRUNCA16X4F32S(abs_inp_x_inp_mul2, abs_inp_x_inp_mul3, 8);
    /* Extra left shift of 1 for using these values in L16_X instruction */
    uh1 = AE_ADD16S(uh1, uh1);

    uh1 = AE_MIN16(uh1, AE_SAT16X4(uint8_max_x2, uint8_max_x2));

    id0 = AE_MOVAD16_3(uh1);
    id1 = AE_MOVAD16_2(uh1);
    id2 = AE_MOVAD16_1(uh1);
    id3 = AE_MOVAD16_0(uh1);

    psigmoid_table_uint16 = (ae_int16 *)sigmoid_table_uint16;

    ae_int16x4 ua1_16 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ua1_16 = AE_SEL16_6543(ua1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1));
    ua1_16 = AE_SEL16_6543(ua1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2));
    ua1_16 = AE_SEL16_6543(ua1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3));

    psigmoid_table_uint16++;

    ae_int16x4 ub1_16 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ub1_16 = AE_SEL16_6543(ub1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id1));
    ub1_16 = AE_SEL16_6543(ub1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id2));
    ub1_16 = AE_SEL16_6543(ub1_16, AE_L16_X((ae_int16 *)psigmoid_table_uint16, id3));

    AE_CVTA32X4F16U(res0, res1, ua0_16, 8);
    AE_CVTA32X4F16U(res2, res3, ua1_16, 8);

    ub0_16 = AE_SUB16(ub0_16, ua0_16);
    ub1_16 = AE_SUB16(ub1_16, ua1_16);

    AE_MULA16X4(res0, res1, ub0_16, ut0);
    AE_MULA16X4(res2, res3, ub1_16, ut1);

    res0 = AE_SUB32(res0, one_q23);
    res1 = AE_SUB32(res1, one_q23);
    res2 = AE_SUB32(res2, one_q23);
    res3 = AE_SUB32(res3, one_q23);

    res0_sub = AE_SUB32S(sub_val, res0);
    res1_sub = AE_SUB32S(sub_val, res1);
    res2_sub = AE_SUB32S(sub_val, res2);
    res3_sub = AE_SUB32S(sub_val, res3);

    x0 = AE_LT32(inp_x_inp_mul0, AE_ZERO32());
    x1 = AE_LT32(inp_x_inp_mul1, AE_ZERO32());
    x2 = AE_LT32(inp_x_inp_mul2, AE_ZERO32());
    x3 = AE_LT32(inp_x_inp_mul3, AE_ZERO32());

    AE_MOVT32X2(res0, res0_sub, x0);
    AE_MOVT32X2(res1, res1_sub, x1);
    AE_MOVT32X2(res2, res2_sub, x2);
    AE_MOVT32X2(res3, res3_sub, x3);

    /* Right shift of (9-1) with asymmetric rounding */
    AE_MULF2P32X16X4RAS(res0, res1, res0, res1, AE_MOVDA16(1 << 7));
    AE_MULF2P32X16X4RAS(res2, res3, res2, res3, AE_MOVDA16(1 << 7));

    ae_int16x4 out0 = AE_SAT16X4(res0, res1);
    ae_int16x4 out1 = AE_SAT16X4(res2, res3);
#endif
    AE_SA16X4X2_IP(out0, out1, out_align, out_ptr);
  }
#if defined(USE_HIFI_ACT_TIE) && defined(AE_TANH16X4X2)
  if(vec_length & 7)
  {
    AE_LAV16X4X2_XP(inp0, inp1, inp_align, in_ptr_align, (vec_length & 7) << 1);
    AE_MUL16X4(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);
    AE_MUL16X4(inp_x_inp_mul2, inp_x_inp_mul3, inp1, inp_mult);

    inp_x_inp_mul0 = AE_SRAA32RS(inp_x_inp_mul0, input_left_shift);
    inp_x_inp_mul1 = AE_SRAA32RS(inp_x_inp_mul1, input_left_shift);
    inp_x_inp_mul2 = AE_SRAA32RS(inp_x_inp_mul2, input_left_shift);
    inp_x_inp_mul3 = AE_SRAA32RS(inp_x_inp_mul3, input_left_shift);

    tanh_in0 = AE_SAT16X4(inp_x_inp_mul0, inp_x_inp_mul1);
    tanh_in1 = AE_SAT16X4(inp_x_inp_mul2, inp_x_inp_mul3);
    ae_int16x4 out0, out1;
    AE_TANH16X4X2(out0, out1, tanh_in0, tanh_in1);
    AE_SAV16X4X2_XP(out0, out1, out_align, out_ptr, (vec_length & 7) << 1);
  } 
  AE_SA128POS_FP(out_align, out_ptr);
  AE_MOVSARA7X2(sar_reg_up_half, sar_reg_low_half);
#else
  AE_SA128POS_FP(out_align, out_ptr);

  p_vec = (ae_int16 *)in_ptr_align;
  p_out = (ae_int16 *)out_ptr;

#pragma loop_count max=7
#pragma concurrent
  for (i = 0; i < (vec_length & 7); i++)
  {
    inp0 = *p_vec++;

    AE_MUL16X4(inp_x_inp_mul0, inp_x_inp_mul1, inp0, inp_mult);

    inp_x_inp_mul0 = AE_SRAA32RS(inp_x_inp_mul0, input_left_shift);

    abs_inp_x_inp_mul0 = AE_ABS32S(inp_x_inp_mul0);

    ut0 = AE_SEL16_6420(AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul0), AE_MOVINT16X4_FROMINT32X2(abs_inp_x_inp_mul0));
    ut0 = AE_AND16(ut0, mask_eight_bit);

    uh_0 = AE_SRAI32(abs_inp_x_inp_mul0, 8);
    /* 1 left shift for using this value in L16_X instruction */
    uh_0 = AE_SLAI32S(uh_0, 1);
    uh_0 = AE_MIN32(uh_0, uint8_max_x2);

    int id0;
    id0 = AE_MOVAD32_H(uh_0);

    ae_int16 *psigmoid_table_uint16 = (ae_int16 *)sigmoid_table_uint16;

    ae_int16x4 sel0 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ua0 = AE_MOVINT32X2_FROMINT16X4(sel0);
    ua0 = AE_SRLI32(ua0, 16);

    psigmoid_table_uint16++;

    sel0 = AE_L16_X((ae_int16 *)psigmoid_table_uint16, id0);
    ub0 = AE_MOVINT32X2_FROMINT16X4(sel0);
    ub0 = AE_SRLI32(ub0, 16);

    res0 = AE_SLAI32S(ua0, 8);

    ub_minus_ua0 = AE_SUB32S(ub0, ua0);

    AE_MULAP32X16X2_H(res0, ub_minus_ua0, ut0);
    res0 = AE_SUB32(res0, one_q23);

    res0_sub = AE_SUB32S(sub_val, res0);

    x0 = AE_LT32(inp_x_inp_mul0, AE_ZERO32());

    AE_MOVT32X2(res0, res0_sub, x0);

    res0 = AE_SRAI32R(res0, 8);

    ae_int16x4 out = AE_SAT16X4(res0, res0);
    *p_out++ = out;
  }
#endif
  return 0;
}
