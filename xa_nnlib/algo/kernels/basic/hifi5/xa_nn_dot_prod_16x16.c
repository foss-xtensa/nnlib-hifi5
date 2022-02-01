/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
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
#include "xa_nnlib_common_macros_hifi5.h"

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2(inp, multiplier, left_shift, right_shift) \
  inp = AE_SLAA32(inp, left_shift); \
  inp = AE_MULFP32X2RAS(inp, AE_MOVDA32(multiplier)); \
  inp = AE_SRAA32SYMS(inp, right_shift);

/*----------------------------Main function---------------------------------*/
WORD32 xa_nn_dot_prod_16x16_asym8s(
    WORD8 * __restrict__ p_out,           /* pointer to output */
    const WORD16 * __restrict__ p_inp1_start,    /* pointer to input1 */
    const WORD16 * __restrict__ p_inp2_start,    /* pointer to input2 */
    const WORD32 * bias_ptr,
    WORD32 vec_length,
    WORD32 out_multiplier,
    WORD32 out_shift,
    WORD32 out_zero_bias,
    WORD32 vec_count)
{
  /* NULL pointer checks */
	XA_NNLIB_ARG_CHK_PTR(p_out, -1);
	XA_NNLIB_ARG_CHK_PTR(p_inp1_start, -1);
	XA_NNLIB_ARG_CHK_PTR(p_inp2_start, -1);
	/* Pointer alignment checks */
	XA_NNLIB_ARG_CHK_ALIGN(p_inp1_start, sizeof(WORD16), -1);
	XA_NNLIB_ARG_CHK_ALIGN(p_inp2_start, sizeof(WORD16), -1);
	/* Basic Parameter checks */
	XA_NNLIB_ARG_CHK_COND((vec_length <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((vec_count <= 0), -1);
	XA_NNLIB_ARG_CHK_COND((out_shift < -31 || out_shift > 31), -1);
	XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
	int left_shift, right_shift;
	int loopcnt;
	const WORD32 bias_buffer[2] = {0, 0};
	const WORD32* p_bias_load;
	WORD32 bias_address_increment = sizeof(WORD32);

	if(bias_ptr == NULL)
	{
		p_bias_load = bias_buffer;
		bias_address_increment = 0;
	}
	else
	{
		p_bias_load = bias_ptr;
	}

	left_shift = out_shift < 0 ? 0 : out_shift;
	right_shift = out_shift > 0 ? 0 : -out_shift;
  
  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);
      
  const ae_int16x4 *pt_inp1, *pt_inp2;		
  ae_valignx2 align_inp1, align_inp2;
  ae_int16x4 d_inp1_0, d_inp1_1;
  ae_int16x4 d_inp2_0, d_inp2_1;
  ae_int64 d_out64_0;
  ae_int64 d_out64_1;
  ae_int32x2 d_out32;
  ae_int32x2 d_bias;
  int i;

  /* handle cases where vec_length is multiple of 8 */
  if(vec_length == 8)
  {
    /* Assumption: 
     * p_inp1_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * p_inp2_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * */
    pt_inp1 = (const ae_int16x4 *)((WORD16 *)p_inp1_start);
    pt_inp2 = (const ae_int16x4 *)((WORD16 *)p_inp2_start);

    align_inp1 = AE_LA128_PP(pt_inp1);
    align_inp2 = AE_LA128_PP(pt_inp2);
    /* TBD: multiple vec_count processing in a single loop can be done */
    for(loopcnt = 0; loopcnt < vec_count; loopcnt++)
    {
      AE_L32_XP(d_bias, (ae_int32 *)p_bias_load, bias_address_increment);

      AE_LA16X4X2_IP(d_inp1_0, d_inp1_1, align_inp1, (ae_int16x8 *)pt_inp1);
      AE_LA16X4X2_IP(d_inp2_0, d_inp2_1, align_inp2, (ae_int16x8 *)pt_inp2);
      AE_MULZAAAA2Q16(d_out64_0, d_out64_1, d_inp1_0, d_inp1_1, d_inp2_0, d_inp2_1);

      d_out64_0 = AE_ADD64S(d_out64_0, d_out64_1);
      d_out32 = AE_SAT32X2(AE_ZERO64(), d_out64_0);
      d_out32 = AE_ADD32S(d_out32, d_bias);

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(d_out32, out_multiplier, left_shift, right_shift)
      d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
      AE_MINMAX32(d_out32, min_int8, max_int8);
      AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(d_out32), (ae_int8 *) p_out, 1);
    }
  }
  else if(vec_length == 32)
  {
    /* Assumption: 
     * p_inp1_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * p_inp2_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * */
    pt_inp1 = (const ae_int16x4 *)((WORD16 *)p_inp1_start);
    pt_inp2 = (const ae_int16x4 *)((WORD16 *)p_inp2_start);

    align_inp1 = AE_LA128_PP(pt_inp1);
    align_inp2 = AE_LA128_PP(pt_inp2);
    /* TBD: multiple vec_count processing in a single loop can be done */
    for(loopcnt = 0; loopcnt < vec_count; loopcnt++)
    {
      AE_L32_XP(d_bias, (ae_int32 *)p_bias_load, bias_address_increment);

      AE_LA16X4X2_IP(d_inp1_0, d_inp1_1, align_inp1, (ae_int16x8 *)pt_inp1);
      AE_LA16X4X2_IP(d_inp2_0, d_inp2_1, align_inp2, (ae_int16x8 *)pt_inp2);
      AE_MULZAAAA2Q16(d_out64_0, d_out64_1, d_inp1_0, d_inp1_1, d_inp2_0, d_inp2_1);

#pragma loop_count min=3
      for(i = 1; i < (vec_length >> 3); i++)
      {
        AE_LA16X4X2_IP(d_inp1_0, d_inp1_1, align_inp1, (ae_int16x8 *)pt_inp1);
        AE_LA16X4X2_IP(d_inp2_0, d_inp2_1, align_inp2, (ae_int16x8 *)pt_inp2);
        AE_MULAAAA2Q16(d_out64_0, d_out64_1, d_inp1_0, d_inp1_1, d_inp2_0, d_inp2_1);
      }
      d_out64_0 = AE_ADD64S(d_out64_0, d_out64_1);
      d_out32 = AE_SAT32X2(AE_ZERO64(), d_out64_0);
      d_out32 = AE_ADD32S(d_out32, d_bias);

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(d_out32, out_multiplier, left_shift, right_shift)
      d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
      AE_MINMAX32(d_out32, min_int8, max_int8);
      AE_S8_0_IP(AE_MOVINT8X8_FROMINT32X2(d_out32), (ae_int8 *) p_out, 1);
    }
  }
  /* handle cases where vec_length is multiple of 8 */
  else if(((vec_length & 7) == 0))
  {
    /* Assumption: 
     * p_inp1_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * p_inp2_start - memory is continuous => vec_count1 end and vect_count2 start are continuous 
     * */
    pt_inp1 = (const ae_int16x4 *)((WORD16 *)p_inp1_start);
    pt_inp2 = (const ae_int16x4 *)((WORD16 *)p_inp2_start);

    align_inp1 = AE_LA128_PP(pt_inp1);
    align_inp2 = AE_LA128_PP(pt_inp2);
    /* TBD: multiple vec_count processing in a single loop can be done */
    for(loopcnt = 0; loopcnt < vec_count; loopcnt++)
    {
      AE_L32_XP(d_bias, (ae_int32 *)p_bias_load, bias_address_increment);

      AE_LA16X4X2_IP(d_inp1_0, d_inp1_1, align_inp1, (ae_int16x8 *)pt_inp1);
      AE_LA16X4X2_IP(d_inp2_0, d_inp2_1, align_inp2, (ae_int16x8 *)pt_inp2);
      AE_MULZAAAA2Q16(d_out64_0, d_out64_1, d_inp1_0, d_inp1_1, d_inp2_0, d_inp2_1);

#pragma no_unroll
      for(i = 1; i < (vec_length >> 3); i++)
      {
        AE_LA16X4X2_IP(d_inp1_0, d_inp1_1, align_inp1, (ae_int16x8 *)pt_inp1);
        AE_LA16X4X2_IP(d_inp2_0, d_inp2_1, align_inp2, (ae_int16x8 *)pt_inp2);
        AE_MULAAAA2Q16(d_out64_0, d_out64_1, d_inp1_0, d_inp1_1, d_inp2_0, d_inp2_1);
      }
      d_out64_0 = AE_ADD64S(d_out64_0, d_out64_1);
      d_out32 = AE_SAT32X2(AE_ZERO64(), d_out64_0);
      d_out32 = AE_ADD32S(d_out32, d_bias);

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(d_out32, out_multiplier, left_shift, right_shift)
      d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
      AE_MINMAX32(d_out32, min_int8, max_int8);
      ae_int8x8 out8_0 = AE_MOVINT8X8_FROMINT32X2(d_out32);
      AE_S8_0_IP(out8_0, (ae_int8 *) p_out, 1);
    }
  }
  else
  {
    for(loopcnt = 0; loopcnt < vec_count; loopcnt++)
    {
      pt_inp1 = (const ae_int16x4 *)((WORD16 *)p_inp1_start + (loopcnt * vec_length));
      pt_inp2 = (const ae_int16x4 *)((WORD16 *)p_inp2_start + (loopcnt * vec_length));
      align_inp1 = AE_LA128_PP(pt_inp1);
      align_inp2 = AE_LA128_PP(pt_inp2);
      int i;
      d_out64_0 = ZERO64;
      d_out64_1 = ZERO64;

      AE_L32_XP(d_bias, (ae_int32 *)p_bias_load, bias_address_increment);

      for(i = 0; i < (vec_length >> 3); i++)
      {
        AE_LA16X4X2_IP(d_inp1_0, d_inp1_1, align_inp1, (ae_int16x8 *)pt_inp1);
        AE_LA16X4X2_IP(d_inp2_0, d_inp2_1, align_inp2, (ae_int16x8 *)pt_inp2);
        AE_MULAAAA2Q16(d_out64_0, d_out64_1, d_inp1_0, d_inp1_1, d_inp2_0, d_inp2_1);
      }
      for(i = 0; i < (vec_length & 7); i++)
      {
        AE_L16_IP(d_inp1_0, (ae_int16 *)pt_inp1, 2);
        AE_L16_IP(d_inp2_0, (ae_int16 *)pt_inp2, 2);
        AE_MULA16_00(d_out64_0, d_inp1_0, d_inp2_0);
      }

      d_out64_0 = AE_ADD64S(d_out64_0, d_out64_1);
      d_out32 = AE_SAT32X2(AE_ZERO64(), d_out64_0);
      d_out32 = AE_ADD32S(d_out32, d_bias);

      MULTIPLYBYQUANTIZEDMULTIPLIER_X2(d_out32, out_multiplier, left_shift, right_shift) 
      d_out32 = AE_ADD32S(d_out32 ,out_zero_bias);
      AE_MINMAX32(d_out32, min_int8, max_int8);
      ae_int8x8 out8_0 = AE_MOVINT8X8_FROMINT32X2(d_out32);
      AE_S8_0_IP(out8_0, (ae_int8 *) p_out, 1);
    }
  }
  return 0;
}
