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
#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"

WORD32 xa_nn_elm_logicaland_boolxbool_bool(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                    const   WORD8 * __restrict__ p_inp2,
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

  int i;
  int rem_length = (num_elm & 15);
  ae_int8x8 m1, m2, m3, m4;

  ae_int8x16 *p_in1  = (ae_int8x16 *)p_inp1;
  ae_int8x16 *p_in2  = (ae_int8x16 *)p_inp2;
  ae_int8x16 *p_o    = (ae_int8x16 *)p_out;

  ae_valignx2 align_src_in1, align_src_in2, align_dst;
  align_src_in1 = AE_LA128_PP(p_in1);
  align_src_in2 = AE_LA128_PP(p_in2);
  align_dst     = AE_ZALIGN128();

  for(i=0; i<(num_elm >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_in1, p_in1);
    AE_LA8X8X2_IP(m3, m4, align_src_in2, p_in2);

    m1 = AE_INT8X8_AND_INT8X8(m1, m3);
	m2 = AE_INT8X8_AND_INT8X8(m2, m4);

    AE_SA8X8X2_IP(m1, m2, align_dst, p_o);
  }

  // remainder loop
  if(rem_length)
  {   
	AE_LAV8X8X2_XP(m1, m2, align_src_in1, p_in1, rem_length);
    AE_LAV8X8X2_XP(m3, m4, align_src_in2, p_in2, rem_length);

    m1 = AE_INT8X8_AND_INT8X8(m1, m3);
	
	if(rem_length > 8)
    {
      m2 = AE_INT8X8_AND_INT8X8(m2, m4);
	}
	AE_SAV8X8X2_XP(m1, m2, align_dst, p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}

WORD32 xa_nn_elm_logicalor_boolxbool_bool(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp1,
                    const   WORD8 * __restrict__ p_inp2,
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

  int i;
  int rem_length = (num_elm & 15);
  ae_int8x8 m1, m2, m3, m4;

  ae_int8x16 *p_in1  = (ae_int8x16 *)p_inp1;
  ae_int8x16 *p_in2  = (ae_int8x16 *)p_inp2;
  ae_int8x16 *p_o    = (ae_int8x16 *)p_out;

  ae_valignx2 align_src_in1, align_src_in2, align_dst;
  align_src_in1 = AE_LA128_PP(p_in1);
  align_src_in2 = AE_LA128_PP(p_in2);
  align_dst     = AE_ZALIGN128();

  for(i=0; i<(num_elm >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_in1, p_in1);
    AE_LA8X8X2_IP(m3, m4, align_src_in2, p_in2);

    m1 = AE_INT8X8_OR_INT8X8(m1, m3);
	m2 = AE_INT8X8_OR_INT8X8(m2, m4);

    AE_SA8X8X2_IP(m1, m2, align_dst, p_o);
  }

  // remainder loop
  if(rem_length)
  {   
	AE_LAV8X8X2_XP(m1, m2, align_src_in1, p_in1, rem_length);
    AE_LAV8X8X2_XP(m3, m4, align_src_in2, p_in2, rem_length);

    m1 = AE_INT8X8_OR_INT8X8(m1, m3);
	
	if(rem_length > 8)
    {
      m2 = AE_INT8X8_OR_INT8X8(m2, m4);
	}
	AE_SAV8X8X2_XP(m1, m2, align_dst, p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}

WORD32 xa_nn_elm_logicalnot_bool_bool(WORD8 * __restrict__ p_out,
                    const   WORD8 * __restrict__ p_inp,
                            WORD32  num_elm)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

  int i;
  int rem_length = (num_elm & 15);
  ae_int8x8 m1, m2;

  ae_int8x16 *p_in  = (ae_int8x16 *)p_inp;
  ae_int8x16 *p_o   = (ae_int8x16 *)p_out;

  ae_valignx2 align_src_in, align_dst;
  align_src_in = AE_LA128_PP(p_in);
  align_dst    = AE_ZALIGN128();
  
  unsigned int two = 2;
  ae_int8x8 two_8x8 = AE_MOVDA8(two);

  for(i=0; i<(num_elm >> 4); i++)
  {
    AE_LA8X8X2_IP(m1, m2, align_src_in, p_in);

    m1 = AE_INT8X8_BNOT(m1);
	m1 = AE_INT8X8_ADD_INT8X8(m1, two_8x8);
	
	m2 = AE_INT8X8_BNOT(m2);
	m2 = AE_INT8X8_ADD_INT8X8(m2, two_8x8);

    AE_SA8X8X2_IP(m1, m2, align_dst, p_o);
  }

  // remainder loop
  if(rem_length)
  {   
	AE_LAV8X8X2_XP(m1, m2, align_src_in, p_in, rem_length);

    m1 = AE_INT8X8_BNOT(m1);
	m1 = AE_INT8X8_ADD_INT8X8(m1, two_8x8);
	
	if(rem_length > 8)
    {
	  m2 = AE_INT8X8_BNOT(m2);
	  m2 = AE_INT8X8_ADD_INT8X8(m2, two_8x8);
	}
	AE_SAV8X8X2_XP(m1, m2, align_dst, p_o, rem_length);
  }
  AE_SA128POS_FP(align_dst, p_o);

  return 0;
}
