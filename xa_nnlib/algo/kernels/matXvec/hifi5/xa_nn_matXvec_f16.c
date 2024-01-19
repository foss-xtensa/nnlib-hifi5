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
/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros_hifi5.h"

#if !HAVE_HP_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(WORD32,xa_nn_matXvec_f16xf16_f16,(
			WORD16  * __restrict__ p_out,          
			const WORD16  * __restrict__ p_mat1,    
			const WORD16  * __restrict__ p_mat2,    
			const WORD16  * __restrict__ p_vec1, 
			const WORD16  * __restrict__ p_vec2, 
			const WORD16  * __restrict__ p_bias,
			WORD32 rows,                    
			WORD32 cols1,                              
			WORD32 cols2,                              
			WORD32 row_stride1,                  
			WORD32 row_stride2                           
			))
#else

#define TRANSPOSEHX4_INPLACE(i1, i2, i3, i4){\
  ae_int16x4 in1 = AE_MOVF16X4_FROMHALFX4(i1);\
  ae_int16x4 in2 = AE_MOVF16X4_FROMHALFX4(i2);\
  ae_int16x4 in3 = AE_MOVF16X4_FROMHALFX4(i3);\
  ae_int16x4 in4 = AE_MOVF16X4_FROMHALFX4(i4);\
  ae_int16x4 out0, out1, out2, out3;\
  ae_int16x4 dsel = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07060302, 0x05040100));\
  AE_DSEL16X4(out0, out1, in1, in2, dsel);\
  AE_DSEL16X4(out2, out3, in3, in4, dsel);\
  dsel = AE_MOVINT16X4_FROMINT32X2(AE_MOVDA32X2(0x07050604, 0x03010200));\
  AE_DSEL16X4(in1, in3, out0, out2, dsel);\
  AE_DSEL16X4(in2, in4, out1, out3, dsel);\
  i1 = AE_MOVHALFX4_FROMF16X4(in1);\
  i2 = AE_MOVHALFX4_FROMF16X4(in2);\
  i3 = AE_MOVHALFX4_FROMF16X4(in3);\
  i4 = AE_MOVHALFX4_FROMF16X4(in4);\
}

/* Internal Function: mtx_vecmpyf_bias_add_aligned()
 * Restrictions:
 *   pmat       : 16-byte aligned
 *   row_stride : Multiple of 8
 *   cols       : Multiple of 8
 *   rows       : Multiple of 4
 */

static void mtx_vecmpyf_bias_add_aligned( WORD16 * pout,
     const WORD16 * pmat,  const WORD16 * pvec, const WORD16 * pbias, 
     int rows, int cols, int row_stride)
{
  WORD32 i, j;

  const xthalfx8 *pb = (const xthalfx8 *)(pbias);
  xthalfx8 *pout0 = (xthalfx8 *)(pout);

  ae_valignx2 alignout0, alignbias;
  alignout0 = AE_ZALIGN128();
  alignbias = AE_LA128_PP(pb);

  /* Eight rows at a time */
  for(i = 0; i < rows>>3; i++)
  {
    const xthalfx8 *pmat0 = (const xthalfx8 *)(pmat+(8*i*row_stride));
    const xthalfx8 *pmat1 = (const xthalfx8 *)((xthalf *)pmat0+row_stride);
    const xthalfx8 *pmat2 = (const xthalfx8 *)((xthalf *)pmat1+row_stride);
    const xthalfx8 *pmat3 = (const xthalfx8 *)((xthalf *)pmat2+row_stride);
    const xthalfx8 *pmat4 = (const xthalfx8 *)((xthalf *)pmat3+row_stride);
    const xthalfx8 *pmat5 = (const xthalfx8 *)((xthalf *)pmat4+row_stride);
    const xthalfx8 *pmat6 = (const xthalfx8 *)((xthalf *)pmat5+row_stride);
    const xthalfx8 *pmat7 = (const xthalfx8 *)((xthalf *)pmat6+row_stride);
    const xthalfx8 *pvec0 = (const xthalfx8 *)(pvec);
    ae_valignx2 alignvec0 = AE_LA128_PP(pvec0);
    /* matrix values */
    xthalfx4 m0_0, m1_0, m2_0, m3_0, m4_0, m5_0, m6_0, m7_0;
    xthalfx4 m0_1, m1_1, m2_1, m3_1, m4_1, m5_1, m6_1, m7_1;
    xthalfx4 m0_2, m1_2, m2_2, m3_2, m4_2, m5_2, m6_2, m7_2;
    xthalfx4 m0_3, m1_3, m2_3, m3_3, m4_3, m5_3, m6_3, m7_3;
    /* vector values */
    xthalfx4 v0_0, v0_1;
    xthalfx4 v0_2, v0_3;
    /* accumulators */
    xthalfx4 acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31,
             acc40, acc41, acc50, acc51, acc60, acc61, acc70, acc71 ;

    acc00 = acc01 = acc10 = acc11 = acc20 = acc21 = acc30 = acc31 = 
    acc40 = acc41 = acc50 = acc51 = acc60 = acc61 = acc70 = acc71 = ZERO_HX4();

    /* Compute for 16 colums per row, i.e. 8x16 * 16x1 */ 
    #pragma no_unroll
    for (j = 0; j < (cols>>4); j++)
    {
      AE_LAHX4X2_IP(v0_0, v0_1, alignvec0, pvec0);
      AE_LAHX4X2_IP(v0_2, v0_3, alignvec0, pvec0);

      AE_LHX4X2_I(m0_2, m0_3, pmat0, sizeof(xthalfx8));
      AE_LHX4X2_IP(m0_0, m0_1, pmat0, 2*sizeof(xthalfx8));

      AE_LHX4X2_I(m1_2, m1_3, pmat1, sizeof(xthalfx8));
      AE_LHX4X2_IP(m1_0, m1_1, pmat1, 2*sizeof(xthalfx8));

      AE_LHX4X2_I(m2_2, m2_3, pmat2, sizeof(xthalfx8));
      AE_LHX4X2_IP(m2_0, m2_1, pmat2, 2*sizeof(xthalfx8));

      AE_LHX4X2_I(m3_2, m3_3, pmat3, sizeof(xthalfx8));
      AE_LHX4X2_IP(m3_0, m3_1, pmat3, 2*sizeof(xthalfx8));

      AE_LHX4X2_I(m4_2, m4_3, pmat4, sizeof(xthalfx8));
      AE_LHX4X2_IP(m4_0, m4_1, pmat4, 2*sizeof(xthalfx8));

      AE_LHX4X2_I(m5_2, m5_3, pmat5, sizeof(xthalfx8));
      AE_LHX4X2_IP(m5_0, m5_1, pmat5, 2*sizeof(xthalfx8));

      AE_LHX4X2_I(m6_2, m6_3, pmat6, sizeof(xthalfx8));
      AE_LHX4X2_IP(m6_0, m6_1, pmat6, 2*sizeof(xthalfx8));

      AE_LHX4X2_I(m7_2, m7_3, pmat7, sizeof(xthalfx8));
      AE_LHX4X2_IP(m7_0, m7_1, pmat7, 2*sizeof(xthalfx8));

      MADD_HX4X2(acc00, acc01, m0_0, m0_1, v0_0, v0_1);
      MADD_HX4X2(acc10, acc11, m1_0, m1_1, v0_0, v0_1);
      MADD_HX4X2(acc20, acc21, m2_0, m2_1, v0_0, v0_1);
      MADD_HX4X2(acc30, acc31, m3_0, m3_1, v0_0, v0_1);
      MADD_HX4X2(acc40, acc41, m4_0, m4_1, v0_0, v0_1);
      MADD_HX4X2(acc50, acc51, m5_0, m5_1, v0_0, v0_1);
      MADD_HX4X2(acc60, acc61, m6_0, m6_1, v0_0, v0_1);
      MADD_HX4X2(acc70, acc71, m7_0, m7_1, v0_0, v0_1);

      MADD_HX4X2(acc00, acc01, m0_2, m0_3, v0_2, v0_3);
      MADD_HX4X2(acc10, acc11, m1_2, m1_3, v0_2, v0_3);
      MADD_HX4X2(acc20, acc21, m2_2, m2_3, v0_2, v0_3);
      MADD_HX4X2(acc30, acc31, m3_2, m3_3, v0_2, v0_3);
      MADD_HX4X2(acc40, acc41, m4_2, m4_3, v0_2, v0_3);
      MADD_HX4X2(acc50, acc51, m5_2, m5_3, v0_2, v0_3);
      MADD_HX4X2(acc60, acc61, m6_2, m6_3, v0_2, v0_3);
      MADD_HX4X2(acc70, acc71, m7_2, m7_3, v0_2, v0_3);
    }

    if(cols&15)
    {
      AE_LHX4X2_IP(m0_0, m0_1, pmat0, sizeof(xthalfx8));
      AE_LHX4X2_IP(m1_0, m1_1, pmat1, sizeof(xthalfx8));
      AE_LHX4X2_IP(m2_0, m2_1, pmat2, sizeof(xthalfx8));
      AE_LHX4X2_IP(m3_0, m3_1, pmat3, sizeof(xthalfx8));
      AE_LHX4X2_IP(m4_0, m4_1, pmat4, sizeof(xthalfx8));
      AE_LHX4X2_IP(m5_0, m5_1, pmat5, sizeof(xthalfx8));
      AE_LHX4X2_IP(m6_0, m6_1, pmat6, sizeof(xthalfx8));
      AE_LHX4X2_IP(m7_0, m7_1, pmat7, sizeof(xthalfx8));
      AE_LAHX4X2_IP(v0_0, v0_1, alignvec0, pvec0);

      MADD_HX4X2(acc00, acc01, m0_0, m0_1, v0_0, v0_1);
      MADD_HX4X2(acc10, acc11, m1_0, m1_1, v0_0, v0_1);
      MADD_HX4X2(acc20, acc21, m2_0, m2_1, v0_0, v0_1);
      MADD_HX4X2(acc30, acc31, m3_0, m3_1, v0_0, v0_1);
      MADD_HX4X2(acc40, acc41, m4_0, m4_1, v0_0, v0_1);
      MADD_HX4X2(acc50, acc51, m5_0, m5_1, v0_0, v0_1);
      MADD_HX4X2(acc60, acc61, m6_0, m6_1, v0_0, v0_1);
      MADD_HX4X2(acc70, acc71, m7_0, m7_1, v0_0, v0_1);
    }
    
    ADD_HX4X2(acc00, acc10, acc00, acc10, acc01, acc11);
    ADD_HX4X2(acc20, acc30, acc20, acc30, acc21, acc31);
    ADD_HX4X2(acc40, acc50, acc40, acc50, acc41, acc51);
    ADD_HX4X2(acc60, acc70, acc60, acc70, acc61, acc71);
    
    TRANSPOSEHX4_INPLACE(acc00, acc10, acc20, acc30);
    TRANSPOSEHX4_INPLACE(acc40, acc50, acc60, acc70);
    ADD_HX4X2(acc00, acc40, acc00, acc40, acc10, acc50);
    ADD_HX4X2(acc00, acc40, acc00, acc40, acc20, acc60);
    ADD_HX4X2(acc00, acc40, acc00, acc40, acc30, acc70);

    xthalfx4 b0 = ZERO_HX4();
    xthalfx4 b1 = ZERO_HX4();
    if(pbias != NULL)
    {
      AE_LAHX4X2_IP(b0, b1, alignbias, pb);
    }
    ADD_HX4X2(acc00, acc40, acc00, acc40, b0, b1);

    AE_SAHX4X2_IP(acc00, acc40, alignout0, pout0);
  }
  AE_SA128POS_FP(alignout0, pout0);

  xthalfx4 *pout0_rem = (xthalfx4 *)pout0;
  xthalfx4 *pb_rem = (xthalfx4 *)pb;
  ae_valign alignout_rem, alignbias_rem;
  alignbias_rem = AE_LA64_PP(pb_rem);
  alignout_rem = AE_ZALIGN64();

  /* Remaining rows, which has to be 4 or 0*/
  if(rows%8)
  {
  /* Process 4 rows only */
    const xthalfx8 *pmat0 = (const xthalfx8 *)(pmat+((rows&~0x7)*row_stride));
    const xthalfx8 *pmat1 = (const xthalfx8 *)((xthalf *)pmat0+row_stride);
    const xthalfx8 *pmat2 = (const xthalfx8 *)((xthalf *)pmat1+row_stride);
    const xthalfx8 *pmat3 = (const xthalfx8 *)((xthalf *)pmat2+row_stride);
    const xthalfx8 *pvec0 = (const xthalfx8 *)(pvec);
    ae_valignx2 alignvec0 = AE_LA128_PP(pvec0);
    /* matrix values */
    xthalfx4 m0_0, m0_1;
    xthalfx4 m1_0, m1_1;
    xthalfx4 m2_0, m2_1;
    xthalfx4 m3_0, m3_1;
    /* vector values */
    xthalfx4 v0_0, v0_1;
   
    xthalfx4 acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31;
    acc00 = acc01 = acc10 = acc11 = acc20 = acc21 = acc30 = acc31 = ZERO_HX4();

    /* Compute for 8 colums per row, i.e. 4x8 * 8x1 */ 
    for (j = 0; j < (cols>>3); j++)
    {
      AE_LHX4X2_IP(m0_0, m0_1, pmat0, sizeof(xthalfx8));
      AE_LHX4X2_IP(m1_0, m1_1, pmat1, sizeof(xthalfx8));
      AE_LHX4X2_IP(m2_0, m2_1, pmat2, sizeof(xthalfx8));
      AE_LHX4X2_IP(m3_0, m3_1, pmat3, sizeof(xthalfx8));
      AE_LAHX4X2_IP(v0_0, v0_1, alignvec0, pvec0);

      MADD_HX4X2(acc00, acc01, m0_0, m0_1, v0_0, v0_1);
      MADD_HX4X2(acc10, acc11, m1_0, m1_1, v0_0, v0_1);
      MADD_HX4X2(acc20, acc21, m2_0, m2_1, v0_0, v0_1);
      MADD_HX4X2(acc30, acc31, m3_0, m3_1, v0_0, v0_1);
    }
    
    acc00 = ADD_HX4(acc00, acc01);
    acc10 = ADD_HX4(acc10, acc11);
    acc20 = ADD_HX4(acc20, acc21);
    acc30 = ADD_HX4(acc30, acc31);
    
    TRANSPOSEHX4_INPLACE(acc00, acc10, acc20, acc30);
    acc00 = ADD_HX4(acc00, acc10);
    acc00 = ADD_HX4(acc00, acc20);
    acc00 = ADD_HX4(acc00, acc30);

    xthalfx4 b0 = ZERO_HX4();
    if(pbias != NULL)
    {
      AE_LAHX4IP(b0, alignbias_rem, pb_rem);
    }
    acc00 = ADD_HX4(acc00, b0);

    AE_SAHX4IP(acc00, alignout_rem, pout0_rem);
  }
  AE_SA64POS_FP(alignout_rem, pout0_rem);
}

static void mtx_vecmpyf_bias_add_generic( WORD16 * pout,
     const WORD16 * pmat,  const WORD16 * pvec, const WORD16 * pbias, 
     int rows, int cols, int row_stride)
{
  WORD32 i, j;

  const xthalfx4 *pb = (const xthalfx4 *)(pbias);
  xthalfx4 *pout0 = (xthalfx4 *)(pout);
  ae_valign alignb = AE_LA64_PP(pb);
  ae_valign alignout = AE_ZALIGN64();

  WORD32 rem_rows=0;

  if((unsigned)pvec%16 == 0)
  {
    /* Since vector is assumed aligned, we can process four rows at a time. */
    for(i = 0; i < rows>>2; i++)
    {
      const xthalfx8 *pmat0 = (const xthalfx8 *)(pmat+(4*i*row_stride));
      const xthalfx8 *pmat1 = (const xthalfx8 *)((xthalf *)pmat0+row_stride);
      const xthalfx8 *pmat2 = (const xthalfx8 *)((xthalf *)pmat1+row_stride);
      const xthalfx8 *pmat3 = (const xthalfx8 *)((xthalf *)pmat2+row_stride);
      const xthalfx8 *pvec0 = (const xthalfx8 *)(pvec);
      /* matrix values */
      xthalfx4 m0_0, m0_1, m0_2, m0_3;
      xthalfx4 m1_0, m1_1, m1_2, m1_3;
      xthalfx4 m2_0, m2_1, m2_2, m2_3;
      xthalfx4 m3_0, m3_1, m3_2, m3_3;
      /* vector values */
      xthalfx4 v0_0, v0_1, v0_2, v0_3;
     
      xthalfx4 acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31;
      acc00 = acc01 = acc10 = acc11 = acc20 = acc21 = acc30 = acc31 = ZERO_HX4();
      xthalfx4 acc02, acc03, acc12, acc13, acc22, acc23, acc32, acc33;
      acc02 = acc03 = acc12 = acc13 = acc22 = acc23 = acc32 = acc33 = ZERO_HX4();
  
      ae_valignx2 alignm0, alignm1, alignm2, alignm3;
      alignm0 = AE_LA128_PP(pmat0);
      alignm1 = AE_LA128_PP(pmat1);
      alignm2 = AE_LA128_PP(pmat2);
      alignm3 = AE_LA128_PP(pmat3);
  
      /* Compute for 16 colums per row, i.e. 4x16 * 16x1 */ 
      for (j = 0; j < (cols>>4); j++)
      {
        AE_LAHX4X2_IP(m0_0, m0_1, alignm0, pmat0);
        AE_LAHX4X2_IP(m1_0, m1_1, alignm1, pmat1);
        AE_LAHX4X2_IP(m2_0, m2_1, alignm2, pmat2);
        AE_LAHX4X2_IP(m3_0, m3_1, alignm3, pmat3);
        AE_LHX4X2_IP(v0_0, v0_1, pvec0, sizeof(xthalfx8));
        AE_LAHX4X2_IP(m0_2, m0_3, alignm0, pmat0);
        AE_LAHX4X2_IP(m1_2, m1_3, alignm1, pmat1);
        AE_LAHX4X2_IP(m2_2, m2_3, alignm2, pmat2);
        AE_LAHX4X2_IP(m3_2, m3_3, alignm3, pmat3);
        AE_LHX4X2_IP(v0_2, v0_3, pvec0, sizeof(xthalfx8));
  
        MADD_HX4X2(acc00, acc01, m0_0, m0_1, v0_0, v0_1);
        MADD_HX4X2(acc10, acc11, m1_0, m1_1, v0_0, v0_1);
        MADD_HX4X2(acc20, acc21, m2_0, m2_1, v0_0, v0_1);
        MADD_HX4X2(acc30, acc31, m3_0, m3_1, v0_0, v0_1);
        MADD_HX4X2(acc02, acc03, m0_2, m0_3, v0_2, v0_3);
        MADD_HX4X2(acc12, acc13, m1_2, m1_3, v0_2, v0_3);
        MADD_HX4X2(acc22, acc23, m2_2, m2_3, v0_2, v0_3);
        MADD_HX4X2(acc32, acc33, m3_2, m3_3, v0_2, v0_3);
      }

      ADD_HX4X2(acc00, acc01, acc00, acc01, acc02, acc03);
      ADD_HX4X2(acc10, acc11, acc10, acc11, acc12, acc13);
      ADD_HX4X2(acc20, acc21, acc20, acc21, acc22, acc23);
      ADD_HX4X2(acc30, acc31, acc30, acc31, acc32, acc33);
  
      if(cols%16 >= 8)
      {
        ae_valignx2 alignv0 = AE_LA128_PP(pvec0);

        AE_LAHX4X2_IP(m0_0, m0_1, alignm0, pmat0);
        AE_LAHX4X2_IP(m1_0, m1_1, alignm1, pmat1);
        AE_LAHX4X2_IP(m2_0, m2_1, alignm2, pmat2);
        AE_LAHX4X2_IP(m3_0, m3_1, alignm3, pmat3);
        AE_LAHX4X2_IP(v0_0, v0_1, alignv0, pvec0);
  
        MADD_HX4X2(acc00, acc01, m0_0, m0_1, v0_0, v0_1);
        MADD_HX4X2(acc10, acc11, m1_0, m1_1, v0_0, v0_1);
        MADD_HX4X2(acc20, acc21, m2_0, m2_1, v0_0, v0_1);
        MADD_HX4X2(acc30, acc31, m3_0, m3_1, v0_0, v0_1);
      }

      if(cols%8)
      {
        int rem_el = (cols%8)*sizeof(xthalf);
        ae_valignx2 alignv0 = AE_LA128_PP(pvec0);

        AE_LAVHX4X2_XP(m0_0, m0_1, alignm0, pmat0, rem_el);
        AE_LAVHX4X2_XP(m1_0, m1_1, alignm1, pmat1, rem_el);
        AE_LAVHX4X2_XP(m2_0, m2_1, alignm2, pmat2, rem_el);
        AE_LAVHX4X2_XP(m3_0, m3_1, alignm3, pmat3, rem_el);
        AE_LAVHX4X2_XP(v0_0, v0_1, alignv0, pvec0, rem_el);
  
        MADD_HX4X2(acc00, acc01, m0_0, m0_1, v0_0, v0_1);
        MADD_HX4X2(acc10, acc11, m1_0, m1_1, v0_0, v0_1);
        MADD_HX4X2(acc20, acc21, m2_0, m2_1, v0_0, v0_1);
        MADD_HX4X2(acc30, acc31, m3_0, m3_1, v0_0, v0_1);
      }
      
      acc00 = ADD_HX4(acc00, acc01);
      acc10 = ADD_HX4(acc10, acc11);
      acc20 = ADD_HX4(acc20, acc21);
      acc30 = ADD_HX4(acc30, acc31);
      
      TRANSPOSEHX4_INPLACE(acc00, acc10, acc20, acc30);
      acc00 = ADD_HX4(acc00, acc10);
      acc00 = ADD_HX4(acc00, acc20);
      acc00 = ADD_HX4(acc00, acc30);
  
      xthalfx4 b0 = ZERO_HX4();
      if(pbias != NULL)
      {
        AE_LAHX4IP(b0, alignb, pb);
      }
      acc00 = ADD_HX4(acc00, b0);
  
      AE_SAHX4IP(acc00, alignout, pout0);
    }
    AE_SA64POS_FP(alignout, pout0);
    rem_rows = rows%4;
  } else {
    /* All pointers are assumed unaligned. Process two rows at a time*/
    for(i = 0; i < rows>>1; i++)
    {
      const xthalfx8 *pmat0 = (const xthalfx8 *)(pmat+(2*i*row_stride));
      const xthalfx8 *pmat1 = (const xthalfx8 *)((xthalf *)pmat0+row_stride);
      const xthalfx8 *pvec0 = (const xthalfx8 *)(pvec);
      /* matrix values */
      xthalfx4 m0_0, m0_1, m0_2, m0_3;
      xthalfx4 m1_0, m1_1, m1_2, m1_3;
      /* vector values */
      xthalfx4 v0_0, v0_1, v0_2, v0_3;
     
      xthalfx4 acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13;
      acc00 = acc01 = acc02 = acc03 = ZERO_HX4();
      acc10 = acc11 = acc12 = acc13 = ZERO_HX4();
  
      ae_valignx2 alignm0, alignm1, alignv0;
      alignm0 = AE_LA128_PP(pmat0);
      alignm1 = AE_LA128_PP(pmat1);
      alignv0 = AE_LA128_PP(pvec0);
  
      /* Compute for 16 colums per row, i.e. 2x16 * 16x1 */ 
      for (j = 0; j < (cols>>4); j++)
      {
        AE_LAHX4X2_IP(m0_0, m0_1, alignm0, pmat0);
        AE_LAHX4X2_IP(m1_0, m1_1, alignm1, pmat1);
        AE_LAHX4X2_IP(v0_0, v0_1, alignv0, pvec0);

        AE_LAHX4X2_IP(m0_2, m0_3, alignm0, pmat0);
        AE_LAHX4X2_IP(m1_2, m1_3, alignm1, pmat1);
        AE_LAHX4X2_IP(v0_2, v0_3, alignv0, pvec0);
  
        MADD_HX4X2(acc00, acc01, m0_0, m0_1, v0_0, v0_1);
        MADD_HX4X2(acc10, acc11, m1_0, m1_1, v0_0, v0_1);
        MADD_HX4X2(acc02, acc03, m0_2, m0_3, v0_2, v0_3);
        MADD_HX4X2(acc12, acc13, m1_2, m1_3, v0_2, v0_3);
      }

      ADD_HX4X2(acc00, acc01, acc00, acc01, acc02, acc03);
      ADD_HX4X2(acc10, acc11, acc10, acc11, acc12, acc13);
  
      if((cols%16) >= 8)
      {
        AE_LAHX4X2_IP(m0_0, m0_1, alignm0, pmat0);
        AE_LAHX4X2_IP(m1_0, m1_1, alignm1, pmat1);
        AE_LAHX4X2_IP(v0_0, v0_1, alignv0, pvec0);
  
        MADD_HX4X2(acc00, acc01, m0_0, m0_1, v0_0, v0_1);
        MADD_HX4X2(acc10, acc11, m1_0, m1_1, v0_0, v0_1);
      }

      if(cols%8)
      {
        int rem_el = (cols%8)*sizeof(xthalf);

        AE_LAVHX4X2_XP(m0_0, m0_1, alignm0, pmat0, rem_el);
        AE_LAVHX4X2_XP(m1_0, m1_1, alignm1, pmat1, rem_el);
        AE_LAVHX4X2_XP(v0_0, v0_1, alignv0, pvec0, rem_el);
  
        MADD_HX4X2(acc00, acc01, m0_0, m0_1, v0_0, v0_1);
        MADD_HX4X2(acc10, acc11, m1_0, m1_1, v0_0, v0_1);
      }
      
      acc00 = ADD_HX4(acc00, acc01);
      acc10 = ADD_HX4(acc10, acc11);
      
      xthalfx4 out_temp= acc00;
      out_temp = ADD_HX4(out_temp, AE_SELH_4321(acc00, acc00));
      out_temp = ADD_HX4(out_temp, AE_SELH_5432(acc00, acc00));
      out_temp = ADD_HX4(out_temp, AE_SELH_6543(acc00, acc00));
      acc00 = out_temp;

      out_temp= acc10;
      out_temp = ADD_HX4(out_temp, AE_SELH_4321(acc10, acc10));
      out_temp = ADD_HX4(out_temp, AE_SELH_5432(acc10, acc10));
      out_temp = ADD_HX4(out_temp, AE_SELH_6543(acc10, acc10));
      acc10 = out_temp;

      xthalfx4 b0 = ZERO_HX4();
      xthalfx4 b1 = ZERO_HX4();
      if(pbias != NULL)
      {
        ae_int16x4 b0_tmp;
        AE_L16_IP(b0_tmp, (ae_int16 *)pb, sizeof(xthalf));
        b0 = AE_MOVHALFX4_FROMF16X4(b0_tmp);
        AE_L16_IP(b0_tmp, (ae_int16 *)pb, sizeof(xthalf));
        b1 = AE_MOVHALFX4_FROMF16X4(b0_tmp);
      }
      acc00 = ADD_HX4(acc00, b0);
      acc10 = ADD_HX4(acc10, b1);
  
      AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(acc00), (ae_int16 *)pout0, sizeof(xthalf));
      AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(acc10), (ae_int16 *)pout0, sizeof(xthalf));
    }

    rem_rows = rows%2;
  }

  WORD32 processed_rows = rows - rem_rows; /* Number of mat-rows already processed by prior loops.*/
  for(i = 0; i < rem_rows; i++)
  {
    const xthalfx8 *pmat0 = (const xthalfx8 *)(pmat+((i+processed_rows)*row_stride));
    const xthalfx8 *pvec0 = (const xthalfx8 *)(pvec);
    /* matrix values */
    xthalfx4 m0_0, m0_1;
    /* vector values */
    xthalfx4 v0_0, v0_1;
   
    xthalfx4 acc00, acc01;
    acc00 = acc01 = ZERO_HX4();

    ae_valignx2 alignm0, alignv0;
    alignm0 = AE_LA128_PP(pmat0);
    alignv0 = AE_LA128_PP(pvec0);

    /* Compute for 8 colums per row, i.e. 4x8 * 8x1 */ 
    for (j = 0; j < (cols>>3); j++)
    {
      AE_LAHX4X2_IP(m0_0, m0_1, alignm0, pmat0);
      AE_LAHX4X2_IP(v0_0, v0_1, alignv0, pvec0);
      MADD_HX4X2(acc00, acc01, m0_0, m0_1, v0_0, v0_1);
    }

    if(cols%8)
    {
      int rem_el = (cols%8)*sizeof(xthalf);
      AE_LAVHX4X2_XP(m0_0, m0_1, alignm0, pmat0, rem_el);
      AE_LAVHX4X2_XP(v0_0, v0_1, alignv0, pvec0, rem_el);

      MADD_HX4X2(acc00, acc01, m0_0, m0_1, v0_0, v0_1);
    }

    acc00 = ADD_HX4(acc00, acc01);
    xthalfx4 out_temp= acc00;

    out_temp = ADD_HX4(out_temp, AE_SELH_4321(acc00, acc00));
    out_temp = ADD_HX4(out_temp, AE_SELH_5432(acc00, acc00));
    out_temp = ADD_HX4(out_temp, AE_SELH_6543(acc00, acc00));
    
    xthalfx4 b0 = ZERO_HX4();
    if(pbias != NULL)
    {
      ae_int16x4 b0_tmp;
      AE_L16_IP(b0_tmp, (ae_int16 *)pb, sizeof(xthalf));
      b0 = AE_MOVHALFX4_FROMF16X4(b0_tmp);
    }
    out_temp = ADD_HX4(out_temp, b0);
    
    AE_S16_0_IP(AE_MOVF16X4_FROMHALFX4(out_temp), (ae_int16 *)pout0, sizeof(xthalf));
  }
}

WORD32 xa_nn_matXvec_f16xf16_f16(
	WORD16  * __restrict__ p_out,                /*!< [out] f32b result: rows x 1 */
	const WORD16  * __restrict__ p_mat1,         /*!< [in] f32b mat1: rows x cols1 */
	const WORD16  * __restrict__ p_mat2,         /*!< [in] f32b mat2: rows x cols2 */
	const WORD16  * __restrict__ p_vec1,         /*!< [in] f32b vec1: cols1 x 1 */
	const WORD16  * __restrict__ p_vec2,         /*!< [in] f32b vec2: cols2 x 1 */
	const WORD16  * __restrict__ p_bias,         /*!< [in] f32b bias: rows x 1 */
	WORD32 rows,                                  /*!< [in] number of rows */
	WORD32 cols1,                                 /*!< [in] number of columns of mat1 */
	WORD32 cols2,                                 /*!< [in] number of columns of mat2 */
	WORD32 row_stride1,                           /*!< [in] row stride for mat1 */
	WORD32 row_stride2                            /*!< [in] row stride for mat2 */
	)
{

  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_mat1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_vec1, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_mat1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_vec1, sizeof(WORD16), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD16), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((rows <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((cols1 <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((row_stride1 < cols1), -1);

  if(p_mat2 != NULL)
  {
    XA_NNLIB_ARG_CHK_PTR(p_vec2, -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_mat2, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_vec2, sizeof(WORD16), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((cols2 <= 0), -1);
    XA_NNLIB_ARG_CHK_COND((row_stride2 < cols2), -1);
  }

  if(p_mat2 != NULL){
    return -1; /* dual matXvec functionality not implemented. */
  }

  /* If matrix rows are aligned, process (rows&~0x03) rows */
  if(((cols1&7) == 0) && ((row_stride1&7) == 0) && ((((unsigned)p_mat1)&15) == 0) )
  {
    WORD32 rows_mul4 = rows&~0x03;
    mtx_vecmpyf_bias_add_aligned(p_out, p_mat1, p_vec1, p_bias, rows_mul4, cols1, row_stride1);
    rows = (rows%4);
    p_out += rows_mul4;
    p_bias += rows_mul4;
    p_mat1 += (rows_mul4*row_stride1);
  }

  /* Generic case. Used also for remaining rows in case of aligned matrix-rows */
  if(rows != 0)
  {
    mtx_vecmpyf_bias_add_generic(p_out, p_mat1, p_vec1, p_bias, rows, cols1, row_stride1);
  }

  return 0;

}
#endif /* !HAVE_HP_VFPU */
