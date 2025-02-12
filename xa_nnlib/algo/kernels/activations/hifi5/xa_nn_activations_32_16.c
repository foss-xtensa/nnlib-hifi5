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
#include "../../../ndsp/hifi5/include/NatureDSP_Signal_math.h"
/*-------------------------------------------------------------------------
  Sigmoid
  The functions compute the sigmoid of input argument. 32-bit fixed-point
  functions accept inputs in Q6.25 and form outputs in Q0.15 format.

  Precision:
  32x16  32-bit inputs, 16-bit output. Accuracy: 2 LSB.

  Input:
  x[N]   input data, Q6.25
  N      length of vectors
  Output:
  y[N]   result, Q0.15

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  return result, Q0.15
-------------------------------------------------------------------------*/
WORD32 xa_nn_vec_sigmoid_32_16(
    WORD16       * __restrict__ y,             /* result, Q0.15 */
    const WORD32 * __restrict__ x,             /* input data, Q6.25 */
    WORD32       N)                            /* length of vectors */
{
    /*
    Reference Matlab code:

    % sigmoid function x in Q6.25, y in Q16.15
    function y=sigmoid_32x16(x)
        % convert Q25 -> Q23 and scale by ln(2)
        % x=round(double(x)*(0.25/log(2)));
        x=round(pow2(double(x)*774541002,-31));
        s=x<0;
        x=abs(x);
        % compute 2^-x
        n=bitshift(x,-23);
        mantQ23=bitand(x,8388607);
        % polynomial for 2^-x, for x=0...1, coeffients in Q23
        polypow2=[57364 -446161 2008107 -5813551 8388608];
        y=polypow2(1);
        y=round(pow2(y.*mantQ23,-23))+polypow2(2);
        y=round(pow2(y.*mantQ23,-23))+polypow2(3);
        y=round(pow2(y.*mantQ23,-23))+polypow2(4);
        y=round(pow2(y.*mantQ23,-23))+polypow2(5);
        x=bitshift(y,-n);

        % iterations to compute 1./(1+x) in Q23
        y=8053064-bitshift(x,-1);  % first approximation 0.96-x/2
        d=8388608-y-round(pow2(y.*x,-23));
        y=y+round(pow2(y.*d,-23));
        d=8388608-y-round(pow2(y.*x,-23));
        y=y+round(pow2(y.*d,-23));

        % apply sign
        y(s)=8388608-y(s);
        % scale to Q15 with rounding
        y=bitshift(y+128,-8);
    */

    XA_NNLIB_ARG_CHK_PTR(y, -1);
    XA_NNLIB_ARG_CHK_PTR(x, -1);
    XA_NNLIB_ARG_CHK_ALIGN(y, sizeof(WORD16), -1);
    XA_NNLIB_ARG_CHK_ALIGN(x, sizeof(WORD32), -1);

    static const int32_t polypow2[] = { 14685184, -114217216 , 514075392, -1488269056, 2147483647, 2061584302 };// coefficients in q31 format
    int n;
    ae_int32x2 Xa, Xb, Ea, Eb, Ya, Yb, Za, Zb, Da, Db;
    ae_f32x2 ta, tb;
    ae_int16x4 Y_16;
    xtbool2 sign_a,sign_b;
    const ae_int32x2 * __restrict pX  = (const ae_int32x2 *)&x[0];
    const ae_int32x2 * __restrict pX1 = (const ae_int32x2 *)&x[0];
          ae_int16 * __restrict pY = (      ae_int16 *)&y[0];
    ae_valignx2 aX;
    ae_valign   aY;
    ae_f32x2 t1,t2;

    ae_int32 * __restrict p_polypow2 = (ae_int32 *)polypow2;
    pX1 = (ae_int32x2 *)((ae_int32 *)pX1 + (N >> 2)*4);

    NASSERT(x);
    NASSERT(y);
    if (N <= 0) return -1;

    ae_int32x2 U/*,V*/;
    U = AE_MOVDA32X2(774541002, 774541002); // ln computation redundant in the loop
//    V =AE_MOVDA32X2(0x007fffff, 0x007fffff);

    t1 = 2061584302;// 0.96 in the accumulator loaded outside from
    t2 = 2061584302;// 0.96 in the accumulator loaded outside from
                    //  0.96-x/2

    if(N >= 4)
    {
        aY = AE_ZALIGN64();
        aX = AE_LA128_PP(pX);

        AE_LA32X2X2_IP(Xa, Xb, aX, (ae_int32x4 *)pX); // try for software pipelining using preloading
        for (n = 0; n < (N >> 2); n++)
        {
            sign_a = AE_LT32(Xa, 0);
            sign_b = AE_LT32(Xb, 0);

            Za = AE_MULFP32X2RAS(Xa, U);
            Xa = AE_ABS32S(Za);
            Zb = AE_MULFP32X2RAS(Xb, U);
            Xb = AE_ABS32S(Zb);

            Ea = AE_SRAI32(Xa, 23);
            Xa = AE_MOVDEXT(Xa, 23, 8);
            Eb = AE_SRAI32(Xb, 23);
            Xb = AE_MOVDEXT(Xb, 23, 8);

            Ya = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 0);
            ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 1);
            tb = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 1);AE_MULAF2P32X4RAS(ta, tb, Xa, Xb, Ya, Ya);Ya = ta;Yb = tb;
            ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 2);
            tb = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 2);AE_MULAF2P32X4RAS(ta, tb, Xa, Xb, Ya, Yb);Ya = ta;Yb = tb;
            ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 3);
            tb = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 3);AE_MULAF2P32X4RAS(ta, tb, Xa, Xb, Ya, Yb);Ya = ta;Yb = tb;
            ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 4);
            tb = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 4);AE_MULAF2P32X4RAS(ta, tb, Xa, Xb, Ya, Yb);Ya = ta;Yb = tb;
            Xa = AE_SRAV32RS(Ya, Ea);
            Xb = AE_SRAV32RS(Yb, Eb);

            Za = AE_MULADDF32RAS(t1,Xa, -1073741824);
            ta = AE_SUB32(2147483647, Za);AE_MULSFP32X2RAS(ta, Za, Xa);Da = ta;
            AE_MULAFP32X2RAS(Za, Za, Da);
            ta = AE_SUB32(2147483647, Za);AE_MULSFP32X2RAS(ta, Za, Xa);Da = ta;
            AE_MULAFP32X2RAS(Za, Za, Da);

            Zb = AE_MULADDF32RAS(t2,Xb, -1073741824);
            tb = AE_SUB32(2147483647, Zb);AE_MULSFP32X2RAS(tb, Zb, Xb);Db = tb;
            AE_MULAFP32X2RAS(Zb, Zb, Db);
            tb = AE_SUB32(2147483647, Zb);AE_MULSFP32X2RAS(tb, Zb, Xb);Db = tb;
            AE_MULAFP32X2RAS(Zb, Zb, Db);

            Za = AE_SRAA32RS(Za, 16);
            Zb = AE_SRAA32RS(Zb, 16);

            //For negative X, sigmoid(X) = 1 - sigmoid(|X|)
            Ya = AE_SUB32(32768, Za);
            AE_MOVT32X2(Za, Ya, sign_a);

            Yb = AE_SUB32(32768, Zb);
            AE_MOVT32X2(Zb, Yb, sign_b);

            AE_LA32X2X2_IP(Xa, Xb, aX, (ae_int32x4 *)pX);

            Y_16  = AE_SAT16X4(Za, Zb);

            AE_SA16X4_IP(Y_16, aY, (ae_int16x4 *)pY);
        }
        AE_SA64POS_FP(aY, pY);
    }

    for(n=0; n < (N & 3); n++)
    {
        AE_L32_IP(Xa, (const ae_int32 *)pX1, 4);
        sign_a = AE_LT32(Xa, 0);

        Za = AE_MULFP32X2RAS(Xa, U);
        Xa = AE_ABS32S(Za);
        Ea = AE_SRAI32(Xa, 23);
        Xa = AE_MOVDEXT(Xa, 23, 8);

        Ya = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 0);
        ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 1);AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
        ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 2);AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
        ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 3);AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
        ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 4);AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
        Xa = AE_SRAV32RS(Ya, Ea);

        Za = AE_MULADDF32RAS(t1, Xa, -1073741824);
        ta = AE_SUB32(2147483647, Za);AE_MULSFP32X2RAS(ta, Za, Xa);Da = ta;
		AE_MULAFP32X2RAS(Za, Za, Da);
        ta = AE_SUB32(2147483647, Za);AE_MULSFP32X2RAS(ta, Za, Xa);Da = ta;
		AE_MULAFP32X2RAS(Za, Za, Da);

        Za = AE_SRAA32RS(Za, 16);

        //For negative X, sigmoid(X) = 1 - sigmoid(|X|)
        Ya = AE_SUB32(32768, Za);
        AE_MOVT32X2(Za, Ya, sign_a);

        Y_16 = AE_SAT16X4(Za, Za);

        *pY++ = Y_16;
    }
    return 0;
} /* xa_nn_vec_sigmoid_32_16() */

/*-------------------------------------------------------------------------
  Hyperbolic Tangent
  The functions compute the hyperbolic tangent of input argument. 32-bit
  fixed-point functions accept inputs in Q6.25 and form outputs in Q0.15
  format.

  Precision:
  32x16  32-bit inputs, 16-bit output. Accuracy: 2 LSB.

  Input:
  x[N]   input data, Q6.25
  N      length of vectors
  Output:
  y[N]   result, Q0.15

  Restriction:
  x,y should not overlap

  Scalar versions:
  ----------------
  return result, Q0.15
-------------------------------------------------------------------------*/
WORD32 xa_nn_vec_tanh_32_16(
    WORD16       * __restrict__ y,             /* result, Q0.15 */
    const WORD32 * __restrict__ x,             /* input data, Q6.25 */
    WORD32       N)                            /* length of vectors */
{
    /*
    Reference Matlab code:
        function y=tanh_32x16(x)
        % convert Q25 -> Q23 and scale by ln(2)
        x=round(pow2(double(x)*774541002*2,-31));
        s=x<0;
        x=abs(x);
        % compute 2^-x
        n=bitshift(x,-23);
        mantQ23=bitand(x,8388607);
        % polynomial for 2^-x, for x=0...1, coeffients in Q23
        polypow2=[57364 -446161 2008107 -5813551 8388608];
        y=polypow2(1);
        y=round(pow2(y.*mantQ23,-23))+polypow2(2);
        y=round(pow2(y.*mantQ23,-23))+polypow2(3);
        y=round(pow2(y.*mantQ23,-23))+polypow2(4);
        y=round(pow2(y.*mantQ23,-23))+polypow2(5);
        x=bitshift(y,-n);

        % iterations to compute 1./(1+x) in Q23
        y=8053064-bitshift(x,-1);  % first approximation 0.96-x/2
        d=8388608-y-round(pow2(y.*x,-23));
        y=y+round(pow2(y.*d,-23));
        d=8388608-y-round(pow2(y.*x,-23));
        y=y+round(pow2(y.*d,-23));
        % scale by (1-x)
        y=round(pow2(y.*(8388608-x),-23));
        % scale to Q15 with rounding
        y=bitshift(y+128,-8);

        % apply sign
        y(s)=-y(s);
    */
    static const int32_t polypow2[] = { 14685184, -114217216 , 514075392, -1488269056, 2147483647 };// coefficients in q31 format
    int n;
    ae_int32x2 Xa, Xb, Xc, Xd ,Ea, Eb, Ec, Ed, Ya, Yb, Yc, Yd, Za, Zb, Zc, Zd, Da, Db, Dc, Dd, Xa_, Xb_, Xc_, Xd_;
    ae_f32x2 ta, tb, tc, td;
    ae_int16x4 Y_16, Y1_16;
    xtbool2 sign;
    const ae_int32x2 * __restrict pX = (const ae_int32x2 *)x;
    const ae_int32x2 * __restrict pX1 = (const ae_int32x2 *)x;
          ae_int16 * __restrict pY = (      ae_int16 *)y;
    ae_valignx2 aX, aX1, aY;
    ae_int32 * __restrict p_polypow2 = (ae_int32 *)polypow2;

    NASSERT(x);
    NASSERT(y);
    if (N <= 0) return -1;

    if(N >= 8)
    {
        aY = AE_ZALIGN128();
        aX = AE_LA128_PP(pX);
        aX1 = AE_LA128_PP(pX1);

        AE_LA32X2X2_IP(Xa_, Xb_, aX, (ae_int32x4 *)pX);
        AE_LA32X2X2_IP(Xc_, Xd_, aX, (ae_int32x4 *)pX);

        for (n = 0; n < (N >> 3); n++)
        {
            Za = AE_MULFP32X2RAS(Xa_, AE_MOVDA32X2(1549082005, 1549082005));
            Xa = AE_ABS32S(Za);
            Zb = AE_MULFP32X2RAS(Xb_, AE_MOVDA32X2(1549082005, 1549082005));
            Xb = AE_ABS32S(Zb);

            Zc = AE_MULFP32X2RAS(Xc_, AE_MOVDA32X2(1549082005, 1549082005));
            Xc = AE_ABS32S(Zc);
            Zd = AE_MULFP32X2RAS(Xd_, AE_MOVDA32X2(1549082005, 1549082005));
            Xd = AE_ABS32S(Zd);

            Ea = AE_SRAI32(Xa, 23);
            Xa = AE_MOVDEXT(Xa, 23, 8);
            Eb = AE_SRAI32(Xb, 23);
            Xb = AE_MOVDEXT(Xb, 23, 8);

            Ec = AE_SRAI32(Xc, 23);
            Xc = AE_MOVDEXT(Xc, 23, 8);
            Ed = AE_SRAI32(Xd, 23);
            Xd = AE_MOVDEXT(Xd, 23, 8);

            // e^x implementation using taylor series
            Ya = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 0);
            ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 1);//AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
            tb = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 1);AE_MULAF2P32X4RAS(ta, tb, Xa, Xb, Ya, Ya);Ya = ta;Yb = tb;
            ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 2);//AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
            tb = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 2);AE_MULAF2P32X4RAS(ta, tb, Xa, Xb, Ya, Yb);Ya = ta;Yb = tb;
            ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 3);//AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
            tb = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 3);AE_MULAF2P32X4RAS(ta, tb, Xa, Xb, Ya, Yb);Ya = ta;Yb = tb;
            ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 4);//AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
            tb = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 4);AE_MULAF2P32X4RAS(ta, tb, Xa, Xb, Ya, Yb);Ya = ta;Yb = tb;
            Xa = AE_SRAV32RS(Ya, Ea);
            Xb = AE_SRAV32RS(Yb, Eb);

            Yc = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 0);
            tc = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 1);//AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
            td = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 1);AE_MULAF2P32X4RAS(tc, td, Xc, Xd, Yc, Yc);Yc = tc;Yd = td;
            tc = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 2);//AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
            td = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 2);AE_MULAF2P32X4RAS(tc, td, Xc, Xd, Yc, Yd);Yc = tc;Yd = td;
            tc = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 3);//AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
            td = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 3);AE_MULAF2P32X4RAS(tc, td, Xc, Xd, Yc, Yd);Yc = tc;Yd = td;
            tc = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 4);//AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
            td = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 4);AE_MULAF2P32X4RAS(tc, td, Xc, Xd, Yc, Yd);Yc = tc;Yd = td;
            Xc = AE_SRAV32RS(Yc, Ec);
            Xd = AE_SRAV32RS(Yd, Ed);

            // 1/(1+x) implementation part
            Za = AE_MULADDF32RAS(2061584302,Xa, -1073741824);
            ta = AE_SUB32(2147483647, Za);
            Zb = AE_MULADDF32RAS(2061584302,Xb, -1073741824);
            tb = AE_SUB32(2147483647, Zb);
            AE_MULSF2P32X4RAS(ta, tb, Za, Zb, Xa, Xb);Da = ta;Db = tb;
            AE_MULAF2P32X4RAS(Za, Zb, Za, Zb, Da, Db);

            Zc = AE_MULADDF32RAS(2061584302,Xc, -1073741824);
            tc = AE_SUB32(2147483647, Zc);
            Zd = AE_MULADDF32RAS(2061584302,Xd, -1073741824);
            td = AE_SUB32(2147483647, Zd);
            AE_MULSF2P32X4RAS(tc, td, Zc, Zd, Xc, Xd);Dc = tc;Dd = td;
            AE_MULAF2P32X4RAS(Zc, Zd, Zc, Zd, Dc, Dd);

            ta = AE_SUB32(2147483647, Za);
            tb = AE_SUB32(2147483647, Zb);
            AE_MULSF2P32X4RAS(ta, tb, Za, Zb, Xa, Xb);Da = ta;Db = tb;

            tc = AE_SUB32(2147483647, Zc);
            td = AE_SUB32(2147483647, Zd);
            AE_MULSF2P32X4RAS(tc, td, Zc, Zd, Xc, Xd);Dc = tc;Dd = td;

            AE_MULAFP32X2RAS(Za, Za, Da);
            AE_MULAFP32X2RAS(Zb, Zb, Db);
            AE_MULAFP32X2RAS(Zc, Zc, Dc);
            AE_MULAFP32X2RAS(Zd, Zd, Dd);

            Ya = AE_SUB32(2147483647, Xa);
            Za = AE_MULFP32X2RAS(Za, Ya);
            Yb = AE_SUB32(2147483647, Xb);
            Zb = AE_MULFP32X2RAS(Zb, Yb);
            Yc = AE_SUB32(2147483647, Xc);
            Zc = AE_MULFP32X2RAS(Zc, Yc);
            Yd = AE_SUB32(2147483647, Xd);
            Zd = AE_MULFP32X2RAS(Zd, Yd);

            AE_LA32X2X2_IP(Xa, Xb, aX1, (ae_int32x4 *)pX1);
            AE_LA32X2X2_IP(Xc, Xd, aX1, (ae_int32x4 *)pX1);

            Za = AE_MOVNEG32S_T(Za, Xa);
            Zb = AE_MOVNEG32S_T(Zb, Xb);
            Zc = AE_MOVNEG32S_T(Zc, Xc);
            Zd = AE_MOVNEG32S_T(Zd, Xd);

            AE_LA32X2X2_IP(Xa_, Xb_, aX, (ae_int32x4 *)pX);
            AE_LA32X2X2_IP(Xc_, Xd_, aX, (ae_int32x4 *)pX);

            Y_16  = AE_ROUND16X4F32SASYM(Za, Zb);
            Y1_16 = AE_ROUND16X4F32SASYM(Zc, Zd);

            AE_SA16X4X2_IP(Y_16, Y1_16, aY, (ae_int16x8 *)pY);
        }
        AE_SA128POS_FP(aY, pY);
    }

    for(n=0;n<(N & 7);n++)
    {
        AE_L32_IP(Xa, (const ae_int32 *)pX1, 4);
        sign = AE_LT32(Xa, 0);

        Za = AE_MULFP32X2RAS(Xa, AE_MOVDA32X2(1549082005, 1549082005));
        Xa = AE_ABS32S(Za);

        Ea = AE_SRAI32(Xa, 23);
        Xa = AE_MOVDEXT(Xa, 23, 8);

	    // e^x implementation using taylor series
        Ya = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 0);
        ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 1);AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
        ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 2);AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
        ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 3);AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
        ta = AE_L32_I((const ae_int32 *)p_polypow2, 4 * 4);AE_MULAFP32X2RAS( ta, Xa, Ya);Ya = ta;
        Xa = AE_SRAV32RS(Ya, Ea);

        // 1/(1+x) implementation part
        Za = AE_MULADDF32RAS(2061584302,Xa, -1073741824);
        ta = AE_SUB32(2147483647, Za);

        AE_MULSFP32X2RAS(ta, Za, Xa);Da = ta;
        AE_MULAFP32X2RAS(Za, Za, Da);

        ta = AE_SUB32(2147483647, Za);
        AE_MULSFP32X2RAS(ta, Za, Xa);Da = ta;

		AE_MULAFP32X2RAS(Za, Za, Da);

        Ya = AE_SUB32(2147483647, Xa);
        Za = AE_MULFP32X2RAS(Za, Ya);

        Xa = AE_NEG32S(Za);
        AE_MOVT32X2(Za, Xa, sign);

        Y_16 = AE_ROUND16X4F32SASYM(Za, Za);

        *pY++ = Y_16;
    }
    return 0;
} /* xa_nn_vec_tanh_32_16() */

