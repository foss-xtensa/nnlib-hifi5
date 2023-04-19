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
#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"

WORD32 xa_nn_memmove_8_8( void *pdst,
    const void *psrc,
    WORD32 n)
{

#if 1
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(pdst, -1);
  XA_NNLIB_ARG_CHK_PTR(psrc, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(pdst, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(psrc, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((n <= 0), -1);

  const WORD8 *x = (const WORD8*)psrc;
  WORD8 *y = (WORD8*)pdst;
  int i;
  ae_int8x8 d0, d1;
  ae_int8x16 *pOut;
  const ae_int8x16 *pInp;
  if(y == x) //no copy needed
    return 0;

  if (y < x)
  {
      pInp = (const  ae_int8x16 *)&x[0];
      pOut = (ae_int8x16 *)&y[0];
    ///check for aligned part
    if( ( (((unsigned)pInp)&15)==0  ) &&  ( (((unsigned)pOut)&15)==0  )   )
    {
        for(i=0;i<n>>4;i++)
        {
            AE_L8X8X2_IP(d0, d1, pInp, 16*sizeof(WORD8));
            AE_S8X8X2_IP(d0, d1, pOut, 16*sizeof(WORD8));
        }
    }
    else
    {
        ae_valignx2 alignIn, alignOut;
        alignIn = AE_LA128_PP(pInp);
        alignOut = AE_ZALIGN128();

        for(i=0;i<n>>4;i++)
        {
            AE_LA8X8X2_IP(d0, d1, alignIn, pInp);
            AE_SA8X8X2_IP(d0, d1, alignOut, pOut);
        }
        AE_SA128POS_FP(alignOut, pOut);
    }

    // i<<=4;//Reminder Loop
    for(i = 0 ;i< (n&15);i++)
    {
        AE_L8_IP(d0, (ae_int8 *)pInp, sizeof(WORD8));
        AE_S8_0_IP(d0, (ae_int8 *)pOut, sizeof(WORD8));
    }
  }
  else
  {
      pInp = (const  ae_int8x16 *)&x[n-16];
      pOut = (ae_int8x16 *)&y[n-16];

        ///check for aligned part
        if( ( (((unsigned)pInp)&15)==0  ) &&  ( (((unsigned)pOut)&15)==0  )   )
        {

            for(i=0;i<(n>>4);i++)
            {
                AE_L8X8X2_IP(d0, d1, pInp, -16*sizeof(WORD8));
                AE_S8X8X2_IP(d0, d1, pOut, -16*sizeof(WORD8));
            }
            // i<<=4;//Reminder Loop
            pInp = (ae_int8x16*)((WORD8*)pInp + 15);
            pOut = (ae_int8x16*)((WORD8*)pOut + 15);
            for(i = 0 ;i<(n&15);i++)
            {
                *(WORD8*)pOut = *(WORD8*)pInp;
                pInp = (ae_int8x16*)((WORD8*)pInp - 1);
                pOut = (ae_int8x16*)((WORD8*)pOut - 1);
            }           

        }
        else
        {
            pInp = (const  ae_int8x16 *)&x[n-1];
            pOut = (ae_int8x16 *)&y[n-1];
            ae_valign alignIn, alignOut;
            alignIn = AE_LA64_PP(pInp);
            alignOut = AE_ZALIGN64();
            for(i=0;i<n>>3;i++)
            {
                AE_LA8X8_RIP(d0, alignIn, (const ae_int8x8 *)pInp);
                AE_SA8X8_RIP(d0, alignOut, (ae_int8x8 *)pOut);
            }
            AE_SA64NEG_FP(alignOut, (void*)pOut);

            // i<<=3;//Reminder Loop
            for(i = 0 ;i<(n&7);i++)
            {
                *(WORD8*)pOut = *(WORD8*)pInp;
                pInp = (ae_int8x16*)((WORD8*)pInp - 1);
                pOut = (ae_int8x16*)((WORD8*)pOut - 1);
            }           
            /*for(;i<n;i++)
            {
                *(WORD8*)pOut = *(WORD8*)pInp;
                pInp = (WORD8*)pInp - 1;
                pOut = (WORD8*)pOut - 1;
            }*/

        }

  }



  return 0;
#endif
}

void *xa_nn_memcpy(void * dest1,const void *src1, size_t n1)
{
  int itr;
  ae_int8x8 di0, di1; \
  ae_int8x16 *__restrict__ pae_i;
  ae_int8x16 *__restrict__ pae_o;
  ae_valignx2 i_a, o_a;
  pae_i = (ae_int8x16 *)(src1);
  pae_o = (ae_int8x16 *)(dest1);
  i_a = AE_LA128_PP(pae_i);
  o_a = AE_ZALIGN128();
  for(itr = 0; itr < (((int)n1)>>4); itr++)
  {
    AE_LA8X8X2_IP(di0, di1, i_a, pae_i);
    AE_SA8X8X2_IP(di0, di1, o_a, pae_o);
  }
  AE_LAV8X8X2_XP(di0, di1, i_a, pae_i, ((n1)&15));
  AE_SAV8X8X2_XP(di0, di1, o_a, pae_o, ((n1)&15));
  AE_SA128POS_FP(o_a, pae_o);
  return (void *)pae_o;
} /* xa_nn_memcpy */


