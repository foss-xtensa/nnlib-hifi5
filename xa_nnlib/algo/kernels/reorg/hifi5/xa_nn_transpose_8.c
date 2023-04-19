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

/*
 * Currently only supports upto 5D input tensors.
 * 1/2/3/4 D input tensors will be scaled up to 5D.
 * For example, 2x3 -> 1x1x1x2x3.
 */

WORD32 xa_nn_transpose_8_8(WORD8 * __restrict__ p_out
                    ,const WORD32 *const p_out_shape
                    ,const WORD8 * __restrict__ p_inp
                    ,const WORD32 *const p_inp_shape
                    ,const WORD32 * __restrict__ p_permute_vec
                    ,WORD32 num_out_dims
                    ,WORD32 num_inp_dims)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_permute_vec, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);

  /* Invalid input checks */
  XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > 5)), -1);
  XA_NNLIB_ARG_CHK_COND((num_out_dims != num_inp_dims), -1);

  int itr = 0;
  for(itr=0; itr < num_inp_dims; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_inp_shape[itr] <= 0), -1);
  }
  for(itr=0; itr < num_out_dims; itr++)
  {
    XA_NNLIB_ARG_CHK_COND((p_out_shape[itr] <= 0), -1);
  }

  /* Output shape provided must be correct based on input
   * shape and permute values */
  for(itr=0; itr < num_out_dims; itr++)
  {
    int output_dim = p_out_shape[itr];
    int expected_dim = p_inp_shape[p_permute_vec[itr]];
    XA_NNLIB_ARG_CHK_COND((output_dim != expected_dim), -1);
  }

  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(WORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_permute_vec, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);

  /* Promoting lesser dim tensors to 5D tensors. 
   * Also updating the permute_vec and shapes as needed for optimization */
  int p_5D_inp_shape[5] = {1, 1, 1, 1, 1};
  int p_5D_out_shape[5] = {1, 1, 1, 1, 1};
  int p_5D_permute_vec[5] = {0, 1, 2, 3, 4};
  
  /* Check if any inner inp dimension is same in the output */
  int last_dim_same = 1, last_n_same_dim = 0;
  itr = num_inp_dims - 1;
  while(itr >= 0)
  {
    last_n_same_dim = (last_dim_same && (p_permute_vec[itr] == itr)) ? (last_n_same_dim + 1) : last_n_same_dim;
    last_dim_same = (p_permute_vec[itr] == itr) ? last_dim_same & 1 : last_dim_same & 0;
    itr--;
  }
  
  int dims_added = 5 - num_inp_dims;
  itr = num_inp_dims - 1;
  int same_count = last_n_same_dim;
  int count = 4;
  while(itr >= 0)
  {
    p_5D_inp_shape[count] = (same_count > 0) ? p_5D_inp_shape[count]*p_inp_shape[itr] : p_inp_shape[itr];
    p_5D_out_shape[count] = (same_count > 0) ? p_5D_out_shape[count]*p_out_shape[itr] : p_out_shape[itr];
    same_count--;
    itr--;
    count = (same_count > 0) ? count : count - 1;
  }
  
  itr = num_inp_dims - 1;
  same_count = (last_n_same_dim) ? num_inp_dims - (last_n_same_dim - 1) : 0;
  count = 4;
  while(itr >= 0)
  {
    p_5D_permute_vec[count] = (same_count > 0) ? p_permute_vec[itr-(last_n_same_dim - 1)] + dims_added + last_n_same_dim - 1 : p_permute_vec[itr] + dims_added;
    same_count--;
    itr--;
    count--;
  }
  
  int out_dim0, out_dim1, out_dim2, out_dim3, out_dim4;
  int inp_dim1, inp_dim2, inp_dim3, inp_dim4;
  int inp_stride[5];

  out_dim0 = p_5D_out_shape[0]; 
  out_dim1 = p_5D_out_shape[1]; 
  out_dim2 = p_5D_out_shape[2]; 
  out_dim3 = p_5D_out_shape[3];
  out_dim4 = p_5D_out_shape[4];

  inp_dim1 = p_5D_inp_shape[1]; 
  inp_dim2 = p_5D_inp_shape[2]; 
  inp_dim3 = p_5D_inp_shape[3];
  inp_dim4 = p_5D_inp_shape[4];

  inp_stride[0] = inp_dim1*inp_dim2*inp_dim3*inp_dim4;
  inp_stride[1] = inp_dim2*inp_dim3*inp_dim4;
  inp_stride[2] = inp_dim3*inp_dim4;
  inp_stride[3] = inp_dim4;
  inp_stride[4] = 1;

  if(last_n_same_dim)
  {
    int itr0, itr1, itr2, itr3, itr4;
    WORD8 *p_inp0 = (WORD8*)p_inp;
    for(itr0 = 0; itr0 < out_dim0; itr0++)
    {
      WORD8 *p_inp1 = p_inp0+(itr0*inp_stride[p_5D_permute_vec[0]]);
#pragma loop_count min=1
      for(itr1 = 0; itr1 < out_dim1; itr1++)
      {
        WORD8 *p_inp2 = p_inp1+(itr1*inp_stride[p_5D_permute_vec[1]]);
#pragma loop_count min=1
        for(itr2 = 0; itr2 < out_dim2; itr2++)
        {
          WORD8 *p_inp3 = p_inp2+(itr2*inp_stride[p_5D_permute_vec[2]]);
#pragma loop_count min=1
          for(itr3 = 0; itr3 < out_dim3; itr3++, p_out+=out_dim4)
          {
            WORD8 *p_inp4 = p_inp3+(itr3*inp_stride[p_5D_permute_vec[3]]);
            ae_int8x16 *__restrict__ pae_i = (ae_int8x16 *)(p_inp4);
            ae_int8x16 *__restrict__ pae_o = (ae_int8x16 *)(p_out);
            ae_valignx2 a_inp = AE_LA128_PP(pae_i);
            ae_valignx2 a_out = AE_ZALIGN128();
            ae_int8x8 d0,d1;
            for(itr4 = 0; itr4 < (out_dim4 >> 4); itr4++)
            {
              AE_LA8X8X2_IP(d0, d1, a_inp, (ae_int8x16*)pae_i);
              AE_SA8X8X2_IP(d0, d1, a_out, (ae_int8x16*)pae_o);
            }
            AE_LAV8X8X2_XP(d0, d1, a_inp, (ae_int8x16*)pae_i, (out_dim4 & 15));
            AE_SAV8X8X2_XP(d0, d1, a_out, (ae_int8x16*)pae_o, (out_dim4 & 15));
            AE_SA128POS_FP(a_out, pae_o);
          }
        }
      }
    }
  }
  else
  {
    int itr0, itr1, itr2, itr3, itr4;
    WORD8 *p_inp0 = (WORD8*)p_inp;
    for(itr0 = 0; itr0 < out_dim0; itr0++)
    {
      WORD8 *p_inp1 = p_inp0+(itr0*inp_stride[p_5D_permute_vec[0]]);
      for(itr1 = 0; itr1 < out_dim1; itr1++)
      {
        WORD8 *p_inp2 = p_inp1+(itr1*inp_stride[p_5D_permute_vec[1]]);
        for(itr2 = 0; itr2 < out_dim2; itr2++)
        {
          WORD8 *p_inp3 = p_inp2+(itr2*inp_stride[p_5D_permute_vec[2]]);
          for(itr3 = 0; itr3 < out_dim3; itr3++)
          {
            WORD8 *p_inp4 = p_inp3+(itr3*inp_stride[p_5D_permute_vec[3]]);

            ae_valign a_out = AE_ZALIGN64();
            for(itr4 = 0; itr4 < (out_dim4 >> 3); itr4++)
            {
              ae_int8x8 d0, d1, d2, d3, d4, d5, d6, d7;
              ae_int8x8 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
              
              d1 = AE_L8_X ((ae_int8*)p_inp4, inp_stride[p_5D_permute_vec[4]]);
              d2 = AE_L8_X ((ae_int8*)p_inp4, 2*inp_stride[p_5D_permute_vec[4]]);
              d3 = AE_L8_X ((ae_int8*)p_inp4, 3*inp_stride[p_5D_permute_vec[4]]);
              d4 = AE_L8_X ((ae_int8*)p_inp4, 4*inp_stride[p_5D_permute_vec[4]]);
              d5 = AE_L8_X ((ae_int8*)p_inp4, 5*inp_stride[p_5D_permute_vec[4]]);
              d6 = AE_L8_X ((ae_int8*)p_inp4, 6*inp_stride[p_5D_permute_vec[4]]);
              d7 = AE_L8_X ((ae_int8*)p_inp4, 7*inp_stride[p_5D_permute_vec[4]]);
              AE_L8_XP(d0, (ae_int8*)p_inp4, 8*inp_stride[p_5D_permute_vec[4]]);
              
              tmp0 = AE_SEL8X8I(d0, d1, 20);
              tmp1 = AE_SEL8X8I(d2, d3, 20);
              tmp2 = AE_SEL8X8I(d4, d5, 20);
              tmp3 = AE_SEL8X8I(d6, d7, 20);
              tmp4 = AE_SEL8X8I(tmp0, tmp1, 24);
              tmp5 = AE_SEL8X8I(tmp2, tmp3, 24);
              tmp6 = AE_SEL8X8I(tmp4, tmp5, 1);
              
              AE_SA8X8_IP(tmp6, a_out, (ae_int8x8 *)p_out);
            }
            AE_SA64POS_FP(a_out, p_out);
#pragma loop_count max=7
            for(itr4 = 0; itr4 < (out_dim4 & 7); itr4++)
            {
              ae_int8x8 d0;
              AE_L8_XP(d0, (ae_int8*)p_inp4, inp_stride[p_5D_permute_vec[4]]);
              AE_S8_0_IP(d0, (ae_int8 *)p_out, 1);
            }
          }
        }
      }
    }
  }

  return 0;
}


