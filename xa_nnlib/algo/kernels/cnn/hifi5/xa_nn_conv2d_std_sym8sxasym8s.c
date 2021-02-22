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
#include "xa_nn_conv2d_std_state.h"

#define MULTIPLYBYQUANTIZEDMULTIPLIER_X2(inp, multiplier, left_shift, right_shift) \
    inp = AE_SLAA32(inp, left_shift); \
    inp = AE_MULFP32X2RAS(inp, AE_MOVDA32(multiplier)); \
    inp = AE_SRAA32SYMS(inp, right_shift);

static WORD32 conv_x_left_pad(
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const WORD32* __restrict__ p_bias,
    WORD8 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 out_width_over_x_pad = (x_padding - kernel_width)/x_stride + 1;
  WORD32 left_shift, right_shift;
  out_width_over_x_pad = out_width_over_x_pad > out_width ? out_width : out_width_over_x_pad;

  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);

  /* When kernel convolves over x-left pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = 0; j < out_width_over_x_pad; j++)
    {
      for(k = 0; k < out_channels; k++)
      {
        left_shift  = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
        ae_int32x2 acc = AE_MOVDA32(p_bias[k]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
        AE_MINMAX32(acc, min_int8, max_int8);
        p_out[i * out_height_offset + j * out_width_offset + k * out_channels_offset] = (UWORD8)AE_MOVAD32_L(acc);
      }
    }
  }
  return out_width_over_x_pad;
}

static WORD32 conv_x_right_pad(
    WORD32 x_padding,
    WORD32 input_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const WORD32* __restrict__ p_bias,
    WORD8 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 idx_out_width_over_x_r_pad = (x_padding + input_width + x_stride - 1)/x_stride;
  WORD32 left_shift, right_shift;
  WORD32 out_width_over_x_r_pad = out_width - idx_out_width_over_x_r_pad;

  ae_int32x2 max_int8 = AE_MOVDA32(127);
  ae_int32x2 min_int8 = AE_MOVDA32(-128);

  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = idx_out_width_over_x_r_pad; j < out_width; j++)
    {
      for(k = 0; k < out_channels; k++)
      {
        left_shift  = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
        ae_int32x2 acc = AE_MOVDA32(p_bias[k]);
        MULTIPLYBYQUANTIZEDMULTIPLIER_X2(acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
        AE_MINMAX32(acc, min_int8, max_int8);
        p_out[i * out_height_offset + j * out_width_offset + k * out_channels_offset] = (UWORD8)AE_MOVAD32_L(acc);
      }
    }
  }
  return out_width_over_x_r_pad;
}
//#define polyphase_debug
#ifdef polyphase_debug
void manipulateinput(void* p_inp, WORD32 input_height, WORD32 input_width, WORD32 input_channels, void* p_ker, WORD32 kernel_height, WORD32 kernel_width, WORD32 output_channels, void* p_bias, WORD32* p_out_multiplier, WORD32* p_out_shift, WORD32* out_zero_bias, WORD32* input_zero_bias)
{
	WORD8* p_inp_debug;
	WORD8* p_ker_debug;
	WORD32* p_bias_debug;

	p_inp_debug  = (WORD8*)p_inp;
	p_ker_debug  = (WORD8*)p_ker;
	p_bias_debug = (WORD32*)p_bias;

	WORD32 iter = 0, i, k, j1, j2;
	for(k=0;k<input_height;k++)
		for(i=0;i<input_width;i++)
		{
			for(j1=0;j1<input_channels;j1++)
			{
				*p_inp_debug = iter;//14*k + 2*i;//iter;
				p_inp_debug++;
			}
			iter++;
		}

	for(j2=0;j2<output_channels;j2++)
		for(k=0;k<kernel_height;k++)
			for(i=0;i<kernel_width;i++)
			{
				for(j1=0;j1<input_channels;j1++)
				{

					{
						*p_ker_debug = 1;
						//if( (k==0) && (i==0) && (j2==1))
							//*p_ker_debug = 1;
						p_ker_debug++;
					}
				}
			}

	for(k=0;k<output_channels;k++)
	{
		p_bias_debug[k] = 0;
		p_out_multiplier[k] = 1073741823;//1073741823;///2147483647;
		p_out_shift[k] = -2;
	}

	*out_zero_bias = 0;
	*input_zero_bias = 0;

}
#endif


WORD32 xa_nn_dilated_conv2d_std_per_chan_sym8sxasym8s(
    WORD8* __restrict__ p_out,
    const WORD8* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD32* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch,
    WORD32 dilation_height,
    WORD32 dilation_width)
{

	WORD8* __restrict__ p_out_base;
	p_out_base = p_out;
	//WORD32 dilation_height = 2, dilation_width = 3;//dilation
	WORD32 circMatrixHeight = 0;

#ifdef polyphase_debug
	/* Filling debug input data*/
	WORD32 base = 0;
	base = base * 5;
	VOID* p_inp_deb = (void*) p_inp;
	VOID* p_kernel_deb = (void*) p_kernel;
	VOID* p_bias_deb = (void*) p_bias;
	WORD8* p_buff_circ_deb;
  	manipulateinput((void*) p_inp_deb, input_height, input_width, input_channels, p_kernel_deb, kernel_height, kernel_width, out_channels, (void*) p_bias_deb, p_out_multiplier, p_out_shift, &out_zero_bias, &input_zero_bias);
#endif

	if(kernel_height==1)
  		dilation_height = 1;
  	if(kernel_width==1)
  		dilation_width = 1;

	WORD32 kernel_height_dilation = kernel_height + ( (dilation_height-1) * (kernel_height-1) );//dilation
	WORD32 kernel_width_dilation = kernel_width + ( (dilation_width-1) * (kernel_width-1) );//dilation
   /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  //XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(UWORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_height <= 0 || dilation_width <= 0), -1);//dilation
  //XA_NNLIB_ARG_CHK_COND((kernel_height_dilation > input_height), -1);//dilation
  //XA_NNLIB_ARG_CHK_COND((kernel_width_dilation > input_width), -1);//dilation
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride != 1 || x_stride != 1), -1);//dilation
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);

  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  WORD32 input_bytewidth = 1;
  VOID *pp_inp = (VOID *)p_inp;
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  WORD32 out_channels_offset = out_data_format ? out_height * out_width : 1;
  WORD32 out_height_offset = out_data_format ? out_width : out_width * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;
  WORD32 x_padding_var = x_padding;
  WORD32 input_channels_pad = input_channels;
  WORD32 dilation_w_offset, dilation_h_offset;
  WORD32 out_iteraions;


  // Initialize start of the circular buffer
  xa_nn_conv2d_dilation_init_state((void*)p_state,(void*)p_kernel, (void*)pp_inp);

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
    if(x_padding_var >= kernel_width_dilation)//dilation
  {
    //out_width_over_x_pad = conv_x_left_pad(x_padding, kernel_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);//dilation
    out_width_over_x_pad = conv_x_left_pad(x_padding, kernel_width_dilation, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
    x_padding_var -= out_width_over_x_pad * x_stride;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = kernel_width_dilation + (out_width - 1) * x_stride - (x_padding + input_width);//dilation
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= kernel_width_dilation)//dilation
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_padding, input_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
  }

  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height_dilation + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  XA_NNLIB_ARG_CHK_COND((kernel_height_dilation > ( y_padding + input_height + y_b_pad)), -1);//dilation
  XA_NNLIB_ARG_CHK_COND((kernel_width_dilation  > ( x_padding + input_width  + x_r_pad)), -1);//dilation

  WORD32 out_width_part_of_convolution = out_width-out_width_over_x_pad-out_width_over_x_r_pad;
  WORD32 out_height_part_of_convolution = out_height;

  for(dilation_w_offset =0; dilation_w_offset<dilation_width; dilation_w_offset++ )
  {
	  /// Calculate number of left padding zeros for this particular width offset
	  WORD32 x_padding_full_dilation = (x_padding/dilation_width) + (WORD32) ( ((x_padding%dilation_width)-1) >= dilation_w_offset);//This is the contribution of zero padding(in total) towards this width offset
	  WORD32 x_padding_dilation_initial_pad = ((x_padding-x_padding_var)/dilation_width) + (WORD32) ( (((x_padding-x_padding_var)%dilation_width)-1) >= dilation_w_offset); /// This offset's contribution which has been absorbed in initial analysis of zero padding
	  WORD32 x_padding_dilation = x_padding_full_dilation - x_padding_dilation_initial_pad;//This is the num of zeros contribution from left padding for this dilation offset

	  /// Calculate number of input data for this particular width offset
	  WORD32 x_padding_plus_input_dilation = ( (x_padding+input_width)/dilation_width) + (WORD32) ( (((x_padding+input_width)%dilation_width)-1) >= dilation_w_offset);//This is the num elements to be convolved for this offset in total(zeropad+input)
	  WORD32 x_input_dilation = x_padding_plus_input_dilation - x_padding_full_dilation;// This is the number of elements from input that can potentially be populated

	  /// Calculate number of right padding zeros for this particular width offset
	  WORD32 x_padding_plus_input_plus_rpadding_dilation = ( (x_padding+input_width+x_r_pad)/dilation_width) + (WORD32) ( (((x_padding+input_width+x_r_pad)%dilation_width)-1) >= dilation_w_offset);//This is the total num of elements to be convolved for this offset in total(zeropad+input+zeroRpad)
	  WORD32 x_r_padding_dilation = x_padding_plus_input_plus_rpadding_dilation - x_padding_plus_input_dilation;

	  WORD32 out_points_for_this_xyoffset = ((x_padding_dilation + x_input_dilation + x_r_padding_dilation) - kernel_width)/x_stride + 1;/// This represents total num of times the conv needs to be called


	  for(dilation_h_offset =0; dilation_h_offset<dilation_height; dilation_h_offset++ )
	  {
		  if( ( dilation_w_offset <= (out_width_part_of_convolution-1)) &&  ( dilation_h_offset <= (out_height_part_of_convolution-1)) )
		  {

			  WORD32 input_padding_consumed =0;
			  WORD32 input_width_consumed = 0;

			  /// Initialize circular buffer end/height/size based on the dilation offset
			  xa_nn_dilated_conv2d_std_init_circ_buf((void*)p_state,(void*)p_kernel,input_height,input_channels,kernel_height_dilation,kernel_width,x_stride,y_stride,y_padding,out_height,-4, dilation_height, dilation_h_offset);//dilation

#ifdef polyphase_debug
			  p_buff_circ_deb = p_state->cir_buf.p_curr;
#endif
			  xa_nn_dilated_conv2d_std_load_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, p_state, -input_zero_bias, dilation_height, dilation_h_offset, dilation_width, dilation_w_offset, x_padding, &input_padding_consumed, &input_width_consumed, (kernel_width - x_stride),1,&circMatrixHeight);

			  ///output index addition corresponding to left padding
			  WORD32 left_pad_offset;
			  for(left_pad_offset=out_width_over_x_pad;left_pad_offset<out_width_over_x_pad+dilation_width;left_pad_offset++)
				  if(((left_pad_offset)%dilation_width) == dilation_w_offset)
					  break;

			  p_out = p_out_base + (dilation_h_offset * out_height_offset) + (left_pad_offset*out_width_offset);//(dilation_w_offset * out_width_offset) + ( (left_pad_offset+out_width_over_x_pad) * out_width_offset);

#ifdef polyphase_debug
			  p_buff_circ_deb = p_state->cir_buf.p_curr;
#endif

			  for(out_iteraions = 0;out_iteraions<out_points_for_this_xyoffset;out_iteraions++)
			  {
				  xa_nn_dilated_conv2d_std_load_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, p_state, -input_zero_bias, dilation_height, dilation_h_offset, dilation_width, dilation_w_offset, x_padding, &input_padding_consumed, &input_width_consumed, x_stride,0,&circMatrixHeight);

#ifdef polyphase_debug
			  p_buff_circ_deb = p_state->cir_buf.p_curr;
#endif

				    // Convolution using matXvec with matrix as circular buffer
				    xa_nn_matXvec_sym8sxasym8s_asym8s_circ
				      (p_out /* output */
				       ,p_state->cir_buf.p_curr/* matrix: rows x cols */
				       ,p_kernel /* vec: cols */
				       ,p_bias /* bias */
				       ,circMatrixHeight-kernel_height+1//out_height /* rows */
				       ,input_channels_pad * kernel_width * kernel_height /* cols */
				       ,input_channels_pad * kernel_width * y_stride/* row_offset */
				       ,out_channels /* vec_count */
				       ,input_channels_pad * kernel_width * kernel_height /* vec_stride */
				       ,out_channels_offset /* out_col_offset */
				       ,out_height_offset * dilation_height /* out_row_offset *//// mul by dilation_height
				       ,input_zero_bias
				       ,p_out_multiplier
				       ,p_out_shift
				       ,out_zero_bias
				      );
			  	  //conv2d_dilation_ptr_reset((void*)p_state, (VOID**)&pp_inp);
				    p_out += out_width_offset*dilation_width;//Mul by dilation width
			  }

		  }
	  }
  }

  return 0;
}


WORD32 xa_nn_conv2d_std_per_chan_sym8sxasym8s(
    WORD8* __restrict__ p_out,
    const WORD8* __restrict__ p_inp,
    const WORD8* __restrict__ p_kernel,
    const WORD32* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch)
{
   /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_bias, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  /* Pointer alignment checks */
  //XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(UWORD8), -1);
  //XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_scratch, ALIGNMENT, -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height > input_height), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_width > input_width), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -127 || input_zero_bias > 128), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < -128 || out_zero_bias > 127), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);

  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }


  WORD32 j;
  WORD32 input_bytewidth = 1;
  VOID *pp_inp = (VOID *)p_inp;

  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  xa_nn_conv2d_std_init_state((void*)p_state,(void*)p_kernel,input_height,input_channels,kernel_height,kernel_width,x_stride,y_stride,y_padding,out_height,-4);

  WORD32 out_channels_offset = out_data_format ? out_height * out_width : 1;
  WORD32 out_height_offset = out_data_format ? out_width : out_width * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;

  WORD32 x_padding_var = x_padding;
  WORD32 input_channels_pad = input_channels;

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
  if(x_padding_var >= kernel_width)
  {
    out_width_over_x_pad = conv_x_left_pad(x_padding, kernel_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
    x_padding_var -= out_width_over_x_pad * x_stride;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = kernel_width + (out_width - 1) * x_stride - (x_padding + input_width);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= kernel_width)
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_padding, input_width, x_stride, out_width, out_height, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
  }

  /* When kernel convolves over input region */
  p_out += out_width_over_x_pad * out_width_offset;
  // Initialize circular buffer
  // Determine y-bottom padding
  WORD32 y_b_pad = kernel_height + (out_height - 1) * y_stride - (y_padding + input_height);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;

  conv2d_std_init_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, p_state, -input_zero_bias);

  // Index to padded input width
  WORD32 idx_beg_inp_width_pad = kernel_width - x_stride;
  idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;


  // Process Loop to compute one output plane [out_height x out_channels] per iteration
  for(j=0;j<out_width-out_width_over_x_pad-out_width_over_x_r_pad;j++)
  {
    // Add x_stride x (input_height x input_channels) new planes to circular buffer
    conv2d_std_update_cir_buf_asym8(input_channels, input_channels_pad, input_bytewidth, input_width, input_height, y_padding, y_b_pad, x_padding_var, kernel_width, x_stride, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state, -input_zero_bias);

    // Update index to input width padded
    idx_beg_inp_width_pad += x_stride;

    // Convolution using matXvec with matrix as circular buffer
    xa_nn_matXvec_sym8sxasym8s_asym8s_circ
      (p_out /* output */
       ,p_state->cir_buf.p_curr/* matrix: rows x cols */
       ,p_kernel /* vec: cols */
       ,p_bias /* bias */
       ,out_height /* rows */
       ,input_channels_pad * kernel_width * kernel_height /* cols */
       ,input_channels_pad * kernel_width * y_stride/* row_offset */
       ,out_channels /* vec_count */
       ,input_channels_pad * kernel_width * kernel_height /* vec_stride */
       ,out_channels_offset /* out_col_offset */
       ,out_height_offset /* out_row_offset */
       ,input_zero_bias
       ,p_out_multiplier
       ,p_out_shift
       ,out_zero_bias
      );

    p_out += out_width_offset;
  }

  return 0;
}

