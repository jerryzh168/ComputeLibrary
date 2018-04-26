/*
* Copyright (c) 2016, 2017 ARM Limited.
*
* SPDX-License-Identifier: MIT
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to
* deal in the Software without restriction, including without limitation the
* rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
* sell copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "helpers.h"

/** Perfoms channel shuffle see https://arxiv.org/pdf/1707.01083.pdf for details. Data Type Supported: F16
*
* @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types: QS8/QS16/F16/F32
* @param[in]  input_stride_x                       Stride of the first source tensor in X dimension (in bytes)
* @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
* @param[in]  input_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
* @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
* @param[in]  input_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
* @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
* @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
* @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
* @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
* @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
* @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
* @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
* @param[in]  output_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
* @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
* @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
*/
__kernel void channel_shuffle(
                              TENSOR3D_DECLARATION(in),
                              TENSOR3D_DECLARATION(out))
{
  // transpose channel C = G x K to K x G
  Tensor3D in  = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(in);
  Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);
  const int current_c = get_global_id(2); // channel id of output
  const int g = current_c / GROUP; // group id
  const int k = current_c % GROUP; // channel id
  // Transpose: (g, k) -> (k, g)
  const int channel_in = k * K + g;
  const float8 coord = get_current_coords();
  vstore4(read_texels4(&in, convert_int8(coord)), channel_in, (__global DATA_TYPE *)out.ptr);
}
