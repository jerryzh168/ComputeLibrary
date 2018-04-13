/*
 * Copyright (c) 2017 ARM Limited.
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

layout(local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = LOCAL_SIZE_Z) in;

#include "helpers_cs.h"

#if defined(DATA_TYPE_FP16)
precision mediump float;
#endif // DATA_TYPE_FP16

/** Performs a pooling function
 *
 * @note The data type must be passed at compile time using "#define DATA_TYPE_NAME". e.g. "#define DATA_TYPE_FP32"
 * @note The pool size must be passed at compile time using "#define POOLING_LAYER_n". e.g. "#define POOLING_LAYER_2"
 *       n must be one of these: 2, 3, 7, N
 *       Pool size must be passed using POOL_SIZE if POOLING_LAYER_N is defined. e.g. POOL_SIZE=13;
 * @note In case of average pooling the following information must be passed at compile time:
 *       POOL_AVG must be provided otherwise max pooling will be performed.
 *       MAX_WIDTH and MAX_HEIGHT which are the maximum accessible indeces in x and y dimensions (width + pad)
 *       STRIDE_X and STRIDE_Y which are the steps of the window along the x and y directions
 *       PAD_X and PAD_Y which are the pooling paddings in x and y dimension
 *
 * @param[in]  src_ptr   Pointer to the source image. Supported data types: F32/F16
 * @param[in]  src_attrs The attributes of the source image
 * @param[out] dst_ptr   Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  src_attrs The attributes of the destination image
 */
SHADER_PARAMS_DECLARATION
{
    Tensor3DAttributes src_attrs;
    Tensor3DAttributes dst_attrs;
    VectorAttributes rois_attrs;
};

#ifdef DATA_TYPE_FP32
#error RoIAlign for FP32 Not Implemented
#elif defined(DATA_TYPE_FP16)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 1, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 1, writeonly);
TENSOR_DECLARATION(3, roisBuffer, uvec2, rois_ptr, rois_shift, 1, readonly);
void main(void)
{
    Tensor3DIterator src_iter   = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter   = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);
    VectorIterator   rois_iter  = CONVERT_TO_VECTOR_ITERATOR(rois_attrs, rois_shift);

    uint pw = gl_GlobalInvocationID.x;
    uint ph = gl_GlobalInvocationID.y;
    //uint c = gl_GlobalInvocationID.z;
    uint c = 0U;
    vec4 roi = LOAD_UNPACK4_HALF(rois_ptr, VECTOR_OFFSET(rois_iter, 0));
    vec4 roi_scaled = SPATIAL_SCALE * roi;
    float roi_start_w = 0.0;//roi_scaled.x;
    float roi_start_h = 0.0;//roi_scaled.y;
    float roi_end_w = 1.0;//roi_scaled.z;
    float roi_end_h = 1.0;//roi_scaled.w;

    float roi_width = max(roi_end_w - roi_start_w, 1.0);
    float roi_height = max(roi_end_h - roi_start_h, 1.0);
    float bin_size_h = roi_height / float(POOLED_H);
    float bin_size_w = roi_width / float(POOLED_W);

    int iy_upper = (SAMPLING_RATIO > 0) ? SAMPLING_RATIO : int(ceil(roi_height / float(POOLED_H)));
    int ix_upper = (SAMPLING_RATIO > 0) ? SAMPLING_RATIO : int(ceil(roi_width / float(POOLED_W)));

    float count = float(ix_upper * iy_upper);

    float res = 0.0;
    float height = float(IN_HEIGHT); // height of src image
    float width = float(IN_WIDTH); // width of src image
    for (int iy = 0; iy < iy_upper; ++iy) {
        for (int ix = 0; ix < ix_upper; ++ix) {
            float y = roi_start_h + float(ph) * bin_size_h + (float(iy) + 0.5) * bin_size_h / float(iy_upper);
            float x = roi_start_w + float(pw) * bin_size_w + (float(ix) + 0.5) * bin_size_w / float(ix_upper);
            if (y < -1.0 || y > height || x < -1.0 || x > width) {
               continue;
            }
            if (y <= 0.0) {
               y = 0.0;
            }
            if (x <= 0.0) {
               x = 0.0;
            }

            int y_low = int(y);
            int x_low = int(x);
            int y_high, x_high;
            if (y_low >= int(height) - 1) {
              y_high = y_low = int(height) - 1;
              y = float(y_low);
            } else {
              y_high = y_low + 1;
            }

            if (x_low >= int(width) - 1) {
              x_high = x_low = int(width) - 1;
              x = float(x_low);
            } else {
              x_high = x_low + 1;
            }

            float ly = y - float(y_low);
            float lx = x - float(x_low);
            float hy = 1.0 - ly, hx = 1.0 - lx;
            float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
            float data1 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x_low, y_low, c)).x;
            float data2 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x_high, y_low, c)).x;
            float data3 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x_low, y_high, c)).x;
            float data4 = LOAD_UNPACK2_HALF(src_ptr, TENSOR3D_OFFSET(src_iter, x_high, y_high, c)).x;
            // TODO: vector?
            res += data1 * w1 + data2 * w2 + data3 * w3 + data4 * w4;
        }
    }
    res /= count;

    // Store result
    STORE(dst_ptr, TENSOR3D_OFFSET(dst_iter, 0, 0, 0), uvec2(uint(res), 0));
}
#else
#error Unrecognized DataType
#endif