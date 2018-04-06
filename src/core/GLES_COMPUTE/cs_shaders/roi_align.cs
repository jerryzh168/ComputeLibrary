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
#error "RoIAlign for FP32 Not Implemented"
#if defined(DATA_TYPE_FP16)
TENSOR_DECLARATION(1, srcBuffer, uvec2, src_ptr, src_shift, 3, readonly);
TENSOR_DECLARATION(2, dstBuffer, uvec2, dst_ptr, dst_shift, 3, writeonly);
TENSOR_DECLARATION(3, roisBuffer, uvec2, rois_ptr, rois_shift, 3, readonly);
void main(void)
{
    Tensor3DIterator src_iter   = CONVERT_TO_TENSOR3D_ITERATOR(src_attrs, src_shift);
    Tensor3DIterator dst_iter   = CONVERT_TO_TENSOR3D_ITERATOR(dst_attrs, dst_shift);
    VectorIterator   rois_iter  = CONVERT_TO_VECTOR_ITERATOR(rois_attrs, rois_shift);

    int pw = int(gl_GlobalInvocationID.x);
    int ph = int(gl_GlobalInvocationID.y);
    int c = int(gl_GlobalInvocationID.z);
    vec4 roi = LOAD_UNPACK4_CURRENT_ITEM_HALF(rois_ptr, rois_iter);
    vec4 roi_scaled = roi * SPATIAL_SCALE;
    float roi_start_w = roi_scaled[0];
    float roi_start_h = roi_scaled[1];
    float roi_end_w = roi_scaled[2];
    float roi_end_h = roi_scaled[3];

    float roi_width = std::max(roi_end_w - roi_start_w, (float)1.);
    float roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    float bin_size_h = roi_height / static_cast<float>(POOLED_H);
    float bin_size_w = roi_width / static_cast<float>(POOLED_W);

    int iy_upper = (SAMPLING_RATIO > 0) ? SAMPLING_RATIO : ceil(roi_height / POOLED_H);
    int ix_upper = (SAMPLING_RATIO > 0) ? SAMPLING_RATIO : ceil(roi_width / POOLED_W);

    float count = roi_bin_grid_h * roi_bin_grid_w;

    float res = 0;
    int height = 0; // height of src image
    int widht = 0; // width of src image
    for (int iy = 0; iy < iy_upper; ++iy) {
        for (int ix = 0; ix < ix_upper; ++ix) {
            float y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / float(roi_bin_grid_h);
            float x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / float(roi_bin_grid_w);
            if (y < -1.0 || y > height || x < -1.0 || x > width) {
               continue;
            }
            if (y <= 0) {
               y = 0;
            }
            if (x <= 0) {
               x = 0;
            }

            int y_low = (int)y;
            int x_low = (int)x;
            int y_high, x_high;
            if (y_low >= height - 1) {
              y_high = y_low = height - 1;
              y = (float)y_low;
            } else {
              y_high = y_low + 1;
            }

            if (x_low >= width - 1) {
              x_high = x_low = width - 1;
              x = (float)x_low;
            } else {
              x_high = x_low + 1;
            }

            float ly = y - y_low;
            float lx = x - x_low;
            float hy = 1. - ly, hx = 1. - lx;
            float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
            float data1 = LOAD(src_ptr, TENSOR3D_OFFSET(src_iter, x_low, y_low, c));
            float data2 = LOAD(src_ptr, TENSOR3D_OFFSET(src_iter, x_high, y_low, c));
            float data3 = LOAD(src_ptr, TENSOR3D_OFFSET(src_iter, x_low, y_high, c));
            float data4 = LOAD(src_ptr, TENSOR3D_OFFSET(src_iter, x_high, y_high, c));
            // TODO: vector?
            res += data1 * w1 + data2 * w2 + data3 * w3 + data4 * w4;
        }
    }
    res /= count;

    // Store result
    STORE_CURRENT_ITEM(dst_ptr, dst_iter, res);
}
