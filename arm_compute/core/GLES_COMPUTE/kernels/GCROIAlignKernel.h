/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_GCROIALIGNKERNEL_H__
#define __ARM_COMPUTE_GCROIALIGNKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the RoIAlign kernel.
 */
class GCROIAlignKernel : public IGCKernel
{
public:
    /** Constructor */
    GCROIAlignKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCROIAlignKernel(const GCROIAlignKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCROIAlignKernel &operator=(const GCROIAlignKernel &) = delete;
    /** Default Move Constructor. */
    GCROIAlignKernel(GCROIAlignKernel &&) = default;
    /** Default move assignment operator. */
    GCROIAlignKernel &operator=(GCROIAlignKernel &&) = default;
    /** Default destructor */
    ~GCROIAlignKernel() = default;

    // [TODO] Update Docs
    /** Set the input and output tensors.
     *
     * @param[in]  input    Source tensor. 3 lower dimensions represent a single input with dimensions [width, height, FM]. batch size should be 1
     *                      The rest are optional and used for representing batches. Data types supported: F16.
     * @param[out] output   Destination tensor. 4D output of shape [pooled_h, pooled_w, FM, R]. The r-th batch element "
     *  "is a pooled feature map cooresponding to the r-th RoI. Data type supported: same as @p input
     * @param[in]  rois     ROIs tensor. 2D input of shape (R, 4 or 5) specifying R RoIs "
     *   "representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
     *   "coordinates are in the coordinate system of the input image. For "
     *   "inputs corresponding to a single image, batch index can be excluded "
     *  "to have just 4 columns. Data types supported: Same as @p input
     * @param[in]  spatial_scale  (float) default 1.0; Spatial scale of the input feature map X "
     *  "relative to the input image. E.g., 0.0625 if X has a stride of 16 "
     *  "w.r.t. the input image
     * @param[in] pooled_h (int) default 1; Pooled output Y's height.
     * @param[in] pooled_w (int) default 1; Pooled output Y's width.
     * @param[in] sampling_ratio "(int) default -1; number of sampling points in the interpolation grid "
     *  "used to compute the output value of each pooled output bin. If > 0, "
     *  "then exactly sampling_ratio x sampling_ratio grid points are used. If "
     *  "<= 0, then an adaptive number of grid points are used (computed as "
     *  "ceil(roi_width / pooled_w), and likewise for height)."
     */
    void configure(const IGCTensor *input, IGCTensor *output, const IGCTensor *rois, float spatial_scale, int pooled_h, int pooled_w, int sampling_ratio);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const IGCTensor *_input;
    IGCTensor       *_output;
    const IGCTensor *_rois;
    float            _spatial_scale;
    int              _pooled_h;
    int              _pooled_w;
    int              _sampling_ratio;
};
}
#endif /*__ARM_COMPUTE_GCROIALIGNKERNEL_H__*/
