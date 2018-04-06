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
     * @param[in]  input    Source tensor. 3 lower dimensions represent a single input with dimensions [width, height, FM].
     *                      The rest are optional and used for representing batches. Data types supported: F16/F32.
     * @param[out] output   Destination tensor. Output will have the same number of dimensions as input. Data type supported: same as @p input
     * @param[in]  mean     Mean values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in]  var      Variance values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in]  beta     Beta values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in]  gamma    Gamma values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in]  epsilon  Small value to avoid division with zero.
     * @param[in]  act_info (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
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
