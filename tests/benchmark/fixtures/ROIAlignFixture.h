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
#ifndef ARM_COMPUTE_TEST_BENCH_ROIALIGN_FIXTURE
#define ARM_COMPUTE_TEST_BENCH_ROIALIGN_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename TensorType, typename Function, typename Accessor, typename T>
class ROIAlignFixture : public framework::Fixture
{
public:
    template <typename...>
      void setup(TensorShape in_shape, TensorShape rois_shape, float spatial_scale, int pooled_h, int pooled_w, int sampling_ratio, DataType data_type)
    {
        TensorShape out_shape;
        out_shape.set(0, pooled_w);
        out_shape.set(1, pooled_h);
        out_shape.set(2, in_shape[2]);
        out_shape.set(3, rois_shape[1]);

        // Create tensors
        src = create_tensor<TensorType>(in_shape, data_type);
        dst = create_tensor<TensorType>(out_shape, data_type);
        rois = create_tensor<TensorType>(rois_shape, data_type);

        // Create and configure function
        roi_align_func.configure(&src, &dst, &rois, spatial_scale, pooled_h, pooled_w, sampling_ratio);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        rois.allocator()->allocate();
    }

    void run()
    {
        roi_align_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

    void teardown()
    {
        src.allocator()->free();
        dst.allocator()->free();
        rois.allocator()->free();
    }

private:
    TensorType src{};
    TensorType dst{};
    TensorType rois{};
    Function   roi_align_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_BENCH_ROIALIGN_FIXTURE */
