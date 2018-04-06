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
#ifndef ARM_COMPUTE_TEST_SCALE_FIXTURE
#define ARM_COMPUTE_TEST_SCALE_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ROIAlign.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ROIAlignValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape in_shape, TensorShape rois_shape, DataType data_type, float spatial_scale, int pooled_h, int pooled_w, int sampling_ratio)
    {
        _in_shape        = in_shape;
        _rois_shape      = rois_shape;
        _data_type       = data_type;
        _spatial_scale   = spatial_scale;
        _pooled_h        = pooled_h;
        _pooled_w        = pooled_w;
        _sampling_ratio  = sampling_ratio;

        _target    = compute_target(in_shape, rois_shape, data_type, spatial_scale, pooled_h, pooled_w, sampling_ratio);
        // TODO later
        //_reference = compute_reference(in_shape, rois_shape, data_type, spatial_scale, pooled_h, pooled_w, sampling_ratio);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape in_shape, const TensorShape rois_shape, const DataType data_type, float spatial_scale, int pooled_h, int pooled_w, int sampling_ratio)
    {
        TensorShape out_shape;
        out_shape.set(0, pooled_w);
        out_shape.set(1, pooled_h);
        out_shape.set(2, in_shape[2]);
        out_shape.set(3, rois_shape[1]);

        // Create tensors
        TensorType src = create_tensor<TensorType>(in_shape, data_type);
        TensorType dst = create_tensor<TensorType>(out_shape, data_type);
        TensorType rois = create_tensor<TensorType>(rois_shape, data_type);

        // Create and configure function
        FunctionType roialign;

        roialign.configure(&src, &dst, &rois, spatial_scale, pooled_h, pooled_w, sampling_ratio);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        rois.allocator()->allocate();

        // Fill tensors
        fill(AccessorType(src));
        fill(AccessorType(rois));

        // Compute function
        roialign.run();

        return dst;
    }

    /* SimpleTensor<T> compute_reference(const TensorShape in_shape, const TensorShape rois_shape, const DataType data_type, float spatial_scale, int pooled_h, int pooled_w, int sampling_ratio) */
    /* { */
    /*     // Create reference */
    /*     SimpleTensor<T> src{ shape, _data_type }; */

    /*     // Fill reference */
    /*     fill(src); */

    /*     return reference::scale<T>(src, scale_x, scale_y, policy, border_mode, constant_border_value, sampling_policy); */
    /* } */

    TensorType          _target{};
    //SimpleTensor<T>     _reference{};
    TensorShape         _in_shape{};
    TensorShape         _rois_shape{};
    DataType            _data_type;
    float               _spatial_scale;
    int                 _pooled_h;
    int                 _pooled_w;
    int                 _sampling_ratio;
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SCALE_FIXTURE */
