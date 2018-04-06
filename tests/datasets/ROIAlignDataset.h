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
#ifndef ARM_COMPUTE_TEST_ROIALIGN_DATASET
#define ARM_COMPUTE_TEST_ROIALIGN_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class ROIAlignDataset {
public:
  using type = std::tuple<TensorShape, TensorShape, float, int, int, int>;

 struct iterator
 {
 iterator(std::vector<TensorShape>::const_iterator         src_it,
          std::vector<TensorShape>::const_iterator         rois_it,
          std::vector<float>::const_iterator               spatial_scale_it,
          std::vector<int>::const_iterator                 pooled_h_it,
          std::vector<int>::const_iterator                 pooled_w_it,
          std::vector<int>::const_iterator                 sampling_ratio_it)
 : _src_it{ std::move(src_it) },
     _rois_it{ std::move(rois_it) },
     _spatial_scale_it { std::move(spatial_scale_it) },
     _pooled_h_it { std::move(pooled_h_it) },
     _pooled_w_it { std::move(pooled_w_it) },
     _sampling_ratio_it { std::move(sampling_ratio_it) }
   {
   }

   std::string description() const
   {
     std::stringstream description;
     description << "In=" << *_src_it << ":";
     description << "Rois=" << *_rois_it << ":";
     description << "Spatial_scale=" << *_spatial_scale_it << ":";
     description << "Pooled_h=" << *_pooled_h_it << ":";
     description << "Pooled_w=" << *_pooled_w_it << ":";
     description << "Sampling_ratio=" << *_sampling_ratio_it;
     return description.str();
   }

   ROIAlignDataset::type operator*() const
   {
     return std::make_tuple(*_src_it, *_rois_it, *_spatial_scale_it, *_pooled_h_it, *_pooled_w_it, *_sampling_ratio_it);
   }

   iterator &operator++()
   {
     ++_src_it;
     ++_rois_it;
     ++_spatial_scale_it;
     ++_pooled_h_it;
     ++_pooled_w_it;
     ++_sampling_ratio_it;

     return *this;
   }

 private:
   std::vector<TensorShape>::const_iterator         _src_it;
   std::vector<TensorShape>::const_iterator         _rois_it;
   std::vector<float>::const_iterator               _spatial_scale_it;
   std::vector<int>::const_iterator                 _pooled_h_it;
   std::vector<int>::const_iterator                 _pooled_w_it;
   std::vector<int>::const_iterator                 _sampling_ratio_it;

 };

 iterator begin() const
 {
   return iterator(_src_shapes.begin(), _rois_shapes.begin(), _spatial_scale.begin(), _pooled_h.begin(), _pooled_w.begin(), _sampling_ratio.begin());
 }

 int size() const
 {
   return std::min(_src_shapes.size(), std::min(_rois_shapes.size(), std::min(_spatial_scale.size(), std::min(_pooled_h.size(), std::min(_pooled_w.size(), _sampling_ratio.size())))));
 }

 void add_config(TensorShape src, TensorShape rois, float spatial_scale, int pooled_h, int pooled_w, int sampling_ratio)
 {
   _src_shapes.emplace_back(std::move(src));
   _rois_shapes.emplace_back(std::move(rois));
   _spatial_scale.emplace_back(std::move(spatial_scale));
   _pooled_h.emplace_back(std::move(pooled_h));
   _pooled_w.emplace_back(std::move(pooled_w));
   _sampling_ratio.emplace_back(std::move(sampling_ratio));
 }

 protected:
 ROIAlignDataset()                     = default;
 ROIAlignDataset(ROIAlignDataset &&)   = default;

 private:
 std::vector<TensorShape>       _src_shapes{};
 std::vector<TensorShape>       _rois_shapes{};
 std::vector<float>             _spatial_scale{};
 std::vector<int>               _pooled_h{};
 std::vector<int>               _pooled_w{};
 std::vector<int>               _sampling_ratio{};
};

/** Data set containing small scale layer shapes. */
class SmallROIAlignShapes final : public ROIAlignDataset
{
public:
    SmallROIAlignShapes()
    {
      for (const auto scale : std::vector<float>{1.0, 2.0, 0.0625}) {
        for (const auto channels : std::vector<size_t>{1, 3, 5, 8}) {
          for (const auto pool : std::vector<size_t>{1, 3, 7}) {
            for (const auto sampling_ratio : std::vector<size_t>{0, 1, 2, 3}) {
              add_config(TensorShape(40U, 40U, channels, 1U), TensorShape(5U, 6U), scale, pool, pool, sampling_ratio);
            }
          }
        }
      }
    }
};

/** Data set containing large scale layer shapes. */
class LargeROIAlignShapes final : public ROIAlignDataset
{
public:
    LargeROIAlignShapes()
    {
      for (const auto num_rois : std::vector<int>{1, 2, 3, 6}) {
        for (const auto scale : std::vector<float>{1.0, 2.0, 0.0625}) {
          for (const auto channels : std::vector<size_t>{3}) {
            for (const auto pool : std::vector<size_t>{3, 7}) {
              for (const auto sampling_ratio : std::vector<size_t>{0, 1, 2, 3}) {
                add_config(TensorShape(160U, 160U, channels, 1), TensorShape(5U, num_rois), scale, pool, pool, sampling_ratio);
              }
            }
          }
        }
      }
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SCALE_LAYER_DATASET */
