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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCROIAlignKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

using namespace arm_compute;

GCROIAlignKernel::GCROIAlignKernel()
  : _input(nullptr), _output(nullptr), _rois(nullptr), _spatial_scale(0.0f), _pooled_h(0), _pooled_w(0), _sampling_ratio(0)
{
}

void GCROIAlignKernel::configure(const IGCTensor *input, IGCTensor *output, const IGCTensor *rois, float spatial_scale, int pooled_h, int pooled_w, int sampling_ratio)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->tensor_shape(), 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, rois);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output, rois);

    _input   = input;
    _output  = output;
    _rois    = rois;
    _spatial_scale = spatial_scale;
    _pooled_h = pooled_h;
    _pooled_w = pooled_w;
    _sampling_ratio = sampling_ratio;

    unsigned int num_elems_processed_per_iteration = 1;
    if(input->info()->data_type() == DataType::F16)
    {
        num_elems_processed_per_iteration = 4;
    }

    // Set build options
    std::set<std::string> build_opts;
    std::string           dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    build_opts.emplace(("#define " + dt_name));
    build_opts.emplace(("#define SPATIAL_SCALE " + float_to_string_with_full_precision(_spatial_scale)));
    build_opts.emplace(("#define POOLED_H " + support::cpp11::to_string(_pooled_h)));
    build_opts.emplace(("#define POOLED_W " + support::cpp11::to_string(_pooled_w)));
    build_opts.emplace(("#define SAMPLING_RATIO " + support::cpp11::to_string(_sampling_ratio)));
    build_opts.emplace(("#define IN_HEIGHT " + support::cpp11::to_string(input->info()->dimension(1))));
    build_opts.emplace(("#define IN_WIDTH " + support::cpp11::to_string(input->info()->dimension(0))));
    build_opts.emplace(("#define LOCAL_SIZE_X " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Y " + support::cpp11::to_string(1)));
    build_opts.emplace(("#define LOCAL_SIZE_Z " + support::cpp11::to_string(1)));

    // Create kernel
    _kernel = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel("roi_align", build_opts));

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowStatic     input_access(input->info(), 0, 0, input->info()->dimension(0), input->info()->dimension(1));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal rois_access(rois->info(), 0, 0, 5);

    update_window_and_padding(win, input_access, output_access, rois_access);

    IGCKernel::configure(win);
}

void GCROIAlignKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    _kernel.use();

    //_output->set_needs_shifting(true);

    Window slice    = window.first_slice_window_3D();
    Window slice_in = window.first_slice_window_3D();
    Window slice_roi = window.first_slice_window_1D();

    unsigned int idx2 = 2 * num_arguments_per_3D_tensor();
    //slice.shift(Window::DimX, -(_output->info()->padding()).left);

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, 1, slice_in);
        add_3D_tensor_argument(idx, _output, 2, slice);
        add_1D_tensor_argument(idx2, _rois, 3, slice_roi);
        _kernel.update_shader_params();
        enqueue(*this, slice);
    }
    while(window.slide_window_slice_3D(slice) &&
          window.slide_window_slice_3D(slice_in) &&
          window.slide_window_slice_1D(slice_roi));
}
