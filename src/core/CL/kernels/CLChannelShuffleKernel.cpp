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
#include "arm_compute/core/CL/kernels/CLChannelShuffleKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>

using namespace arm_compute;

CLChannelShuffleKernel::CLChannelShuffleKernel()
  : _input(nullptr), _output(nullptr), _group(1)
{
}

void CLChannelShuffleKernel::configure(const ICLTensor *input, ICLTensor *output, int group)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1,DataType::S16);

    _input = input;
    _output = output;
    _group = group;

    // Set kernel build options
    std::set<std::string> build_opts;
    build_opts.insert("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("channel_shuffle", build_opts));

    unsigned int idx = num_arguments_per_3D_tensor() * 2;
    int channels = _input->info()->dimension(2);
    _kernel.setArg<cl_int>(idx++, channels);
    _kernel.setArg<cl_int>(idx++, _group);


    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 4;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, input->info()->valid_region());

    ICLKernel::configure(win);
}

void CLChannelShuffleKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_3D();
    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_3D(slice));
}
