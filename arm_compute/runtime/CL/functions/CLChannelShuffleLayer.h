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
#ifndef __ARM_COMPUTE_CLCHANNELSHUFFLELAYER_H__
#define __ARM_COMPUTE_CLCHANNELSHUFFLELAYER_H__

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to run @ref CLChannelShuffleLayer
 *
 * @note The tensor data types for the inputs must be S16.
 * @note The function performs a channel shuffle operation on the input tensor. given NCHW tensor with group G, it will
 * first dive the channels into G groups, C = (G * C'), and perform a transpose of the channel, which gives C = (C' * G).
 * for more details see: https://arxiv.org/pdf/1707.01083.pdf
 */
class CLChannelShuffleLayer : public ICLSimpleFunction
{
public:
    /** Initialize the function
     *
     * @param[in]  input Input tensor. Data types supported: S16
     * @param[out] output Output tensor. Data types supported: S16
     */
  void configure(const ICLTensor *input, ICLTensor *output, int group);
};
}
#endif /* __ARM_COMPUTE_CLCHANNELSHUFFLELAYER_H__ */
