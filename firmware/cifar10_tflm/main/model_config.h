#pragma once

namespace cifar10
{

constexpr int kInputBatch =
    1;

constexpr int kInputHeight =
    32;

constexpr int kInputWidth =
    32;

constexpr int kInputChannels =
    3;

constexpr int kInputElementCount =
    kInputHeight
    * kInputWidth
    * kInputChannels;

constexpr int kOutputClassCount =
    10;

constexpr float kInputScale =
    0.0039215688593685627f;

constexpr int kInputZeroPoint =
    -128;

constexpr float kOutputScale =
    0.00390625f;

constexpr int kOutputZeroPoint =
    -128;

}  // namespace cifar10
