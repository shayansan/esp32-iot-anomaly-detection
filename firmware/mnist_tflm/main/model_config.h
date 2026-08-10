#pragma once

namespace mnist {

constexpr int kInputBatch = 1;
constexpr int kInputHeight = 28;
constexpr int kInputWidth = 28;
constexpr int kInputChannels = 1;

constexpr int kInputElementCount =
    kInputHeight * kInputWidth * kInputChannels;

constexpr int kOutputClassCount = 10;

constexpr float kInputScale =
    0.0039215688593685627f;

constexpr int kInputZeroPoint =
    -128;

constexpr float kOutputScale =
    0.00390625f;

constexpr int kOutputZeroPoint =
    -128;

}  // namespace mnist
