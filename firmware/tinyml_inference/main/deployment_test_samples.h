#pragma once

#include <cstddef>

#include "model_config.h"

struct DeploymentTestSample
{
    const char* name;
    float features[kFeatureCount];
    int expected_label;
};

inline constexpr DeploymentTestSample
    kDeploymentTestSamples[] = {
    {
        "normal",
        {24.36f, 49.52f, 663.17f},
        0
    },
    {
        "temperature_spike",
        {32.75f, 49.77f, 453.39f},
        1
    },
    {
        "humidity_drop",
        {22.73f, 20.91f, 638.01f},
        1
    },
    {
        "light_saturation",
        {24.18f, 49.79f, 1023.0f},
        1
    },
};

inline constexpr std::size_t
    kDeploymentTestSampleCount =
        sizeof(kDeploymentTestSamples)
        / sizeof(kDeploymentTestSamples[0]);
