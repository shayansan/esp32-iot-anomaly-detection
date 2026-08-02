#pragma once

#include <cstddef>

inline constexpr std::size_t kFeatureCount =
    3;

inline constexpr const char* kFeatureNames[
    kFeatureCount
] = {
    "temperature_c",
    "humidity_percent",
    "light_raw"
};

inline constexpr float kFeatureMean[
    kFeatureCount
] = {
    24.1320833f,
    49.5023633f,
    557.675385f
};

inline constexpr float kFeatureStandardDeviation[
    kFeatureCount
] = {
    1.41350399f,
    3.79266774f,
    219.769585f
};

inline constexpr float kClassificationThreshold =
    0.99609375f;
