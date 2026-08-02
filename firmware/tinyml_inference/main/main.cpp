#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "esp_log.h"
#include "esp_timer.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_config.h"
#include "model_data.h"


namespace
{
constexpr char TAG[] = "tinyml";

constexpr std::uint32_t SAMPLE_INTERVAL_MS = 100;
constexpr float PI_VALUE = 3.14159265f;

constexpr std::size_t TENSOR_ARENA_SIZE =
    32 * 1024;

alignas(16) std::uint8_t tensorArena[
    TENSOR_ARENA_SIZE
];

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;

std::uint32_t randomState = 42;
std::uint64_t sampleNumber = 0;


struct SensorSample
{
    float temperature;
    float humidity;
    float light;
    bool groundTruthAnomaly;
};


std::uint32_t nextRandom()
{
    randomState =
        1664525U * randomState
        + 1013904223U;

    return randomState;
}


float randomFloat(
    float minimum,
    float maximum
)
{
    constexpr float denominator =
        16777215.0f;

    const float ratio =
        static_cast<float>(
            nextRandom() & 0x00FFFFFFU
        ) / denominator;

    return minimum
        + ratio * (maximum - minimum);
}


float clampValue(
    float value,
    float minimum,
    float maximum
)
{
    if (value < minimum)
    {
        return minimum;
    }

    if (value > maximum)
    {
        return maximum;
    }

    return value;
}


SensorSample generateSensorSample(
    std::uint64_t timestampMilliseconds
)
{
    const float timeSeconds =
        static_cast<float>(
            timestampMilliseconds
        ) / 1000.0f;

    float temperature =
        24.0f
        + 1.2f * std::sinf(
            2.0f
            * PI_VALUE
            * timeSeconds
            / 30.0f
        )
        + randomFloat(-0.2f, 0.2f);

    float humidity =
        50.0f
        - 0.7f * (
            temperature - 24.0f
        )
        + randomFloat(-0.5f, 0.5f);

    float light =
        550.0f
        + 300.0f * std::sinf(
            2.0f
            * PI_VALUE
            * timeSeconds
            / 20.0f
        )
        + randomFloat(-15.0f, 15.0f);

    bool anomaly = false;

    if (
        sampleNumber >= 100
        && sampleNumber % 100 < 5
    )
    {
        anomaly = true;

        const std::uint8_t anomalyType =
            static_cast<std::uint8_t>(
                ((sampleNumber / 100) - 1) % 3
            );

        switch (anomalyType)
        {
            case 0:
                temperature += 8.0f;
                break;

            case 1:
                humidity -= 30.0f;
                break;

            case 2:
                light = 1023.0f;
                break;

            default:
                break;
        }
    }

    temperature = clampValue(
        temperature,
        -20.0f,
        80.0f
    );

    humidity = clampValue(
        humidity,
        0.0f,
        100.0f
    );

    light = clampValue(
        light,
        0.0f,
        1023.0f
    );

    return {
        temperature,
        humidity,
        light,
        anomaly
    };
}


std::int8_t quantizeInput(
    float normalizedValue
)
{
    const float scale =
        inputTensor->params.scale;

    const int zeroPoint =
        inputTensor->params.zero_point;

    long quantizedValue =
        std::lround(
            normalizedValue / scale
        ) + zeroPoint;

    if (quantizedValue > 127)
    {
        quantizedValue = 127;
    }
    else if (quantizedValue < -128)
    {
        quantizedValue = -128;
    }

    return static_cast<std::int8_t>(
        quantizedValue
    );
}


bool initializeModel()
{
    model = tflite::GetModel(
        g_model_data
    );

    if (
        model->version()
        != TFLITE_SCHEMA_VERSION
    )
    {
        ESP_LOGE(
            TAG,
            "Schema mismatch: model=%d runtime=%d",
            model->version(),
            TFLITE_SCHEMA_VERSION
        );

        return false;
    }

    static tflite::MicroMutableOpResolver<2>
        resolver;

    if (
        resolver.AddFullyConnected()
        != kTfLiteOk
    )
    {
        ESP_LOGE(
            TAG,
            "Could not register FullyConnected"
        );

        return false;
    }

    if (
        resolver.AddLogistic()
        != kTfLiteOk
    )
    {
        ESP_LOGE(
            TAG,
            "Could not register Logistic"
        );

        return false;
    }

    static tflite::MicroInterpreter
        staticInterpreter(
            model,
            resolver,
            tensorArena,
            TENSOR_ARENA_SIZE
        );

    interpreter = &staticInterpreter;

    if (
        interpreter->AllocateTensors()
        != kTfLiteOk
    )
    {
        ESP_LOGE(
            TAG,
            "AllocateTensors failed"
        );

        return false;
    }

    inputTensor = interpreter->input(0);
    outputTensor = interpreter->output(0);

    if (
        inputTensor == nullptr
        || outputTensor == nullptr
    )
    {
        ESP_LOGE(
            TAG,
            "Input or output tensor missing"
        );

        return false;
    }

    if (
        inputTensor->type != kTfLiteInt8
        || outputTensor->type != kTfLiteInt8
    )
    {
        ESP_LOGE(
            TAG,
            "Model tensors are not int8"
        );

        return false;
    }

    ESP_LOGI(
        TAG,
        "Model initialized successfully"
    );

    ESP_LOGI(
        TAG,
        "Model size: %u bytes",
        g_model_data_len
    );

    ESP_LOGI(
        TAG,
        "Threshold: %.9f",
        kClassificationThreshold
    );

    return true;
}


bool runInference(
    const SensorSample& sample,
    float& anomalyScore,
    std::int64_t& inferenceTimeMicroseconds
)
{
    const float rawFeatures[
        kFeatureCount
    ] = {
        sample.temperature,
        sample.humidity,
        sample.light
    };

    for (
        std::size_t index = 0;
        index < kFeatureCount;
        ++index
    )
    {
        const float normalizedValue =
            (
                rawFeatures[index]
                - kFeatureMean[index]
            )
            / kFeatureStandardDeviation[index];

        inputTensor->data.int8[index] =
            quantizeInput(
                normalizedValue
            );
    }

    const std::int64_t startTime =
        esp_timer_get_time();

    const TfLiteStatus status =
        interpreter->Invoke();

    const std::int64_t endTime =
        esp_timer_get_time();

    inferenceTimeMicroseconds =
        endTime - startTime;

    if (status != kTfLiteOk)
    {
        return false;
    }

    const std::int8_t quantizedOutput =
        outputTensor->data.int8[0];

    anomalyScore =
        (
            static_cast<int>(
                quantizedOutput
            )
            - outputTensor->params.zero_point
        )
        * outputTensor->params.scale;

    return true;
}
}


extern "C" void app_main(void)
{
    ESP_LOGI(
        TAG,
        "Starting real-time anomaly detection"
    );

    if (!initializeModel())
    {
        ESP_LOGE(
            TAG,
            "Model initialization failed"
        );

        return;
    }

    std::printf(
        "timestamp_ms,"
        "temperature_c,"
        "humidity_percent,"
        "light_raw,"
        "ground_truth,"
        "predicted_anomaly,"
        "anomaly_score,"
        "inference_time_us\n"
    );

    std::fflush(stdout);

    TickType_t previousWakeTime =
        xTaskGetTickCount();

    while (true)
    {
        const std::uint64_t timestampMilliseconds =
            static_cast<std::uint64_t>(
                esp_timer_get_time() / 1000
            );

        const SensorSample sample =
            generateSensorSample(
                timestampMilliseconds
            );

        float anomalyScore = 0.0f;
        std::int64_t inferenceTimeUs = 0;

        const bool inferenceSucceeded =
            runInference(
                sample,
                anomalyScore,
                inferenceTimeUs
            );

        if (!inferenceSucceeded)
        {
            ESP_LOGE(
                TAG,
                "Inference failed"
            );
        }
        else
        {
            const int predictedAnomaly =
                anomalyScore
                    >= kClassificationThreshold
                ? 1
                : 0;

            std::printf(
                "%llu,%.2f,%.2f,%.2f,"
                "%d,%d,%.6f,%lld\n",
                static_cast<unsigned long long>(
                    timestampMilliseconds
                ),
                sample.temperature,
                sample.humidity,
                sample.light,
                sample.groundTruthAnomaly
                    ? 1
                    : 0,
                predictedAnomaly,
                anomalyScore,
                static_cast<long long>(
                    inferenceTimeUs
                )
            );

            std::fflush(stdout);
        }

        ++sampleNumber;

        vTaskDelayUntil(
            &previousWakeTime,
            pdMS_TO_TICKS(
                SAMPLE_INTERVAL_MS
            )
        );
    }
}