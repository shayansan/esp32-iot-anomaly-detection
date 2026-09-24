#include <cstddef>
#include <cstdint>
#include <climits>

#include "esp_log.h"
#include "esp_timer.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "test_sample.h"


namespace
{

constexpr char TAG[] = "n_baiot_tflm";

constexpr std::size_t INPUT_ELEMENT_COUNT = 60;
constexpr std::size_t OUTPUT_ELEMENT_COUNT = 1;

/*
 * Initial bring-up arena.
 *
 * We intentionally start larger than necessary.
 * After successful inference, we will reduce it
 * and measure the real requirement.
 */
constexpr std::size_t TENSOR_ARENA_SIZE =
    1792;

alignas(16) std::uint8_t tensorArena[
    TENSOR_ARENA_SIZE
];

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;


bool initializeModel()
{
    model = tflite::GetModel(
        g_model
    );

    if (model == nullptr)
    {
        ESP_LOGE(
            TAG,
            "Could not load model"
        );

        return false;
    }

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

    /*
     * The N-BaIoT model uses two unique operators:
     *
     * FULLY_CONNECTED
     * LOGISTIC
     */
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

    ESP_LOGI(
        TAG,
        "Tensor arena used: %u / %u bytes",
        static_cast<unsigned>(
            interpreter->arena_used_bytes()
        ),
        static_cast<unsigned>(
            TENSOR_ARENA_SIZE
        )
    );

    inputTensor =
        interpreter->input(0);

    outputTensor =
        interpreter->output(0);

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
        inputTensor->type
            != kTfLiteInt8
        || outputTensor->type
            != kTfLiteInt8
    )
    {
        ESP_LOGE(
            TAG,
            "Expected INT8 input and output"
        );

        return false;
    }

    if (
        inputTensor->bytes
        < INPUT_ELEMENT_COUNT
    )
    {
        ESP_LOGE(
            TAG,
            "Input tensor is too small"
        );

        return false;
    }

    if (
        outputTensor->bytes
        < OUTPUT_ELEMENT_COUNT
    )
    {
        ESP_LOGE(
            TAG,
            "Output tensor is too small"
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
        g_model_len
    );

    ESP_LOGI(
        TAG,
        "Tensor arena allocated: %u bytes",
        static_cast<unsigned>(
            TENSOR_ARENA_SIZE
        )
    );

    ESP_LOGI(
        TAG,
        "Input scale=%.10f zero_point=%d",
        inputTensor->params.scale,
        inputTensor->params.zero_point
    );

    ESP_LOGI(
        TAG,
        "Output scale=%.10f zero_point=%d",
        outputTensor->params.scale,
        outputTensor->params.zero_point
    );

    return true;
}


void loadTestInput()
{
    for (
        std::size_t index = 0;
        index < INPUT_ELEMENT_COUNT;
        ++index
    )
    {
        inputTensor->data.int8[index] =
            g_test_input[index];
    }
}


float dequantizeOutput(
    std::int8_t value
)
{
    return (
        static_cast<int>(value)
        - outputTensor->params.zero_point
    )
        * outputTensor->params.scale;
}


int getPredictedLabel()
{
    const std::int8_t quantizedOutput =
        outputTensor->data.int8[0];

    const float probability =
        dequantizeOutput(
            quantizedOutput
        );

    return probability >= 0.5f
        ? 1
        : 0;
}


bool runTestInference()
{
    loadTestInput();

    ESP_LOGI(
        TAG,
        "Running N-BaIoT inference..."
    );

    const std::int64_t startTime =
        esp_timer_get_time();

    if (
        interpreter->Invoke()
        != kTfLiteOk
    )
    {
        ESP_LOGE(
            TAG,
            "Invoke failed"
        );

        return false;
    }

    const std::int64_t endTime =
        esp_timer_get_time();

    const std::int64_t inferenceTime =
        endTime - startTime;

    const std::int8_t quantizedOutput =
        outputTensor->data.int8[0];

    const float probability =
        dequantizeOutput(
            quantizedOutput
        );

    const int predictedLabel =
        getPredictedLabel();

    ESP_LOGI(
        TAG,
        "Output int8: %d",
        static_cast<int>(
            quantizedOutput
        )
    );

    ESP_LOGI(
        TAG,
        "Attack probability: %.6f",
        probability
    );

    ESP_LOGI(
        TAG,
        "Expected label: %d",
        g_expected_label
    );

    ESP_LOGI(
        TAG,
        "Host predicted label: %d",
        g_host_predicted_label
    );

    ESP_LOGI(
        TAG,
        "ESP32 predicted label: %d",
        predictedLabel
    );

    ESP_LOGI(
        TAG,
        "Inference time: %lld us",
        static_cast<long long>(
            inferenceTime
        )
    );

    const bool correct =
        predictedLabel
        == g_expected_label;

    const bool parity =
        predictedLabel
        == g_host_predicted_label;

    if (correct && parity)
    {
        ESP_LOGI(
            TAG,
            "N-BaIoT inference test: PASS"
        );

        return true;
    }

    ESP_LOGE(
        TAG,
        "N-BaIoT inference test: FAIL"
    );

    return false;
}


void runLatencyBenchmark()
{
    constexpr int WARMUP_RUNS = 5;
    constexpr int BENCHMARK_RUNS = 100;

    ESP_LOGI(
        TAG,
        "Starting latency benchmark"
    );

    for (
        int run = 0;
        run < WARMUP_RUNS;
        ++run
    )
    {
        loadTestInput();

        if (
            interpreter->Invoke()
            != kTfLiteOk
        )
        {
            ESP_LOGE(
                TAG,
                "Warm-up inference failed"
            );

            return;
        }
    }

    std::int64_t totalTime = 0;
    std::int64_t minimumTime = INT64_MAX;
    std::int64_t maximumTime = 0;

    for (
        int run = 0;
        run < BENCHMARK_RUNS;
        ++run
    )
    {
        loadTestInput();

        const std::int64_t startTime =
            esp_timer_get_time();

        if (
            interpreter->Invoke()
            != kTfLiteOk
        )
        {
            ESP_LOGE(
                TAG,
                "Benchmark inference failed"
            );

            return;
        }

        const std::int64_t endTime =
            esp_timer_get_time();

        const std::int64_t elapsedTime =
            endTime - startTime;

        totalTime += elapsedTime;

        if (elapsedTime < minimumTime)
        {
            minimumTime = elapsedTime;
        }

        if (elapsedTime > maximumTime)
        {
            maximumTime = elapsedTime;
        }
    }

    const double averageTime =
        static_cast<double>(
            totalTime
        )
        / BENCHMARK_RUNS;

    ESP_LOGI(
        TAG,
        "Latency benchmark runs: %d",
        BENCHMARK_RUNS
    );

    ESP_LOGI(
        TAG,
        "Latency min: %lld us",
        static_cast<long long>(
            minimumTime
        )
    );

    ESP_LOGI(
        TAG,
        "Latency max: %lld us",
        static_cast<long long>(
            maximumTime
        )
    );

    ESP_LOGI(
        TAG,
        "Latency average: %.2f us",
        averageTime
    );
}

}  // namespace


extern "C" void app_main()
{
    ESP_LOGI(
        TAG,
        "Starting N-BaIoT TFLite Micro"
    );

    if (!initializeModel())
    {
        ESP_LOGE(
            TAG,
            "Initialization failed"
        );

        return;
    }

    const bool testPassed =
        runTestInference();

    if (testPassed)
    {
        runLatencyBenchmark();
    }
    else
    {
        ESP_LOGE(
            TAG,
            "Skipping benchmark because correctness test failed"
        );
    }

    while (true)
    {
        vTaskDelay(
            pdMS_TO_TICKS(1000)
        );
    }
}