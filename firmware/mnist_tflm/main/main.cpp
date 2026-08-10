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

#include "model_config.h"
#include "model_data.h"
#include "test_data.h"


namespace
{

constexpr char TAG[] = "mnist_tflm";

/*
 * Initial bring-up arena.
 *
 * This is intentionally larger than necessary.
 * After successful inference, we will reduce it
 * and measure the actual memory requirement.
 */
constexpr std::size_t TENSOR_ARENA_SIZE =
    12 * 1024;

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
        g_mnist_model
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
     * Analyzer showed eight unique operators:
     *
     * CONV_2D
     * MAX_POOL_2D
     * SHAPE
     * STRIDED_SLICE
     * PACK
     * RESHAPE
     * FULLY_CONNECTED
     * SOFTMAX
     */
    static tflite::MicroMutableOpResolver<8>
        resolver;

    if (
        resolver.AddConv2D()
        != kTfLiteOk
    )
    {
        ESP_LOGE(
            TAG,
            "Could not register Conv2D"
        );

        return false;
    }

    if (
        resolver.AddMaxPool2D()
        != kTfLiteOk
    )
    {
        ESP_LOGE(
            TAG,
            "Could not register MaxPool2D"
        );

        return false;
    }

    if (
        resolver.AddShape()
        != kTfLiteOk
    )
    {
        ESP_LOGE(
            TAG,
            "Could not register Shape"
        );

        return false;
    }

    if (
        resolver.AddStridedSlice()
        != kTfLiteOk
    )
    {
        ESP_LOGE(
            TAG,
            "Could not register StridedSlice"
        );

        return false;
    }

    if (
        resolver.AddPack()
        != kTfLiteOk
    )
    {
        ESP_LOGE(
            TAG,
            "Could not register Pack"
        );

        return false;
    }

    if (
        resolver.AddReshape()
        != kTfLiteOk
    )
    {
        ESP_LOGE(
            TAG,
            "Could not register Reshape"
        );

        return false;
    }

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
        resolver.AddSoftmax()
        != kTfLiteOk
    )
    {
        ESP_LOGE(
            TAG,
            "Could not register Softmax"
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
        < mnist::kInputElementCount
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
        < mnist::kOutputClassCount
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
        g_mnist_model_len
    );

    ESP_LOGI(
        TAG,
        "Tensor arena: %u bytes",
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
        int index = 0;
        index < mnist::kInputElementCount;
        ++index
    )
    {
        inputTensor->data.int8[index] =
            g_mnist_test_input[index];
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


int findPredictedClass()
{
    int bestClass = 0;

    std::int8_t bestValue =
        outputTensor->data.int8[0];

    for (
        int classIndex = 1;
        classIndex
            < mnist::kOutputClassCount;
        ++classIndex
    )
    {
        const std::int8_t value =
            outputTensor->data.int8[
                classIndex
            ];

        if (value > bestValue)
        {
            bestValue = value;
            bestClass = classIndex;
        }
    }

    return bestClass;
}


bool runTestInference()
{
    loadTestInput();

    ESP_LOGI(
        TAG,
        "Running inference..."
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

    const int predictedClass =
        findPredictedClass();

    ESP_LOGI(
        TAG,
        "Output scores:"
    );

    for (
        int classIndex = 0;
        classIndex
            < mnist::kOutputClassCount;
        ++classIndex
    )
    {
        const std::int8_t quantized =
            outputTensor->data.int8[
                classIndex
            ];

        const float score =
            dequantizeOutput(
                quantized
            );

        ESP_LOGI(
            TAG,
            "class=%d int8=%d score=%.4f",
            classIndex,
            static_cast<int>(
                quantized
            ),
            score
        );
    }

    ESP_LOGI(
        TAG,
        "Expected label: %d",
        g_mnist_test_expected_label
    );

    ESP_LOGI(
        TAG,
        "Predicted label: %d",
        predictedClass
    );

    ESP_LOGI(
        TAG,
        "Inference time: %lld us",
        static_cast<long long>(
            inferenceTime
        )
    );

    const bool correct =
        predictedClass
        == g_mnist_test_expected_label;

    if (correct)
    {
        ESP_LOGI(
            TAG,
            "MNIST inference test: PASS"
        );
    }
    else
    {
        ESP_LOGE(
            TAG,
            "MNIST inference test: FAIL"
        );
    }

    return correct;
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
        static_cast<double>(totalTime)
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
        "Starting MNIST TFLite Micro"
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
