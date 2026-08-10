#include <Arduino.h>
#include <math.h>

#ifndef DATASET_SEED
#define DATASET_SEED 42UL
#endif

// Generate one sample every 100 milliseconds.
constexpr unsigned long SAMPLE_INTERVAL_MS = 100;

// Used for the sine-wave calculations.
constexpr float PI_VALUE = 3.14159265f;

unsigned long lastSampleTime = 0;
unsigned long sampleNumber = 0;

/**
 * Generate a floating-point random value between minValue and maxValue.
 */
float randomFloat(float minValue, float maxValue)
{
    const float ratio = random(0, 10001) / 10000.0f;
    return minValue + ratio * (maxValue - minValue);
}

void setup()
{
    Serial.begin(115200);

    // Give the serial monitor enough time to connect.
    delay(1000);

    /*
     * The seed is selected from platformio.ini.
     *
     * train      -> 42
     * validation -> 123
     * test       -> 999
     */
    randomSeed(DATASET_SEED);

    Serial.printf(
        "# dataset_seed=%lu\n",
        static_cast<unsigned long>(DATASET_SEED)
    );

    Serial.println(
        "timestamp_ms,temperature_c,humidity_percent,light_raw,anomaly"
    );
}

void loop()
{
    const unsigned long currentTime = millis();

    // Return immediately if 100 ms have not passed.
    if (currentTime - lastSampleTime < SAMPLE_INTERVAL_MS)
    {
        return;
    }

    lastSampleTime = currentTime;

    const float timeSeconds = currentTime / 1000.0f;

    /*
     * Normal simulated temperature:
     * - Average: approximately 24°C
     * - Slow periodic variation
     * - Small random noise
     */
    float temperature =
        24.0f +
        1.2f * sinf(2.0f * PI_VALUE * timeSeconds / 30.0f) +
        randomFloat(-0.2f, 0.2f);

    /*
     * Humidity is partially related to temperature.
     * When temperature increases, humidity decreases slightly.
     */
    float humidity =
        50.0f -
        0.7f * (temperature - 24.0f) +
        randomFloat(-0.5f, 0.5f);

    /*
     * Simulated light level in ADC-like units:
     * 0 means very dark.
     * 1023 means maximum simulated brightness.
     */
    float light =
        550.0f +
        300.0f * sinf(2.0f * PI_VALUE * timeSeconds / 20.0f) +
        randomFloat(-15.0f, 15.0f);

    bool anomaly = false;

    /*
     * Create an anomaly every 100 samples.
     *
     * 100 samples × 100 ms = 10 seconds
     *
     * The anomaly lasts for five samples:
     * 5 × 100 ms = 500 ms
     */
    if (sampleNumber >= 100 && sampleNumber % 100 < 5)
    {
        anomaly = true;

        const uint8_t anomalyType =
            ((sampleNumber / 100) - 1) % 3;

        switch (anomalyType)
        {
            case 0:
                // Sudden temperature spike
                temperature += 8.0f;
                break;

            case 1:
                // Sudden humidity drop
                humidity -= 30.0f;
                break;

            case 2:
                // Sudden maximum light intensity
                light = 1023.0f;
                break;
        }
    }

    // Keep values inside reasonable limits.
    temperature = constrain(temperature, -20.0f, 80.0f);
    humidity = constrain(humidity, 0.0f, 100.0f);
    light = constrain(light, 0.0f, 1023.0f);

    // Print one CSV row.
    Serial.printf(
        "%lu,%.2f,%.2f,%.2f,%d\n",
        currentTime,
        temperature,
        humidity,
        light,
        anomaly ? 1 : 0
    );

    sampleNumber++;
}