# ESP32 IoT Anomaly Detection

Research project focused on TinyML, IoT anomaly detection, and machine learning deployment on the ESP32-S3.

The main goal is to study the complete workflow from model training and optimization to real inference on a resource-constrained embedded device.

## Current Progress

### Synthetic Anomaly Detection

A complete anomaly-detection pipeline has been implemented using simulated sensor data.

Completed steps:

* sensor data generation on ESP32
* dataset collection and preprocessing
* neural network training and evaluation
* TensorFlow Lite conversion
* full INT8 quantization
* TensorFlow Lite Micro deployment
* inference on ESP32-S3
* initial RAM, Flash, and model-size measurements

### MNIST TinyML Benchmark

The MNIST benchmark has been completed from training to ESP32 deployment.

Results:

| Metric                 |       Result |
| ---------------------- | -----------: |
| Keras accuracy         |       97.75% |
| INT8 accuracy          |       97.82% |
| INT8 model size        | 14,072 bytes |
| Tensor arena used      |      ~9.8 KB |
| Firmware RAM           |     ~28.3 KB |
| Firmware Flash         |      ~277 KB |
| Average inference time |     ~8.04 ms |

### CIFAR-10 TinyML Benchmark

The CIFAR-10 host-side pipeline has also been completed.

Results:

| Model          | Accuracy |          Size |
| -------------- | -------: | ------------: |
| Keras          |   60.93% | 229,550 bytes |
| Float32 TFLite |   60.93% |  66,496 bytes |
| INT8 TFLite    |   61.10% |  24,192 bytes |

The next step is to deploy the CIFAR-10 INT8 model on the ESP32-S3 and measure its memory usage and inference time.

## Project Structure

```text
artifacts/
├── anomaly_detection/
├── mnist/
└── cifar10/

docs/
└── reports/

firmware/
├── anomaly_tflm/
├── dataset_generator/
└── mnist_tflm/

ml/
├── anomaly_detection/
└── benchmarks/
    ├── mnist/
    └── cifar10/
```

## Next Steps

1. Complete CIFAR-10 deployment and benchmarking on ESP32-S3.
2. Compare N-BaIoT and TON_IoT for the next anomaly-detection experiment.
3. Select and document a suitable IoT security dataset or subset.
4. Apply the same training, INT8 quantization, deployment, and benchmarking workflow to the selected dataset.

## Technologies

* ESP32-S3
* C/C++
* Python
* TensorFlow / Keras
* TensorFlow Lite
* TensorFlow Lite Micro
* PlatformIO
* TinyML
* Machine Learning
* IoT Security

## Documentation

Weekly progress reports and experiment details are available in:

```text
docs/reports/
```
