# ESP32 IoT Anomaly Detection

## Description

Research project focused on TinyML-based IoT anomaly detection using the ESP32-S3.

The project studies the complete workflow from model training and optimization to TensorFlow Lite Micro deployment and embedded benchmarking.

## Current Progress

### Synthetic Anomaly Detection Prototype
- Data preprocessing and model training completed
- Float32 and INT8 TensorFlow Lite conversion completed
- INT8 model deployed on ESP32-S3
- Initial memory and inference analysis completed

### MNIST TinyML Benchmark
- Keras accuracy: 97.75%
- Float32 TFLite accuracy: 97.75%
- INT8 TFLite accuracy: 97.82%
- INT8 model size: 14,072 bytes
- Tensor arena used: 9,836 bytes
- Firmware RAM: 28,300 bytes
- Firmware Flash: 277,185 bytes
- Average inference latency: 8.04 ms
- ESP32 correctness test: PASS

### CIFAR-10 TinyML Benchmark
- Keras accuracy: 60.93%
- Float32 TFLite accuracy: 60.93%
- INT8 TFLite accuracy: 61.10%
- INT8 model size: 24,192 bytes
- Tensor arena used: 21,276 bytes
- Minimum tested working arena: 21,504 bytes
- Final configured arena: 22,528 bytes
- Firmware RAM: 38,540 bytes
- Firmware Flash: 291,209 bytes
- Average inference latency: 18.58 ms
- ESP32 correctness test: PASS

## Next Phase

- Compare N-BaIoT and TON_IoT
- Select the dataset most suitable for the project
- Define a reproducible subset if required
- Prepare and preprocess IoT attack data
- Train an anomaly detection model
- Convert the model to TensorFlow Lite
- Apply full INT8 quantization
- Deploy the model on ESP32-S3
- Measure detection performance, model size, RAM, Flash, and inference latency

## Technologies

- ESP32-S3
- ESP-IDF
- PlatformIO
- C/C++
- Python
- TensorFlow / Keras
- TensorFlow Lite
- TensorFlow Lite Micro
- TinyML
- Machine Learning
- IoT Security
- Network Anomaly Detection