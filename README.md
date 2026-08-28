# ESP32 IoT Anomaly Detection

## Description

Research project focused on TinyML-based IoT anomaly detection using the ESP32-S3.

The project studies the complete workflow from dataset preparation and model training to TensorFlow Lite Micro deployment and embedded benchmarking.

## Current Progress

### Synthetic Anomaly Detection Prototype

* Data preprocessing and model training completed
* Float32 and INT8 TensorFlow Lite conversion completed
* INT8 model deployed on ESP32-S3
* Initial memory and inference analysis completed

### MNIST TinyML Benchmark

* Keras accuracy: 97.75%
* Float32 TFLite accuracy: 97.75%
* INT8 TFLite accuracy: 97.82%
* INT8 model size: 14,072 bytes
* Tensor arena used: 9,836 bytes
* Firmware RAM: 28,300 bytes
* Firmware Flash: 277,185 bytes
* Average inference latency: 8.04 ms
* ESP32 correctness test: PASS

### CIFAR-10 TinyML Benchmark

* Keras accuracy: 60.93%
* Float32 TFLite accuracy: 60.93%
* INT8 TFLite accuracy: 61.10%
* INT8 model size: 24,192 bytes
* Tensor arena used: 21,276 bytes
* Minimum tested working arena: 21,504 bytes
* Final configured arena: 22,528 bytes
* Firmware RAM: 38,540 bytes
* Firmware Flash: 291,209 bytes
* Average inference latency: 18.58 ms
* ESP32 correctness test: PASS

### N-BaIoT IoT Anomaly Detection

The project has now moved to the IoT anomaly-detection phase using the N-BaIoT dataset.

#### Initial Experimental Setup

* Selected device: Danmini Doorbell
* Problem formulation: Binary classification
* Label 0: Benign
* Label 1: Attack
* Attack families: Gafgyt/BASHLITE and Mirai
* Original input features: 115

#### Dataset Characterization

For the Danmini Doorbell device:

* Benign samples: 49,548
* Attack samples: 968,750
* Total samples: 1,018,298

The original dataset is strongly imbalanced towards attack traffic.

#### Initial Reproducible Subset

A balanced subset of 80,000 samples was created:

* Benign: 40,000
* Attack: 40,000

The attack class contains equal contributions from the ten available attack subtypes.

Dataset split:

* Training: 56,000 samples
* Validation: 12,000 samples
* Test: 12,000 samples

The source CSV files are sequential, so training, validation, and test samples are selected from separated regions of each source file instead of randomly mixing the original traffic sequence.

Unused gaps are kept between regions where possible to reduce the risk of data leakage.

#### Feature Analysis

Feature analysis was performed using the training set only.

Results:

* NaN values: 0
* Infinite values: 0
* Constant features: 0
* Near-constant features: 0
* Highly correlated feature pairs with |r| >= 0.99: 160

A reproducible correlation-based feature-selection procedure reduced the input from:

* 115 original features
* 60 selected features
* 55 removed redundant features

Two configurations will be compared during training:

1. Full 115-feature baseline
2. Reduced 60-feature baseline

The goal is to evaluate whether the reduced feature set can maintain similar detection performance while lowering ESP32 resource requirements.

## Next Steps

* Build the preprocessing pipeline
* Fit preprocessing parameters using training data only
* Prepare 115-feature and 60-feature dataset versions
* Train the first binary N-BaIoT baseline models
* Evaluate accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC
* Compare full and reduced feature configurations
* Convert the selected model to TensorFlow Lite
* Apply full INT8 quantization
* Deploy the model on ESP32-S3
* Measure model size, RAM, Flash, tensor arena usage, and inference latency
* Compare detection performance with embedded resource requirements

## Technologies

* ESP32-S3
* ESP-IDF
* PlatformIO
* C/C++
* Python
* NumPy
* Pandas
* TensorFlow / Keras
* TensorFlow Lite
* TensorFlow Lite Micro
* TinyML
* Machine Learning
* IoT Security
* Network Anomaly Detection
* N-BaIoT
