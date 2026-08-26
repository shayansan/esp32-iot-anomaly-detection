# IoT Dataset Selection

## Candidate Datasets

### N-BaIoT
- IoT-focused dataset
- Approximately 7.06 million samples
- 115 features
- 9 real IoT devices
- Mirai and BASHLITE botnet attacks
- Supports benign vs malicious anomaly detection
- Suitable for creating a smaller reproducible embedded-oriented subset

### TON_IoT
- IoT and IIoT dataset
- Includes IoT telemetry
- Includes network traffic
- Includes Linux telemetry
- Includes Windows telemetry
- Includes multiple attack types
- More heterogeneous and complex

## Selected Dataset

N-BaIoT

## Reason

N-BaIoT is more directly focused on IoT botnet anomaly detection and provides a simpler starting point for applying the complete TinyML pipeline on the ESP32-S3.

Because the full dataset is very large, the next step will be to define a smaller reproducible subset of devices, attack classes, samples, and features.