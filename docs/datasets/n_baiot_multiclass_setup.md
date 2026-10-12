# N-BaIoT Multiclass Experimental Setup

## Goal

The second N-BaIoT experiment extends the initial binary
Benign-vs-Attack baseline to multiclass attack classification.

The objective is to determine whether a compact model can
distinguish between individual N-BaIoT attack subtypes while
remaining suitable for ESP32 deployment.

## Device

Danmini Doorbell

## Classes

The experiment contains 11 classes:

0. Benign
1. Gafgyt Combo
2. Gafgyt Junk
3. Gafgyt Scan
4. Gafgyt TCP
5. Gafgyt UDP
6. Mirai ACK
7. Mirai Scan
8. Mirai SYN
9. Mirai UDP
10. Mirai UDPplain

## Initial Subset

The first multiclass baseline will use a balanced subset.

Samples per class:

- 8,000 samples

Total:

- 88,000 samples

## Split

For each class:

- Training: 5,600
- Validation: 1,200
- Test: 1,200

Total dataset split:

- Training: 61,600
- Validation: 13,200
- Test: 13,200

## Leakage Control

Samples will not be randomly split directly from the original
sequential source files.

For every source file, separated source regions will be assigned
to training, validation, and test sets.

Unused gaps will be kept between the selected regions where
possible.

Samples will be shuffled only after the split has already been
created.

## Feature Configurations

Two feature configurations will eventually be compared:

1. Full 115-feature input
2. Reduced feature input derived from training data only

Feature-selection decisions must not use validation or test data.

## Evaluation

The multiclass experiment will report:

- Accuracy
- Per-class precision
- Per-class recall
- Per-class F1-score
- Macro F1-score
- Weighted F1-score
- Confusion matrix

The test set will be used only for final evaluation.

## Deployment Goal

After host-side evaluation, the best compact candidate will be:

Keras
-> Float32 TFLite
-> Full INT8 TFLite
-> TensorFlow Lite Micro
-> ESP32-S3

Embedded comparison will include:

- Model size
- Tensor arena
- RAM
- Flash
- Inference latency
- Host-device parity