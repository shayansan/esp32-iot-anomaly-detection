# N-BaIoT Experimental Setup

## Objective

Evaluate the feasibility of binary IoT intrusion detection on the ESP32-S3 using the N-BaIoT dataset.

## Device Comparison

| Device | Benign | Attack | Attack Families |
|---|---:|---:|---|
| Danmini Doorbell | 49,548 | 968,750 | Mirai + Gafgyt |
| Ecobee Thermostat | 13,113 | 822,763 | Mirai + Gafgyt |
| Ennio Doorbell | 39,100 | 316,400 | Gafgyt only |
| Philips B120N/10 Baby Monitor | 175,240 | 923,437 | Mirai + Gafgyt |
| Provision PT-737E Camera | 62,154 | 766,106 | Mirai + Gafgyt |
| Provision PT-838 Camera | 98,514 | 738,377 | Mirai + Gafgyt |
| Samsung SNH-1011-N Webcam | 52,150 | 323,072 | Gafgyt only |
| SimpleHome XCS7-1002 Camera | 46,585 | 816,471 | Mirai + Gafgyt |
| SimpleHome XCS7-1003 Camera | 19,528 | 831,298 | Mirai + Gafgyt |

## Initial Device

Danmini Doorbell

### Rationale

Danmini Doorbell contains benign traffic and both major N-BaIoT botnet families, Mirai and Gafgyt/BASHLITE.

It also contains all available attack subtypes for these families and provides enough benign and malicious samples for the first experiment.

Using one device initially keeps the experimental workflow manageable and allows the complete preprocessing, training, quantization, deployment, and benchmarking pipeline to be established before considering multi-device experiments.

## Initial Problem Formulation

Binary classification:

- 0 = Benign
- 1 = Attack

The first experiment will combine all selected Mirai and Gafgyt attack samples into the Attack class.

Multiclass attack classification will be considered as a later extension.

## Features

The original N-BaIoT representation contains 115 numerical features.

Initial analysis was performed using only the training set.

The training data contained:

- 0 NaN values
- 0 infinite values
- 0 constant features
- 0 near-constant features

A correlation analysis using a threshold of |r| >= 0.99 identified 160 highly correlated feature pairs.

A deterministic correlation-redundancy selection procedure was then applied using the training set only.

Results:

- Original features: 115
- Selected features: 60
- Removed redundant features: 55

Two feature configurations will therefore be evaluated:

1. Full baseline: 115 features
2. Correlation-reduced baseline: 60 features

The reduced feature set will only be considered preferable if it maintains acceptable detection performance while improving embedded resource requirements such as model size, RAM/tensor arena usage, Flash usage, and inference latency.

## Class Imbalance

For Danmini Doorbell:

- Benign: 49,548
- Attack: 968,750
- Total: 1,018,298

The original distribution is strongly imbalanced towards attack traffic.

A smaller reproducible subset will therefore be defined for the first experiment.

## Dataset Split

The dataset is sequential, so random row-level splitting of the complete data will be avoided.

Each source CSV will be handled separately and split into contiguous sections for:

- Training
- Validation
- Test

Preprocessing parameters and feature-selection decisions will be fitted using training data only.

Validation data will be used for model selection and threshold decisions.

The test set will only be used for final evaluation.

## Reproducibility

The complete subset-generation procedure will be implemented in code.

The following will be fixed and documented:

- Selected device
- Selected attack classes
- Number of samples per class
- Sampling/splitting method
- Random seed where required
- Selected features
- Preprocessing parameters