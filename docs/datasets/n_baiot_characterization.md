# N-BaIoT Dataset Characterization

## Initial Candidate Device

Danmini Doorbell

## Features

All available CSV files for the Danmini Doorbell contain 115 input features.

## Available Traffic

### Benign

- Benign: 49,548 samples

### Gafgyt / BASHLITE

- Combo: 59,718
- Junk: 29,068
- Scan: 29,849
- TCP: 92,141
- UDP: 105,874

Total Gafgyt samples: 316,650

### Mirai

- ACK: 102,195
- Scan: 107,685
- SYN: 122,573
- UDP: 237,665
- UDPplain: 81,982

Total Mirai samples: 652,100

## Binary Distribution

- Benign: 49,548
- Attack: 968,750
- Total: 1,018,298

Approximate distribution:

- Benign: 4.87%
- Attack: 95.13%

The raw Danmini Doorbell data is therefore strongly imbalanced towards attack traffic.

## Initial Problem Formulation

The first experiment will use binary classification:

- 0 = Benign
- 1 = Attack

Both Gafgyt/BASHLITE and Mirai traffic will initially be treated as attack samples.

No final subset has been selected yet.

## Feature Characterization

Feature analysis was performed on the training subset only.

- Samples analyzed: 56,000
- Original features: 115
- NaN values: 0
- Infinite values: 0
- Constant features: 0
- Near-constant features: 0

Using an absolute Pearson correlation threshold of 0.99:

- Highly correlated feature pairs: 160
- Features retained: 60
- Features removed as redundant: 55

The selected 60-feature configuration will be compared experimentally against the full 115-feature baseline.