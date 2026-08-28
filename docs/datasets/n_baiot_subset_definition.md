# N-BaIoT Initial Subset Definition

## Objective

Create a smaller and reproducible dataset for the first binary
benign-vs-attack experiment.

## Device

Danmini Doorbell

## Classification

Binary classification:

- 0 = Benign
- 1 = Attack

## Features

All 115 original N-BaIoT features will initially be retained.

Feature reduction will be evaluated after establishing the first baseline.

## Total Dataset Size

80,000 samples

- Benign: 40,000
- Attack: 40,000

## Attack Distribution

Each attack subtype contributes 4,000 samples.

### Gafgyt / BASHLITE

- Combo: 4,000
- Junk: 4,000
- Scan: 4,000
- TCP: 4,000
- UDP: 4,000

### Mirai

- ACK: 4,000
- Scan: 4,000
- SYN: 4,000
- UDP: 4,000
- UDPplain: 4,000

## Training Set

Total: 56,000

- Benign: 28,000
- Attack: 28,000
- Each attack subtype: 2,800

## Validation Set

Total: 12,000

- Benign: 6,000
- Attack: 6,000
- Each attack subtype: 600

## Test Set

Total: 12,000

- Benign: 6,000
- Attack: 6,000
- Each attack subtype: 600

## Split Strategy

The source CSV files are sequential and will not be randomly mixed before splitting.

For each source file, samples will be selected from separated regions of the sequence:

- Training samples from an early region
- Validation samples from a later region
- Test samples from the final region

Unused gaps will be kept between these regions where possible to reduce temporal similarity between the splits.

For every attack subtype:

- Training: 2,800 samples
- Validation: 600 samples
- Test: 600 samples

For benign traffic:

- Training: 28,000 samples
- Validation: 6,000 samples
- Test: 6,000 samples

The exact row ranges used for every source file will be generated and recorded by the subset-generation script.

## Reproducibility

The subset will be generated automatically by a Python script.

The script will record:

- Device
- Source files
- Sample counts
- Split strategy
- Feature names
- Class labels
- Dataset shapes