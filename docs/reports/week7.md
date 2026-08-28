
# Week 7 Report

This week, I started working with the N-BaIoT dataset for the IoT anomaly-detection part of the project.

I first compared N-BaIoT with TON_IoT and selected N-BaIoT as the initial dataset because it is directly focused on IoT botnet traffic and fits the goal of the project well.

For the first experiment, I selected the Danmini Doorbell device. It contains benign traffic together with both Gafgyt/BASHLITE and Mirai attacks. I inspected the available CSV files and confirmed that all of them use the same 115 numerical input features.

I also analyzed the class distribution for this device. The original data contains 49,548 benign samples and 968,750 attack samples, so the raw dataset is highly imbalanced towards attack traffic.

Following this analysis, I defined the first experiment as a binary classification problem:

* 0 = Benign
* 1 = Attack

All selected Gafgyt and Mirai attack types are grouped into the Attack class.

To create a more manageable and reproducible first experiment, I prepared a subset containing 80,000 samples:

* 40,000 benign samples
* 40,000 attack samples

The attack samples are distributed equally between the ten available attack subtypes, with 4,000 samples from each subtype.

The subset was divided into:

* Training: 56,000 samples
* Validation: 12,000 samples
* Test: 12,000 samples

Instead of randomly splitting the original sequential traffic data, I selected separated regions from each source file for training, validation, and testing. Unused gaps were left between the regions where possible to reduce the risk of data leakage. The final samples are shuffled only after the split has been created.

I implemented a Python script to generate this subset automatically and save metadata describing the source files, row ranges, class distribution, feature names, and split configuration.

After creating the subset, I started analyzing the 115 input features using only the training data. The analysis showed:

* No missing values
* No infinite values
* No constant features
* No near-constant features

I then performed a correlation analysis to identify redundant features. Using an absolute Pearson correlation threshold of 0.99, I found 160 highly correlated feature pairs.

Based on this analysis, I created a reproducible feature-selection procedure. It reduced the input from 115 features to 60 features by removing 55 highly correlated features.

For the next experiments, I plan to compare two configurations:

* Full model using all 115 features
* Reduced model using the selected 60 features

This will allow me to check whether the reduced input can maintain similar detection performance while reducing the embedded resource requirements on the ESP32.

I also documented the dataset selection, device choice, sample distribution, subset design, split strategy, and feature-analysis decisions in the repository.

At this point, the dataset characterization and the initial subset definition are mostly completed. The next step is to prepare the preprocessing pipeline for the 115-feature and 60-feature versions and then start training the first N-BaIoT binary detection models.
