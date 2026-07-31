In this stage of the project, I created the first part of the data pipeline required for training and deploying an anomaly-detection model on the ESP32-S3.

First, I implemented an ESP32 program that generates simulated temperature, humidity, and light sensor values every 100 milliseconds. The normal values contain periodic changes and random noise to make the generated data more realistic. The program also generates three types of anomalies: a sudden temperature increase, a sudden humidity decrease, and maximum light intensity.

Each generated sample is printed through the Serial port in CSV format. The output contains the timestamp, temperature, humidity, light level, and anomaly label. The anomaly label is 0 for normal samples and 1 for anomalous samples.

To create separate datasets, I configured three PlatformIO environments with different random seeds:

Training dataset: seed 42
Validation dataset: seed 123
Test dataset: seed 999

Using different seeds ensures that the datasets have similar behavior but different random noise. This prevents the model from being trained and tested on exactly the same generated sequence.

I also created a Python virtual environment and installed the required libraries, including pyserial, pandas, and numpy. A Python script was implemented to read the ESP32 Serial output, validate each received line, and save valid samples into a CSV file.

Before collecting the complete datasets, I performed a smoke test with 300 samples. The collected dataset had the expected structure with five columns, no missing values, and correct numeric data types. It contained 290 normal samples and 10 anomalous samples, which matched the anomaly-generation logic in the ESP32 code.

A dataset-validation script was also prepared to automatically check the CSV structure, number of rows, labels, sensor limits, timestamps, missing values, anomaly counts, and the presence of all three anomaly types.

At the end of this stage, the ESP32 data generator, Serial data collector, reproducible PlatformIO configurations, and basic dataset-validation pipeline were completed successfully. The next step is to collect the complete training, validation, and test datasets before starting neural-network training.
