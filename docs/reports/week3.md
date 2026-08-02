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

In the next phase, I analyzed the training, validation, and test datasets before starting the neural-network training process. The main purpose was to confirm that the datasets had similar behavior and that all generated anomaly types were represented correctly.

I created a Python analysis script using pandas, NumPy, and Matplotlib. The script loaded the three datasets, calculated their main statistics, identified each synthetic anomaly type, and generated CSV reports and visual plots.

The analysis confirmed that the training dataset contained 6000 samples, while the validation and test datasets contained 3000 samples each. The number of normal and anomalous samples matched the expected anomaly-generation logic. All three anomaly types were present: temperature spikes, humidity drops, and maximum light intensity.

Time-series plots were generated for temperature, humidity, and light. These plots showed that each anomaly appeared at the expected sample positions. Temperature anomalies produced clear increases, humidity anomalies produced large decreases, and light anomalies reached the maximum value of 1023.

Histograms were also created to compare the distributions of normal and anomalous samples. The anomalous values formed separate regions from the normal values for their affected sensor. Scatter plots were used to examine the relationships between the features and showed that different anomaly types appeared in different parts of the feature space.

The normal feature distributions of the training, validation, and test datasets were also compared. Their means and standard deviations were similar, which confirmed that the datasets represented the same simulated system while still containing different random noise due to their different seeds.

At the end of this phase, the datasets were successfully analyzed and visually verified. No major problems were found, and the data are ready for feature preparation, normalization, and neural-network training.

In this stage of the project, I prepared the collected sensor data for neural-network training and deployed the final model on the ESP32-S3.

First, I selected temperature, humidity, and light as the model input features. The timestamp was excluded because the generated anomalies occurred at predictable time intervals, and using it could cause data leakage. The anomaly column was kept as the target label, where 0 represented normal data and 1 represented anomalous data.

The input features had different numerical ranges, especially the light values, which were much larger than the temperature and humidity values. For this reason, I applied z-score normalization using the mean and standard deviation calculated only from the training dataset. The same normalization parameters were then used for the validation and test datasets and were saved for later use on the ESP32.

After preprocessing, I created a small neural network with three input values, two hidden layers with 8 and 4 neurons, and one sigmoid output. The output represented the probability that the current sensor sample was anomalous. Class weights were used during training because the number of normal samples was much larger than the number of anomaly samples.

The model was trained using the training dataset and monitored using the validation dataset. Early stopping and model checkpointing were used to stop training when the validation loss stopped improving and to save the best model.

After training, I evaluated the model using precision, recall, F1-score, accuracy, a confusion matrix, and separate results for each anomaly type. The classification threshold was selected using the validation dataset, while the test dataset was kept separate for final evaluation.

The trained Keras model was then converted into both float32 and fully quantized int8 LiteRT models. The int8 model was selected for deployment because it required less memory and was more suitable for embedded inference. During this process, I found that the threshold selected for the float model was higher than the maximum output value of the int8 model. Therefore, I calculated a separate deployment threshold using the int8 validation predictions.

The int8 model, normalization parameters, feature order, and classification threshold were converted into C++ files and added to a separate PlatformIO ESP-IDF project. TensorFlow Lite Micro was used to load the model, allocate the tensor memory, normalize and quantize the sensor values, run inference, and convert the output back into an anomaly score.

Finally, I implemented a real-time inference loop on the ESP32-S3. The device generated temperature, humidity, and light samples every 100 milliseconds, passed them into the neural network, and printed the ground-truth label, predicted anomaly, anomaly score, and inference time through the Serial port.

At the end of this stage, the complete pipeline from feature preprocessing and model training to int8 quantization and real-time ESP32 inference was implemented successfully.
