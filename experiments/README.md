# Classical Machine Learning on ESP32

In this experiment, I started learning how classical machine learning can be used on an ESP32.

The main idea I learned is that the model is usually trained on a computer because training needs more memory and processing power. After training, only the learned parameters are transferred to the ESP32, where the device performs inference on new sensor data.

For the first experiment, I focused on time-series data and K-means clustering. Time-series data is a sequence of sensor measurements collected over time. Instead of analysing only one sample, I collect multiple samples inside a fixed-size window.

From each window, I can extract simple statistical features such as:

* Mean
* Minimum
* Maximum
* Standard deviation

These features describe the behaviour of the signal while reducing the amount of data that the model needs to process.

K-means is an unsupervised machine learning algorithm. It groups similar data samples into clusters without requiring predefined labels. Each cluster has a centre called a centroid.

For anomaly detection, the model can be trained using normal sensor data. When the ESP32 receives a new window of data, it extracts the same features and calculates the distance between the new feature vector and the cluster centroids.

If the distance from the nearest centroid is higher than a selected threshold, the sample can be considered anomalous.

The complete workflow is:

1. Collect time-series sensor data.
2. Divide the data into fixed-size windows.
3. Extract statistical features from each window.
4. Scale the features.
5. Train the K-means model on a computer using Python and scikit-learn.
6. Export the centroids and scaling parameters to C++.
7. Use the ESP32 to calculate the features and perform anomaly detection.

This experiment helped me understand the difference between training and inference. It also showed me how a lightweight classical machine learning model can be used on a resource-constrained embedded device such as the ESP32.

The ESP32 will print information such as:

* Extracted feature values
* Nearest cluster
* Distance from the centroid
* Normal or anomaly result

This is the first step toward implementing a complete IoT anomaly-detection system on the ESP32.
