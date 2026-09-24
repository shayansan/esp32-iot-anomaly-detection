# Week 8 Report

This week, I continued the N-BaIoT binary anomaly-detection experiment and moved from dataset preparation into preprocessing, model training, evaluation, and deployment preparation.

I first created a preprocessing pipeline for both feature configurations:

Full version with 115 features
Reduced version with 60 selected features
For both versions, standardization parameters were calculated using only the training data in order to avoid data leakage. The same training mean and standard deviation were then applied to the validation and test sets.
The preprocessing results were checked for all splits. No NaN or infinite values were found, and the class balance remained correct:
Training: 28,000 benign / 28,000 attack
Validation: 6,000 benign / 6,000 attack
Test: 6,000 benign / 6,000 attack
After preprocessing, I trained two neural-network baseline models using the same architecture so that the feature configurations could be compared fairly.

The network architecture was:
Input layer
Dense layer with 32 neurons and ReLU
Dense layer with 16 neurons and ReLU
Output layer with 1 neuron and Sigmoid
The full 115-feature model used 4,257 parameters, while the reduced 60-feature version used 2,497 parameters.
The final test results were very similar for both models.
For the 115-feature model:
Accuracy: 99.91%
Precision: 99.90%
Recall: 99.92%
F1-score: 99.91%
False positives: 6
False negatives: 5
For the 60-feature model:
Accuracy: 99.85%
Precision: 99.78%
Recall: 99.92%
F1-score: 99.85%
False positives: 13
False negatives: 5
The reduced model therefore kept almost the same attack-detection performance while using significantly fewer input features and fewer model parameters.
I then converted both trained Keras models into TensorFlow Lite Float32 and full INT8 versions.
The model sizes were:
115-feature model
Float32 TFLite: 19,176 bytes
INT8 TFLite: 8,040 bytes
60-feature model
Float32 TFLite: 12,136 bytes
INT8 TFLite: 6,280 bytes
I also evaluated all TFLite models on the full test set to check whether conversion or quantization caused a significant loss in detection performance.
The 60-feature INT8 model achieved:
Accuracy: 99.83%
Precision: 99.77%
Recall: 99.90%
F1-score: 99.83%
False positives: 14
False negatives: 6
This shows that full INT8 quantization caused only a very small change in performance while reducing the model size to about 6.3 KB.
Based on these results, the 60-feature INT8 model is currently the main candidate for ESP32 deployment because it provides a good trade-off between detection performance and embedded resource requirements.
At the end of the week, I started preparing this model for TensorFlow Lite Micro deployment by converting the .tflite model into a C/C++ byte array that can be included directly in the ESP32 firmware.
The next step is to build the N-BaIoT TensorFlow Lite Micro firmware, run a fixed test sample on the ESP32, verify host-device prediction parity, and then measure tensor arena usage, RAM, Flash, and inference latency.

