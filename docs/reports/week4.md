## Weekly Progress Report

During this week, I completed the remaining analysis of the synthetic sensor datasets and validated the first complete TinyML prototype of the project.

First, I completed the analysis and visualization of the training, validation, and test datasets. The analysis included class distributions, sensor-value distributions, time-series plots, feature relationships, anomaly-type counts, and statistical comparisons between the three datasets. The results confirmed that the normal data distributions were consistent across the training, validation, and test sets and that all three synthetic anomaly types were represented correctly.

The training dataset contains 6000 samples, while the validation and test datasets contain 3000 samples each. Approximately 5% of the samples are anomalous. The analysis also confirmed that the three anomaly types—temperature spikes, humidity drops, and light saturation—appear as expected in the generated datasets.

After validating the data, I completed the first anomaly-detection machine-learning pipeline. Temperature, humidity, and light were used as the model inputs, while the anomaly label was used as the prediction target. The data were normalized using statistics calculated only from the training dataset, and the same normalization parameters were reused for validation, testing, and embedded inference.

A small neural network was trained to classify each sensor sample as normal or anomalous. The model contains only 73 trainable parameters, making it suitable for experimentation on a resource-constrained device such as the ESP32-S3.

The trained model was evaluated using the validation and test datasets. The classification threshold was selected using validation data, while the test dataset was kept separate for final evaluation. On the current synthetic dataset, the model correctly classified all test samples. Because the dataset is generated and the anomaly patterns are deliberately well separated from the normal data, these results are considered a validation of the pipeline rather than evidence of real-world anomaly-detection performance.

The trained Keras model was then converted to TensorFlow Lite in both float32 and fully quantized int8 formats. The original Keras model was approximately 27.9 KB, the float32 TensorFlow Lite model was approximately 2.8 KB, and the int8 model was approximately 3.3 KB.

The converted models were evaluated again on the host computer to verify that model conversion and quantization did not change the expected predictions. A separate deployment threshold was also selected for the int8 model because quantization changes the possible output values of the model.

The int8 model, normalization parameters, feature order, and anomaly threshold were then converted into C++ source files and integrated into an ESP32-S3 firmware project using TensorFlow Lite Micro.

The ESP32 firmware was successfully built and the model was able to perform inference on the device. The current firmware build uses approximately 48.5 KB of RAM and 216.6 KB of Flash, while the TensorFlow Lite Micro tensor arena is configured with 32 KB of memory. These measurements provide an initial reference for later TinyML benchmarking.

In addition to the machine-learning work, I reorganized the repository so that source code, datasets, generated artifacts, and firmware are clearly separated. The existing anomaly-detection implementation is now maintained as an initial prototype, while the repository structure is prepared for the upcoming MNIST and CIFAR-10 experiments.

I also reviewed the main TinyML deployment options for ESP32 development. For the next stages, the project will continue using a TensorFlow/Keras to int8 TensorFlow Lite to TensorFlow Lite Micro workflow, with ESP32-specific optimizations considered where appropriate.

At the end of this stage, the first end-to-end anomaly-detection prototype—from dataset preparation and model training to quantization and ESP32 inference—has been successfully validated.

The next phase will follow the benchmark workflow suggested by Professor La Rosa. I will begin with MNIST to study the complete TinyML workflow in a controlled environment, including training, model optimization, deployment on the ESP32-S3, model size, memory usage, and inference time. After completing the MNIST experiment, I will continue with CIFAR-10 before returning to the project-specific anomaly-detection dataset.
