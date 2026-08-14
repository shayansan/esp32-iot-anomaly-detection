## Weekly Progress Report

During this week, I completed the full MNIST TinyML benchmark workflow on the ESP32-S3 and started the CIFAR-10 benchmark.

For MNIST, I inspected and preprocessed the dataset, trained a compact CNN with 7,834 parameters, and achieved approximately 97.75% test accuracy. The model was then converted to float32 and fully quantized int8 TensorFlow Lite formats. The int8 model achieved approximately 97.82% accuracy and was reduced to about 14.1 KB.

The int8 model was successfully deployed on the ESP32-S3 using TensorFlow Lite Micro. A test sample with expected label 7 was correctly classified as 7. I also measured the memory requirements and reduced the tensor arena from 64 KB to 12 KB after confirming that the model used approximately 9.8 KB.

With the final configuration, the firmware used approximately 28.3 KB of RAM and 277.2 KB of Flash. A latency benchmark with 5 warm-up runs and 100 measured inferences produced an average inference time of approximately 8.04 ms at 160 MHz, with very stable results between runs.

After completing the MNIST benchmark, I started the CIFAR-10 experiment. The dataset was inspected and preprocessed, and a compact CNN was trained as an embedded-oriented baseline. The current Keras model achieved approximately 60.93% test accuracy, correctly classifying 6,093 out of 10,000 test images. The model size is approximately 229.6 KB.

The next step is to convert the CIFAR-10 model to float32 and fully quantized int8 TensorFlow Lite formats, evaluate the converted models on the host, and then deploy the int8 model on the ESP32-S3 to measure memory usage and inference latency.
