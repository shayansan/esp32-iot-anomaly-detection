# Week 6 Report

This week, I continued the CIFAR-10 TinyML experiment by deploying the INT8 model on the ESP32-S3.

I created a separate TensorFlow Lite Micro firmware project for CIFAR-10 and adapted the previous MNIST deployment code for the CIFAR-10 model, input data, and output classes.

During the first test, the initial 12 KB tensor arena was too small and `AllocateTensors()` failed. I temporarily increased it to 64 KB and confirmed that the model could initialize and run correctly.

The ESP32 correctly classified the test sample:

* Expected class: 3
* Predicted class: 3
* Result: PASS

The actual tensor arena usage was 21,276 bytes. I then reduced the allocated arena step by step. Both 24 KB and 22 KB worked correctly, and I also tested 21 KB successfully. Since 21 KB leaves only 228 bytes of free space, I decided to keep 22 KB as the final configuration to provide a small safety margin.

The final benchmark results are:

* INT8 model size: 24,192 bytes
* Tensor arena used: 21,276 bytes
* Minimum tested working arena: 21,504 bytes
* Final allocated arena: 22,528 bytes
* Firmware RAM with 22 KB arena: 38,540 bytes
* Firmware Flash usage: 291,209 bytes
* CPU frequency: 160 MHz
* PSRAM: not used

I also ran 100 inference measurements. The results were:

* Minimum inference time: 18.566 ms
* Maximum inference time: 18.591 ms
* Average inference time: about 18.580 ms

At this point, the CIFAR-10 TinyML workflow has been completed from training and INT8 quantization to ESP32 deployment, correctness testing, memory measurement, and latency benchmarking.