# Week 5 Report

This week, I completed the MNIST TinyML benchmark and continued working on CIFAR-10.

For MNIST, I trained and evaluated the model, converted it to Float32 and INT8 TensorFlow Lite formats, and deployed the INT8 model on the ESP32-S3 using TensorFlow Lite Micro.

The final results were:

* Keras accuracy: **97.75%**
* Float32 TFLite accuracy: **97.75%**
* INT8 TFLite accuracy: **97.82%**
* INT8 model size: **14,072 bytes**

I also tested the model directly on the ESP32-S3. The prediction matched the expected result.

The main ESP32 measurements were:

* Tensor arena allocated: **12,288 bytes**
* Tensor arena used: about **9,836 bytes**
* RAM usage: about **28.3 KB**
* Flash usage: about **277 KB**
* Average inference time: about **8.04 ms**

For the timing test, I used 5 warm-up runs followed by 100 measured inference runs.

After finishing MNIST, I continued with the CIFAR-10 benchmark.

I prepared and normalized the CIFAR-10 dataset and trained a small CNN designed to be suitable for later ESP32 deployment. The Keras model reached a test accuracy of **60.93%**.

I then converted the model to Float32 TFLite and fully quantized INT8 TFLite.

The results were:

| Model          | Accuracy |          Size |
| -------------- | -------: | ------------: |
| Keras          |   60.93% | 229,550 bytes |
| Float32 TFLite |   60.93% |  66,496 bytes |
| INT8 TFLite    |   61.10% |  24,192 bytes |

The INT8 version reduced the model size a lot while keeping almost the same accuracy, which is important for running the model on the ESP32.

At this point, the CIFAR-10 model has been trained, converted, quantized, and tested on the host machine.

## Next Step

The next step is to deploy the CIFAR-10 INT8 model on the ESP32-S3 and measure:

* inference correctness
* tensor arena usage
* RAM usage
* Flash usage
* inference time

After completing the CIFAR-10 benchmark, I will move to an IoT-specific anomaly-detection dataset.
