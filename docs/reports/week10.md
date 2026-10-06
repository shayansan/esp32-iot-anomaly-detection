# Week 10 Report

This week, I continued optimizing and validating the N-BaIoT ESP32 deployment.

The main focus was to determine a more accurate minimum tensor-arena size for the reduced 60-feature INT8 model.

At the end of last week, I had already confirmed that 1,664 bytes worked while 1,536 bytes failed. This week, I continued narrowing this range by testing smaller arena sizes.

The additional results were:

* 1,632 bytes: PASS
* 1,616 bytes: PASS
* 1,608 bytes: PASS
* 1,600 bytes: FAIL

This means that the minimum tested working tensor arena is now 1,608 bytes, while 1,600 bytes is the largest tested configuration that fails during `AllocateTensors()`.

Although 1,608 bytes works, I decided not to use the absolute minimum as the final configuration. Instead, I selected 1,792 bytes as the recommended tensor-arena size to keep a small safety margin.

With the recommended 1,792-byte arena, the firmware results were approximately:

* Model size: 6,280 bytes
* Tensor arena used: 1,228 bytes
* Tensor arena allocated: 1,792 bytes
* RAM: 17,548 bytes
* Flash: about 217 KB
* Average inference latency: about 77.5 microseconds
* CPU frequency: 160 MHz

Compared with the initial 16 KB tensor arena, the RAM usage was significantly reduced while the model size, Flash usage, prediction result, and inference latency remained almost unchanged.

I also extended the host-device parity test. Previously, I had only tested an attack sample. This week, I generated a benign sample and ran the same test on both the host and the ESP32.

For the benign sample:

* Expected label: Benign (0)
* Host predicted label: 0
* ESP32 predicted label: 0
* Output probability: 0.0
* Result: PASS

The attack sample had already passed previously:

* Expected label: Attack (1)
* Host predicted label: 1
* ESP32 predicted label: 1
* Result: PASS

This means that host-device parity has now been confirmed for both classes.

I also created a JSON benchmark artifact containing the final deployment information, including:

* ESP32 board and CPU frequency
* Model size
* Minimum tested tensor-arena size
* Largest tested failing arena size
* Recommended arena size
* RAM and Flash usage
* Host-device correctness results
* Latency benchmark results

At this point, the first N-BaIoT binary baseline has been completed end-to-end from dataset preparation and model training to INT8 quantization, ESP32 deployment, memory optimization, correctness testing, and latency benchmarking.

The next step is to finalize the repository documentation and then decide on the next N-BaIoT experiment, such as extending the work to multiclass attack classification, testing generalization across different IoT devices, or studying the practical cost of computing the selected input features in a real embedded deployment.
