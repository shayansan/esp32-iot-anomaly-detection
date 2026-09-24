# Week 9 Report

This week, I focused on deploying the reduced N-BaIoT INT8 model on the ESP32-S3 using TensorFlow Lite Micro.

The selected deployment model is the reduced 60-feature binary classifier. The full INT8 TFLite model is 6,280 bytes and uses only two unique TensorFlow Lite operators:

* Fully Connected
* Logistic

I first converted the INT8 TFLite model into a C/C++ byte array so that it could be included directly in the ESP32 firmware.

I also generated a fixed INT8 test sample from the N-BaIoT test set. On the host, this sample had:

* Expected label: Attack (1)
* Host predicted label: Attack (1)
* Output probability: approximately 0.996

This same quantized input was then used on the ESP32 to check host-device prediction parity.

I created a new TensorFlow Lite Micro firmware project for N-BaIoT based on the previously tested ESP32-S3 setup. The firmware loads the INT8 model, registers only the required operators, allocates the tensor arena, copies the fixed test input into the input tensor, runs inference, dequantizes the output, and compares the ESP32 prediction with the host result.

The first deployment was successful.

The ESP32 produced:

* Expected label: 1
* Host predicted label: 1
* ESP32 predicted label: 1
* Result: PASS

This confirmed host-device inference parity for the fixed sample.

I then added an on-device latency benchmark using 5 warm-up runs followed by 100 measured inference runs.

With the initial 16 KB tensor arena, the results were:

* Model size: 6,280 bytes
* Tensor arena used: 1,228 bytes
* Tensor arena allocated: 16,384 bytes
* RAM: 32,140 bytes
* Flash: 217,109 bytes
* Average inference latency: approximately 77.4 microseconds
* CPU frequency: 160 MHz

Because the actual tensor arena usage was much smaller than the initial 16 KB allocation, I started reducing the arena size to find the minimum practical value.

The following configurations were tested:

* 16,384 bytes: PASS
* 4,096 bytes: PASS
* 2,048 bytes: PASS
* 1,792 bytes: PASS
* 1,664 bytes: PASS
* 1,536 bytes: FAIL

At 1,536 bytes, TensorFlow Lite Micro failed during `AllocateTensors()`, so the minimum working arena is currently known to be between 1,536 and 1,664 bytes.

At 1,664 bytes, the model still passed the correctness test and achieved:

* Tensor arena used: 1,228 bytes
* RAM: 17,420 bytes
* Flash: 217,109 bytes
* Average latency: approximately 77.5 microseconds

This means the RAM requirement was significantly reduced compared with the initial 16 KB arena, while model size, Flash usage, prediction correctness, and inference latency remained essentially unchanged.

The next step is to continue narrowing the tensor-arena boundary, choose a safe final arena size, save the final benchmark results, and document the N-BaIoT ESP32 deployment in the repository.
