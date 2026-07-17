# report week 1 (the first implemental part of project)
# 14 jul 2026

In the first step, I installed all the required tools for developing projects with the ESP32-S3-DevKitC on Ubuntu Linux.

Then, I learned how PlatformIO works, including the process it follows from building a project to uploading the firmware to the ESP32. After understanding its workflow, I installed PlatformIO in Visual Studio Code and created my first project with it.

After that, I connected the ESP32-S3-DevKitC to my laptop and wrote a simple program to verify that the communication between the laptop and the board was working correctly.

I also tried to turn on the onboard LED by writing a simple program, but I realized that this specific version of the ESP32-S3-DevKitC does not include a user-programmable onboard LED.

After completing these steps, I still had some questions about the development process. I clarified them by reading the documentation and searching through reliable references.

Experience: Missing setup() Messages in the Serial Monitor

I figured out that the ESP32 does not ignore the messages inside setup(). The real problem is that the board starts running immediately after upload, while the Serial Monitor still needs some time to connect.

Because of this, messages printed at the beginning of setup() may be sent before the Serial Monitor is ready. However, messages inside loop() are visible because loop() keeps running repeatedly.

Adding a delay gives the Serial Monitor enough time to connect:

void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.println("Setup started");
}

Pressing the RESET button also works because the Serial Monitor is already open when the ESP32 starts again.

Point I learned: the message is not skipped; it is sent before the Serial Monitor starts listening.
