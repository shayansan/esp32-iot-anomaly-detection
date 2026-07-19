# difference between delay command and millis:

This week, I learned the difference between blocking and non-blocking timing on the ESP32. The delay() function stops the program for a specific time, so other tasks cannot run during that period. In contrast, millis() checks how much time has passed without stopping the program. This allows the ESP32 to read sensors, maintain Wi-Fi, receive MQTT messages, and perform other tasks at the same time. I also learned the standard timing condition:
if (millis() - previousTime >= interval)
