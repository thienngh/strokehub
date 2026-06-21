// ESP32 FSR grip-strength stream for save_trials.py

const int FSR_PIN = 34;

const unsigned long SAMPLE_INTERVAL_US = 12500; // 80 Hz
unsigned long nextSampleUs = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  analogReadResolution(12);
  nextSampleUs = micros();

  Serial.println("FSR_READY");
}

void loop() {
  unsigned long currentUs = micros();

  if ((long)(currentUs - nextSampleUs) >= 0) {
    nextSampleUs += SAMPLE_INTERVAL_US;

    int fsrRaw = analogRead(FSR_PIN);

    Serial.print("FSR,");
    Serial.print(currentUs);
    Serial.print(",");
    Serial.println(fsrRaw);
  }
}