#include <ESP_I2S.h>

const int PIN_BCLK = 25;
const int PIN_WS = 26;
const int PIN_DIN = 4;
const int PIN_LED = 2;

const uint32_t SAMPLE_RATE = 16000;
const uint32_t RECORD_SECONDS = 3;
const size_t NUM_SAMPLES = SAMPLE_RATE * RECORD_SECONDS;
const size_t CHUNK = 256;

I2SClass i2s;
int16_t *audio = nullptr;
int32_t chunkBuf[CHUNK];

void record() {
  digitalWrite(PIN_LED, HIGH);
  for (int i = 0; i < 10; i++) {
    i2s.readBytes((char *)chunkBuf, sizeof(chunkBuf));
  }
  size_t n = 0;
  while (n < NUM_SAMPLES) {
    size_t got = i2s.readBytes((char *)chunkBuf, sizeof(chunkBuf));
    size_t count = got / sizeof(int32_t);
    for (size_t i = 0; i < count && n < NUM_SAMPLES; i++) {
      audio[n++] = (int16_t)(chunkBuf[i] >> 16);
    }
  }
  digitalWrite(PIN_LED, LOW);
}

void printStats() {
  int32_t mn = 32767, mx = -32768;
  double sum = 0;
  for (size_t i = 0; i < NUM_SAMPLES; i++) {
    int32_t s = audio[i];
    if (s < mn) mn = s;
    if (s > mx) mx = s;
    sum += s;
  }
  double mean = sum / NUM_SAMPLES;
  double sumSq = 0;
  for (size_t i = 0; i < NUM_SAMPLES; i++) {
    double d = audio[i] - mean;
    sumSq += d * d;
  }
  double rms = sqrt(sumSq / NUM_SAMPLES);
  double peak = fmax(fabs((double)mn), fabs((double)mx));
  double rmsDb = 20.0 * log10(fmax(rms, 1e-9) / 32768.0);
  double peakDb = 20.0 * log10(fmax(peak, 1.0) / 32768.0);

  Serial.printf("min %ld  max %ld  mean %.1f  rms %.1f (%.1f dBFS)  peak %.1f dBFS\n", (long)mn, (long)mx, mean, rms, rmsDb, peakDb);
  Serial.printf("raw words: %08lx %08lx %08lx %08lx\n",
                (unsigned long)chunkBuf[0], (unsigned long)chunkBuf[1],
                (unsigned long)chunkBuf[2], (unsigned long)chunkBuf[3]);
}

void sendAudio() {
  uint32_t bytes = NUM_SAMPLES * sizeof(int16_t);
  Serial.write((const uint8_t *)"AUDIO", 5);
  Serial.write((const uint8_t *)&bytes, 4);
  Serial.write((const uint8_t *)audio, bytes);
  Serial.flush();
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_LED, OUTPUT);
  delay(500);

  audio = (int16_t *)malloc(NUM_SAMPLES * sizeof(int16_t));
  if (!audio) {
    Serial.println("ERROR: not enough memory for the audio buffer");
    while (true) delay(1000);
  }

  i2s.setPins(PIN_BCLK, PIN_WS, -1, PIN_DIN);
  if (!i2s.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT,
                 I2S_SLOT_MODE_MONO, I2S_STD_SLOT_RIGHT)) {
    Serial.println("ERROR: I2S failed to start");
    while (true) delay(1000);
  }

  Serial.printf("Ready. Free heap: %lu bytes\n", (unsigned long)ESP.getFreeHeap());
  Serial.println("Send t to test levels, r to record and send audio.");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 't') {
      record();
      printStats();
    } else if (c == 'r') {
      record();
      sendAudio();
    }
  }
}