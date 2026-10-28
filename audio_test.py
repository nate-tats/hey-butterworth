import math
import sys
import time
import wave
import struct
from array import array

import serial
import serial.tools.list_ports

BAUD = 115200
SAMPLE_RATE = 16000
FULL_SCALE = 32768
OUTFILE = "test_recording2.wav"
MARKER = b"AUDIO"
MAX_BYTES = SAMPLE_RATE * 2 * 20
MARKER_TIMEOUT_S = 15
IDLE_TIMEOUT_S = 3.0
TOTAL_TIMEOUT_S = 60.0


def to_dbfs(x):
    return 20 * math.log10(max(x, 1e-9) / FULL_SCALE)


def find_port():
    if len(sys.argv) > 1:
        return sys.argv[1]
    names = ("usbserial", "wchusbserial", "SLAB_USBtoUART", "usbmodem")
    ports = [p.device for p in serial.tools.list_ports.comports()
             if p.device.startswith("/dev/cu.") and any(n in p.device for n in names)]
    if len(ports) == 1:
        return ports[0]
    if not ports:
        sys.exit("No board found. Plug it in, or pass the port: python audio_test.py /dev/cu.xxx")
    sys.exit("Several ports found, pass one of these:\n  " + "\n  ".join(ports))


def read_exact(ser, n):
    data = bytearray()
    start = last = time.time()
    while len(data) < n:
        chunk = ser.read(n - len(data))
        now = time.time()
        if chunk:
            data += chunk
            last = now
        if now - last > IDLE_TIMEOUT_S:
            sys.exit(f"Stalled: got {len(data)} of {n} bytes, nothing new for {IDLE_TIMEOUT_S} s")
        if now - start > TOTAL_TIMEOUT_S:
            sys.exit(f"Too slow: got {len(data)} of {n} bytes in {TOTAL_TIMEOUT_S} s")
    return bytes(data)


def wait_for_marker(ser):
    window = b""
    deadline = time.time() + MARKER_TIMEOUT_S
    while time.time() < deadline:
        b = ser.read(1)
        if b:
            window = (window + b)[-len(MARKER):]
            if window == MARKER:
                return
    sys.exit("Timed out waiting for audio. Check the sketch is uploaded and the Serial Monitor is closed.")


def main():
    port = find_port()
    print(f"Opening {port} at {BAUD} baud")
    ser = serial.Serial(port, BAUD, timeout=1)

    time.sleep(2.5)
    ser.reset_input_buffer()

    print("Recording 3 seconds, speak now...")
    ser.write(b"r")

    wait_for_marker(ser)
    raw_len = read_exact(ser, 4)
    length = struct.unpack("<I", raw_len)[0]
    print(f"Marker found, length bytes {raw_len.hex()} -> {length}")
    if length == 0 or length > MAX_BYTES or length % 2:
        sys.exit("Bad length, so bytes were lost before this point. Try another USB cable or port.")

    t0 = time.time()
    audio = read_exact(ser, length)
    dt = time.time() - t0
    print(f"Received {len(audio)} bytes in {dt:.1f} s ({len(audio) / dt / 1000:.1f} KB/s)")
    ser.close()

    with wave.open(OUTFILE, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio)

    samples = array("h")
    samples.frombytes(audio)
    mean = sum(samples) / len(samples)
    rms = (sum((s - mean) ** 2 for s in samples) / len(samples)) ** 0.5
    peak = max(abs(min(samples)), abs(max(samples)))

    print(f"Saved {OUTFILE}: {len(samples)} samples, {len(samples) / SAMPLE_RATE:.2f} s")
    print(f"min {min(samples)}  max {max(samples)}  mean {mean:.1f}  rms {rms:.1f}")
    print(f"rms {to_dbfs(rms):.1f} dBFS   peak {to_dbfs(peak):.1f} dBFS")
    if peak >= 32767:
        print("Warning: clipped, move back from the mic")
    if rms < 5:
        print("Warning: almost silent, check the mic wiring and L/R pin")


if __name__ == "__main__":
    main()