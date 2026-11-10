import argparse
import re
import struct
import sys
import time
import wave
from array import array
from pathlib import Path

import serial
import serial.tools.list_ports

BAUD = 115200
SAMPLE_RATE = 16000
MIC = "mems"  # mic tag in the filename
MARKER = b"AUDIO"
MAX_SECONDS = 10  # must match MAX_SECONDS in the firmware
SEND_CHUNK = 256  # must match SEND_CHUNK in the firmware; we ack each chunk with b"A"
MARKER_TIMEOUT_S = 15
IDLE_TIMEOUT_S = 3.0
OUT_DIR = Path(__file__).parent.parent / "dataset" / "raw" / "recordings"


def find_port(requested):
    if requested:
        return requested
    names = ("usbserial", "wchusbserial", "SLAB_USBtoUART", "usbmodem",  # macOS /dev/cu.*
             "ttyUSB", "ttyACM")                                          # Linux
    ports = [p.device for p in serial.tools.list_ports.comports()
             if not p.device.startswith("/dev/tty.") and any(n in p.device for n in names)]
    if len(ports) == 1:
        return ports[0]
    if not ports:
        sys.exit("No board found. Plug it in, or pass --port /dev/cu.xxx")
    sys.exit("Several ports found, pass one with --port:\n  " + "\n  ".join(ports))


def read_exact(ser, n, total_timeout):
    data = bytearray()
    start = last = time.time()
    while len(data) < n:
        chunk = ser.read(n - len(data))
        now = time.time()
        if chunk:
            data += chunk
            last = now
        if now - last > IDLE_TIMEOUT_S:
            raise RuntimeError(f"stalled: got {len(data)} of {n} bytes")
        if now - start > total_timeout:
            raise RuntimeError(f"too slow: got {len(data)} of {n} bytes in {total_timeout:.0f} s")
    return bytes(data)


def wait_for_marker(ser, seconds):
    window = b""
    seen = bytearray()
    deadline = time.time() + seconds + MARKER_TIMEOUT_S
    while time.time() < deadline:
        b = ser.read(1)
        if b:
            seen += b
            window = (window + b)[-len(MARKER):]
            if window == MARKER:
                return
    heard = seen[-200:].decode("ascii", errors="replace").strip() or "nothing"
    raise RuntimeError(f"timed out waiting for audio. Board said: {heard!r}. "
                       "Check the latest firmware is flashed and the Serial Monitor is closed.")


def record_one(ser, seconds):
    tenths = round(seconds * 10)
    ser.reset_input_buffer()
    ser.write(b"R" + bytes([tenths]))
    wait_for_marker(ser, seconds)
    length = struct.unpack("<I", read_exact(ser, 4, 5))[0]
    expected = SAMPLE_RATE // 10 * 2 * tenths
    if length != expected:
        raise RuntimeError(f"bad length {length}, expected {expected}")
    # The firmware sends SEND_CHUNK bytes, then waits (up to 2 s) for an b"A"
    # ack before sending the next chunk, including after the last one.
    audio = bytearray()
    while len(audio) < length:
        n = min(SEND_CHUNK, length - len(audio))
        audio += read_exact(ser, n, 5)
        ser.write(b"A")
    return bytes(audio)


def next_index(out_dir, prefix):
    used = [int(p.stem.rsplit("_", 1)[1]) for p in out_dir.glob(f"{prefix}_*.wav")
            if p.stem.rsplit("_", 1)[1].isdigit()]
    return max(used, default=-1) + 1


def save_wav(path, audio):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio)


def main():
    ap = argparse.ArgumentParser(description="Record a batch of fixed-length clips from the ESP32 mic.")
    ap.add_argument("--num", type=int, required=True, help="how many recordings")
    ap.add_argument("--length", type=float, default=1.5, help=f"seconds per recording, in 0.1 steps (0.1-{MAX_SECONDS})")
    ap.add_argument("--label", required=True, choices=("ww", "sp", "bg"),
                    help="ww = wake word, sp = other speech, bg = background")
    ap.add_argument("--note", help="describe the setting of recording, e.g. kitchen (letters, digits, hyphens)")
    ap.add_argument("--gap", type=float, default=0.0, help="countdown seconds before each recording")
    ap.add_argument("--port", help="serial port (auto-detected if omitted)")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="folder to save clips in (default: dataset/raw/recordings)")
    args = ap.parse_args()

    if args.note and not re.fullmatch(r"[A-Za-z0-9-]+", args.note):
        sys.exit("--note may only contain letters, digits and hyphens (no underscores or spaces)")
    if args.num < 1:
        sys.exit("--num must be at least 1")
    if not 0.1 <= args.length <= MAX_SECONDS or abs(args.length * 10 - round(args.length * 10)) > 1e-6:
        sys.exit(f"--length must be 0.1-{MAX_SECONDS} in steps of 0.1")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "_".join(filter(None, (args.label, MIC, args.note)))
    start_idx = next_index(out_dir, prefix)

    port = find_port(args.port)
    print(f"Opening {port}, saving to {out_dir}")
    ser = serial.Serial(port, BAUD, timeout=1)
    time.sleep(2.5)  # board resets when the port opens

    saved = 0
    try:
        for i in range(args.num):
            print(f"\n[{i + 1}/{args.num}] get ready...")
            for t in range(int(args.gap), 0, -1):
                print(f"  {t}...", flush=True)
                time.sleep(1)
            print(f"  RECORDING {args.length} s", flush=True)
            try:
                audio = record_one(ser, args.length)
            except RuntimeError as e:
                print(f"  failed: {e}; skipping")
                continue

            samples = array("h")
            samples.frombytes(audio)
            peak = max(abs(min(samples)), abs(max(samples)))
            path = out_dir / f"{prefix}_{start_idx + saved:04d}.wav"
            save_wav(path, audio)
            saved += 1
            note = "  (clipped!)" if peak >= 32767 else "  (near silent!)" if peak < 50 else ""
            print(f"  saved {path.name}, peak {peak}{note}")
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        ser.close()

    print(f"\nDone: {saved}/{args.num} clips in {out_dir}")


if __name__ == "__main__":
    main()