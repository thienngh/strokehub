#!/usr/bin/env python3
import argparse
import csv
import json
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import serial
from scipy.signal import butter, sosfiltfilt, welch


TRIAL_DURATION_S = 20
FSR_CALIBRATION_S = 2
FSR_MIN_ZERO_THRESHOLD = 500
FSR_MAX_ZERO_THRESHOLD = 1000
FSR_REST_MARGIN = 100
FSR_MAX_RAW = 4000
TREMOR_LOW_HZ = 3.0
TREMOR_HIGH_HZ = 12.0


def request_json(base_url, path, attempts=3):
    last_error = None
    for attempt in range(attempts):
        try:
            with urlopen(f"{base_url}{path}", timeout=5) as response:
                return json.load(response)
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1)
    raise last_error


def control(base_url, command):
    result = request_json(base_url, f"/control?cmd={command}")
    if not result.get("result"):
        raise RuntimeError(f"phyphox rejected the {command!r} command")


def get_acceleration(base_url):
    query = "/get?acc_time=full&accX=full&accY=full&accZ=full"
    data = request_json(base_url, query)["buffer"]

    required = ("acc_time", "accX", "accY", "accZ")
    missing = [name for name in required if name not in data]
    if missing:
        raise RuntimeError(
            "Missing phyphox buffers: "
            + ", ".join(missing)
            + ". Select 'Acceleration (without g)' in phyphox."
        )

    arrays = [np.asarray(data[name]["buffer"], dtype=float) for name in required]
    length = min(map(len, arrays))
    if length < 10:
        raise RuntimeError("Not enough accelerometer samples were received")

    return tuple(array[:length] for array in arrays)


def calculate_tremor(t, x, y, z):
    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    t, x, y, z = t[valid], x[valid], y[valid], z[valid]

    keep = np.concatenate(([True], np.diff(t) > 0))
    t, x, y, z = t[keep], x[keep], y[keep], z[keep]

    sample_rate = 1.0 / np.median(np.diff(t))
    uniform_t = np.arange(t[0], t[-1], 1.0 / sample_rate)
    axes = np.vstack(
        [np.interp(uniform_t, t, values) for values in (x, y, z)]
    )

    high_hz = min(TREMOR_HIGH_HZ, sample_rate * 0.45)
    if high_hz <= TREMOR_LOW_HZ:
        raise RuntimeError(f"Sampling rate {sample_rate:.1f} Hz is too low")

    sos = butter(
        4,
        [TREMOR_LOW_HZ, high_hz],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    filtered = np.vstack([sosfiltfilt(sos, axis) for axis in axes])
    tremor_magnitude = np.sqrt(np.sum(filtered**2, axis=0))
    acceleration_magnitude = np.sqrt(np.sum(axes**2, axis=0))

    segment_length = min(1024, filtered.shape[1])
    frequencies, power_x = welch(filtered[0], sample_rate, nperseg=segment_length)
    _, power_y = welch(filtered[1], sample_rate, nperseg=segment_length)
    _, power_z = welch(filtered[2], sample_rate, nperseg=segment_length)
    total_power = power_x + power_y + power_z
    tremor_band = (frequencies >= TREMOR_LOW_HZ) & (frequencies <= high_hz)
    dominant_hz = frequencies[tremor_band][np.argmax(total_power[tremor_band])]

    return (
        uniform_t - uniform_t[0],
        axes,
        acceleration_magnitude,
        filtered,
        tremor_magnitude,
        sample_rate,
        dominant_hz,
    )


def read_fsr_sample(device):
    line = device.readline().decode("ascii", errors="ignore").strip()
    if not line:
        return None

    fields = line.split(",")
    try:
        if fields[0] == "FSR" and len(fields) == 3:
            return int(fields[2])
        if len(fields) == 2:
            return int(fields[1])
    except ValueError:
        return None
    return None


def calibrate_fsr(device):
    print(f"Keep the grip pad untouched for {FSR_CALIBRATION_S} seconds...")
    device.reset_input_buffer()
    samples = []
    end_time = time.monotonic() + FSR_CALIBRATION_S
    while time.monotonic() < end_time:
        value = read_fsr_sample(device)
        if value is not None:
            samples.append(value)

    if not samples:
        raise RuntimeError("No FSR data received from the ESP32")

    rest_limit = int(np.percentile(samples, 99))
    threshold = max(FSR_MIN_ZERO_THRESHOLD, rest_limit + FSR_REST_MARGIN)
    threshold = min(threshold, FSR_MAX_ZERO_THRESHOLD)
    print(
        f"FSR calibrated: resting values up to {threshold} count as zero grip"
    )
    return threshold


def record_fsr(device, duration_s):
    device.reset_input_buffer()
    start_time = time.monotonic()
    times = []
    values = []

    while time.monotonic() - start_time < duration_s:
        value = read_fsr_sample(device)
        if value is not None:
            times.append(time.monotonic() - start_time)
            values.append(value)

    if not values:
        raise RuntimeError("No FSR samples were recorded")

    return np.asarray(times), np.asarray(values, dtype=float)


def save_trial(output_dir, trial_number, results, fsr_data, fsr_threshold):
    t, axes, accel_mag, filtered, tremor_mag, sample_rate, dominant_hz = results
    fsr_time, fsr_raw_samples = fsr_data
    fsr_raw = np.interp(t, fsr_time, fsr_raw_samples)
    fsr_calibrated = np.clip(fsr_raw - fsr_threshold, 0, None)
    grip_percent = np.clip(
        100.0 * fsr_calibrated / (FSR_MAX_RAW - fsr_threshold),
        0,
        100,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"iphone_trial_{trial_number:03d}_{timestamp}.csv"

    header = [
        "time_s",
        "fsr_raw",
        "fsr_zero_threshold",
        "grip_calibrated",
        "grip_percent",
        "accel_x_m_s2",
        "accel_y_m_s2",
        "accel_z_m_s2",
        "accel_magnitude_m_s2",
        "tremor_x_m_s2",
        "tremor_y_m_s2",
        "tremor_z_m_s2",
        "tremor_magnitude_m_s2",
        "dominant_tremor_hz",
    ]

    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for index in range(len(t)):
            writer.writerow(
                [
                    f"{t[index]:.6f}",
                    f"{fsr_raw[index]:.1f}",
                    str(fsr_threshold),
                    f"{fsr_calibrated[index]:.1f}",
                    f"{grip_percent[index]:.3f}",
                    f"{axes[0, index]:.6f}",
                    f"{axes[1, index]:.6f}",
                    f"{axes[2, index]:.6f}",
                    f"{accel_mag[index]:.6f}",
                    f"{filtered[0, index]:.6f}",
                    f"{filtered[1, index]:.6f}",
                    f"{filtered[2, index]:.6f}",
                    f"{tremor_mag[index]:.6f}",
                    f"{dominant_hz:.3f}",
                ]
            )

    return path, sample_rate, dominant_hz


def read_key():
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main():
    parser = argparse.ArgumentParser(description="Record iPhone tremor using phyphox")
    parser.add_argument(
        "url",
        nargs="?",
        help="Remote-access URL shown by phyphox",
    )
    parser.add_argument(
        "--serial-port",
        default="/dev/cu.SLAB_USBtoUART",
        help="ESP32 serial port",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--output", type=Path, default=Path("trials"))
    args = parser.parse_args()

    phyphox_url = args.url or input(
        "Enter the phyphox remote-access URL: "
    ).strip()
    if not phyphox_url:
        raise SystemExit("A phyphox remote-access URL is required")
    if not phyphox_url.startswith(("http://", "https://")):
        phyphox_url = "http://" + phyphox_url

    base_url = phyphox_url.rstrip("/")
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        config = request_json(base_url, "/config")
        print(f"Connected to phyphox experiment: {config.get('localTitle', 'unknown')}")
    except (URLError, TimeoutError) as error:
        raise SystemExit(f"Cannot connect to phyphox at {base_url}: {error}")

    try:
        esp32 = serial.Serial(args.serial_port, args.baud, timeout=0.05)
        time.sleep(2)
        esp32.reset_input_buffer()
        print(f"Connected to ESP32 at {args.serial_port}")
    except serial.SerialException as error:
        raise SystemExit(f"Cannot open ESP32 at {args.serial_port}: {error}")

    trial_number = 0
    fsr_threshold = calibrate_fsr(esp32)
    print("Press SPACE for a 20-second trial. Press C to recalibrate. Press Q to quit.")

    try:
        while True:
            key = read_key()
            if key.lower() == "q":
                break
            if key.lower() == "c":
                fsr_threshold = calibrate_fsr(esp32)
                continue
            if key != " ":
                continue

            trial_number += 1
            print(f"Trial {trial_number} recording...")
            try:
                control(base_url, "clear")
                control(base_url, "start")
                fsr_data = record_fsr(esp32, TRIAL_DURATION_S)
                control(base_url, "stop")

                results = calculate_tremor(*get_acceleration(base_url))
                path, sample_rate, dominant_hz = save_trial(
                    args.output,
                    trial_number,
                    results,
                    fsr_data,
                    fsr_threshold,
                )
                print(
                    f"Saved {path} | {sample_rate:.1f} Hz | "
                    f"dominant tremor {dominant_hz:.2f} Hz"
                )
            except (URLError, TimeoutError, OSError, RuntimeError) as error:
                try:
                    control(base_url, "stop")
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
                print(f"Trial {trial_number} failed: {error}")
                print("Check that phyphox is open with remote access enabled.")
            print("Press SPACE for another trial, C to recalibrate, or Q to quit.")
    finally:
        esp32.close()


if __name__ == "__main__":
    main()
