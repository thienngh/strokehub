import argparse
import csv
import json
import math
import random
import threading
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tkinter import messagebox, simpledialog
from urllib.request import urlopen

import serial
import joblib
import pandas as pd

from train_nervaflex_model import extract_trial_features


WIDTH = 960
HEIGHT = 650
GAME_HEIGHT = 540
PANEL_WIDTH = 360
FPS_MS = 16

BIRD_ASSET = (
    Path(__file__).resolve().parents[1]
    / "flappy-motion-model"
    / "assets"
    / "bird.png"
)

# Adjustable assessment settings.
GRIP_THRESHOLD = 0.01
FSR_CALIBRATION_SECONDS = 2.0
FSR_MIN_ZERO_THRESHOLD = 500
FSR_MAX_ZERO_THRESHOLD = 1000
FSR_REST_MARGIN = 100
FSR_MAX_RAW = 4000
DEFAULT_SERIAL_PORT = "/dev/cu.SLAB_USBtoUART"
CAMERA_MIN_VISIBILITY = 0.75
CAMERA_MAX_FUSION_WEIGHT = 0.20
CAMERA_MAX_DISAGREEMENT_FRACTION = 0.30
ML_DATASET_NAME = "nervaflex_ml_trials.csv"
PREDICTION_DATASET_NAME = "nervaflex_trial_predictions.csv"
VALIDATION_DATASET_NAME = "nervaflex_validation_trials.csv"
VALIDATION_PREDICTION_NAME = "nervaflex_validation_predictions.csv"
SCENARIO_LABELS = [
    "normal",
    "weak_grip",
    "intermittent_grip",
    "limited_movement",
    "unstable_grip",
    "jerky_movement",
    "rapid_oscillation",
    "combined",
]
ML_FIELDS = [
    "trial_id", "attempt_number", "trial_started_at", "game",
    "scenario_label", "outcome",
    "trial_duration_s", "final_assessment_score", "final_flappy_score",
    "final_pong_hits", "final_pong_misses", "grip_pause_count",
    "time_s", "accel_x_g", "accel_y_g", "accel_z_g", "tilt", "target_y",
    "fsr_raw", "fsr_zero_threshold", "grip_calibrated", "grip_percent",
    "paused", "speed_px_s", "tremor_px", "smoothness", "difficulty",
    "flappy_score", "pong_hits", "pong_misses", "camera_tracking",
    "camera_forearm_angle_deg", "camera_elbow_angle_deg",
    "camera_shoulder_elevation_deg", "camera_visibility",
    "camera_shoulder_x", "camera_shoulder_y", "camera_elbow_x",
    "camera_elbow_y", "camera_wrist_x", "camera_wrist_y",
    "phone_target_y", "camera_target_y", "fused_target_y",
    "camera_fusion_weight", "camera_phone_disagreement",
]

# Pixel-based prototype standards. Recalibrate when MMA8451Q/FSR values are real
# acceleration/force units rather than keyboard simulator values.
HEALTHY_STANDARD = {
    "controlled_speed_px_per_sec": 650,
    "excessive_speed_px_per_sec": 1100,
    "excellent_tremor_px": 4,
    "upper_tremor_px": 18,
    "smoothness_warning_jerk": 1800,
}


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def lerp(start, end, amount):
    return start + (end - start) * amount


def score_speed(speed):
    controlled = HEALTHY_STANDARD["controlled_speed_px_per_sec"]
    excessive = HEALTHY_STANDARD["excessive_speed_px_per_sec"]
    if speed <= controlled:
        return 1
    return clamp(1 - (speed - controlled) / (excessive - controlled), 0, 1)


def score_tremor(tremor):
    excellent = HEALTHY_STANDARD["excellent_tremor_px"]
    upper = HEALTHY_STANDARD["upper_tremor_px"]
    if tremor <= excellent:
        return 1
    return clamp(1 - (tremor - excellent) / (upper - excellent), 0, 1)


def score_smoothness(jerk):
    return clamp(1 - jerk / HEALTHY_STANDARD["smoothness_warning_jerk"], 0, 1)


def calculate_assessment_score(speed_score, stability_score, smoothness_score):
    return 10 * (speed_score * 0.25 + stability_score * 0.35 + smoothness_score * 0.40)


class SimulatedSensorProvider:
    """Keyboard demo for later replacement with Arduino/serial FSR + MMA8451Q readings.

    Demo controls:
    - Space simulates FSR grip pressure.
    - Up/W simulates forearm lifting upward from the elbow.
    - Down/S simulates forearm lowering downward from the elbow.
    """

    def __init__(self, root):
        self.tilt_value = 0.0
        self.grip_strength = 0.0
        self.up_pressed = False
        self.down_pressed = False
        self.grip_pressed = False

        root.bind("<KeyPress-Up>", lambda _event: self.set_up(True))
        root.bind("<KeyRelease-Up>", lambda _event: self.set_up(False))
        root.bind("<KeyPress-w>", lambda _event: self.set_up(True))
        root.bind("<KeyRelease-w>", lambda _event: self.set_up(False))

        root.bind("<KeyPress-Down>", lambda _event: self.set_down(True))
        root.bind("<KeyRelease-Down>", lambda _event: self.set_down(False))
        root.bind("<KeyPress-s>", lambda _event: self.set_down(True))
        root.bind("<KeyRelease-s>", lambda _event: self.set_down(False))

        root.bind("<KeyPress-space>", lambda _event: self.set_grip(True))
        root.bind("<KeyRelease-space>", lambda _event: self.set_grip(False))

    def set_up(self, pressed):
        self.up_pressed = pressed

    def set_down(self, pressed):
        self.down_pressed = pressed

    def set_grip(self, pressed):
        self.grip_pressed = pressed

    def update(self, dt):
        if self.up_pressed:
            self.tilt_value -= 1.8 * dt
        if self.down_pressed:
            self.tilt_value += 1.8 * dt
        if not self.up_pressed and not self.down_pressed:
            self.tilt_value = lerp(self.tilt_value, 0, 0.045)

        target_grip = 0.86 if self.grip_pressed else 0.12
        self.grip_strength = lerp(self.grip_strength, target_grip, 0.14)
        self.tilt_value = clamp(self.tilt_value, -1, 1)
        self.grip_strength = clamp(self.grip_strength, 0, 1)

    def read(self):
        # Approximate gravity vector for forearm pitch around a tucked elbow.
        angle = self.tilt_value * math.radians(70)
        return {
            "tilt": self.tilt_value,
            "accel_x": math.sin(angle),
            "accel_y": 0.0,
            "accel_z": math.cos(angle),
            "grip": self.grip_strength,
        }


# ================================================================
# ESP32 SENSOR INTEGRATION POINT
# ================================================================
# Replace SimulatedSensorProvider with this class when the real hardware is ready.
# Expected ESP32 serial packet shape can be simple CSV, for example:
#   DATA,ax,ay,az,grip
#
# Required normalized output from read():
#   {
#       "tilt": float from -1.0 to +1.0,
#       "accel_x": normalized MMA8451Q X acceleration in g,
#       "accel_y": normalized MMA8451Q Y acceleration in g,
#       "accel_z": normalized MMA8451Q Z acceleration in g,
#       "grip": float from 0.0 to 1.0 from the FSR 402
#   }
#
# Suggested real tilt calculation:
#   1. Read MMA8451Q raw X/Y/Z over I2C on the ESP32.
#   2. Convert counts to g using the selected MMA8451Q range (+/-2g, +/-4g, +/-8g).
#   3. Compute forearm pitch around the tucked elbow, for example:
#        pitch_rad = atan2(accel_x, accel_z)
#   4. During calibration, neutral is the forearm held at 90 degrees to the body.
#      The app maps calibrated up/down pitch values into game target position.
#
# To use it:
#   self.sensor = ESP32SensorProvider("COM5")
# in FlappySensorAssessmentApp.__init__ instead of SimulatedSensorProvider(root).
class ESP32SensorProvider:
    def __init__(self, port, baud=115200):
        raise NotImplementedError("Connect ESP32 serial code here when hardware is ready.")

    def update(self, dt):
        pass

    def read(self):
        return {
            "tilt": 0.0,
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 1.0,
            "grip": 0.0,
        }


class PhoneGripSensorProvider:
    """Streams iPhone acceleration from phyphox and FSR grip from ESP32."""

    def __init__(self, phyphox_url, serial_port=DEFAULT_SERIAL_PORT, baud=115200):
        self.base_url = self.normalize_url(phyphox_url)
        self.serial_port = serial_port
        self.baud = baud
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._accel = (0.0, 0.0, 1.0)
        self._fsr_raw = 0
        self._fsr_threshold = FSR_MIN_ZERO_THRESHOLD
        self._grip_ready = False
        self._calibrating = False
        self._calibration_start = None
        self._calibration_samples = []
        self.phyphox_error = "Connecting..."
        self.serial_error = "Connecting..."
        self.phyphox_receiving = False
        self.serial_receiving = False
        self._last_phyphox = 0.0
        self._last_serial = 0.0
        threading.Thread(target=self._phyphox_loop, daemon=True).start()
        threading.Thread(target=self._serial_loop, daemon=True).start()

    @staticmethod
    def normalize_url(value):
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            value = "http://" + value
        return value

    def _phyphox_loop(self):
        get_url = self.base_url + "/get?accX&accY&accZ"
        try:
            try:
                urlopen(self.base_url + "/control?cmd=start", timeout=2).close()
            except Exception:
                pass
            while not self._stop.is_set():
                try:
                    with urlopen(get_url, timeout=2) as response:
                        data = json.load(response)["buffer"]
                    values = []
                    for name in ("accX", "accY", "accZ"):
                        buffer = data[name]["buffer"]
                        if not buffer:
                            raise ValueError("No data in phyphox buffer " + name)
                        values.append(float(buffer[-1]) / 9.80665)
                    with self._lock:
                        self._accel = tuple(values)
                        self._last_phyphox = time.monotonic()
                        self.phyphox_receiving = True
                        self.phyphox_error = ""
                except Exception as error:
                    with self._lock:
                        self.phyphox_receiving = False
                        self.phyphox_error = str(error)
                self._stop.wait(0.0125)
        finally:
            self.phyphox_receiving = False

    @staticmethod
    def _parse_fsr(line):
        fields = line.strip().split(",")
        try:
            if len(fields) == 3 and fields[0] == "FSR":
                return int(fields[2])
            if len(fields) == 2:
                return int(fields[1])
        except ValueError:
            pass
        return None

    def _finish_grip_calibration(self):
        values = sorted(self._calibration_samples)
        if not values:
            return
        index = min(len(values) - 1, int(len(values) * 0.99))
        threshold = max(FSR_MIN_ZERO_THRESHOLD,
                        values[index] + FSR_REST_MARGIN)
        self._fsr_threshold = min(threshold, FSR_MAX_ZERO_THRESHOLD)
        self._grip_ready = True
        self._calibrating = False

    def _serial_loop(self):
        device = None
        try:
            device = serial.Serial(self.serial_port, self.baud, timeout=0.05)
            time.sleep(2.0)
            device.reset_input_buffer()
            self.serial_error = ""
            while not self._stop.is_set():
                line = device.readline().decode("ascii", "ignore")
                value = self._parse_fsr(line)
                if value is None:
                    continue
                now = time.monotonic()
                with self._lock:
                    self._fsr_raw = value
                    self._last_serial = now
                    self.serial_receiving = True
                    if self._calibrating:
                        if self._calibration_start is None:
                            self._calibration_start = now
                        self._calibration_samples.append(value)
                        if now - self._calibration_start >= FSR_CALIBRATION_SECONDS:
                            self._finish_grip_calibration()
        except Exception as error:
            with self._lock:
                self.serial_receiving = False
                self.serial_error = str(error)
        finally:
            if device is not None:
                device.close()

    def begin_grip_calibration(self):
        with self._lock:
            self._grip_ready = False
            self._calibrating = True
            self._calibration_start = None
            self._calibration_samples = []

    def grip_ready(self):
        with self._lock:
            return self._grip_ready

    def update(self, dt):
        del dt

    def read(self):
        with self._lock:
            ax, ay, az = self._accel
            raw = self._fsr_raw
            threshold = self._fsr_threshold
            ready = self._grip_ready
            phy_ok = time.monotonic() - self._last_phyphox < 1.0
            fsr_ok = time.monotonic() - self._last_serial < 1.0
        calibrated = max(0, raw - threshold)
        grip = clamp(calibrated / max(1, FSR_MAX_RAW - threshold), 0, 1)
        angle = math.degrees(math.atan2(ax, math.sqrt(ay * ay + az * az) or 1e-9))
        return {
            "tilt": clamp(angle / 90.0, -1, 1),
            "accel_x": ax,
            "accel_y": ay,
            "accel_z": az,
            "grip": grip if ready and fsr_ok else 0.0,
            "grip_raw": raw,
            "grip_calibrated": calibrated,
            "grip_threshold": threshold,
            "grip_ready": ready,
            "phyphox_ready": phy_ok,
            "fsr_ready": fsr_ok,
        }

    def status(self):
        sample = self.read()
        if not sample["phyphox_ready"]:
            return "Phone: " + (self.phyphox_error or "no data")
        if not sample["fsr_ready"]:
            return "FSR: " + (self.serial_error or "no data")
        if not sample["grip_ready"]:
            return "FSR connected; grip not calibrated"
        return "Phone + FSR connected"

    def close(self):
        self._stop.set()


class CameraBridge:
    """Serves the webcam tracker and receives right-arm pose measurements."""

    def __init__(self, html_path, port=8765):
        self.html_path = Path(html_path)
        self.port = port
        self._lock = threading.Lock()
        self._server = None
        self._thread = None
        self._last_received = 0.0
        self._sample = self.empty_sample()

    @staticmethod
    def empty_sample():
        return {
            "tracking": False,
            "forearm_angle_deg": math.nan,
            "elbow_angle_deg": math.nan,
            "shoulder_elevation_deg": math.nan,
            "visibility": 0.0,
            "shoulder_x": math.nan,
            "shoulder_y": math.nan,
            "elbow_x": math.nan,
            "elbow_y": math.nan,
            "wrist_x": math.nan,
            "wrist_y": math.nan,
        }

    def start(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in ("/", "/webcam_tracker.html"):
                    self.send_error(404)
                    return
                try:
                    content = bridge.html_path.read_bytes()
                except OSError as error:
                    self.send_error(500, str(error))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def do_POST(self):
                if self.path != "/camera":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    bridge.update(payload)
                    self.send_response(204)
                    self.end_headers()
                except (ValueError, json.JSONDecodeError) as error:
                    self.send_error(400, str(error))

            def log_message(self, format_string, *args):
                del format_string, args

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        webbrowser.open(f"http://127.0.0.1:{self.port}/")

    def update(self, payload):
        sample = self.empty_sample()
        sample["tracking"] = bool(payload.get("tracking", False))
        for key in sample:
            if key == "tracking":
                continue
            value = payload.get(key)
            if value is not None:
                try:
                    sample[key] = float(value)
                except (TypeError, ValueError):
                    pass
        with self._lock:
            self._sample = sample
            self._last_received = time.monotonic()

    def sample(self):
        with self._lock:
            sample = dict(self._sample)
            fresh = time.monotonic() - self._last_received < 1.0
        if not fresh:
            sample["tracking"] = False
        return sample

    def status(self):
        sample = self.sample()
        if sample["tracking"]:
            return (
                f"Camera: right arm tracked, visibility "
                f"{sample['visibility']:.2f}"
            )
        return "Camera: waiting for tracker"

    def close(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


class AccelerometerCalibration:
    def __init__(self):
        self.neutral = 0.0
        self.up = -1.0
        self.down = 1.0

    def set_neutral(self, value):
        self.neutral = value

    def set_up(self, value):
        self.up = value

    def set_down(self, value):
        self.down = value

    def target_y_from_tilt(self, tilt_value):
        denominator = self.down - self.up
        if abs(denominator) < 0.05:
            denominator = 0.05 if denominator >= 0 else -0.05
        ratio = clamp((tilt_value - self.up) / denominator, 0, 1)
        return lerp(72, GAME_HEIGHT - 72, ratio)


class MovementAnalyzer:
    def __init__(self):
        self.samples = []
        self.history = []
        self.max_window_ms = 1800
        self.max_history_ms = 12000

    def reset(self):
        self.samples.clear()
        self.history.clear()

    def add_sample(self, target_y, timestamp_ms):
        self.samples.append({"target_y": target_y, "timestamp": timestamp_ms})
        oldest_allowed = timestamp_ms - self.max_window_ms
        self.samples = [
            sample for sample in self.samples if sample["timestamp"] >= oldest_allowed
        ]

    def get_metrics(self):
        if len(self.samples) < 3:
            return {
                "hand_speed": 0,
                "speed_score": 1,
                "stability_score": 1,
                "smoothness_score": 1,
                "tremor_pixels": 0,
                "jerk_index": 0,
                "assessment_score": 10,
            }

        positions = [sample["target_y"] for sample in self.samples]
        tremor_pixels = self.calculate_tremor(positions)
        speeds = []

        for index in range(1, len(self.samples)):
            current = self.samples[index]
            previous = self.samples[index - 1]
            dt = max((current["timestamp"] - previous["timestamp"]) / 1000, 0.001)
            speeds.append(abs(current["target_y"] - previous["target_y"]) / dt)

        average_speed = sum(speeds) / len(speeds)
        jerk_signals = [
            abs(speeds[index] - speeds[index - 1]) for index in range(1, len(speeds))
        ]
        jerk_index = sum(jerk_signals) / len(jerk_signals) if jerk_signals else 0

        speed_score = score_speed(average_speed)
        stability_score = score_tremor(tremor_pixels)
        smoothness_score = score_smoothness(jerk_index)
        assessment_score = calculate_assessment_score(
            speed_score, stability_score, smoothness_score
        )

        return {
            "hand_speed": average_speed,
            "speed_score": speed_score,
            "stability_score": stability_score,
            "smoothness_score": smoothness_score,
            "tremor_pixels": tremor_pixels,
            "jerk_index": jerk_index,
            "assessment_score": assessment_score,
        }

    def calculate_tremor(self, positions):
        trend = positions[0]
        residuals = []
        for y in positions:
            trend = trend * 0.82 + y * 0.18
            residuals.append(y - trend)
        return math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))

    def record_history(self, timestamp_ms, metrics):
        self.history.append(
            {
                "timestamp": timestamp_ms,
                "speed": metrics["hand_speed"],
                "tremor": metrics["tremor_pixels"],
                "smoothness": metrics["smoothness_score"],
            }
        )
        oldest_allowed = timestamp_ms - self.max_history_ms
        self.history = [
            sample for sample in self.history if sample["timestamp"] >= oldest_allowed
        ]


class DifficultyAdapter:
    def __init__(self):
        self.difficulty = 0.5
        self.performance = 0.5

    def reset(self):
        self.difficulty = 0.5
        self.performance = 0.5

    def update(self, metrics, game_stats, dt):
        survival_score = clamp(game_stats["time_alive"] / 30, 0, 1)
        pipe_accuracy_score = game_stats["pipe_accuracy_score"]
        performance = (
            survival_score * 0.25
            + pipe_accuracy_score * 0.35
            + metrics["speed_score"] * 0.15
            + metrics["stability_score"] * 0.15
            + metrics["smoothness_score"] * 0.10
        )
        self.performance = lerp(self.performance, performance, 0.03)
        target_difficulty = clamp(self.performance, 0.12, 0.95)
        self.difficulty = lerp(self.difficulty, target_difficulty, dt * 0.08)
        return self.difficulty


class GameEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bird = {"x": 210, "y": GAME_HEIGHT / 2, "velocity_y": 0, "radius": 18}
        self.pipes = []
        self.score = 0
        self.time_alive = 0
        self.pipe_timer = 0
        self.pipe_accuracy_score = 1
        self.is_game_over = False
        self.spawn_pipe(0.5)

    def update(self, dt, target_y, difficulty):
        if self.is_game_over:
            return
        self.time_alive += dt
        self.update_bird(dt, target_y)
        self.update_pipes(dt, difficulty)
        self.track_accuracy(difficulty)
        self.check_collisions()

    def update_bird(self, dt, target_y):
        follow_strength = 34
        damping = 0.86
        error = target_y - self.bird["y"]
        self.bird["velocity_y"] += error * follow_strength * dt
        self.bird["velocity_y"] *= damping
        self.bird["y"] += self.bird["velocity_y"] * dt

    def update_pipes(self, dt, difficulty):
        pipe_speed = lerp(185, 365, difficulty)
        pipe_spacing_time = lerp(2.05, 1.2, difficulty)
        self.pipe_timer += dt
        if self.pipe_timer >= pipe_spacing_time:
            self.pipe_timer = 0
            self.spawn_pipe(difficulty)

        for pipe in self.pipes:
            pipe["x"] -= pipe_speed * dt
            pipe["phase"] += dt * pipe["move_speed"]
            pipe["current_gap_y"] = (
                pipe["gap_y"] + math.sin(pipe["phase"]) * pipe["move_range"]
            )
            if not pipe["passed"] and pipe["x"] + pipe["width"] < self.bird["x"]:
                pipe["passed"] = True
                self.score += 1

        self.pipes = [pipe for pipe in self.pipes if pipe["x"] + pipe["width"] > -20]

    def spawn_pipe(self, difficulty):
        gap_size = lerp(220, 118, difficulty)
        margin = 76 + gap_size / 2
        gap_y = lerp(margin, GAME_HEIGHT - margin, random.random())
        self.pipes.append(
            {
                "x": WIDTH + 30,
                "width": 74,
                "gap_y": gap_y,
                "current_gap_y": gap_y,
                "gap_size": gap_size,
                "passed": False,
                "phase": random.random() * math.pi * 2,
                "move_range": lerp(0, 54, difficulty),
                "move_speed": lerp(0, 1.8, difficulty),
            }
        )

    def track_accuracy(self, difficulty):
        active_pipe = None
        for pipe in self.pipes:
            if pipe["x"] + pipe["width"] >= self.bird["x"] - self.bird["radius"]:
                active_pipe = pipe
                break
        if not active_pipe:
            return

        distance = abs(self.bird["y"] - active_pipe["current_gap_y"])
        allowed_distance = active_pipe["gap_size"] / 2
        instant_accuracy = clamp(1 - distance / allowed_distance, 0, 1)
        smoothing = lerp(0.035, 0.07, difficulty)
        self.pipe_accuracy_score = lerp(
            self.pipe_accuracy_score, instant_accuracy, smoothing
        )

    def check_collisions(self):
        bird = self.bird
        if bird["y"] - bird["radius"] < 0 or bird["y"] + bird["radius"] > GAME_HEIGHT:
            bird["y"] = clamp(bird["y"], bird["radius"],
                              GAME_HEIGHT - bird["radius"])
            bird["velocity_y"] = 0

        for pipe in self.pipes:
            in_pipe_x = (
                bird["x"] + bird["radius"] > pipe["x"]
                and bird["x"] - bird["radius"] < pipe["x"] + pipe["width"]
            )
            if not in_pipe_x:
                continue
            top_pipe_bottom = pipe["current_gap_y"] - pipe["gap_size"] / 2
            bottom_pipe_top = pipe["current_gap_y"] + pipe["gap_size"] / 2
            if bird["y"] - bird["radius"] < top_pipe_bottom:
                self.is_game_over = True
            if bird["y"] + bird["radius"] > bottom_pipe_top:
                self.is_game_over = True

    def get_stats(self):
        return {
            "time_alive": self.time_alive,
            "pipe_accuracy_score": self.pipe_accuracy_score,
        }


class PongEngine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.paddle_x = 42
        self.paddle_y = GAME_HEIGHT / 2
        self.paddle_w = 18
        self.paddle_h = 96
        self.ball_x = WIDTH * 0.66
        self.ball_y = GAME_HEIGHT / 2
        self.ball_vx = -250
        self.ball_vy = random.choice([-150, 150])
        self.ball_r = 10
        self.hits = 0
        self.misses = 0
        self.time_alive = 0
        self.is_game_over = False

    def update(self, dt, target_y, engaged):
        if self.is_game_over:
            return
        self.paddle_y = lerp(self.paddle_y, target_y, 0.22)
        self.paddle_y = clamp(self.paddle_y, self.paddle_h / 2 + 12, GAME_HEIGHT - self.paddle_h / 2 - 12)

        if not engaged:
            return

        self.time_alive += dt
        self.ball_x += self.ball_vx * dt
        self.ball_y += self.ball_vy * dt

        if self.ball_y - self.ball_r <= 18:
            self.ball_y = 18 + self.ball_r
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y + self.ball_r >= GAME_HEIGHT - 28:
            self.ball_y = GAME_HEIGHT - 28 - self.ball_r
            self.ball_vy = -abs(self.ball_vy)

        paddle_top = self.paddle_y - self.paddle_h / 2
        paddle_bottom = self.paddle_y + self.paddle_h / 2
        hits_paddle = (
            self.ball_x - self.ball_r <= self.paddle_x + self.paddle_w
            and self.ball_x + self.ball_r >= self.paddle_x
            and paddle_top <= self.ball_y <= paddle_bottom
            and self.ball_vx < 0
        )

        if hits_paddle:
            relative_hit = (self.ball_y - self.paddle_y) / (self.paddle_h / 2)
            self.ball_x = self.paddle_x + self.paddle_w + self.ball_r
            self.ball_vx = abs(self.ball_vx) * 1.035
            self.ball_vy = relative_hit * 260
            self.hits += 1

        if self.ball_x + self.ball_r >= WIDTH - 18:
            self.ball_x = WIDTH - 18 - self.ball_r
            self.ball_vx = -abs(self.ball_vx)

        if self.ball_x < -30:
            self.misses += 1
            self.is_game_over = True

    def reset_ball(self):
        speed = clamp(250 + self.hits * 8, 250, 430)
        self.ball_x = WIDTH * 0.72
        self.ball_y = random.randint(80, GAME_HEIGHT - 80)
        self.ball_vx = -speed
        self.ball_vy = random.choice([-170, -120, 120, 170])

    def accuracy(self):
        total = self.hits + self.misses
        if total == 0:
            return 1
        return self.hits / total


class FlappySensorAssessmentApp:
    def __init__(self, root, phyphox_url=None, serial_port=DEFAULT_SERIAL_PORT,
                 study_mode="training", enable_camera=False, camera_port=8765,
                 camera_weight=CAMERA_MAX_FUSION_WEIGHT):
        self.root = root
        self.study_mode = study_mode
        self.root.title(
            f"NervaFlex - iPhone Motion + FSR Grip ({study_mode.title()})"
        )
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            root,
            width=WIDTH + PANEL_WIDTH,
            height=HEIGHT,
            bg="#eaf5ff",
            highlightthickness=0,
        )
        self.canvas.pack()

        self.sensor = None
        self.initial_phyphox_url = phyphox_url
        self.serial_port = serial_port
        self.camera = None
        self.camera_max_fusion_weight = clamp(camera_weight, 0.0, 0.5)
        self.latest_camera = CameraBridge.empty_sample()
        if enable_camera:
            try:
                camera_html = Path(__file__).resolve().parent / "webcam_tracker.html"
                self.camera = CameraBridge(camera_html, camera_port)
                self.camera.start()
            except OSError as error:
                print(f"WARNING: camera bridge could not start: {error}")
        self.calibration = AccelerometerCalibration()
        self.camera_calibration = AccelerometerCalibration()
        self.camera_calibration_points = set()
        self.analyzer = MovementAnalyzer()
        self.difficulty_adapter = DifficultyAdapter()
        self.game = GameEngine()
        self.pong = PongEngine()

        self.phase = "start_screen"
        self.click_zones = []
        self.placeholder_title = ""
        self.menu_message = ""
        self.grip_failures = 0
        self.pause_started_at = None
        self.final_reason = ""
        self.output_dir = Path(__file__).resolve().parent / "trials"
        if study_mode == "validation":
            self.dataset_path = self.output_dir / VALIDATION_DATASET_NAME
            self.prediction_path = self.output_dir / VALIDATION_PREDICTION_NAME
        else:
            self.dataset_path = self.output_dir / ML_DATASET_NAME
            self.prediction_path = self.output_dir / PREDICTION_DATASET_NAME
        self.model_path = Path(__file__).resolve().parent / "nervaflex_rf_model.pkl"
        self.model_bundle = self.load_model_bundle()
        self.trial_samples = []
        self.trial_started_at = None
        self.trial_started_at_iso = None
        self.current_trial_id = None
        self.current_attempt_number = None
        self.current_scenario_label = "unlabeled"
        self.next_attempt_number = 1
        self.last_csv_path = None
        self.last_prediction = None
        self.initialize_ml_dataset()
        self.last_time = time.perf_counter()
        self.bird_image = self.load_bird_image()
        self.latest_metrics = self.analyzer.get_metrics()
        self.latest_difficulty = 0.5
        self.latest_target_y = GAME_HEIGHT / 2
        self.latest_phone_target_y = GAME_HEIGHT / 2
        self.latest_camera_target_y = math.nan
        self.camera_fusion_weight = 0.0
        self.camera_phone_disagreement = math.nan
        self.latest_sensor = {
            "tilt": 0, "accel_x": 0, "accel_y": 0, "accel_z": 1,
            "grip": 0, "grip_raw": 0, "grip_calibrated": 0,
            "grip_threshold": FSR_MIN_ZERO_THRESHOLD, "grip_ready": False,
            "phyphox_ready": False, "fsr_ready": False,
        }

        root.bind("<Return>", self.handle_enter)
        self.canvas.bind("<Button-1>", self.handle_click)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(FPS_MS, self.loop)

    def connect_sensors(self):
        if self.sensor is not None:
            return True
        url = self.initial_phyphox_url
        self.initial_phyphox_url = None
        if not url:
            url = simpledialog.askstring(
                "Connect iPhone",
                "Paste the phyphox Remote Access URL shown on the iPhone:",
                parent=self.root,
            )
        if not url:
            self.menu_message = "A phyphox URL is required. Press Enter to try again."
            return False
        normalized = PhoneGripSensorProvider.normalize_url(url)
        try:
            with urlopen(normalized + "/config", timeout=5) as response:
                config = json.load(response)
        except Exception as error:
            messagebox.showerror(
                "phyphox connection failed",
                f"Could not connect to {normalized}\n\n{error}\n\n"
                "Keep phyphox open with Remote Access enabled.",
                parent=self.root,
            )
            return False
        title = config.get("localTitle", config.get("title", ""))
        if "with g" not in title.lower():
            messagebox.showwarning(
                "Select Acceleration with g",
                f"Current experiment: {title}\n\n"
                "Pong and Flappy require phyphox 'Acceleration with g'.",
                parent=self.root,
            )
        self.sensor = PhoneGripSensorProvider(normalized, self.serial_port)
        self.menu_message = "Connecting to iPhone and ESP32..."
        return True

    def close(self):
        if self.trial_started_at is not None:
            game = "pong" if self.phase.startswith("pong_") else "flappy"
            self.save_trial_csv(game, "Window closed")
        if self.sensor is not None and hasattr(self.sensor, "close"):
            self.sensor.close()
        if self.camera is not None:
            self.camera.close()
        self.root.destroy()

    def initialize_ml_dataset(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.dataset_path.exists():
            try:
                with self.dataset_path.open(newline="") as csv_file:
                    reader = csv.DictReader(csv_file)
                    rows = list(reader)
                    existing_fields = reader.fieldnames or []
                if existing_fields != ML_FIELDS:
                    temporary_path = self.dataset_path.with_suffix(".tmp")
                    with temporary_path.open("w", newline="") as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=ML_FIELDS)
                        writer.writeheader()
                        for row in rows:
                            if not row.get("scenario_label"):
                                row["scenario_label"] = "legacy_unlabeled"
                            writer.writerow({field: row.get(field, "")
                                             for field in ML_FIELDS})
                    temporary_path.replace(self.dataset_path)
                attempts = [int(row["attempt_number"])
                            for row in rows if row.get("attempt_number")]
                self.next_attempt_number = max(attempts, default=0) + 1
            except (OSError, ValueError, KeyError):
                self.next_attempt_number = 1
            return
        with self.dataset_path.open("w", newline="") as csv_file:
            csv.DictWriter(csv_file, fieldnames=ML_FIELDS).writeheader()

    def load_model_bundle(self):
        if not self.model_path.exists():
            return None
        try:
            return joblib.load(self.model_path)
        except Exception as error:
            print(f"WARNING: could not load ML model: {error}")
            return None

    def load_bird_image(self):
        try:
            source = tk.PhotoImage(file=str(BIRD_ASSET))
            scale = max(1, round(source.width() / 78))
            return source.subsample(scale, scale)
        except tk.TclError:
            return None

    def handle_enter(self, _event=None):
        tilt = self.latest_sensor["tilt"]
        if self.phase == "start_screen":
            if self.connect_sensors():
                self.phase = "game_select"
        elif self.phase == "welcome":
            self.start_grip_calibration("calibrate_grip")
        elif self.phase == "calibrate_grip":
            if self.latest_sensor["grip_ready"]:
                self.phase = "calibrate_neutral"
            else:
                self.menu_message = "Grip calibration is still running. Keep the pad untouched."
        elif self.phase == "calibrate_neutral":
            self.calibration.set_neutral(tilt)
            self.capture_camera_calibration("neutral")
            self.phase = "calibrate_up"
        elif self.phase == "calibrate_up":
            self.calibration.set_up(tilt)
            self.capture_camera_calibration("up")
            self.phase = "calibrate_down"
        elif self.phase == "calibrate_down":
            self.calibration.set_down(tilt)
            self.capture_camera_calibration("down")
            self.start_assessment()
        elif self.phase == "pong_welcome":
            self.start_grip_calibration("pong_calibrate_grip")
        elif self.phase == "pong_calibrate_grip":
            if self.latest_sensor["grip_ready"]:
                self.phase = "pong_calibrate_neutral"
            else:
                self.menu_message = "Grip calibration is still running. Keep the pad untouched."
        elif self.phase == "pong_calibrate_neutral":
            self.calibration.set_neutral(tilt)
            self.capture_camera_calibration("neutral")
            self.phase = "pong_calibrate_up"
        elif self.phase == "pong_calibrate_up":
            self.calibration.set_up(tilt)
            self.capture_camera_calibration("up")
            self.phase = "pong_calibrate_down"
        elif self.phase == "pong_calibrate_down":
            self.calibration.set_down(tilt)
            self.capture_camera_calibration("down")
            self.start_pong_assessment()
        elif self.phase == "pong_practice":
            self.start_pong_assessment()
        elif self.phase == "test":
            self.start_assessment()
        elif self.phase == "results":
            self.reset_all()
        elif self.phase == "pong_results":
            self.phase = "game_select"
        elif self.phase == "coming_soon":
            self.phase = "game_select"

    def handle_click(self, event):
        for name, x1, y1, x2, y2 in self.click_zones:
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                if name == "start":
                    if self.connect_sensors():
                        self.phase = "game_select"
                elif name == "flappy":
                    self.menu_message = ""
                    self.phase = "welcome"
                elif name == "another":
                    self.start_pong_welcome()
                elif name == "mystery":
                    self.placeholder_title = "Mystery Game"
                    self.phase = "coming_soon"
                elif name == "back_to_menu":
                    self.phase = "game_select"
                elif name == "back_home":
                    self.phase = "start_screen"
                elif name == "quit_flappy":
                    self.quit_to_game_select()
                return

    def start_grip_calibration(self, phase):
        if self.sensor is None:
            return
        self.sensor.begin_grip_calibration()
        self.camera_calibration_points = set()
        self.phase = phase
        self.menu_message = "Keep the grip pad untouched for two seconds."

    def capture_camera_calibration(self, position):
        sample = self.latest_camera
        angle = sample["forearm_angle_deg"]
        if not (
            sample["tracking"]
            and sample["visibility"] >= CAMERA_MIN_VISIBILITY
            and math.isfinite(angle)
        ):
            self.menu_message = (
                f"Camera {position} point unavailable; phone-only fallback active."
            )
            return
        if position == "neutral":
            self.camera_calibration.set_neutral(angle)
        elif position == "up":
            self.camera_calibration.set_up(angle)
        elif position == "down":
            self.camera_calibration.set_down(angle)
        self.camera_calibration_points.add(position)

    def quit_to_game_select(self):
        if self.trial_started_at is not None:
            game = "pong" if self.phase.startswith("pong_") else "flappy"
            self.save_trial_csv(game, "User quit")
        self.phase = "game_select"
        self.grip_failures = 0
        self.pause_started_at = None
        self.final_reason = ""
        self.game.reset()
        self.analyzer.reset()
        self.difficulty_adapter.reset()
        self.latest_metrics = self.analyzer.get_metrics()
        self.latest_difficulty = 0.5
        self.pong.reset()

    def start_pong_welcome(self):
        self.menu_message = ""
        self.phase = "pong_welcome"
        self.pong.reset()
        self.analyzer.reset()
        self.grip_failures = 0
        self.pause_started_at = None
        self.final_reason = ""

    def start_pong_practice(self):
        self.phase = "pong_practice"
        self.pong.reset()
        self.analyzer.reset()

    def start_pong_assessment(self):
        self.phase = "pong_assessment"
        self.pong.reset()
        self.analyzer.reset()
        self.grip_failures = 0
        self.pause_started_at = None
        self.final_reason = ""
        self.begin_trial_recording()

    def start_test_phase(self):
        self.phase = "test"
        self.game.reset()
        self.game.pipes.clear()
        self.analyzer.reset()

    def start_assessment(self):
        self.phase = "assessment"
        self.grip_failures = 0
        self.pause_started_at = None
        self.final_reason = ""
        self.game.reset()
        self.analyzer.reset()
        self.difficulty_adapter.reset()
        self.begin_trial_recording()

    def begin_trial_recording(self):
        if self.study_mode == "validation":
            self.current_scenario_label = "validation_unlabeled"
        else:
            self.current_scenario_label = self.prompt_scenario_label()
        self.trial_samples = []
        self.trial_started_at = time.perf_counter()
        started = datetime.now()
        self.trial_started_at_iso = started.isoformat(timespec="milliseconds")
        self.current_trial_id = started.strftime("%Y%m%d_%H%M%S_%f")
        self.current_attempt_number = self.next_attempt_number
        self.next_attempt_number += 1

    def prompt_scenario_label(self):
        choices = "\n".join(f"- {label}" for label in SCENARIO_LABELS)
        while True:
            value = simpledialog.askstring(
                "Trial scenario label",
                "Select the pattern you intend to perform:\n\n" + choices,
                initialvalue="normal",
                parent=self.root,
            )
            if value is None:
                return "unlabeled"
            normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
            if normalized in SCENARIO_LABELS:
                return normalized
            messagebox.showerror(
                "Invalid scenario label",
                "Enter one of the labels shown in the list.",
                parent=self.root,
            )

    def record_trial_sample(self, paused):
        if self.trial_started_at is None:
            return
        sensor = self.latest_sensor
        self.trial_samples.append({
            "time_s": time.perf_counter() - self.trial_started_at,
            "accel_x_g": sensor["accel_x"],
            "accel_y_g": sensor["accel_y"],
            "accel_z_g": sensor["accel_z"],
            "tilt": sensor["tilt"],
            "target_y": self.latest_target_y,
            "fsr_raw": sensor["grip_raw"],
            "fsr_zero_threshold": sensor["grip_threshold"],
            "grip_calibrated": sensor["grip_calibrated"],
            "grip_percent": sensor["grip"] * 100.0,
            "paused": int(paused),
            "speed_px_s": self.latest_metrics["hand_speed"],
            "tremor_px": self.latest_metrics["tremor_pixels"],
            "smoothness": self.latest_metrics["smoothness_score"],
            "difficulty": self.latest_difficulty,
            "flappy_score": self.game.score,
            "pong_hits": self.pong.hits,
            "pong_misses": self.pong.misses,
            "camera_tracking": int(self.latest_camera["tracking"]),
            "camera_forearm_angle_deg": self.latest_camera["forearm_angle_deg"],
            "camera_elbow_angle_deg": self.latest_camera["elbow_angle_deg"],
            "camera_shoulder_elevation_deg": self.latest_camera[
                "shoulder_elevation_deg"
            ],
            "camera_visibility": self.latest_camera["visibility"],
            "camera_shoulder_x": self.latest_camera["shoulder_x"],
            "camera_shoulder_y": self.latest_camera["shoulder_y"],
            "camera_elbow_x": self.latest_camera["elbow_x"],
            "camera_elbow_y": self.latest_camera["elbow_y"],
            "camera_wrist_x": self.latest_camera["wrist_x"],
            "camera_wrist_y": self.latest_camera["wrist_y"],
            "phone_target_y": self.latest_phone_target_y,
            "camera_target_y": self.latest_camera_target_y,
            "fused_target_y": self.latest_target_y,
            "camera_fusion_weight": self.camera_fusion_weight,
            "camera_phone_disagreement": self.camera_phone_disagreement,
        })

    def predict_trial(self, metadata):
        if self.model_bundle is None or not self.trial_samples:
            return None
        try:
            frame_table = pd.DataFrame(
                [{**metadata, **sample} for sample in self.trial_samples]
            )
            trial_features = extract_trial_features(frame_table)
            feature_columns = self.model_bundle["model_features"]
            label_columns = self.model_bundle["label_columns"]
            model = self.model_bundle["model"]
            model_input = trial_features.reindex(columns=feature_columns)

            predicted_values = model.predict(model_input)[0]
            probability_arrays = model.predict_proba(model_input)
            estimators = model.named_steps["classifier"].estimators_
            probabilities = {}
            predictions = {}
            for index, label in enumerate(label_columns):
                classes = list(estimators[index].classes_)
                if 1 in classes:
                    positive_index = classes.index(1)
                    probability = float(probability_arrays[index][0][positive_index])
                else:
                    probability = float(classes[0] == 1)
                probabilities[label] = probability
                predictions[label] = int(predicted_values[index])

            if metadata["game"] == "flappy":
                game_score = 100.0 * min(metadata["final_flappy_score"] / 10.0, 1.0)
            else:
                game_score = 100.0 * min(metadata["final_pong_hits"] / 15.0, 1.0)

            movement_labels = [
                "limited_movement", "unstable_movement", "high_tremor"
            ]
            grip_labels = ["low_grip", "unstable_grip", "grip_loss"]
            movement_score = 100.0 * (
                1.0 - sum(probabilities.get(label, 0.0)
                          for label in movement_labels) / len(movement_labels)
            )
            grip_score = 100.0 * (
                1.0 - sum(probabilities.get(label, 0.0)
                          for label in grip_labels) / len(grip_labels)
            )
            final_score = (
                0.40 * game_score + 0.30 * movement_score + 0.30 * grip_score
            )
            if final_score >= 80:
                result_band = "strong_gameplay_motor_control"
            elif final_score >= 60:
                result_band = "moderate_gameplay_motor_control"
            else:
                result_band = "gameplay_patterns_need_improvement"

            result = {
                "trial_id": metadata["trial_id"],
                "attempt_number": metadata["attempt_number"],
                "scenario_label": metadata["scenario_label"],
                "game": metadata["game"],
                "outcome": metadata["outcome"],
                "game_score": round(game_score, 3),
                "movement_score": round(movement_score, 3),
                "grip_score": round(grip_score, 3),
                "final_score": round(final_score, 3),
                "result_band": result_band,
            }
            for label in label_columns:
                result[f"predicted_{label}"] = predictions[label]
                result[f"probability_{label}"] = round(probabilities[label], 6)
            self.append_prediction(result)
            return result
        except Exception as error:
            print(f"WARNING: trial prediction failed: {error}")
            return None

    def append_prediction(self, result):
        fieldnames = list(result.keys())
        write_header = not self.prediction_path.exists()
        with self.prediction_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(result)

    def save_trial_csv(self, game_name, outcome):
        if not self.trial_samples:
            self.trial_started_at = None
            return None
        duration = self.trial_samples[-1]["time_s"]
        metadata = {
            "trial_id": self.current_trial_id,
            "attempt_number": self.current_attempt_number,
            "trial_started_at": self.trial_started_at_iso,
            "game": game_name,
            "scenario_label": self.current_scenario_label,
            "outcome": outcome,
            "trial_duration_s": duration,
            "final_assessment_score": self.latest_metrics["assessment_score"],
            "final_flappy_score": self.game.score,
            "final_pong_hits": self.pong.hits,
            "final_pong_misses": self.pong.misses,
            "grip_pause_count": self.grip_failures,
        }
        with self.dataset_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=ML_FIELDS)
            for sample in self.trial_samples:
                writer.writerow({**metadata, **sample})
        self.last_prediction = self.predict_trial(metadata)
        self.last_csv_path = self.dataset_path
        self.trial_started_at = None
        self.current_trial_id = None
        self.current_attempt_number = None
        self.current_scenario_label = "unlabeled"
        return self.dataset_path

    def update_fused_target(self):
        phone_target = self.calibration.target_y_from_tilt(
            self.latest_sensor["tilt"]
        )
        camera_target = math.nan
        disagreement = math.nan
        weight = 0.0
        required_points = {"neutral", "up", "down"}
        sample = self.latest_camera

        if (
            required_points.issubset(self.camera_calibration_points)
            and sample["tracking"]
            and sample["visibility"] >= CAMERA_MIN_VISIBILITY
            and math.isfinite(sample["forearm_angle_deg"])
        ):
            camera_target = self.camera_calibration.target_y_from_tilt(
                sample["forearm_angle_deg"]
            )
            disagreement = abs(camera_target - phone_target)
            if disagreement <= CAMERA_MAX_DISAGREEMENT_FRACTION * GAME_HEIGHT:
                quality = clamp(
                    (sample["visibility"] - CAMERA_MIN_VISIBILITY)
                    / (1.0 - CAMERA_MIN_VISIBILITY),
                    0.0,
                    1.0,
                )
                weight = self.camera_max_fusion_weight * quality

        self.latest_phone_target_y = phone_target
        self.latest_camera_target_y = camera_target
        self.camera_fusion_weight = weight
        self.camera_phone_disagreement = disagreement
        self.latest_target_y = (
            phone_target * (1.0 - weight)
            + (camera_target * weight if math.isfinite(camera_target) else 0.0)
        )

    def reset_all(self):
        self.phase = "start_screen"
        self.grip_failures = 0
        self.pause_started_at = None
        self.final_reason = ""
        self.placeholder_title = ""
        self.menu_message = ""
        self.game.reset()
        self.pong.reset()
        self.analyzer.reset()
        self.difficulty_adapter.reset()

    def loop(self):
        now = time.perf_counter()
        dt = clamp(now - self.last_time, 0, 1 / 30)
        self.last_time = now
        timestamp_ms = now * 1000

        if self.sensor is not None:
            self.sensor.update(dt)
            self.latest_sensor = self.sensor.read()
        if self.camera is not None:
            self.latest_camera = self.camera.sample()
        self.update_fused_target()

        if self.phase in ("test", "assessment", "pong_practice", "pong_assessment"):
            self.analyzer.add_sample(self.latest_target_y, timestamp_ms)
            self.latest_metrics = self.analyzer.get_metrics()
            self.analyzer.record_history(timestamp_ms, self.latest_metrics)

        if self.phase == "test":
            self.game.update(dt, self.latest_target_y, 0.2)
            self.game.pipes.clear()
            self.game.is_game_over = False
        elif self.phase == "assessment":
            self.update_assessment(dt)
        elif self.phase == "grip_pause":
            self.update_grip_pause(now)
        elif self.phase == "pong_practice":
            self.pong.update(dt, self.latest_target_y, True)
        elif self.phase == "pong_assessment":
            self.update_pong_assessment(dt)
        elif self.phase == "pong_grip_pause":
            self.update_pong_grip_pause(now)

        if self.phase in ("assessment", "grip_pause"):
            self.record_trial_sample(self.phase == "grip_pause")
        elif self.phase in ("pong_assessment", "pong_grip_pause"):
            self.record_trial_sample(self.phase == "pong_grip_pause")

        self.draw(now)
        self.root.after(FPS_MS, self.loop)

    def update_assessment(self, dt):
        grip = self.latest_sensor["grip"]
        if grip < GRIP_THRESHOLD:
            self.grip_failures += 1
            self.phase = "grip_pause"
            self.pause_started_at = time.perf_counter()
            return

        self.latest_difficulty = self.difficulty_adapter.update(
            self.latest_metrics, self.game.get_stats(), dt
        )
        self.game.update(dt, self.latest_target_y, self.latest_difficulty)
        if self.game.is_game_over:
            self.end_assessment("Pipe collision.")

    def update_grip_pause(self, now):
        del now
        grip = self.latest_sensor["grip"]
        if grip >= GRIP_THRESHOLD:
            self.phase = "assessment"
            self.pause_started_at = None

    def end_assessment(self, reason):
        self.phase = "results"
        self.final_reason = reason
        self.save_trial_csv("flappy", reason)

    def update_pong_assessment(self, dt):
        grip = self.latest_sensor["grip"]
        if grip < GRIP_THRESHOLD:
            self.grip_failures += 1
            self.phase = "pong_grip_pause"
            self.pause_started_at = time.perf_counter()
            return

        self.pong.update(dt, self.latest_target_y, True)
        if self.pong.is_game_over:
            self.end_pong_assessment("Ball missed.")

    def update_pong_grip_pause(self, now):
        del now
        grip = self.latest_sensor["grip"]
        if grip >= GRIP_THRESHOLD:
            self.phase = "pong_assessment"
            self.pause_started_at = None

    def end_pong_assessment(self, reason):
        self.phase = "pong_results"
        self.final_reason = reason
        self.save_trial_csv("pong", reason)

    def draw(self, now):
        self.click_zones = []
        self.canvas.delete("all")
        if self.phase.startswith("pong_"):
            self.draw_pong_background()
        else:
            self.draw_background()

        if self.phase in ("test", "assessment", "grip_pause", "results"):
            self.draw_pipes()
            self.draw_target_line()
            self.draw_bird()
            self.draw_ground()
        elif self.phase in ("pong_practice", "pong_assessment", "pong_grip_pause", "pong_results"):
            self.draw_pong_game()

        if not self.phase.startswith("pong_"):
            self.draw_hud()
        self.draw_bottom_definitions()
        self.draw_panel(now)

        if self.phase == "start_screen":
            self.draw_start_menu()
        elif self.phase == "game_select":
            self.draw_game_select_menu()
        elif self.phase == "coming_soon":
            self.draw_coming_soon()
        elif self.phase == "welcome":
            self.draw_quit_button()
            self.draw_center_message(
                "neuroFlap Assessment",
                "The iPhone controls arm motion and the ESP32 pad measures grip.",
                "Press Enter to calibrate grip and arm movement.",
            )
        elif self.phase == "pong_welcome":
            self.draw_quit_button()
            self.draw_center_message(
                "accelPong Assessment",
                "The iPhone controls the paddle and the ESP32 pad measures grip.",
                "Press Enter to calibrate grip and arm movement.",
            )
        elif self.phase in ("calibrate_grip", "pong_calibrate_grip"):
            self.draw_quit_button()
            ready = self.latest_sensor["grip_ready"]
            self.draw_center_message(
                "Grip Calibration",
                "Leave the FSR pad completely untouched while resting noise is measured.",
                "Press Enter to continue." if ready else "Calibrating for two seconds...",
            )
        elif self.phase == "calibrate_neutral":
            self.draw_quit_button()
            self.draw_center_message(
                "iPhone Calibration: Neutral",
                "Keep elbow tucked to the body. Hold forearm at 90 degrees to the body.",
                "Press Enter to save neutral.",
            )
        elif self.phase == "pong_calibrate_neutral":
            self.draw_quit_button()
            self.draw_center_message(
                "accelPong iPhone Calibration: Neutral",
                "Keep elbow tucked to the body. Hold forearm at 90 degrees to the body.",
                "Press Enter to save neutral.",
            )
        elif self.phase == "calibrate_up":
            self.draw_quit_button()
            self.draw_center_message(
                "iPhone Calibration: Forearm Up",
                "Keep elbow tucked. Lift your forearm upward around the elbow center.",
                "Press Enter to save upward range.",
            )
        elif self.phase == "pong_calibrate_up":
            self.draw_quit_button()
            self.draw_center_message(
                "accelPong iPhone Calibration: Forearm Up",
                "Keep elbow tucked. Lift your forearm upward around the elbow center.",
                "Press Enter to save upward range.",
            )
        elif self.phase == "calibrate_down":
            self.draw_quit_button()
            self.draw_center_message(
                "iPhone Calibration: Forearm Down",
                "Keep elbow tucked. Lower your forearm downward around the elbow center.",
                "Press Enter to save downward range.",
            )
        elif self.phase == "pong_calibrate_down":
            self.draw_quit_button()
            self.draw_center_message(
                "accelPong iPhone Calibration: Forearm Down",
                "Keep elbow tucked. Lower your forearm downward around the elbow center.",
                "Press Enter to save downward range.",
            )
        elif self.phase == "test":
            self.draw_quit_button()
            self.draw_top_banner(
                "Test phase: move the bird by raising/lowering the forearm from a tucked elbow. Press Enter to start."
            )
        elif self.phase == "pong_practice":
            self.draw_quit_button()
            self.draw_top_banner(
                "accelPong practice: move the paddle by raising/lowering the forearm. Press Enter to start."
            )
        elif self.phase == "grip_pause":
            self.draw_quit_button()
            self.draw_center_message(
                "Assessment Paused",
                "Grip the pad above its calibrated zero level to continue.",
                "The game will remain paused until grip returns.",
            )
        elif self.phase == "pong_grip_pause":
            self.draw_quit_button()
            self.draw_center_message(
                "accelPong Paused",
                "Grip the pad above its calibrated zero level to continue.",
                "The game will remain paused until grip returns.",
            )
        elif self.phase == "results":
            self.draw_quit_button()
            self.draw_results()
        elif self.phase == "pong_results":
            self.draw_quit_button()
            self.draw_pong_results()

    def add_click_zone(self, name, x1, y1, x2, y2):
        self.click_zones.append((name, x1, y1, x2, y2))

    def draw_quit_button(self):
        x1, y1, x2, y2 = WIDTH - 158, 82, WIDTH - 18, 122
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#ffffff", outline="#d6e2f0", width=2)
        self.canvas.create_text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            text="QUIT TO GAMES",
            fill="#d9480f",
            font=("Arial", 10, "bold"),
        )
        self.add_click_zone("quit_flappy", x1, y1, x2, y2)

    def draw_start_menu(self):
        self.canvas.create_rectangle(0, 0, WIDTH, GAME_HEIGHT, fill="#d8f1ff", outline="")
        self.canvas.create_text(
            WIDTH / 2,
            170,
            text="StrokeLess",
            fill="#172033",
            font=("Courier New", 34, "bold"),
        )
        self.canvas.create_text(
            WIDTH / 2,
            220,
            text="a stroke recovery assessment platform based on FSR grip gate and digital accelerometer",
            fill="#435168",
            font=("Times New Roman", 16, "italic"),
        )

        x1, y1, x2, y2 = WIDTH / 2 - 105, 286, WIDTH / 2 + 105, 346
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#1767e8", outline="#0f56c6", width=2)
        self.canvas.create_text(
            WIDTH / 2,
            (y1 + y2) / 2,
            text="START / ENTER",
            fill="#ffffff",
            font=("Courier New", 20, "bold"),
        )
        self.add_click_zone("start", x1, y1, x2, y2)

    def draw_game_select_menu(self):
        self.canvas.create_rectangle(0, 0, WIDTH, GAME_HEIGHT, fill="#d8f1ff", outline="")
        self.canvas.create_text(
            WIDTH / 2,
            90,
            text="Choose Assessment Game",
            fill="#172033",
            font=("Arial", 32, "bold"),
        )
        self.canvas.create_text(
            WIDTH / 2,
            130,
            text="Both games use the same calibrated FSR grip gate and iPhone forearm-angle workflow.",
            fill="#435168",
            font=("Arial", 14),
        )

        cards = [
            ("flappy", 90, "neuroFlap", "Current assessment game", "PLAY"),
            ("another", 365, "accelPong", "Forearm-angle Pong game", "PLAY"),
            ("mystery", 640, "?", "Mystery Game", "COMING SOON"),
        ]

        for name, x, title, subtitle, action in cards:
            self.draw_game_card(name, x, 190, 230, 220, title, subtitle, action)

        if self.menu_message:
            self.canvas.create_text(
                WIDTH / 2,
                GAME_HEIGHT - 100,
                text=self.menu_message,
                fill="#435168",
                font=("Arial", 12, "bold"),
            )

        x1, y1, x2, y2 = WIDTH / 2 - 110, GAME_HEIGHT - 78, WIDTH / 2 + 110, GAME_HEIGHT - 30
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#ffffff", outline="#d6e2f0", width=2)
        self.canvas.create_text(
            WIDTH / 2,
            (y1 + y2) / 2,
            text="HOME",
            fill="#1767e8",
            font=("Arial", 13, "bold"),
        )
        self.add_click_zone("back_home", x1, y1, x2, y2)

    def draw_game_card(self, name, x, y, width, height, title, subtitle, action):
        self.canvas.create_rectangle(
            x,
            y,
            x + width,
            y + height,
            fill="#ffffff",
            outline="#d6e2f0",
            width=2,
        )

        if name == "mystery":
            center_x = x + width / 2
            self.canvas.create_oval(
                center_x - 36,
                y + 26,
                center_x + 36,
                y + 98,
                fill="#172033",
                outline="",
            )
            self.canvas.create_text(
                center_x,
                y + 62,
                text="?",
                fill="#ffffff",
                font=("Arial", 34, "bold"),
            )
            self.canvas.create_text(
                center_x,
                y + 125,
                text="Mystery Game",
                fill="#172033",
                font=("Arial", 18, "bold"),
            )
            self.canvas.create_text(
                center_x,
                y + 154,
                text="Hidden assessment game",
                fill="#435168",
                font=("Arial", 11),
            )
        else:
            self.canvas.create_text(
                x + width / 2,
                y + 58,
                text=title,
                fill="#172033",
                font=("Arial", 22, "bold"),
            )
            self.canvas.create_text(
                x + width / 2,
                y + 94,
                text=subtitle,
                fill="#435168",
                font=("Arial", 12),
                width=width - 34,
            )

        self.canvas.create_rectangle(
            x + 30,
            y + height - 62,
            x + width - 30,
            y + height - 24,
            fill="#1767e8" if name in ("flappy", "another") else "#eef4fb",
            outline="#d6e2f0",
        )
        self.canvas.create_text(
            x + width / 2,
            y + height - 43,
            text=action,
            fill="#ffffff" if name in ("flappy", "another") else "#435168",
            font=("Arial", 11, "bold"),
        )
        self.add_click_zone(name, x, y, x + width, y + height)

    def draw_coming_soon(self):
        title = self.placeholder_title or "Game"
        self.canvas.create_rectangle(0, 0, WIDTH, GAME_HEIGHT, fill="#d8f1ff", outline="")
        self.canvas.create_text(
            WIDTH / 2,
            156,
            text=title,
            fill="#172033",
            font=("Arial", 34, "bold"),
        )
        self.canvas.create_oval(
            WIDTH / 2 - 42,
            206,
            WIDTH / 2 + 42,
            290,
            fill="#172033",
            outline="",
        )
        self.canvas.create_text(
            WIDTH / 2,
            248,
            text="?",
            fill="#ffffff",
            font=("Arial", 40, "bold"),
        )
        self.canvas.create_text(
            WIDTH / 2,
            330,
            text="This game slot is ready for a future Python game.",
            fill="#435168",
            font=("Arial", 15),
        )

        x1, y1, x2, y2 = WIDTH / 2 - 120, 376, WIDTH / 2 + 120, 426
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="#1767e8", outline="#0f56c6", width=2)
        self.canvas.create_text(
            WIDTH / 2,
            (y1 + y2) / 2,
            text="BACK TO GAMES",
            fill="#ffffff",
            font=("Arial", 14, "bold"),
        )
        self.add_click_zone("back_to_menu", x1, y1, x2, y2)

    def draw_background(self):
        self.canvas.create_rectangle(0, 0, WIDTH, GAME_HEIGHT, fill="#9ad8ff", outline="")
        self.canvas.create_oval(82, 70, 178, 128, fill="#ffffff", outline="")
        self.canvas.create_oval(480, 48, 590, 110, fill="#ffffff", outline="")
        self.canvas.create_oval(770, 102, 900, 174, fill="#ffffff", outline="")

    def draw_pong_background(self):
        self.canvas.create_rectangle(0, 0, WIDTH, GAME_HEIGHT, fill="#090d13", outline="")
        for y in range(30, GAME_HEIGHT - 30, 34):
            self.canvas.create_rectangle(WIDTH / 2 - 2, y, WIDTH / 2 + 2, y + 16, fill="#2d3744", outline="")

    def draw_pong_game(self):
        self.canvas.create_rectangle(20, 20, WIDTH - 20, GAME_HEIGHT - 26, outline="#2d3744", width=2)

        pong = self.pong
        self.canvas.create_rectangle(
            pong.paddle_x,
            pong.paddle_y - pong.paddle_h / 2,
            pong.paddle_x + pong.paddle_w,
            pong.paddle_y + pong.paddle_h / 2,
            fill="#38bdf8",
            outline="",
        )
        self.canvas.create_oval(
            pong.ball_x - pong.ball_r,
            pong.ball_y - pong.ball_r,
            pong.ball_x + pong.ball_r,
            pong.ball_y + pong.ball_r,
            fill="#f8fafc",
            outline="",
        )

        if self.phase == "pong_grip_pause":
            self.canvas.create_text(
                WIDTH / 2,
                GAME_HEIGHT / 2 - 24,
                text="PAUSED",
                fill="#f97316",
                font=("Arial", 36, "bold"),
            )
            self.canvas.create_text(
                WIDTH / 2,
                GAME_HEIGHT / 2 + 24,
                text="Grip harder to resume",
                fill="#cbd5e1",
                font=("Arial", 16, "bold"),
            )

    def draw_pipes(self):
        if self.phase == "test":
            return
        for pipe in self.game.pipes:
            x = pipe["x"]
            width = pipe["width"]
            gap_top = pipe["current_gap_y"] - pipe["gap_size"] / 2
            gap_bottom = pipe["current_gap_y"] + pipe["gap_size"] / 2
            self.draw_pipe_rect(x, 0, width, gap_top)
            self.draw_pipe_rect(x, gap_bottom, width, GAME_HEIGHT - gap_bottom)

    def draw_pipe_rect(self, x, y, width, height):
        self.canvas.create_rectangle(
            x, y, x + width, y + height, fill="#2fb85e", outline="#137736", width=4
        )
        self.canvas.create_rectangle(
            x + 8, y + 4, x + 20, max(y + 4, y + height - 4), fill="#6bd987", outline=""
        )

    def draw_target_line(self):
        for x in range(0, WIDTH, 28):
            self.canvas.create_line(
                x, self.latest_target_y, x + 12, self.latest_target_y, fill="#1767e8", width=3
            )

    def draw_bird(self):
        bird = self.game.bird
        if self.bird_image:
            self.canvas.create_image(bird["x"], bird["y"], image=self.bird_image)
        else:
            r = bird["radius"]
            self.canvas.create_oval(
                bird["x"] - r,
                bird["y"] - r,
                bird["x"] + r,
                bird["y"] + r,
                fill="#ffd33d",
                outline="#5a3d00",
                width=3,
            )

    def draw_ground(self):
        self.canvas.create_rectangle(
            0, GAME_HEIGHT - 24, WIDTH, GAME_HEIGHT, fill="#67b64c", outline=""
        )

    def draw_hud(self):
        if self.phase.startswith("pong_"):
            second_label = "Difficulty"
            score_text = f"{self.pong.hits}/{self.pong.hits + self.pong.misses}"
            difficulty_text = f"Ball speed {round(abs(self.pong.ball_vx))}"
        else:
            second_label = "Difficulty"
            score_text = str(self.game.score)
            difficulty_text = f"{round(self.latest_difficulty * 100)}%"

        items = [
            ("Score", score_text),
            (second_label, difficulty_text),
            ("Stability", f"{round(self.latest_metrics['stability_score'] * 100)}%"),
            ("Speed", f"{round(self.latest_metrics['hand_speed'])}"),
        ]
        for index, (label, value) in enumerate(items):
            x = 12 + index * 176
            self.canvas.create_rectangle(
                x, 12, x + 164, 68, fill="#ffffff", outline="#d6e2f0", width=2
            )
            self.canvas.create_text(
                x + 12,
                25,
                text=label.upper(),
                anchor="w",
                fill="#53627a",
                font=("Arial", 9, "bold"),
            )
            self.canvas.create_text(
                x + 12,
                49,
                text=value,
                anchor="w",
                fill="#172033",
                font=("Arial", 19, "bold"),
            )

    def draw_bottom_definitions(self):
        self.canvas.create_rectangle(
            0, GAME_HEIGHT, WIDTH, HEIGHT, fill="#f8fbff", outline="#d6e2f0"
        )
        if self.phase.startswith("pong_"):
            total = self.pong.hits + self.pong.misses
            definitions = [
                ("Score", f"{self.pong.hits}/{total}", "Paddle hits compared with total ball attempts."),
                ("Difficulty", f"{round(abs(self.pong.ball_vx))} px/s", "Ball speed increases after successful hits."),
                ("Stability", f"{round(self.latest_metrics['stability_score'] * 100)}%", "How little tremor or shaky paddle motion is detected."),
                ("Speed", f"{round(self.latest_metrics['hand_speed'])} px/s", "Average vertical forearm-angle paddle target speed."),
            ]
        else:
            definitions = [
                ("Score", f"{self.game.score}", "Pipes passed during the assessment."),
                ("Difficulty", f"{round(self.latest_difficulty * 100)}%", "Pipe speed, gap size, and movement challenge."),
                ("Stability", f"{round(self.latest_metrics['stability_score'] * 100)}%", "How little tremor or shaky motion is detected."),
                ("Speed", f"{round(self.latest_metrics['hand_speed'])} px/s", "Average vertical forearm-angle bird target speed."),
            ]
        column_width = WIDTH / 4
        for index, (label, value, definition) in enumerate(definitions):
            x = index * column_width + 16
            self.canvas.create_text(
                x,
                GAME_HEIGHT + 18,
                text=f"{label}: {value}",
                anchor="w",
                fill="#172033",
                font=("Arial", 12, "bold"),
            )
            self.canvas.create_text(
                x,
                GAME_HEIGHT + 45,
                text=definition,
                anchor="w",
                width=column_width - 28,
                fill="#435168",
                font=("Arial", 10),
            )

    def draw_panel(self, now):
        left = WIDTH
        panel_title = "accelPong Assessment" if self.phase.startswith("pong_") else "Sensor Assessment"
        self.canvas.create_rectangle(left, 0, WIDTH + PANEL_WIDTH, HEIGHT, fill="#eaf5ff", outline="")
        self.canvas.create_rectangle(
            left + 16,
            16,
            WIDTH + PANEL_WIDTH - 16,
            HEIGHT - 16,
            fill="#ffffff",
            outline="#d6e2f0",
            width=2,
        )
        self.canvas.create_text(
            left + 32, 42, text=panel_title, anchor="w", fill="#172033", font=("Arial", 21, "bold")
        )
        self.canvas.create_text(
            left + 32, 72, text=f"Phase: {self.phase.replace('_', ' ').title()}", anchor="w", fill="#435168", font=("Arial", 11, "bold")
        )

        self.draw_status_bar(left + 32, 100, "FSR Grip", self.latest_sensor["grip"], GRIP_THRESHOLD)
        self.draw_status_bar(left + 32, 148, "Forearm Angle", (self.latest_sensor["tilt"] + 1) / 2, 0.5)
        self.canvas.create_text(
            left + 32,
            194,
            text=(
                f"iPhone accel: "
                f"X={self.latest_sensor['accel_x']:+.2f}g  "
                f"Y={self.latest_sensor['accel_y']:+.2f}g  "
                f"Z={self.latest_sensor['accel_z']:+.2f}g"
            ),
            anchor="w",
            fill="#435168",
            font=("Arial", 9, "bold"),
        )

        self.canvas.create_text(
            left + 32,
            220,
            text=(f"FSR raw: {self.latest_sensor['grip_raw']}   "
                  f"zero: {self.latest_sensor['grip_threshold']}"),
            anchor="w",
            fill="#172033",
            font=("Arial", 11, "bold"),
        )
        sensor_status = self.sensor.status() if self.sensor is not None else "Not connected"
        self.canvas.create_text(
            left + 32, 242, text=sensor_status[:48], anchor="w",
            fill="#435168", font=("Arial", 9, "bold"),
        )
        camera_status = (
            self.camera.status() if self.camera is not None else "Camera: disabled"
        )
        if self.camera is not None:
            camera_status += f", fusion {self.camera_fusion_weight * 100:.0f}%"
        self.canvas.create_text(
            left + 32, 257, text=camera_status[:48], anchor="w",
            fill="#435168", font=("Arial", 9, "bold"),
        )
        self.canvas.create_text(
            left + 32,
            280,
            text=f"Assessment score: {self.latest_metrics['assessment_score']:.1f}/10",
            anchor="w",
            fill="#172033",
            font=("Arial", 18, "bold"),
        )

        chart_specs = [
            ("Speed", "speed", HEALTHY_STANDARD["excessive_speed_px_per_sec"], "#1767e8", 320),
            ("Tremor", "tremor", HEALTHY_STANDARD["upper_tremor_px"] * 1.4, "#d9480f", 428),
            ("Smoothness", "smoothness", 1, "#2f9e44", 536),
        ]
        for title, key, maximum, color, y in chart_specs:
            self.canvas.create_text(left + 32, y, text=title, anchor="w", fill="#172033", font=("Arial", 11, "bold"))
            self.draw_chart(left + 32, y + 16, 296, 70, key, maximum, color)

        if self.phase in ("grip_pause", "pong_grip_pause"):
            self.canvas.create_text(
                left + 32,
                HEIGHT - 42,
                text="Paused at zero grip; squeeze to continue.",
                anchor="w",
                fill="#d9480f",
                font=("Arial", 11, "bold"),
            )

    def draw_status_bar(self, x, y, label, value, threshold):
        width = 296
        height = 18
        self.canvas.create_text(x, y, text=f"{label}: {value:.2f}", anchor="w", fill="#172033", font=("Arial", 11, "bold"))
        self.canvas.create_rectangle(x, y + 18, x + width, y + 18 + height, fill="#eef4fb", outline="#d6e2f0")
        self.canvas.create_rectangle(x, y + 18, x + width * clamp(value, 0, 1), y + 18 + height, fill="#2f9e44", outline="")
        threshold_x = x + width * clamp(threshold, 0, 1)
        self.canvas.create_line(threshold_x, y + 14, threshold_x, y + 40, fill="#d9480f", width=2)

    def draw_chart(self, x, y, width, height, key, maximum, color):
        self.canvas.create_rectangle(x, y, x + width, y + height, fill="#f8fbff", outline="#d6e2f0")
        history = self.analyzer.history
        if len(history) < 2:
            return
        first_time = history[0]["timestamp"]
        last_time = history[-1]["timestamp"]
        span = max(last_time - first_time, 1)
        points = []
        for sample in history:
            point_x = x + ((sample["timestamp"] - first_time) / span) * width
            normalized = clamp(sample[key] / maximum, 0, 1)
            point_y = y + height - normalized * height
            points.extend([point_x, point_y])
        if len(points) >= 4:
            self.canvas.create_line(*points, fill=color, width=3, smooth=True)

    def draw_top_banner(self, message):
        self.canvas.create_rectangle(18, 82, WIDTH - 18, 126, fill="#ffffff", outline="#d6e2f0", width=2)
        self.canvas.create_text(WIDTH / 2, 104, text=message, fill="#172033", font=("Arial", 13, "bold"))

    def draw_center_message(self, title, body, action):
        self.canvas.create_rectangle(156, 178, WIDTH - 156, 372, fill="#ffffff", outline="#d6e2f0", width=2)
        self.canvas.create_text(WIDTH / 2, 224, text=title, fill="#172033", font=("Arial", 28, "bold"))
        self.canvas.create_text(WIDTH / 2, 276, text=body, fill="#435168", font=("Arial", 15), width=560)
        self.canvas.create_text(WIDTH / 2, 326, text=action, fill="#1767e8", font=("Arial", 17, "bold"))

    def draw_results(self):
        self.draw_session_summary("flappy")

    def draw_pong_results(self):
        self.draw_session_summary("pong")

    def dominant_pattern_note(self):
        if not self.last_prediction:
            return "Model pattern note unavailable."
        descriptions = {
            "limited_movement": "reduced movement range",
            "low_grip": "lower sustained grip",
            "unstable_grip": "variable grip force",
            "grip_loss": "periods of grip release",
            "unstable_movement": "less smooth movement control",
            "high_tremor": "a higher rapid-movement/tremor proxy",
        }
        probabilities = {
            label: self.last_prediction.get(f"probability_{label}", 0.0)
            for label in descriptions
        }
        label = max(probabilities, key=probabilities.get)
        probability = probabilities[label]
        if probability < 0.50:
            return "No dominant negative gameplay pattern was predicted."
        note = f"Most prominent measured pattern: {descriptions[label]} ({probability:.0%})."
        if label == "high_tremor":
            note += " This is a gameplay motion proxy, not a clinical tremor finding."
        return note

    def draw_session_summary(self, game_name):
        grips = [sample["grip_percent"] for sample in self.trial_samples]
        grip_avg = sum(grips) / len(grips) if grips else 0
        grip_max = max(grips) if grips else 0
        prediction = self.last_prediction or {}
        movement_score = prediction.get("movement_score", 0.0)
        grip_score = prediction.get("grip_score", 0.0)
        final_score = prediction.get("final_score")
        band = prediction.get("result_band", "unavailable")
        band_details = {
            "strong_gameplay_motor_control": ("Great control", "#2f9e44"),
            "moderate_gameplay_motor_control": ("Moderate control", "#d97706"),
            "gameplay_patterns_need_improvement": ("Poor control", "#d9480f"),
            "unavailable": ("Score unavailable", "#53627a"),
        }
        band_text, band_color = band_details.get(band, band_details["unavailable"])
        if game_name == "flappy":
            game_score_text = f"{self.game.score} pipes passed"
            session_time = self.game.time_alive
            enter_text = "Enter: home"
        else:
            game_score_text = f"{self.pong.hits} paddle hits"
            session_time = self.pong.time_alive
            enter_text = "Enter: games"

        self.canvas.create_rectangle(
            72, 82, WIDTH - 72, 506, fill="#ffffff", outline="#d6e2f0", width=2
        )
        self.canvas.create_text(
            112, 118, text="Session Summary", anchor="w",
            fill="#172033", font=("Arial", 28, "bold")
        )
        self.canvas.create_text(
            112, 157, text=f"Game ended: {self.final_reason}", anchor="w",
            fill="#435168", font=("Arial", 13)
        )
        self.canvas.create_text(
            112, 190,
            text=f"Overall Control Score: {self.latest_metrics['assessment_score']:.1f} / 10",
            anchor="w", fill="#172033", font=("Arial", 16, "bold")
        )
        self.canvas.create_text(
            112, 218, text=f"Game Score: {game_score_text}", anchor="w",
            fill="#172033", font=("Arial", 14)
        )
        self.canvas.create_text(
            112, 244, text=f"Session Time: {session_time:.1f} seconds", anchor="w",
            fill="#172033", font=("Arial", 14)
        )

        self.canvas.create_text(
            112, 282, text="Grip Control", anchor="w",
            fill="#1767e8", font=("Arial", 15, "bold")
        )
        self.canvas.create_text(
            112, 308, text=f"Average grip: {grip_avg:.1f}%", anchor="w",
            fill="#172033", font=("Arial", 13)
        )
        self.canvas.create_text(
            112, 332, text=f"Maximum grip: {grip_max:.1f}%", anchor="w",
            fill="#172033", font=("Arial", 13)
        )

        self.canvas.create_text(
            430, 282, text="Movement Control", anchor="w",
            fill="#1767e8", font=("Arial", 15, "bold")
        )
        self.canvas.create_text(
            430, 308, text=f"Movement score: {movement_score:.0f} / 100", anchor="w",
            fill="#172033", font=("Arial", 13)
        )
        self.canvas.create_text(
            430, 332, text=f"Grip score: {grip_score:.0f} / 100", anchor="w",
            fill="#172033", font=("Arial", 13)
        )

        ai_text = "AI Motor Control Score: unavailable"
        if final_score is not None:
            ai_text = f"AI Motor Control Score: {final_score:.1f} / 100 - {band_text}"
        self.canvas.create_text(
            WIDTH / 2, 376, text=ai_text, fill=band_color,
            font=("Arial", 21, "bold")
        )
        self.canvas.create_text(
            WIDTH / 2, 414, text=self.dominant_pattern_note(),
            fill="#53627a", font=("Arial", 10), width=700
        )
        saved_name = self.last_csv_path.name if self.last_csv_path else "not saved"
        self.canvas.create_text(
            WIDTH / 2, 458, text=f"Saved to: {saved_name}",
            fill="#172033", font=("Arial", 12, "bold")
        )
        self.canvas.create_text(
            WIDTH / 2, 484,
            text=f"{enter_text} | Research gameplay result - not a medical diagnosis",
            fill="#1767e8", font=("Arial", 10)
        )


def main():
    parser = argparse.ArgumentParser(description="NervaFlex rehab-style games")
    parser.add_argument(
        "phyphox_url",
        nargs="?",
        help="phyphox Remote Access URL, for example http://192.168.68.60",
    )
    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument(
        "--study-mode",
        choices=("training", "validation"),
        default="training",
        help="training appends to model data; validation keeps new trials separate",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="launch the localhost right-arm webcam tracker",
    )
    parser.add_argument("--camera-port", type=int, default=8765)
    parser.add_argument(
        "--camera-weight",
        type=float,
        default=CAMERA_MAX_FUSION_WEIGHT,
        help="maximum camera fusion weight from 0.0 (shadow) to 0.5",
    )
    args = parser.parse_args()
    root = tk.Tk()
    FlappySensorAssessmentApp(
        root,
        args.phyphox_url,
        args.serial_port,
        args.study_mode,
        args.camera,
        args.camera_port,
        args.camera_weight,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
