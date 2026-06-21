# StrokeHub

**StrokeHub** is a rehab-style game platform developed by **Thien “Timmy” Nguyen Huu, Charles Zhang, Ryunosuke Suzuki, and Lily Duong**. The system turns movement and grip tracking into interactive gameplay using phone accelerometer motion, FSR grip sensing, camera-based motion capture, and machine learning to provide personalized feedback on motor-control patterns during simple games like **Flappy Bird** and **Pong**.

StrokeHub was built as a hackathon proof of concept for game-based movement assessment and feedback. It is designed to make motor-control tracking more engaging, accessible, and easier to understand.

> **Disclaimer:** StrokeHub is not a medical diagnostic tool. It does not diagnose stroke, Parkinson’s disease, tremor disorders, or any neurological condition. It provides gameplay-based movement and grip feedback only.

---

## Team

StrokeHub was developed by:

* Thien “Timmy” Nguyen Huu
* Charles Zhang
* Ryunosuke Suzuki
* Lily Duong

---
Notes: All the file names are Nervaflex because that was intended in the first place, but now they are all part of the StrokeHub now. 
---

## Overview

Traditional motor assessments can feel repetitive, clinical, and difficult to track over time. StrokeHub explores a more interactive approach: using games to create structured movement tasks while collecting real-time sensor data.

During gameplay, the player controls the game using arm or forearm movement while maintaining grip on an FSR sensor. The system records movement, grip, game performance, stability, smoothness, camera-based motion agreement, and difficulty progression. After each trial, StrokeHub summarizes the session and uses machine learning to classify gameplay-based motor-control patterns.

StrokeHub combines **skin-contact sensing** and **visual motion capture**. The phone accelerometer provides direct movement data from the user’s arm, while the camera runs in shadow mode as a visual validation layer. This creates a sensor-fusion approach that improves confidence in the movement feedback without relying completely on one signal source.

---

## Key Features

* Two interactive assessment games:

  * **NeuroFlap**: a Flappy Bird-style movement game
  * **AccelPong**: a Pong-style reaction and movement game
* Phone accelerometer-based arm movement tracking
* FSR grip sensing through ESP32
* Camera-based motion capture running in shadow mode
* Sensor fusion between phone accelerometer motion, FSR grip sensing, and camera pose tracking
* Camera confidence scoring based on visibility, motion agreement, and pose-tracking confidence
* Optional 10% camera contribution to the final result when tracking quality is high
* Personalized calibration for movement range and grip strength
* Frame-by-frame gameplay and sensor data logging
* Trial-level feature extraction for machine learning
* Random Forest-based multi-label classification
* Applied AI model for motion capture, sensor fusion, and gameplay-based data analysis
* User-friendly session feedback and score breakdown
* Expandable design for future EMG integration

---

## How It Works

StrokeHub uses gameplay as a structured motor-control task.

1. The player chooses a game.
2. The player calibrates their arm movement range and grip range.
3. The game begins.
4. Phone accelerometer data controls the game character or paddle.
5. The FSR sensor measures grip engagement.
6. The camera runs in shadow mode to visually track arm movement.
7. The system logs sensor, camera, and gameplay data every frame.
8. When the trial ends, the data is saved to CSV.
9. StrokeHub extracts trial-level features.
10. A machine learning model predicts movement and grip pattern labels.
11. A sensor-fusion layer checks agreement between phone motion and camera motion.
12. The player receives a readable performance summary.

---

## Current Sensors

### Phone Accelerometer

The phone accelerometer is used to estimate arm or forearm movement. During calibration, the player records their lowest and highest comfortable movement range. During gameplay, live accelerometer data is mapped into a normalized movement score.

The phone accelerometer is the primary movement signal because it is attached directly to the user’s arm or forearm.

### FSR Grip Sensor

The FSR sensor measures relative grip pressure. It is used to track whether the player maintains grip during the game and whether grip control changes over time.

Grip data helps determine whether the player is actively engaging with the task and whether grip stability changes during gameplay.

### Camera Capture Motion

StrokeHub also includes a camera-based motion capture section that runs in **shadow mode**. In this mode, the webcam does not replace the phone accelerometer or FSR grip sensor. Instead, it acts as a visual validation layer that tracks arm movement through pose estimation and compares it with the phone-based motion signal.

The goal of shadow mode is to improve confidence in the movement measurement by checking whether the visual motion captured by the camera agrees with the skin-contact motion captured by the phone accelerometer.

When camera visibility is high and the camera-phone motion agreement is strong, the camera score can contribute up to **10%** of the final result. If visibility is poor, tracking confidence is low, or the camera and phone signals do not match well, the camera contribution is reduced or ignored.

This allows StrokeHub to combine three complementary sensing methods:

* **Phone accelerometer:** skin-contact motion tracking
* **Camera capture:** visual motion tracking
* **FSR grip sensor:** grip engagement and grip stability

Together, these signals create a sensor-fusion approach for more reliable movement feedback.

### Future EMG Support

StrokeHub is designed to support EMG integration later. Future EMG features may include muscle activation level, activation timing, fatigue-like trends, and effort-to-movement ratio.

Current EMG-related columns are planned as placeholders and can be added without redesigning the full machine learning pipeline.

---

## Games

### StrokeFlap

StrokeFlap is a Flappy Bird-style game where arm movement controls the bird’s vertical position. The player must guide the bird through pipes while maintaining grip control.

Tracked metrics include:

* Pipes passed
* Survival time
* Movement speed
* Movement smoothness
* Movement stability
* Grip control
* Difficulty progression
* Camera-phone movement agreement

### StrokePong

StrokePong is a Pong-style game where arm movement controls the paddle. The player must return the ball while maintaining grip and responding to changing ball movement.

Tracked metrics include:

* Paddle hits
* Misses
* Reaction-style movement response
* Movement range
* Movement smoothness
* Grip stability
* Game difficulty progression
* Camera-phone movement agreement

---

## Camera Shadow Mode

Camera shadow mode is a secondary motion-capture system that runs alongside the main phone accelerometer and FSR sensor setup.

The camera uses pose tracking to estimate arm position visually. This visual signal is then compared with the phone accelerometer signal. If both signals agree, the system gains more confidence that the measured movement reflects the player’s actual arm motion.

Camera shadow mode is useful because phone and camera sensing each have different strengths:

| Signal              | Strength                            | Limitation                                               |
| ------------------- | ----------------------------------- | -------------------------------------------------------- |
| Phone accelerometer | Direct skin-contact motion tracking | Can be affected by phone placement                       |
| Camera capture      | Visual confirmation of arm movement | Can be affected by lighting, occlusion, and camera angle |
| FSR grip sensor     | Measures grip engagement            | Does not measure arm position                            |

By combining these signals, StrokeHub aims to improve motion tracking reliability while keeping the system accessible and low-cost.

---

## Sensor Fusion Logic

StrokeHub is designed to combine skin-contact sensing and visual motion capture. The phone accelerometer provides the primary movement signal because it is attached directly to the user’s arm or forearm. The camera runs in the background as a secondary validation signal.

The camera contribution is based on three conditions:

1. **Visibility:** the camera must clearly detect the arm landmarks.
2. **Motion matching:** the camera-based arm angle should agree with the phone accelerometer movement trend.
3. **Confidence:** the pose-tracking model must report stable tracking confidence.

If these conditions are met, the camera adds a small weighted contribution to the final result:

```text
Final Score = 90% sensor/game model + 10% camera validation score
```

If the camera signal is unreliable, StrokeHub falls back to the phone accelerometer, FSR grip sensor, and gameplay features only.

This fusion design allows StrokeHub to improve accuracy without depending entirely on camera tracking, which can be affected by lighting, visibility, camera angle, and background conditions.

---

## Machine Learning Pipeline

StrokeHub uses a trial-based machine learning pipeline. The raw CSV contains frame-by-frame sensor, camera, and gameplay data, but the machine learning model is trained on trial-level summary features.

Pipeline:

```text
Raw gameplay CSV
→ group by trial_id
→ extract trial-level features
→ create rule-based labels
→ train multi-label Random Forest model
→ predict gameplay-based motor-control patterns
→ apply camera shadow-mode validation
→ generate user-friendly feedback
```

### Example Features

Movement features:

* Movement range used
* Average movement score
* Movement smoothness
* Movement variance
* Tremor-like motion proxy
* Movement speed
* Reaction-style response timing

Grip features:

* Average grip strength
* Maximum grip strength
* Grip stability
* Grip active percentage
* Grip loss events

Camera features:

* Camera visibility
* Forearm angle from pose tracking
* Elbow angle from pose tracking
* Shoulder elevation estimate
* Camera-phone motion agreement
* Camera confidence score

Game features:

* Game score
* Survival time
* Difficulty level
* Hits or pipes passed
* Failure reason
* Final assessment score

Future EMG features:

* Mean EMG activation
* Peak EMG activation
* EMG onset time
* EMG variance
* Fatigue-like EMG slope
* Effort-to-movement ratio

### Example Labels

The current model predicts gameplay-based labels such as:

* `limited_movement`
* `low_grip`
* `unstable_grip`
* `grip_loss`
* `unstable_movement`
* `high_tremor`
* `sensor_control_good`
* `good_control`

These labels describe gameplay and sensor patterns. They are not clinical diagnoses.

---

## Repository Structure

```text
StrokeHub/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── games/
│   │   ├── flappy_motion_sensor.py
│   │   └── pong_iphone.py
│   ├── sensors/
│   │   ├── microcontroller_setup.ino
│   │   └── webcam_tracker.html
│   ├── ml/
│   │   ├── train_strokehub_model.py
│   │   ├── predict_strokehub_study.py
│   │   └── feature_extraction.py
│   └── reports/
│       └── strokehub_analysis_report.txt
├── data/
│   ├── raw/
│   ├── validation/
│   └── processed/
├── models/
│   └── strokehub_rf_model.pkl
├── results/
│   ├── strokehub_trial_predictions.csv
│   ├── strokehub_validation_predictions.csv
│   └── strokehub_rf_classification_report.txt
└── docs/
    ├── project_overview.md
    ├── ml_pipeline.md
    └── hardware_setup.md
```

---

## Tech Stack

* Python
* Pygame
* Tkinter
* ESP32
* FSR grip sensor
* Phone accelerometer
* Camera-based pose tracking
* pandas
* NumPy
* scikit-learn
* joblib
* HTML / JavaScript webcam tracking prototype

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/StrokeHub.git
cd StrokeHub
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Games

Run the Flappy Bird-style game:

```bash
python src/games/flappy_motion_sensor.py
```

Run the Pong-style game:

```bash
python src/games/pong_iphone.py
```

Run the prediction pipeline:

```bash
python src/ml/predict_strokehub_study.py
```

Open the webcam tracker:

```text
src/sensors/webcam_tracker.html
```

The webcam tracker should be served locally or opened in a browser environment that allows camera access.

---

## Hardware Setup

Current hardware:

* ESP32
* FSR grip sensor
* Phone accelerometer
* Optional webcam tracker prototype
* Future EMG sensor support

Suggested ESP32 analog pin mapping:

```text
FSR signal → GPIO34
EMG signal → GPIO35
SDA        → GPIO21
SCL        → GPIO22
GND        → common GND
VCC        → 3.3V
```

The EMG channel is planned for future expansion and is not required for the current version.

---

## Data Collection

Each trial saves frame-level data such as:

* Timestamp
* Game type
* Trial ID
* Accelerometer values
* Grip value
* Movement score
* Camera visibility
* Camera forearm angle
* Camera-phone agreement score
* Game score
* Difficulty level
* Smoothness
* Stability
* Outcome

The raw data is then summarized into trial-level features for machine learning.

---

## Model Notes

The current model is a proof-of-concept Random Forest classifier trained on gameplay-derived features. It is designed for small tabular datasets and interpretable feature engineering.

The model combines:

* Movement features
* Grip features
* Game performance features
* Optional camera validation features

The camera shadow-mode score is designed to support the final result only when visibility, tracking confidence, and camera-phone agreement are high.

The model should be evaluated carefully because current data may come from limited users, limited trials, and simulated or prototype conditions. Future work should include more participants, standardized protocols, and external validation.

---

## Future Improvements

* Add EMG muscle activation sensing
* Improve reaction time event detection
* Add more games and motor-control tasks
* Improve webcam-based movement validation
* Add participant IDs and longitudinal tracking
* Create a cleaner dashboard for session history
* Expand dataset size across more users
* Compare sensor-based movement with camera-based pose tracking
* Improve sensor-fusion weighting between phone, FSR, camera, and future EMG
* Build a web app version for easier deployment

---

## Project Goal

StrokeHub aims to make movement tracking more interactive and understandable by combining games, sensors, camera-based motion capture, and machine learning. Instead of presenting raw sensor values, it translates movement and grip data into user-friendly feedback that can help visualize motor-control performance during structured gameplay.

The long-term goal is to build a flexible sensor-fusion platform that supports both skin-contact motion sensing and visual motion capture for accessible rehab-style movement feedback.
