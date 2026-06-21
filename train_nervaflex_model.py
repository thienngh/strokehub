#!/usr/bin/env python3
"""Train a proof-of-concept NervaFlex gameplay-pattern classifier.

This model classifies rule-defined movement and grip patterns. It is not a
clinical diagnostic model and must not be used to diagnose a health condition.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


EMG_FEATURES = [
    "mean_emg_activation",
    "peak_emg_activation",
    "emg_onset_time",
    "emg_variance",
    "fatigue_like_emg_slope",
    "effort_to_movement_ratio",
]

CAMERA_FEATURES = [
    "camera_available",
    "camera_tracking_percent",
    "camera_forearm_range_deg",
    "camera_forearm_mean_deg",
    "camera_forearm_std_deg",
    "camera_forearm_min_deg",
    "camera_forearm_max_deg",
    "camera_elbow_range_deg",
    "camera_elbow_mean_deg",
    "camera_elbow_std_deg",
    "camera_shoulder_range_deg",
    "camera_shoulder_mean_deg",
    "camera_shoulder_std_deg",
    "camera_visibility_mean",
    "camera_visibility_min",
    "camera_iphone_angle_correlation",
]

MODEL_FEATURES = [
    "game",
    "outcome",
    "trial_duration_s",
    "final_assessment_score",
    "tilt_range",
    "tilt_mean",
    "tilt_std",
    "tilt_min",
    "tilt_max",
    "avg_grip_percent",
    "max_grip_percent",
    "min_grip_percent",
    "grip_std",
    "grip_active_percent",
    "paused_percent",
    "avg_smoothness",
    "min_smoothness",
    "avg_tremor_px",
    "max_tremor_px",
    "avg_speed_px_s",
    "max_speed_px_s",
    "max_difficulty",
    "final_flappy_score",
    "final_pong_hits",
    "final_pong_misses",
    *CAMERA_FEATURES,
    *EMG_FEATURES,
]

LABEL_COLUMNS = [
    "limited_movement",
    "low_grip",
    "unstable_grip",
    "grip_loss",
    "unstable_movement",
    "high_tremor",
    "sensor_control_good",
    "good_control",
]

CATEGORICAL_FEATURES = ["game", "outcome"]
NUMERIC_FEATURES = [
    column for column in MODEL_FEATURES if column not in CATEGORICAL_FEATURES
]


def load_data(csv_path):
    """Load frame-level data and add safe placeholders for key identifiers."""
    data = pd.read_csv(csv_path, low_memory=False)
    if data.empty:
        raise ValueError(f"No rows found in {csv_path}")

    if "trial_id" not in data.columns:
        if "attempt_number" in data.columns:
            data["trial_id"] = "attempt_" + data["attempt_number"].astype(str)
        else:
            data["trial_id"] = "trial_0001"
    data["trial_id"] = data["trial_id"].fillna("unknown_trial").astype(str)
    return data


def _numeric(group, column):
    if column not in group.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(group[column], errors="coerce").dropna()


def _first_text(group, column, default="unknown"):
    if column not in group.columns:
        return default
    values = group[column].dropna()
    return str(values.iloc[0]) if not values.empty else default


def _first_number(group, column, default=np.nan):
    values = _numeric(group, column)
    return float(values.iloc[0]) if not values.empty else default


def _stat(values, operation, default=np.nan):
    if values.empty:
        return default
    return float(getattr(values, operation)())


def extract_trial_features(frame_data):
    """Convert frame-level rows into one feature row for each trial_id."""
    trials = []

    for trial_id, group in frame_data.groupby("trial_id", sort=False):
        tilt = _numeric(group, "tilt")
        grip = _numeric(group, "grip_percent")
        paused = _numeric(group, "paused")
        smoothness = _numeric(group, "smoothness")
        tremor = _numeric(group, "tremor_px")
        speed = _numeric(group, "speed_px_s")
        difficulty = _numeric(group, "difficulty")
        time_values = _numeric(group, "time_s")
        camera_tracking = _numeric(group, "camera_tracking")
        camera_forearm = _numeric(group, "camera_forearm_angle_deg")
        camera_elbow = _numeric(group, "camera_elbow_angle_deg")
        camera_shoulder = _numeric(group, "camera_shoulder_elevation_deg")
        camera_visibility = _numeric(group, "camera_visibility")

        duration = _first_number(group, "trial_duration_s")
        if np.isnan(duration) and not time_values.empty:
            duration = float(time_values.max() - time_values.min())

        trial = {
            "trial_id": trial_id,
            "attempt_number": _first_number(group, "attempt_number"),
            "trial_started_at": _first_text(group, "trial_started_at", "unknown"),
            "scenario_label": _first_text(group, "scenario_label", "unlabeled"),
            "game": _first_text(group, "game"),
            "outcome": _first_text(group, "outcome"),
            "trial_duration_s": duration,
            "final_assessment_score": _first_number(
                group, "final_assessment_score"
            ),
            "tilt_range": (
                float(tilt.max() - tilt.min()) if not tilt.empty else np.nan
            ),
            "tilt_mean": _stat(tilt, "mean"),
            "tilt_std": _stat(tilt, "std", 0.0),
            "tilt_min": _stat(tilt, "min"),
            "tilt_max": _stat(tilt, "max"),
            "avg_grip_percent": _stat(grip, "mean"),
            "max_grip_percent": _stat(grip, "max"),
            "min_grip_percent": _stat(grip, "min"),
            "grip_std": _stat(grip, "std", 0.0),
            "grip_active_percent": (
                float((grip > 0).mean() * 100) if not grip.empty else np.nan
            ),
            "paused_percent": (
                float((paused > 0).mean() * 100) if not paused.empty else np.nan
            ),
            "avg_smoothness": _stat(smoothness, "mean"),
            "min_smoothness": _stat(smoothness, "min"),
            "avg_tremor_px": _stat(tremor, "mean"),
            "max_tremor_px": _stat(tremor, "max"),
            "avg_speed_px_s": _stat(speed, "mean"),
            "max_speed_px_s": _stat(speed, "max"),
            "max_difficulty": _stat(difficulty, "max"),
            "final_flappy_score": _first_number(group, "final_flappy_score", 0),
            "final_pong_hits": _first_number(group, "final_pong_hits", 0),
            "final_pong_misses": _first_number(group, "final_pong_misses", 0),
            "camera_available": int(
                not camera_tracking.empty and camera_tracking.gt(0).any()
            ),
            "camera_tracking_percent": (
                float(camera_tracking.gt(0).mean() * 100)
                if not camera_tracking.empty else np.nan
            ),
            "camera_forearm_range_deg": (
                float(camera_forearm.max() - camera_forearm.min())
                if not camera_forearm.empty else np.nan
            ),
            "camera_forearm_mean_deg": _stat(camera_forearm, "mean"),
            "camera_forearm_std_deg": _stat(camera_forearm, "std", 0.0),
            "camera_forearm_min_deg": _stat(camera_forearm, "min"),
            "camera_forearm_max_deg": _stat(camera_forearm, "max"),
            "camera_elbow_range_deg": (
                float(camera_elbow.max() - camera_elbow.min())
                if not camera_elbow.empty else np.nan
            ),
            "camera_elbow_mean_deg": _stat(camera_elbow, "mean"),
            "camera_elbow_std_deg": _stat(camera_elbow, "std", 0.0),
            "camera_shoulder_range_deg": (
                float(camera_shoulder.max() - camera_shoulder.min())
                if not camera_shoulder.empty else np.nan
            ),
            "camera_shoulder_mean_deg": _stat(camera_shoulder, "mean"),
            "camera_shoulder_std_deg": _stat(camera_shoulder, "std", 0.0),
            "camera_visibility_mean": _stat(camera_visibility, "mean"),
            "camera_visibility_min": _stat(camera_visibility, "min"),
            "camera_iphone_angle_correlation": np.nan,
        }

        if "camera_forearm_angle_deg" in group and "tilt" in group:
            paired = group[["camera_forearm_angle_deg", "tilt"]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if len(paired) >= 3:
                trial["camera_iphone_angle_correlation"] = float(
                    paired["camera_forearm_angle_deg"].corr(paired["tilt"])
                )

        # These remain NaN today. If identically named EMG columns are added to
        # the raw CSV later, the same pipeline will aggregate and use them.
        for column in EMG_FEATURES:
            trial[column] = _stat(_numeric(group, column), "mean")

        trials.append(trial)

    columns = [
        "trial_id", "attempt_number", "trial_started_at", "scenario_label",
        *MODEL_FEATURES,
    ]
    return pd.DataFrame(trials).reindex(columns=columns)


def create_rule_labels(trial_features):
    """Create transparent proof-of-concept labels from requested thresholds."""
    labels = pd.DataFrame(index=trial_features.index)
    labels["limited_movement"] = trial_features["tilt_range"].lt(0.6)
    labels["low_grip"] = trial_features["avg_grip_percent"].lt(30)
    labels["unstable_grip"] = trial_features["grip_std"].gt(25)
    labels["grip_loss"] = trial_features["paused_percent"].gt(20)
    labels["unstable_movement"] = trial_features["avg_smoothness"].lt(0.75)
    labels["high_tremor"] = trial_features["avg_tremor_px"].gt(12)

    negative_columns = [
        "limited_movement",
        "low_grip",
        "unstable_grip",
        "grip_loss",
        "unstable_movement",
        "high_tremor",
    ]
    has_negative_label = labels[negative_columns].any(axis=1)
    high_score = trial_features["final_assessment_score"].ge(7.0)
    labels["sensor_control_good"] = ~has_negative_label
    labels["good_control"] = labels["sensor_control_good"].astype(bool) & high_score
    return labels.fillna(False).astype(int).reindex(columns=LABEL_COLUMNS)


def train_model(trial_features, labels, random_state=42):
    """Train and evaluate a multi-label Random Forest proof of concept."""
    if len(trial_features) < 2:
        raise ValueError("At least two trials are required to train a model")

    inputs = trial_features.reindex(columns=MODEL_FEATURES)
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0.0,
                    keep_empty_features=True,
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=random_state,
                        min_samples_leaf=1,
                    )
                ),
            ),
        ]
    )

    test_size = max(1, int(round(len(inputs) * 0.30)))
    test_size = min(test_size, len(inputs) - 1)
    x_train, x_test, y_train, y_test = train_test_split(
        inputs,
        labels,
        test_size=test_size,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        target_names=LABEL_COLUMNS,
        zero_division=0,
    )
    return model, report, len(x_train), len(x_test)


def save_outputs(
    trial_features,
    labels,
    model,
    report,
    feature_path,
    model_path,
    report_path,
):
    """Save features, rule labels, trained pipeline, and evaluation report."""
    output_table = pd.concat([trial_features, labels], axis=1)
    output_table.to_csv(feature_path, index=False)

    model_bundle = {
        "model": model,
        "model_features": MODEL_FEATURES,
        "label_columns": LABEL_COLUMNS,
        "camera_features": CAMERA_FEATURES,
        "emg_features": EMG_FEATURES,
        "purpose": "Gameplay-pattern classification; not medical diagnosis",
    }
    joblib.dump(model_bundle, model_path)
    report_path.write_text(report)


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train the NervaFlex gameplay-pattern Random Forest"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "trials" / "nervaflex_ml_trials.csv",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=project_dir / "nervaflex_trial_features.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_dir / "nervaflex_rf_model.pkl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_dir / "nervaflex_rf_classification_report.txt",
    )
    args = parser.parse_args()

    frame_data = load_data(args.input)
    trial_features = extract_trial_features(frame_data)
    labels = create_rule_labels(trial_features)
    model, report, train_count, test_count = train_model(trial_features, labels)

    for path in (args.features, args.model, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    save_outputs(
        trial_features,
        labels,
        model,
        report,
        args.features,
        args.model,
        args.report,
    )

    print(f"Trials: {len(trial_features)} ({train_count} train, {test_count} test)")
    print(f"Saved features: {args.features}")
    print(f"Saved model: {args.model}")
    print(f"Saved report: {args.report}")
    print("\nProof of concept only: this model does not diagnose medical conditions.")


if __name__ == "__main__":
    main()
