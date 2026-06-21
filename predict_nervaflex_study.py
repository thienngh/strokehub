#!/usr/bin/env python3
"""Generate research-only gameplay-pattern predictions for all trials."""

import argparse
import csv
from pathlib import Path

import joblib

from train_nervaflex_model import extract_trial_features, load_data


MOVEMENT_LABELS = ["limited_movement", "unstable_movement", "high_tremor"]
GRIP_LABELS = ["low_grip", "unstable_grip", "grip_loss"]


def positive_probabilities(bundle, model_input):
    model = bundle["model"]
    labels = bundle["label_columns"]
    predictions = model.predict(model_input)[0]
    probability_arrays = model.predict_proba(model_input)
    estimators = model.named_steps["classifier"].estimators_

    output = {}
    for index, label in enumerate(labels):
        classes = list(estimators[index].classes_)
        if 1 in classes:
            probability = float(
                probability_arrays[index][0][classes.index(1)]
            )
        else:
            probability = float(classes[0] == 1)
        output[label] = (int(predictions[index]), probability)
    return output


def score_trial(row, outputs):
    if row["game"] == "flappy":
        game_score = 100.0 * min(float(row["final_flappy_score"]) / 10.0, 1.0)
    else:
        game_score = 100.0 * min(float(row["final_pong_hits"]) / 15.0, 1.0)

    movement_score = 100.0 * (
        1.0 - sum(outputs[label][1] for label in MOVEMENT_LABELS)
        / len(MOVEMENT_LABELS)
    )
    grip_score = 100.0 * (
        1.0 - sum(outputs[label][1] for label in GRIP_LABELS)
        / len(GRIP_LABELS)
    )
    final_score = 0.40 * game_score + 0.30 * movement_score + 0.30 * grip_score
    if final_score >= 80:
        band = "strong_gameplay_motor_control"
    elif final_score >= 60:
        band = "moderate_gameplay_motor_control"
    else:
        band = "gameplay_patterns_need_improvement"
    return game_score, movement_score, grip_score, final_score, band


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Score all NervaFlex trials")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "trials" / "nervaflex_ml_trials.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_dir / "nervaflex_rf_model.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "trials" / "nervaflex_trial_predictions.csv",
    )
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    features = extract_trial_features(load_data(args.input))
    results = []

    for _, row in features.iterrows():
        model_input = row.to_frame().T.reindex(columns=bundle["model_features"])
        outputs = positive_probabilities(bundle, model_input)
        game, movement, grip, final, band = score_trial(row, outputs)
        result = {
            "trial_id": row["trial_id"],
            "attempt_number": row["attempt_number"],
            "scenario_label": row["scenario_label"],
            "game": row["game"],
            "outcome": row["outcome"],
            "game_score": round(game, 3),
            "movement_score": round(movement, 3),
            "grip_score": round(grip, 3),
            "final_score": round(final, 3),
            "result_band": band,
        }
        for label in bundle["label_columns"]:
            prediction, probability = outputs[label]
            result[f"predicted_{label}"] = prediction
            result[f"probability_{label}"] = round(probability, 6)
        results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Scored {len(results)} trials")
    print(f"Saved {args.output}")
    print("Research-only gameplay classification; not medical diagnosis.")


if __name__ == "__main__":
    main()
