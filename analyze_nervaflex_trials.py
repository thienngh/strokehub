#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ABORTED_OUTCOMES = {"User quit", "Window closed"}


def number(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values):
    return statistics.fmean(values) if values else 0.0


def stdev(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def rms(values):
    return math.sqrt(mean([value * value for value in values])) if values else 0.0


def slope(xs, ys):
    if len(xs) < 2:
        return 0.0
    x_mean, y_mean = mean(xs), mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def extract_trial(rows):
    first = rows[0]
    grip = [number(row, "grip_percent") for row in rows]
    paused = [number(row, "paused") for row in rows]
    tremor = [number(row, "tremor_px") for row in rows]
    speed = [number(row, "speed_px_s") for row in rows]
    smoothness = [number(row, "smoothness") for row in rows]
    tilt = [number(row, "tilt") for row in rows]
    difficulty = [number(row, "difficulty") for row in rows]
    acceleration = [
        math.sqrt(
            number(row, "accel_x_g") ** 2
            + number(row, "accel_y_g") ** 2
            + number(row, "accel_z_g") ** 2
        )
        for row in rows
    ]

    return {
        "trial_id": first["trial_id"],
        "attempt_number": int(number(first, "attempt_number")),
        "trial_started_at": first["trial_started_at"],
        "game": first["game"],
        "outcome": first["outcome"],
        "completed": int(first["outcome"] not in ABORTED_OUTCOMES),
        "sample_count": len(rows),
        "duration_s": number(first, "trial_duration_s"),
        "assessment_score": number(first, "final_assessment_score"),
        "flappy_score": int(number(first, "final_flappy_score")),
        "pong_hits": int(number(first, "final_pong_hits")),
        "pong_misses": int(number(first, "final_pong_misses")),
        "grip_pause_count": int(number(first, "grip_pause_count")),
        "grip_mean_percent": mean(grip),
        "grip_median_percent": statistics.median(grip),
        "grip_p95_percent": percentile(grip, 0.95),
        "grip_max_percent": max(grip, default=0.0),
        "grip_active_fraction": mean([value > 0 for value in grip]),
        "pause_fraction": mean(paused),
        "tremor_rms_px": rms(tremor),
        "tremor_p95_px": percentile(tremor, 0.95),
        "speed_mean_px_s": mean(speed),
        "speed_p95_px_s": percentile(speed, 0.95),
        "smoothness_mean": mean(smoothness),
        "tilt_range": max(tilt, default=0.0) - min(tilt, default=0.0),
        "tilt_stdev": stdev(tilt),
        "accel_magnitude_mean_g": mean(acceleration),
        "accel_magnitude_stdev_g": stdev(acceleration),
        "difficulty_mean": mean(difficulty),
    }


def fmt(value, digits=2):
    return f"{value:.{digits}f}"


def build_report(features, source_path):
    completed = [trial for trial in features if trial["completed"]]
    lines = [
        "NervaFlex Research-Only Motor Performance Report",
        "=" * 52,
        "",
        "DIAGNOSTIC CONCLUSION: NOT DETERMINABLE",
        "This dataset cannot diagnose stroke, tremor type, Parkinson's disease,",
        "or another neurological condition.",
        "",
        f"Source: {source_path}",
        f"Recorded attempts: {len(features)}",
        f"Completed attempts: {len(completed)}",
        f"Aborted attempts: {len(features) - len(completed)}",
        "Known independent participants: 1 (inferred from current collection)",
        "Clinician-provided diagnostic labels: 0",
        "EMG channels: 0",
        "",
        "Observed within-user performance",
        "-" * 32,
    ]

    if not completed:
        lines.append("No completed trials are available.")
        return "\n".join(lines) + "\n"

    lines.extend([
        f"Assessment score: mean {fmt(mean([t['assessment_score'] for t in completed]))}/10, "
        f"range {fmt(min(t['assessment_score'] for t in completed))}-"
        f"{fmt(max(t['assessment_score'] for t in completed))}",
        f"Grip strength: mean {fmt(mean([t['grip_mean_percent'] for t in completed]))}%, "
        f"mean trial maximum {fmt(mean([t['grip_max_percent'] for t in completed]))}%",
        f"Tremor proxy: mean RMS {fmt(mean([t['tremor_rms_px'] for t in completed]))} pixels",
        f"Smoothness: mean {fmt(100 * mean([t['smoothness_mean'] for t in completed]))}%",
        f"Active grip time: mean {fmt(100 * mean([t['grip_active_fraction'] for t in completed]))}%",
        f"Paused time: mean {fmt(100 * mean([t['pause_fraction'] for t in completed]))}%",
        "",
        "Per-game trend model",
        "-" * 20,
    ])

    for game in sorted({trial["game"] for trial in completed}):
        game_trials = sorted(
            [trial for trial in completed if trial["game"] == game],
            key=lambda trial: trial["attempt_number"],
        )
        attempts = [trial["attempt_number"] for trial in game_trials]
        scores = [trial["assessment_score"] for trial in game_trials]
        score_slope = slope(attempts, scores)
        lines.append(
            f"{game}: {len(game_trials)} trials; score trend "
            f"{score_slope:+.3f} points per attempt"
        )

    lines.extend([
        "",
        "Interpretation limits",
        "-" * 21,
        "- The trend regression describes these recorded game attempts only.",
        "- Game difficulty, learning, fatigue, phone placement, and grip-pad setup",
        "  can change the measurements independently of neurological status.",
        "- There is no healthy control cohort, patient cohort, clinical ground truth,",
        "  participant diversity, held-out test set, or external validation.",
        "- The tremor value is a game-motion proxy, not a clinical tremor diagnosis.",
        "",
        "Requirements before diagnostic-model research",
        "-" * 44,
        "1. Pre-register the intended condition and clinical use.",
        "2. Collect clinician-confirmed labels and standardized clinical scores.",
        "3. Add participant IDs and recruit independent control/patient cohorts.",
        "4. Split evaluation by participant, never by sensor row.",
        "5. Validate prospectively at another site and report uncertainty/failures.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Analyze NervaFlex trial data")
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "trials" / "nervaflex_ml_trials.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis",
    )
    args = parser.parse_args()

    grouped = defaultdict(list)
    with args.dataset.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("trial_id"):
                grouped[row["trial_id"]].append(row)

    features = [extract_trial(rows) for rows in grouped.values()]
    features.sort(key=lambda trial: trial["attempt_number"])
    if not features:
        raise SystemExit("No trial data found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "ml_trial_features.csv"
    with feature_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(features[0].keys()))
        writer.writeheader()
        writer.writerows(features)

    report_path = args.output_dir / "nervaflex_analysis_report.txt"
    report_path.write_text(build_report(features, args.dataset))
    print(f"Wrote {feature_path}")
    print(f"Wrote {report_path}")
    print("Diagnostic conclusion: NOT DETERMINABLE")


if __name__ == "__main__":
    main()
