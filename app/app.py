from pathlib import Path
import json
import os
import sys

import numpy as np
from flask import Flask, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "results" / "saved_weights" / "set_b_normal.npz"
SUMMARY_PATH = PROJECT_ROOT / "results" / "run_summary.json"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from regression import hypothesis, clip_nonnegative

app = Flask(__name__)
_weights = None

CONTRIBUTORS = [
    {"login": "laibasan", "url": "https://github.com/laibasan"},
    {"login": "zohaib-2548", "url": "https://github.com/zohaib-2548"},
]


def load_weights():
    global _weights
    if _weights is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"Missing {WEIGHTS_PATH}. Run python src/train_eval.py first."
            )
        data = np.load(WEIGHTS_PATH, allow_pickle=True)
        _weights = {
            "theta": data["theta"],
            "mean": data["mean"],
            "std": data["std"],
            "feature_names": [str(x) for x in data["feature_names"]],
        }
    return _weights


def peak_train_ac():
    if not SUMMARY_PATH.exists():
        return None
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return float(summary["peak_train_ac"])


def predict_ac_power(hour, sw_radiation, temp_2m, cloud_cover):
    w = load_weights()
    sin_hour = np.sin(2 * np.pi * hour / 24.0)
    cos_hour = np.cos(2 * np.pi * hour / 24.0)
    raw = np.array(
        [sw_radiation, temp_2m, cloud_cover, sin_hour, cos_hour],
        dtype=float,
    )
    scaled = (raw - w["mean"]) / w["std"]
    x = np.concatenate([[1.0], scaled])
    pred = clip_nonnegative(hypothesis(x, w["theta"]))
    return float(pred)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    pct_of_peak = None
    error = None
    form = {
        "hour": "12",
        "sw_radiation": "800",
        "temp_2m": "32",
        "cloud_cover": "20",
    }
    peak = peak_train_ac()
    if request.method == "POST":
        try:
            hour = float(request.form["hour"])
            sw_radiation = float(request.form["sw_radiation"])
            temp_2m = float(request.form["temp_2m"])
            cloud_cover = float(request.form["cloud_cover"])
            form = {
                "hour": request.form["hour"],
                "sw_radiation": request.form["sw_radiation"],
                "temp_2m": request.form["temp_2m"],
                "cloud_cover": request.form["cloud_cover"],
            }
            if hour < 0 or hour > 23:
                raise ValueError("Hour must be between 0 and 23.")
            if sw_radiation < 0:
                raise ValueError("Shortwave radiation cannot be negative.")
            if cloud_cover < 0 or cloud_cover > 100:
                raise ValueError("Cloud cover must be between 0 and 100.")
            prediction = predict_ac_power(hour, sw_radiation, temp_2m, cloud_cover)
            if peak and peak > 0:
                pct_of_peak = 100.0 * prediction / peak
        except Exception as exc:
            error = str(exc)
    return render_template(
        "index.html",
        prediction=prediction,
        pct_of_peak=pct_of_peak,
        peak=peak,
        error=error,
        form=form,
        contributors=CONTRIBUTORS,
    )


if __name__ == "__main__":
    load_weights()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
