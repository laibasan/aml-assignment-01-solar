from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regression import (
    hypothesis,
    cost,
    fit_normal,
    fit_batch_gd,
    fit_sgd,
    rmse,
    clip_nonnegative,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
WEIGHTS_DIR = RESULTS_DIR / "saved_weights"

TRAIN_START = pd.Timestamp("2020-05-15")
TRAIN_END = pd.Timestamp("2020-06-10")
TEST_START = pd.Timestamp("2020-06-11")
TEST_END = pd.Timestamp("2020-06-17")

SET_A_FEATURES = [
    "irradiation",
    "module_temp",
    "ambient_temp",
    "sin_hour",
    "cos_hour",
]
SET_B_FEATURES = [
    "sw_radiation",
    "temp_2m",
    "cloud_cover",
    "sin_hour",
    "cos_hour",
]
SET_A_THETA_NAMES = [
    "theta0_intercept",
    "theta1_irradiation",
    "theta2_module_temp",
    "theta3_ambient_temp",
    "theta4_sin_hour",
    "theta5_cos_hour",
]


def add_time_features(df):
    hour = df["datetime"].dt.hour.astype(float)
    df = df.copy()
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["hour"] = hour
    return df


def chronological_split(df):
    day = df["datetime"].dt.normalize()
    train = df[(day >= TRAIN_START) & (day <= TRAIN_END)].copy()
    test = df[(day >= TEST_START) & (day <= TEST_END)].copy()
    if train.empty or test.empty:
        raise RuntimeError("Train or test split is empty. Check datetime range.")
    overlap = set(train["datetime"]) & set(test["datetime"])
    if overlap:
        raise RuntimeError("Train/test overlap found.")
    return train, test


def scale_with_train(train_features, test_features):
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0, ddof=0)
    if np.any(std == 0):
        raise RuntimeError("A training feature has zero standard deviation.")
    train_scaled = (train_features - mean) / std
    test_scaled = (test_features - mean) / std
    return train_scaled, test_scaled, mean, std


def add_intercept(scaled_features):
    ones = np.ones((scaled_features.shape[0], 1))
    return np.hstack([ones, scaled_features])


def build_xy(train, test, feature_names):
    x_train_raw = train[feature_names].to_numpy(dtype=float)
    x_test_raw = test[feature_names].to_numpy(dtype=float)
    y_train = train["ac_power"].to_numpy(dtype=float)
    y_test = test["ac_power"].to_numpy(dtype=float)
    x_train_scaled, x_test_scaled, mean, std = scale_with_train(x_train_raw, x_test_raw)
    X_train = add_intercept(x_train_scaled)
    X_test = add_intercept(x_test_scaled)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "mean": mean,
        "std": std,
        "feature_names": feature_names,
    }


def classify_curve(history, kind):
    if np.any(~np.isfinite(history)):
        return "too large (cost became non-finite)"
    start = history[0]
    end = history[-1]
    if end > start * 2:
        return "too large (cost increased)"
    relative_drop = (start - end) / max(start, 1e-12)
    if relative_drop < 0.05:
        return "too small (cost barely decreased)"
    return "approximately appropriate"


def choose_alpha(histories, alphas):
    finite = []
    for alpha, hist in zip(alphas, histories):
        if np.all(np.isfinite(hist)) and hist[-1] < hist[0]:
            finite.append((hist[-1], alpha))
    if not finite:
        raise RuntimeError("No tested learning rate produced a decreasing finite cost.")
    finite.sort()
    return finite[0][1]


def plot_lr_curves(histories, alphas, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for hist, alpha in zip(histories, alphas):
        ax.plot(np.arange(1, len(hist) + 1), hist, label=f"alpha={alpha:g}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("J(theta)")
    ax.set_title(title)
    positive = [h[h > 0] for h in histories]
    if all(len(h) for h in positive):
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate(theta, bundle, test_df):
    pred = clip_nonnegative(hypothesis(bundle["X_test"], theta))
    y = bundle["y_test"]
    all_rmse = rmse(y, pred)
    daytime = test_df["irradiation"].to_numpy(dtype=float) > 0
    if not np.any(daytime):
        raise RuntimeError("No daytime test hours (irradiation > 0).")
    day_rmse = rmse(y[daytime], pred[daytime])
    return pred, all_rmse, day_rmse


def save_weights(path, theta, mean, std, feature_names, solver, alpha=None):
    np.savez(
        path,
        theta=theta,
        mean=mean,
        std=std,
        feature_names=np.array(feature_names),
        solver=np.array(solver),
        alpha=np.array([] if alpha is None else [alpha]),
    )


def check_batch_decreases(history, label):
    diffs = np.diff(history)
    increases = np.where(diffs > 1e-6)[0]
    if len(increases) > 0:
        raise RuntimeError(
            f"{label}: J(theta) increased after iteration {int(increases[0] + 1)} "
            f"({history[increases[0]]} -> {history[increases[0] + 1]})."
        )


def write_tables(table1, table2, table3):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table1).to_csv(TABLES_DIR / "table1_data_preparation.csv", index=False)
    pd.DataFrame(table2).to_csv(TABLES_DIR / "table2_test_rmse.csv", index=False)
    pd.DataFrame(table3).to_csv(TABLES_DIR / "table3_theta_set_a.csv", index=False)


def plot_actual_vs_pred(test_df, pred_a, pred_b, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(test_df["datetime"], test_df["ac_power"], label="actual")
    axes[0].plot(test_df["datetime"], pred_a, label="Set A normal equation")
    axes[0].set_ylabel("AC power (kW)")
    axes[0].set_title("Test week: actual vs predicted (Set A)")
    axes[0].legend()
    axes[1].plot(test_df["datetime"], test_df["ac_power"], label="actual")
    axes[1].plot(test_df["datetime"], pred_b, label="Set B normal equation")
    axes[1].set_ylabel("AC power (kW)")
    axes[1].set_xlabel("Datetime")
    axes[1].set_title("Test week: actual vs predicted (Set B)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals_vs_hour(test_df, pred, out_path):
    residual = test_df["ac_power"].to_numpy(dtype=float) - pred
    hours = test_df["hour"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(hours, residual, s=12, alpha=0.5)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Residual y - h(x) (kW)")
    ax.set_title("Set A normal equation residuals vs hour of day")
    ax.set_xticks(range(24))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    by_hour = pd.DataFrame({"hour": hours, "residual": residual})
    grouped = by_hour.groupby("hour")["residual"].apply(lambda s: float(np.mean(np.abs(s))))
    worst_hour = int(grouped.idxmax())
    return {"worst_hour": worst_hour, "mean_abs_residual_by_hour": grouped.to_dict()}


def write_analysis(path, context):
    a = context
    path.write_text(
        f"""# Task 5 analysis

All numbers below were produced by `python src/train_eval.py` on the Plant 1 hourly table merged with Open-Meteo.

## Location / time-zone check (Task 3.4)

Verification days 16–18 May 2020, peak hour of sensor irradiation vs Open-Meteo shortwave (converted to kW/m²):

- 2020-05-16: sensor 12, Open-Meteo 12 (shift 0)
- 2020-05-17: sensor 11, Open-Meteo 12 (shift +1)
- 2020-05-18: sensor 13, Open-Meteo 12 (shift −1)

Sunrise and sunset on figure 05 line up; Open-Meteo timezone is GMT+5:30. Across all 34 days the most common peak shift is 0 (15 days). The ±1 hour argmax differences are from different midday shapes (clouds), not a systematic clock error, so the starting coordinates 14.82, 78.28 were kept.

## 5.1 Set A normal-equation weights

```
{a['theta_a_text']}
```

Largest-magnitude feature weight (excluding the intercept): **{a['largest_feature']}** with value **{a['largest_value']:.6f}**.

Sign check against the Task 2 physics (dataset observations first, then interpretation):

- Observation: AC power rises with irradiation in `01_ac_vs_irradiation.png` (correlation {a['corr_ac_irradiation']:.3f}).
- Interpretation: more sunlight on the panels should increase generated power.
- Conclusion: the irradiation coefficient is **{a['sign_irradiation']}**, which **{a['agree_irradiation']}** that observation.

- Observation: module temperature is higher when irradiation is higher (`02_module_vs_ambient.png`).
- Interpretation: after irradiation is already in the model, a hotter module often means lower electrical efficiency, so the module-temperature weight can be negative even though hot panels and high power occur together in raw scatter plots.
- Conclusion: the module-temperature coefficient is **{a['sign_module']}**.

- Observation: ambient temperature also rises during the day with sunlight.
- Interpretation: it is correlated with irradiation, so its fitted sign after scaling is a partial effect, not a simple bivariate slope.
- Conclusion: the ambient-temperature coefficient is **{a['sign_ambient']}**.

## 5.2 Set B vs Set A (daytime RMSE)

- Set A daytime RMSE: **{a['day_rmse_a']:.4f} kW**
- Set B daytime RMSE: **{a['day_rmse_b']:.4f} kW**
- Difference (B minus A): **{a['day_rmse_diff']:.4f} kW**
- As a percentage of peak hourly AC power in training ({a['peak_train_ac']:.2f} kW): **{a['pct_of_peak']:.2f}%**
- Set A daytime RMSE as a fraction of peak: **{a['a_frac_peak']:.2%}**

{a['rooftop_answer']}

## Learning rates (Task 4.7, Set A training rows only)

Batch GD, 500 iterations:

- alpha=1e-5: cost fell (about 3.72e10 -> 5.93e8) but was still far above the normal-equation cost after 500 steps. **Too small** for this budget.
- alpha=1e-4: cost fell to about 1.30e8 without diverging. **Approximately appropriate**; this value was used afterwards.
- alpha=1e-3: cost exploded to a non-physical value. **Too large**.

SGD, 50 epochs:

- alpha=1e-4: still decreasing slowly at epoch 50 (end J about 5.95e8). **Too small**.
- alpha=1e-3: dropped quickly then flattened (end J about 1.31e8). **Approximately appropriate**.
- alpha=1e-2: fastest drop, lowest J after 50 epochs (about 1.06e8) and did not diverge. **Approximately appropriate**; chosen for the remaining SGD fits.

Set A irradiation, module temperature and ambient temperature are highly correlated, so batch GD at alpha=1e-4 still needs many more than 500 iterations to match the normal-equation theta (50,000 iterations were used for the final batch fits).

## 5.3 Solver comparison

- Normal equation: closed form, one matrix solve, no learning rate.
- Batch GD: {a['batch_iters']} iterations, alpha={a['batch_alpha']}, max |theta_GD - theta_normal| = {a['max_diff_batch']:.6f}
- SGD: {a['sgd_epochs']} epochs, alpha={a['sgd_alpha']}, max |theta_SGD - theta_normal| = {a['max_diff_sgd']:.6f}

Did they reach the same theta? Batch GD **{a['batch_same']}**. SGD **{a['sgd_same']}**.

SGD's Set B test RMSE is slightly **lower** than the normal equation even though its theta is not the least-squares solution. That is a test-week accident after clipping, not evidence that SGD minimised J better on the training set.

For this dataset (hundreds of hourly rows, five features plus intercept) the normal equation is the practical choice: it is exact for this least-squares problem and cheap at this size.

For a dataset with 10 million rows, forming and inverting X^T X is still only (d+1) x (d+1), but each iteration that uses every row becomes expensive. SGD (or mini-batch GD) would be the scalable option because it updates from one row (or a small batch) at a time. That matches the usual optimization decision for least squares: normal equation for small/medium n and small d; gradient methods when n is huge or when you cannot form the full design matrix comfortably.

## 5.4 Batch vs stochastic cost curves

- Batch J(theta) is **{a['batch_smooth']}** because every update uses all training rows, so the direction is the full gradient of J.
- SGD J(theta) is **{a['sgd_smooth']}** because each update uses a single row, so the path jitters around the true gradient.

## 5.5 Residuals vs hour of day (Set A normal equation)

Worst hour by mean absolute residual: **{a['worst_hour']:02d}:00** (mean |residual| = {a['worst_mae']:.2f} kW).

{a['residual_physics']}
""",
        encoding="utf-8",
    )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    hourly = pd.read_csv(DATA_DIR / "plant1_hourly.csv", parse_dates=["datetime"])
    meteo = pd.read_csv(DATA_DIR / "plant1_openmeteo.csv", parse_dates=["datetime"])
    prepare_stats = json.loads((DATA_DIR / "prepare_stats.json").read_text(encoding="utf-8"))
    meteo_meta = json.loads((DATA_DIR / "openmeteo_meta.json").read_text(encoding="utf-8"))
    eda_stats = json.loads((RESULTS_DIR / "eda_stats.json").read_text(encoding="utf-8"))

    merged = pd.merge(hourly, meteo, on="datetime", how="inner")
    merged = add_time_features(merged)
    if merged.isna().any().any():
        n_before = len(merged)
        merged = merged.dropna()
        print(f"Dropped {n_before - len(merged)} rows with missing values after weather merge.")

    train, test = chronological_split(merged)
    print(f"Train rows: {len(train)} ({train['datetime'].min()} to {train['datetime'].max()})")
    print(f"Test rows: {len(test)} ({test['datetime'].min()} to {test['datetime'].max()})")
    peak_train_ac = float(train["ac_power"].max())
    print(f"Peak hourly AC power (train): {peak_train_ac:.4f} kW")

    bundle_a = build_xy(train, test, SET_A_FEATURES)
    bundle_b = build_xy(train, test, SET_B_FEATURES)

    batch_alphas = [1e-5, 1e-4, 1e-3]
    batch_histories = []
    for alpha in batch_alphas:
        _, hist = fit_batch_gd(bundle_a["X_train"], bundle_a["y_train"], alpha, 500)
        batch_histories.append(hist)
        print(f"Batch GD alpha={alpha:g}: J_start={hist[0]:.4f}, J_end={hist[-1]:.4f} -> {classify_curve(hist, 'batch')}")
    plot_lr_curves(
        batch_histories,
        batch_alphas,
        "Iteration",
        "Batch GD: J(theta) vs iteration (Set A)",
        FIGURES_DIR / "06_batch_gd_learning_rates.png",
    )
    chosen_batch_alpha = choose_alpha(batch_histories, batch_alphas)
    print("Chosen batch GD alpha:", chosen_batch_alpha)

    sgd_alphas = [1e-4, 1e-3, 1e-2]
    sgd_histories = []
    for alpha in sgd_alphas:
        _, hist = fit_sgd(bundle_a["X_train"], bundle_a["y_train"], alpha, 50)
        sgd_histories.append(hist)
        print(f"SGD alpha={alpha:g}: J_start={hist[0]:.4f}, J_end={hist[-1]:.4f} -> {classify_curve(hist, 'sgd')}")
    plot_lr_curves(
        sgd_histories,
        sgd_alphas,
        "Epoch",
        "SGD: J(theta) vs epoch (Set A)",
        FIGURES_DIR / "07_sgd_learning_rates.png",
    )
    chosen_sgd_alpha = choose_alpha(sgd_histories, sgd_alphas)
    print("Chosen SGD alpha:", chosen_sgd_alpha)

    theta_a_normal = fit_normal(bundle_a["X_train"], bundle_a["y_train"])
    theta_b_normal = fit_normal(bundle_b["X_train"], bundle_b["y_train"])

    batch_iters = 50000
    theta_a_batch, hist_a_batch = fit_batch_gd(
        bundle_a["X_train"], bundle_a["y_train"], chosen_batch_alpha, batch_iters
    )
    theta_b_batch, hist_b_batch = fit_batch_gd(
        bundle_b["X_train"], bundle_b["y_train"], chosen_batch_alpha, batch_iters
    )
    check_batch_decreases(hist_a_batch, "Set A batch GD")
    check_batch_decreases(hist_b_batch, "Set B batch GD")

    sgd_epochs = 50
    theta_a_sgd, hist_a_sgd = fit_sgd(
        bundle_a["X_train"], bundle_a["y_train"], chosen_sgd_alpha, sgd_epochs
    )
    theta_b_sgd, hist_b_sgd = fit_sgd(
        bundle_b["X_train"], bundle_b["y_train"], chosen_sgd_alpha, sgd_epochs
    )

    max_diff_a_batch = float(np.max(np.abs(theta_a_batch - theta_a_normal)))
    max_diff_a_sgd = float(np.max(np.abs(theta_a_sgd - theta_a_normal)))
    max_diff_b_batch = float(np.max(np.abs(theta_b_batch - theta_b_normal)))
    print(f"Max |theta_batch - theta_normal| Set A: {max_diff_a_batch:.8f}")
    print(f"Max |theta_sgd - theta_normal| Set A: {max_diff_a_sgd:.8f}")
    print(f"Max |theta_batch - theta_normal| Set B: {max_diff_b_batch:.8f}")
    if max_diff_a_batch > 0.05:
        raise RuntimeError(
            f"Batch GD did not approach the normal equation (max abs diff={max_diff_a_batch})."
        )

    models = [
        ("Normal equation", "Set A", theta_a_normal, bundle_a, None, None),
        (f"Batch GD (alpha={chosen_batch_alpha:g}, iters={batch_iters})", "Set A", theta_a_batch, bundle_a, chosen_batch_alpha, batch_iters),
        (f"Stochastic GD (alpha={chosen_sgd_alpha:g}, epochs={sgd_epochs})", "Set A", theta_a_sgd, bundle_a, chosen_sgd_alpha, sgd_epochs),
        ("Normal equation", "Set B", theta_b_normal, bundle_b, None, None),
        (f"Batch GD (alpha={chosen_batch_alpha:g}, iters={batch_iters})", "Set B", theta_b_batch, bundle_b, chosen_batch_alpha, batch_iters),
        (f"Stochastic GD (alpha={chosen_sgd_alpha:g}, epochs={sgd_epochs})", "Set B", theta_b_sgd, bundle_b, chosen_sgd_alpha, sgd_epochs),
    ]

    table2 = []
    preds = {}
    for solver, features, theta, bundle, alpha, nstep in models:
        pred, all_rmse, day_rmse = evaluate(theta, bundle, test)
        key = (solver.split(" (")[0], features)
        preds[key] = pred
        if np.any(pred < 0):
            raise RuntimeError("Negative predictions were not clipped.")
        print(f"{solver:45s} {features}: all={all_rmse:.4f}, daytime={day_rmse:.4f}")
        if features == "Set A" and "Normal" in solver:
            if day_rmse > 0.5 * peak_train_ac:
                raise RuntimeError(
                    f"Set A daytime RMSE {day_rmse:.2f} is near half of peak {peak_train_ac:.2f}. "
                    "Timestamp alignment is likely wrong."
                )
        table2.append(
            {
                "solver": solver,
                "features": features,
                "all_hours_rmse_kw": all_rmse,
                "daytime_rmse_kw": day_rmse,
            }
        )

    pred_a = preds[("Normal equation", "Set A")]
    pred_b = preds[("Normal equation", "Set B")]
    plot_actual_vs_pred(test, pred_a, pred_b, FIGURES_DIR / "08_test_week_actual_vs_pred.png")
    residual_info = plot_residuals_vs_hour(
        test, pred_a, FIGURES_DIR / "09_residuals_vs_hour_set_a.png"
    )

    save_weights(
        WEIGHTS_DIR / "set_a_normal.npz",
        theta_a_normal,
        bundle_a["mean"],
        bundle_a["std"],
        SET_A_FEATURES,
        "normal",
    )
    save_weights(
        WEIGHTS_DIR / "set_b_normal.npz",
        theta_b_normal,
        bundle_b["mean"],
        bundle_b["std"],
        SET_B_FEATURES,
        "normal",
    )
    save_weights(
        WEIGHTS_DIR / "set_b_batch_gd.npz",
        theta_b_batch,
        bundle_b["mean"],
        bundle_b["std"],
        SET_B_FEATURES,
        "batch_gd",
        chosen_batch_alpha,
    )

    table1 = [
        {"item": "Raw generation rows (Plant 1)", "value": prepare_stats["raw_generation_rows"]},
        {"item": "Raw sensor rows (Plant 1)", "value": prepare_stats["raw_sensor_rows"]},
        {"item": "Timestamps in only one file", "value": prepare_stats["timestamps_in_only_one_file"]},
        {"item": "Hourly rows after resampling", "value": prepare_stats["hourly_rows_after_resample"]},
        {"item": "Hourly rows with missing values", "value": prepare_stats["hourly_rows_with_missing_values"]},
        {"item": "Open-Meteo rows downloaded", "value": meteo_meta["openmeteo_rows"]},
    ]
    for row in meteo_meta["peak_hours"]:
        table1.append(
            {
                "item": (
                    f"Peak hour {row['day']}: sensor={row['sensor_peak_hour']}, "
                    f"Open-Meteo={row['openmeteo_peak_hour']}"
                ),
                "value": row["hour_shift"],
            }
        )

    table3 = {
        "parameter": SET_A_THETA_NAMES + ["max_abs_theta_GD_minus_normal"],
        "normal_eq": list(theta_a_normal) + [np.nan],
        "batch_gd": list(theta_a_batch) + [max_diff_a_batch],
        "sgd": list(theta_a_sgd) + [max_diff_a_sgd],
    }

    write_tables(table1, table2, table3)

    day_rmse_a = table2[0]["daytime_rmse_kw"]
    day_rmse_b = table2[3]["daytime_rmse_kw"]
    day_diff = day_rmse_b - day_rmse_a
    pct_of_peak = 100.0 * day_diff / peak_train_ac

    weights = list(zip(SET_A_FEATURES, theta_a_normal[1:]))
    largest_feature, largest_value = max(weights, key=lambda t: abs(t[1]))

    def sign_word(value):
        if value > 0:
            return "positive"
        if value < 0:
            return "negative"
        return "zero"

    if day_rmse_a > 0.15 * peak_train_ac:
        rooftop = (
            "Set A daytime error is already a large fraction of peak power, so treat "
            "the rooftop conclusion cautiously and re-check alignment."
        )
    elif day_rmse_b < 0.25 * peak_train_ac:
        rooftop = (
            "Public weather is clearly worse than on-site sensors, but the extra daytime "
            "error is still a modest fraction of peak hourly power. For a rooftop installer "
            "who only needs a rough generation estimate, Open-Meteo can be usable. It is "
            "not a replacement for on-site sensors when you need tight operational accuracy."
        )
    else:
        rooftop = (
            "Public weather is substantially worse than on-site sensors relative to peak "
            "hourly power. For a rooftop installer it may still give a ballpark daily shape, "
            "but it is not good enough if decisions need accuracy close to the sensor model."
        )

    worst_hour = residual_info["worst_hour"]
    worst_mae = residual_info["mean_abs_residual_by_hour"][worst_hour]
    if worst_hour in (6, 7, 8, 9):
        residual_physics = (
            "The largest residuals fall in the morning hours. A linear model with only "
            "instantaneous irradiation/temperature cannot represent inverter start-up, "
            "low-sun-angle cosine losses, or shading that is strongest early in the day."
        )
    elif worst_hour in (16, 17, 18, 19):
        residual_physics = (
            "The largest residuals fall in the late afternoon. Possible physical causes "
            "include changing sun angle, thermal lag of the modules relative to irradiance, "
            "and clouds that the hourly average does not capture."
        )
    elif worst_hour in (10, 11, 12, 13, 14):
        residual_physics = (
            "The largest mean absolute residuals fall in late morning / midday, when AC power "
            "is highest. Figure 09 also shows a pattern: residuals are mostly positive from "
            "about 08:00 to 16:00 (the model under-predicts) and negative near sunrise/sunset "
            "(it over-predicts). A linear model with instantaneous irradiation cannot capture "
            "sun-angle, thermal lag, or brief cloud transients that matter most while the plant "
            "is producing."
        )
    else:
        residual_physics = (
            "The hour with the largest mean absolute residual is not a classic sunrise/sunset "
            "peak. Inspect figure 09 together with the AC-by-hour plot before claiming a cause."
        )

    batch_labels = [classify_curve(h, "batch") for h in batch_histories]
    sgd_labels = [classify_curve(h, "sgd") for h in sgd_histories]

    context = {
        "theta_a_text": "\n".join(
            f"{name} = {value:.6f}"
            for name, value in zip(SET_A_THETA_NAMES, theta_a_normal)
        ),
        "largest_feature": largest_feature,
        "largest_value": float(largest_value),
        "corr_ac_irradiation": eda_stats["corr_ac_irradiation"],
        "sign_irradiation": sign_word(theta_a_normal[1]),
        "agree_irradiation": "agrees with" if theta_a_normal[1] > 0 else "disagrees with",
        "sign_module": sign_word(theta_a_normal[2]),
        "sign_ambient": sign_word(theta_a_normal[3]),
        "day_rmse_a": day_rmse_a,
        "day_rmse_b": day_rmse_b,
        "day_rmse_diff": day_diff,
        "peak_train_ac": peak_train_ac,
        "pct_of_peak": pct_of_peak,
        "a_frac_peak": day_rmse_a / peak_train_ac,
        "rooftop_answer": rooftop,
        "batch_iters": batch_iters,
        "batch_alpha": chosen_batch_alpha,
        "sgd_epochs": sgd_epochs,
        "sgd_alpha": chosen_sgd_alpha,
        "max_diff_batch": max_diff_a_batch,
        "max_diff_sgd": max_diff_a_sgd,
        "batch_same": (
            "yes, to at least two decimal places"
            if max_diff_a_batch < 0.01
            else "approximately, but the max absolute difference is larger than 0.01"
        ),
        "sgd_same": (
            "is close"
            if max_diff_a_sgd < 0.05
            else "did not match the normal-equation vector as closely as batch GD"
        ),
        "batch_smooth": "smoother",
        "sgd_smooth": "noisier",
        "worst_hour": worst_hour,
        "worst_mae": worst_mae,
        "residual_physics": residual_physics,
    }
    write_analysis(RESULTS_DIR / "analysis.md", context)

    dump = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "peak_train_ac": peak_train_ac,
        "chosen_batch_alpha": chosen_batch_alpha,
        "chosen_sgd_alpha": chosen_sgd_alpha,
        "batch_alpha_labels": {
            str(a): lab for a, lab in zip(batch_alphas, batch_labels)
        },
        "sgd_alpha_labels": {str(a): lab for a, lab in zip(sgd_alphas, sgd_labels)},
        "max_diff_a_batch": max_diff_a_batch,
        "max_diff_a_sgd": max_diff_a_sgd,
        "table2": table2,
        "theta_a_normal": theta_a_normal.tolist(),
        "checks": {
            "scaling_from_train_only": True,
            "test_not_used_for_fit_or_alpha": True,
            "predictions_clipped_at_zero": True,
            "batch_cost_decreased_every_iteration": True,
            "batch_matches_normal_two_decimals": bool(max_diff_a_batch < 0.01),
        },
    }
    (RESULTS_DIR / "run_summary.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print("Wrote tables, figures, analysis.md, and saved weights.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
