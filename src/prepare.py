from pathlib import Path
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_raw


def inspect_datetime_strings(series, name, n=8):
    print(f"\n{name} DATE_TIME samples (dtype={series.dtype}):")
    print(series.head(n).tolist())
    print("last samples:", series.tail(3).tolist())
    print("unique string lengths:", sorted(series.astype(str).str.len().unique().tolist()))


def parse_plant1_datetime(series, file_name):
    sample = str(series.iloc[0])
    if sample[2] == "-" and sample[5] == "-":
        parsed = pd.to_datetime(series, dayfirst=True, format="mixed")
        print(f"{file_name}: parsed with dayfirst=True (sample={sample})")
    else:
        parsed = pd.to_datetime(series)
        print(f"{file_name}: parsed with pandas default (sample={sample})")
    return parsed


def plant_level_power(gen):
    return (
        gen.groupby("datetime", as_index=False)[["AC_POWER", "DC_POWER"]]
        .sum()
        .rename(columns={"AC_POWER": "ac_power", "DC_POWER": "dc_power"})
    )


def exclusive_timestamps(power, sensor):
    power_set = set(power["datetime"])
    sensor_set = set(sensor["datetime"])
    only_power = sorted(power_set - sensor_set)
    only_sensor = sorted(sensor_set - power_set)
    return only_power, only_sensor


def resample_hourly(merged):
    hourly = (
        merged.set_index("datetime")
        .resample("h")
        .mean(numeric_only=True)
        .reset_index()
    )
    return hourly


def handle_missing(hourly):
    n_missing_before = int(hourly.isna().any(axis=1).sum())
    hourly = hourly.sort_values("datetime").set_index("datetime")
    hourly = hourly.interpolate(method="time", limit=2)
    hourly = hourly.reset_index()
    n_missing_after_interp = int(hourly.isna().any(axis=1).sum())
    hourly = hourly.dropna()
    return hourly, n_missing_before, n_missing_after_interp


def prepare_plant1(data_dir, out_csv):
    raw = load_raw(data_dir)
    gen = raw["gen1"].copy()
    sensor = raw["sensor1"].copy()

    inspect_datetime_strings(gen["DATE_TIME"], "Plant 1 generation")
    inspect_datetime_strings(sensor["DATE_TIME"], "Plant 1 sensor")

    gen["datetime"] = parse_plant1_datetime(gen["DATE_TIME"], "Plant 1 generation")
    sensor["datetime"] = parse_plant1_datetime(sensor["DATE_TIME"], "Plant 1 sensor")

    print("\nPlant 1 generation first timestamp:", gen["datetime"].min())
    print("Plant 1 generation last timestamp:", gen["datetime"].max())
    print("Plant 1 sensor first timestamp:", sensor["datetime"].min())
    print("Plant 1 sensor last timestamp:", sensor["datetime"].max())

    power = plant_level_power(gen)
    sensor_keep = sensor[
        ["datetime", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    ].rename(
        columns={
            "AMBIENT_TEMPERATURE": "ambient_temp",
            "MODULE_TEMPERATURE": "module_temp",
            "IRRADIATION": "irradiation",
        }
    )

    only_power, only_sensor = exclusive_timestamps(power, sensor_keep)
    n_only_one = len(only_power) + len(only_sensor)
    print(f"\nTimestamps only in generation (plant-level): {len(only_power)}")
    print(f"Timestamps only in sensor: {len(only_sensor)}")
    print(f"Timestamps in only one file: {n_only_one}")
    if only_power[:5]:
        print("example generation-only:", only_power[:5])
    if only_sensor[:5]:
        print("example sensor-only:", only_sensor[:5])

    merged = pd.merge(power, sensor_keep, on="datetime", how="outer")
    hourly = resample_hourly(merged)
    n_hourly_raw = len(hourly)
    n_missing_rows = int(hourly.isna().any(axis=1).sum())
    hourly, n_missing_before, n_missing_after_interp = handle_missing(hourly)

    hourly = hourly[
        ["datetime", "ac_power", "dc_power", "ambient_temp", "module_temp", "irradiation"]
    ]
    hourly = hourly.sort_values("datetime")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out_csv, index=False)

    stats = {
        "raw_generation_rows": int(len(gen)),
        "raw_sensor_rows": int(len(sensor)),
        "timestamps_only_generation": int(len(only_power)),
        "timestamps_only_sensor": int(len(only_sensor)),
        "timestamps_in_only_one_file": int(n_only_one),
        "hourly_rows_after_resample": int(n_hourly_raw),
        "hourly_rows_with_missing_values": int(n_missing_rows),
        "hourly_rows_saved": int(len(hourly)),
        "missing_rows_after_time_interpolation": int(n_missing_after_interp),
        "missing_handling": (
            "Hourly means were computed from an outer merge of plant-level "
            "power and sensor readings. Rows with missing values were filled "
            "with time-based linear interpolation (limit=2 hours). Any remaining "
            "incomplete rows were dropped."
        ),
    }
    stats_path = out_csv.parent / "prepare_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nHourly rows after resampling: {n_hourly_raw}")
    print(f"Hourly rows with missing values: {n_missing_rows}")
    print(f"Missing rows after interpolation: {n_missing_after_interp}")
    print(f"Hourly rows saved: {len(hourly)}")
    print(f"Saved {out_csv}")
    return hourly, stats


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    prepare_plant1(project_root / "data", project_root / "data" / "plant1_hourly.csv")
