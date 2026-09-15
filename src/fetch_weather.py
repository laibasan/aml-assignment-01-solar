from pathlib import Path
import json
import pandas as pd
import requests
import matplotlib.pyplot as plt


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE = 14.82
LONGITUDE = 78.28
START_DATE = "2020-05-15"
END_DATE = "2020-06-17"
TIMEZONE = "Asia/Kolkata"


def build_url(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
        "timezone": TIMEZONE,
    }
    return ARCHIVE_URL, params


def fetch_openmeteo(lat=LATITUDE, lon=LONGITUDE):
    url, params = build_url(lat, lon)
    print("GET", url)
    print("params", params)
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    hourly = payload["hourly"]
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hourly["time"]),
            "sw_radiation": hourly["shortwave_radiation"],
            "temp_2m": hourly["temperature_2m"],
            "cloud_cover": hourly["cloud_cover"],
        }
    )
    return df, payload


def peak_hour_for_day(series_by_hour):
    return int(series_by_hour.idxmax())


def verify_three_days(hourly_plant, openmeteo, figures_dir, days=None):
    merged = pd.merge(
        hourly_plant[["datetime", "irradiation"]],
        openmeteo,
        on="datetime",
        how="inner",
    )
    merged["sw_radiation_kw"] = merged["sw_radiation"] / 1000.0

    if days is None:
        days = [
            pd.Timestamp("2020-05-16").date(),
            pd.Timestamp("2020-05-17").date(),
            pd.Timestamp("2020-05-18").date(),
        ]

    peak_rows = []
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=False)
    for ax, day in zip(axes, days):
        day_df = merged[merged["datetime"].dt.date == day].copy()
        if day_df.empty:
            raise ValueError(f"No overlapping rows for verification day {day}")
        day_df["hour"] = day_df["datetime"].dt.hour
        sensor_peak = peak_hour_for_day(
            day_df.set_index("hour")["irradiation"]
        )
        meteo_peak = peak_hour_for_day(
            day_df.set_index("hour")["sw_radiation_kw"]
        )
        peak_rows.append(
            {
                "day": str(day),
                "sensor_peak_hour": sensor_peak,
                "openmeteo_peak_hour": meteo_peak,
                "hour_shift": meteo_peak - sensor_peak,
            }
        )
        ax.plot(day_df["hour"], day_df["irradiation"], label="sensor irradiation (kW/m²)")
        ax.plot(
            day_df["hour"],
            day_df["sw_radiation_kw"],
            label="Open-Meteo shortwave (kW/m²)",
        )
        ax.set_title(f"{day}")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Radiation (kW/m²)")
        ax.set_xticks(range(0, 24, 2))
        ax.legend()

    fig.suptitle("Sensor irradiation vs Open-Meteo shortwave radiation")
    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "05_irradiation_vs_openmeteo_3days.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved", out_path)
    return peak_rows


def run_fetch(data_dir, figures_dir):
    data_dir = Path(data_dir)
    weather, payload = fetch_openmeteo()
    out_csv = data_dir / "plant1_openmeteo.csv"
    weather.to_csv(out_csv, index=False)
    print(f"Open-Meteo rows: {len(weather)}")
    print(f"Saved {out_csv}")
    print(
        "Units: Open-Meteo shortwave_radiation is W/m²; "
        "the on-site IRRADIATION sensor is kW/m². Divide by 1000 to compare."
    )

    hourly_path = data_dir / "plant1_hourly.csv"
    if not hourly_path.exists():
        raise FileNotFoundError("Run prepare.py before the location check.")
    hourly_plant = pd.read_csv(hourly_path, parse_dates=["datetime"])
    peaks = verify_three_days(hourly_plant, weather, Path(figures_dir))
    for row in peaks:
        print(
            f"{row['day']}: sensor peak hour={row['sensor_peak_hour']}, "
            f"Open-Meteo peak hour={row['openmeteo_peak_hour']}, "
            f"shift={row['hour_shift']}"
        )
        if abs(row["hour_shift"]) >= 1:
            print(
                "WARNING: peak shifted by an hour or more. "
                "Coordinates or time zone may be wrong."
            )

    meta = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "openmeteo_rows": int(len(weather)),
        "peak_hours": peaks,
        "timezone_abbreviation": payload.get("timezone_abbreviation"),
    }
    (data_dir / "openmeteo_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return weather, meta


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    run_fetch(project_root / "data", project_root / "results" / "figures")
