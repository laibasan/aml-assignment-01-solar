from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd


def load_hourly(path):
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df


def plot_ac_vs_irradiation(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["irradiation"], df["ac_power"], s=8, alpha=0.45)
    ax.set_xlabel("Irradiation (kW/m²)")
    ax.set_ylabel("AC power (kW)")
    ax.set_title("AC power vs irradiation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_module_vs_ambient(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        df["ambient_temp"],
        df["module_temp"],
        c=df["irradiation"],
        s=8,
        alpha=0.7,
        cmap="viridis",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Irradiation (kW/m²)")
    ax.set_xlabel("Ambient temperature (°C)")
    ax.set_ylabel("Module temperature (°C)")
    ax.set_title("Module temperature vs ambient temperature")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ac_vs_dc(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["dc_power"], df["ac_power"], s=8, alpha=0.45)
    ax.set_xlabel("DC power (kW)")
    ax.set_ylabel("AC power (kW)")
    ax.set_title("AC power vs DC power")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hourly_average_ac(df, out_path):
    by_hour = df.groupby(df["datetime"].dt.hour)["ac_power"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(by_hour.index, by_hour.values, marker="o")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average AC power (kW)")
    ax.set_title("Average AC power by hour of day")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return by_hour


def summarize_for_observations(df, by_hour):
    daytime = df[df["irradiation"] > 0]
    corr_irr = df["ac_power"].corr(df["irradiation"])
    corr_mod_amb = df["module_temp"].corr(df["ambient_temp"])
    ratio = (daytime["ac_power"] / daytime["dc_power"]).replace(
        [float("inf"), -float("inf")], pd.NA
    )
    peak_hour = int(by_hour.idxmax())
    return {
        "corr_ac_irradiation": float(corr_irr),
        "corr_module_ambient": float(corr_mod_amb),
        "mean_ac_dc_ratio_daytime": float(ratio.mean()),
        "peak_average_hour": peak_hour,
        "peak_average_ac": float(by_hour.max()),
        "night_mean_ac": float(df[df["irradiation"] == 0]["ac_power"].mean()),
    }


def run_eda(hourly_csv, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_hourly(hourly_csv)

    plot_ac_vs_irradiation(df, figures_dir / "01_ac_vs_irradiation.png")
    plot_module_vs_ambient(df, figures_dir / "02_module_vs_ambient.png")
    plot_ac_vs_dc(df, figures_dir / "03_ac_vs_dc.png")
    by_hour = plot_hourly_average_ac(df, figures_dir / "04_avg_ac_by_hour.png")
    stats = summarize_for_observations(df, by_hour)
    stats_path = figures_dir.parent / "eda_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("EDA figures saved to", figures_dir)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    run_eda(
        project_root / "data" / "plant1_hourly.csv",
        project_root / "results" / "figures",
    )
