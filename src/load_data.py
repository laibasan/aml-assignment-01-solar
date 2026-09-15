from pathlib import Path
import pandas as pd

FILES = {
    "gen1": "Plant_1_Generation_Data.csv",
    "sensor1": "Plant_1_Weather_Sensor_Data.csv",
    "gen2": "Plant_2_Generation_Data.csv",
    "sensor2": "Plant_2_Weather_Sensor_Data.csv",
}


def load_raw(data_dir="data"):
    data_dir = Path(data_dir)
    frames = {}
    for key, name in FILES.items():
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download the dataset first."
            )
        frames[key] = pd.read_csv(path)
    return frames


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    raw = load_raw(project_root / "data")
    for key, df in raw.items():
        print(f"{key}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(df.head(3).to_string(), end="\n\n")
