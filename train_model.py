import os
import pickle
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import json

warnings.filterwarnings("ignore")

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)
    
config = load_config()

DATA_DIR = config["data_dir"]
MODEL_PATH = os.path.join(DATA_DIR, config["model_file"])
META_PATH = os.path.join(DATA_DIR, config["metadata_file"])

def load_encounters():
    encounters = pd.read_csv(os.path.join(DATA_DIR, config["encounters_file"]))
    encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce")
    encounters = encounters[encounters["START"].notna()].copy()
    return encounters


def build_hourly_series(encounters):
    hourly = encounters.copy()
    hourly["HOUR"] = hourly["START"].dt.floor("h")
    hourly = hourly.groupby("HOUR").size().reset_index(name="TOTAL_ENCOUNTERS")
    hourly = hourly.sort_values("HOUR").reset_index(drop=True)

    if hourly.empty:
        return pd.DataFrame(columns=["HOUR", "TOTAL_ENCOUNTERS"])

    full_hours = pd.date_range(
        start=hourly["HOUR"].min(),
        end=hourly["HOUR"].max(),
        freq="h"
    )
    hourly = hourly.set_index("HOUR").reindex(full_hours, fill_value=0).reset_index()
    hourly.columns = ["HOUR", "TOTAL_ENCOUNTERS"]
    return hourly

def make_features(hourly):
    df = hourly.copy()

    df["lag_1"] = df["TOTAL_ENCOUNTERS"].shift(1)
    df["lag_2"] = df["TOTAL_ENCOUNTERS"].shift(2)
    df["lag_3"] = df["TOTAL_ENCOUNTERS"].shift(3)
    df["lag_24"] = df["TOTAL_ENCOUNTERS"].shift(24)

    df["rolling_mean_6"] = df["TOTAL_ENCOUNTERS"].shift(1).rolling(6).mean()
    df["rolling_mean_24"] = df["TOTAL_ENCOUNTERS"].shift(1).rolling(24).mean()

    df["hour_of_day"] = df["HOUR"].dt.hour
    df["day_of_week"] = df["HOUR"].dt.dayofweek

    df = df.dropna().reset_index(drop=True)
    return df

def train_model():
    encounters = load_encounters()
    hourly = build_hourly_series(encounters)
    feature_df = make_features(hourly)

    if feature_df.empty:
        raise ValueError("Not enough data to train the model.")

    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_24",
        "rolling_mean_6", "rolling_mean_24",
        "hour_of_day", "day_of_week"
    ]

    X = feature_df[feature_cols]
    y = feature_df["TOTAL_ENCOUNTERS"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "feature_cols": feature_cols,
        "last_trained_hour": hourly["HOUR"].max(),
        "training_rows": len(feature_df)
    }

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("Model trained and saved successfully.")
    print(f"Model path: {MODEL_PATH}")
    print(f"Metadata path: {META_PATH}")
    print(f"Training rows: {len(feature_df)}")
    print(f"Last trained hour: {hourly['HOUR'].max()}")
    
    
if __name__ == "__main__":
    train_model()
    
