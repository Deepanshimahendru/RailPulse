from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/railpulse_model.pkl")

# Load processed data
df = pd.read_parquet("data/processed/railway_features.parquet")

# Sort each train journey
df = df.sort_values(
    ["train_number", "start_date", "seq"]
).copy()

# Keep journeys with at least two stations
df["next_station"] = (
    df.groupby(["train_number", "start_date"])["station_code"]
    .shift(-1)
)
df["actual_next_station_delay"] = (
    df.groupby(["train_number", "start_date"])["arrival_delay_min"]
    .shift(-1)
)

# Select rows where the next station exists
predictions_df = df[df["next_station"].notna()].copy()

# Create prediction features
predictions_df["current_station_delay"] = (
    predictions_df["arrival_delay_min"]
)
predictions_df["scheduled_arrival_hour"] = (
    predictions_df["scheduled_arrival"].dt.hour
)

predictions_df["scheduled_departure_hour"] = (
    predictions_df["scheduled_departure"].dt.hour
)


feature_columns = [
    "category",
    "station_code",
    "state",
    "seq",
    "scheduled_arrival_hour",
    "scheduled_departure_hour",
    "previous_station_delay",
    "current_station_delay",
    "rush_hour_flag",
    "weekend_flag",
    "holiday_flag",
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "weather_missing"
]

# Predict delay at the next station
X = predictions_df[feature_columns].copy()

predictions_df["predicted_next_station_delay"] = (
    model.predict(X)
)

# Display results
results = predictions_df[
    [
        "train_number",
        "start_date",
        "train_name",
        "station_code",
        "next_station",
        "seq",
        "predicted_next_station_delay",
        "actual_next_station_delay"
    ]
].copy()
# Evaluate only on the unseen test date
test_results = results[
    (results["start_date"].astype(str).str[:10] == "2026-10-10")
    & results["actual_next_station_delay"].notna()
].copy()

y_true = test_results["actual_next_station_delay"]
y_pred = test_results["predicted_next_station_delay"]

print("\nTEST-DATE EVALUATION (10 OCTOBER)")
print("--------------------------------")
print("Records evaluated:", len(test_results))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("R2:", r2_score(y_true, y_pred))
print("\nRAILPULSE NEXT STATION PREDICTIONS")
print("---------------------------------")
print(results.head(20).to_string(index=False))

# Save results
results.to_csv("reports/predictions.csv", index=False)

print("\nPredictions saved successfully!")