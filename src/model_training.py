
import pandas as pd
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor


# 1. Load engineered railway dataset
df = pd.read_parquet("data/processed/railway_features.parquet")

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# 2. Sort stations within each train journey
df["start_date"] = pd.to_datetime(df["start_date"])

df = df.sort_values(
    ["train_number", "start_date", "seq"]
).copy()


# 3. Create next-station target
df["next_station_delay"] = (
    df.groupby(["train_number", "start_date"])["arrival_delay_min"]
    .shift(-1)
)

# Current station's observed arrival delay
df["current_station_delay"] = df["arrival_delay_min"]

print("\nTarget variable: next_station_delay")
print("Available target records:", df["next_station_delay"].notna().sum())
print("Missing target records:", df["next_station_delay"].isna().sum())


# 4. Remove rows without a next-station target
df = df.dropna(subset=["next_station_delay"]).copy()

print("\nAfter removing missing target values:")
print("Shape:", df.shape)
print("Missing target values:", df["next_station_delay"].isna().sum())


# 5. Identify columns excluded from model inputs
leakage_columns = [
    "arrival_delay_min",
    "actual_arrival",
    "actual_departure",
    "departure_delay_min",
    "arrival_reported",
    "departure_reported",
    "run_quality",
    "next_station_delay"
]

print("\nColumns excluded to prevent data leakage:")
print(leakage_columns)


# 6. Create time-based features
df["scheduled_arrival_hour"] = df["scheduled_arrival"].dt.hour
df["scheduled_departure_hour"] = df["scheduled_departure"].dt.hour


# 7. Select model features
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

X = df[feature_columns].copy()
y = df["next_station_delay"].copy()

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nSelected features:")
print(X.columns.tolist())

print("\nMissing values in selected features:")
print(X.isna().sum())


# 8. Handle missing categorical values
X["state"] = X["state"].fillna("Unknown")

categorical_features = [
    "category",
    "station_code",
    "state"
]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

print("\nCategorical features:", categorical_features)
print("Numerical features:", numerical_features)

print("\nRemaining missing values:")
print(X.isna().sum())


# 9. Check date distribution
print("\nAvailable journey dates:")
print(df["start_date"].value_counts().sort_index())

print("\nEarliest date:", df["start_date"].min())
print("Latest date:", df["start_date"].max())


# 10. Chronological train-test split
train_mask = df["start_date"] < "2026-10-10"
test_mask = df["start_date"] == "2026-10-10"

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining dates:")
print(df.loc[train_mask, "start_date"].value_counts().sort_index())

print("\nTesting dates:")
print(df.loc[test_mask, "start_date"].value_counts().sort_index())


# 11. Preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

print("\nPreprocessing pipeline created successfully!")


# 12. Random Forest model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nRandom Forest model pipeline created successfully!")


# 13. Train model
print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")


# 14. Make predictions and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("-------------------------")
print(f"Mean Absolute Error: {mae:.2f} minutes")
print(f"Root Mean Squared Error: {rmse:.2f} minutes")
print(f"R2 Score: {r2:.4f}")


# 15. Baseline model
baseline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DummyRegressor(strategy="mean"))
])

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_mae = mean_absolute_error(y_test, baseline_pred)

print("\nBASELINE MODEL")
print("-------------------------")
print(f"Baseline MAE: {baseline_mae:.2f} minutes")
print(f"Random Forest MAE: {mae:.2f} minutes")


# 16. Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/railpulse_model.pkl")

print("\nModel saved successfully!")


# 17. Feature importance
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importances = model.named_steps["regressor"].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTOP 15 IMPORTANT FEATURES")
print("-------------------------")
print(importance_df.head(15).to_string(index=False))

os.makedirs("reports", exist_ok=True)

importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")
