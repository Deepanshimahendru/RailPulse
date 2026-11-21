
import pandas as pd

# Load original dataset
df = pd.read_parquet("data/raw/2026-10-10.parquet")

# Sort stations within each train journey
df = df.sort_values(
    ["train_number", "start_date", "seq"]
).copy()

# Feature 1: Previous station delay
df["previous_station_delay"] = (
    df.groupby(["train_number", "start_date"])["arrival_delay_min"]
    .shift(1)
)

# Feature 2: Rush hour flag
hour = df["scheduled_departure"].dt.hour

df["rush_hour_flag"] = (
    ((hour >= 6) & (hour < 10)) |
    ((hour >= 17) & (hour < 20))
).astype(int)

# Feature 3: Weekend flag
df["weekend_flag"] = (
    pd.to_datetime(df["start_date"]).dt.day_of_week >= 5
).astype(int)
# Feature 4: Station state
station_mapping = pd.read_csv(
    "data/processed/station_state_mapping.csv"
)

df["station_code"] = df["station_code"].astype(str)

df = df.merge(
    station_mapping[["station_code", "state"]],
    on="station_code",
    how="left"
)

print("Stations with state:", df["state"].notna().sum())
print("Stations without state:", df["state"].isna().sum())

# Feature 5: State-wise holiday flag

holiday_calendar = pd.read_csv(
    "data/processed/holiday_calendar.csv"
)

holiday_calendar["date"] = pd.to_datetime(
    holiday_calendar["date"]
).dt.date

df["departure_date"] = (
    df["scheduled_departure"].dt.date
)

df = df.merge(
    holiday_calendar.assign(holiday_flag=1),
    left_on=["departure_date", "state"],
    right_on=["date", "state"],
    how="left"
)

df["holiday_flag"] = (
    df["holiday_flag"].fillna(0).astype(int)
)
# Remove temporary holiday merge columns
df = df.drop(
    columns=["departure_date", "date", "holiday_name"]
)
print("Holiday records:", df["holiday_flag"].sum())

# Save processed dataset

# Merge hourly weather data
weather_df = pd.read_parquet(
    "data/processed/weather_hourly.parquet"
)

# Convert weather timestamps to Indian local time
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

# Convert railway departure timestamps to local hourly timestamps
df["weather_timestamp"] = (
    df["scheduled_departure"]
    .dt.tz_localize(None)
    .dt.floor("h")
)

# Merge weather using station and hour
df = df.merge(
    weather_df,
    left_on=["station_code", "weather_timestamp"],
    right_on=["station_code", "timestamp"],
    how="left"
)
df["weather_missing"] = df["temperature"].isna().astype("int8")
# Remove temporary timestamp columns
df = df.drop(columns=["weather_timestamp", "timestamp"])

print("\nWeather integration completed!")
print("Temperature missing:", df["temperature"].isna().sum())
print("Humidity missing:", df["humidity"].isna().sum())
print("Precipitation missing:", df["precipitation"].isna().sum())
print("Wind speed missing:", df["wind_speed"].isna().sum())

# Save processed dataset
df.to_parquet(
    "data/processed/railway_features.parquet",
    index=False
)

print("Final dataset shape:", df.shape)
print("Saved to data/processed/railway_features.parquet")


# Display results
print("Feature engineering completed!")
print("Dataset shape:", df.shape)
print("Features created: previous_station_delay, rush_hour_flag, weekend_flag")
print("Missing previous station delays:", df["previous_station_delay"].isna().sum())
print("Weekend records:", df["weekend_flag"].sum())
print("Rush hour records:", df["rush_hour_flag"].sum())
print("Saved to data/processed/railway_features.parquet")
