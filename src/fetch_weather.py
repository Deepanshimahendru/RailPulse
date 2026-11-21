
import json
import time
import urllib.request
import urllib.parse
import pandas as pd

# Load railway data
df = pd.read_parquet("data/raw/2026-10-10.parquet")

# Load station coordinates
with open("data/raw/stations.json", "r", encoding="utf-8") as file:
    stations = json.load(file)

station_codes = set(df["station_code"].dropna().unique())

weather_locations = []

for station in stations:
    if station["code"] in station_codes:
        coordinates = station.get("coordinates", {})

        lat = coordinates.get("latitude")
        lon = coordinates.get("longitude")

        if lat and lon and lat != 0 and lon != 0:
            weather_locations.append({
                "station_code": station["code"],
                "latitude": lat,
                "longitude": lon
            })

locations_df = pd.DataFrame(weather_locations)

# Remove duplicate coordinates
locations_df = locations_df.drop_duplicates(
    subset=["latitude", "longitude"]
).reset_index(drop=True)

print("Unique weather locations:", len(locations_df))

# API settings
batch_size = 20
weather_records = []

for start in range(0, len(locations_df), batch_size):

    batch = locations_df.iloc[start:start + batch_size]

    params = {
        "latitude": ",".join(map(str, batch["latitude"])),
        "longitude": ",".join(map(str, batch["longitude"])),
        "start_date": "2026-10-07",
        "end_date": "2026-10-10",
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),
        "timezone": "Asia/Kolkata"
    }

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        + urllib.parse.urlencode(params)
    )

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "RailPulse/1.0"}
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)

        if isinstance(result, dict):
            result = [result]

        if len(result) != len(batch):
            raise ValueError("API returned an unexpected number of locations.")

        for location, weather in zip(batch.to_dict("records"), result):

            hourly = weather["hourly"]

            for i, timestamp in enumerate(hourly["time"]):
                weather_records.append({
                    "station_code": location["station_code"],
                    "timestamp": timestamp,
                    "temperature": hourly["temperature_2m"][i],
                    "humidity": hourly["relative_humidity_2m"][i],
                    "precipitation": hourly["precipitation"][i],
                    "wind_speed": hourly["wind_speed_10m"][i]
                })

        print(
            f"Completed {min(start + batch_size, len(locations_df))}"
            f"/{len(locations_df)} locations"
        )

        time.sleep(1)

    except Exception as error:
        print(f"Batch starting at {start} failed: {error}")
        raise

# Save weather data
weather_df = pd.DataFrame(weather_records)

weather_df.to_parquet(
    "data/processed/weather_hourly.parquet",
    index=False
)

print("\nWeather fetching completed!")
print("Weather records:", len(weather_df))
print("Unique stations:", weather_df["station_code"].nunique())
print("Saved to data/processed/weather_hourly.parquet")
