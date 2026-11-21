import pandas as pd
import matplotlib.pyplot as plt
# Load railway dataset
df = pd.read_parquet("data/raw/2026-10-10.parquet")
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Records:")
print("Total duplicates:", df.duplicated().sum())
print("\nArrival Delay Statistics:")
print(df["arrival_delay_min"].describe())

print("\nDeparture Delay Statistics:")
print(df["departure_delay_min"].describe())
print("\nExtreme Arrival Delays:")

extreme_delays = df[df["arrival_delay_min"] < -60]

print("Records below -60 minutes:", len(extreme_delays))
print("\nSuspicious Arrival Timestamps:")

suspicious = df[
    df["scheduled_arrival"] > df["scheduled_departure"]
]

print("Inconsistent records:", len(suspicious))
suspicious.to_csv(
    "reports/timestamp_anomalies.csv",
    index=False
)

print("Anomaly report saved successfully!")
print("\nActual Timing Availability:")

print("Missing actual arrivals:", df["actual_arrival"].isnull().sum())

print("Missing actual departures:", df["actual_departure"].isnull().sum())
print("\nMissing Timing Percentage:")

print(
    "Arrival:",
    round(df["actual_arrival"].isnull().mean() * 100, 2),
    "%"
)

print(
    "Departure:",
    round(df["actual_departure"].isnull().mean() * 100, 2),
    "%"
)
print("\nTrain Status Distribution:")

print(df["status"].value_counts())
print("\nTrain Category Distribution:")

print(df["category"].value_counts())
print("\nUnique Train Analysis:")

print("Unique train numbers:", df["train_number"].nunique())

print("Unique train names:", df["train_name"].nunique())

print("Unique stations:", df["station_code"].nunique())
print("\nMedian Arrival Delay by Train Category:")

completed_trains = df[df["status"] == "completed"]

category_delays = completed_trains.groupby("category")["arrival_delay_min"].median()

print(category_delays.sort_values(ascending=False))
category_analysis = completed_trains.groupby("category")["arrival_delay_min"].agg(
    ["count", "median", "mean"]
)

print("\nCategory-wise Delay Analysis:")
print(category_analysis.sort_values(by="median", ascending=False))
station_analysis = completed_trains.groupby("station_code")["arrival_delay_min"].agg(
    ["count", "median", "mean"]
)

station_analysis = station_analysis[station_analysis["count"] >= 30]

print("\nTop 15 Most Delayed Stations:")

print(station_analysis.sort_values(
    by="median", ascending=False
).head(15))
print("\nTop 15 Least Delayed Stations:")

print(station_analysis.sort_values(
    by="median", ascending=True
).head(15))
print("\nGenerating Station Delay Visualisations...")
top_15 = station_analysis.sort_values(
    by="median", ascending=False
).head(15)
least_15 = station_analysis.sort_values(
    by="median", ascending=True
).head(15)
plt.figure(figsize=(12, 6))
plt.bar(top_15.index, top_15["median"])
plt.title("Top 15 Most Delayed Stations")
plt.title("Top 15 Most Delayed Stations")
plt.ylabel("Median Arrival Delay (Minutes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/top_delayed_stations.png") 
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(least_15.index, least_15["median"])
plt.title("Top 15 Least Delayed Stations")
plt.ylabel("Median Arrival Delay (Minutes)")
plt.xlabel("Station Code")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/least_delayed_stations.png")
plt.figure(figsize=(10, 6))
plt.bar(category_analysis.index, category_analysis["median"])
plt.title("Median Arrival Delay by Train Category")
plt.xlabel("Train Category")
plt.ylabel("Median Arrival Delay (Minutes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/category_delay.png")
plt.show()