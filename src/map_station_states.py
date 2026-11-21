
import pandas as pd
import geopandas as gpd
import json

# Load railway dataset
df = pd.read_parquet("data/raw/2026-10-10.parquet")

# Get all unique station codes
railway_codes = sorted(
    df["station_code"].dropna().astype(str).str.strip().unique()
)

# Load station master
with open("data/raw/stations.json", encoding="utf-8") as f:
    stations = pd.DataFrame(json.load(f))

stations["code"] = stations["code"].astype(str).str.strip()
stations = stations.drop_duplicates("code")

stations = stations[stations["code"].isin(railway_codes)].copy()

# Extract coordinates
stations["longitude"] = stations["coordinates"].apply(
    lambda c: c.get("longitude", 0) if isinstance(c, dict) else 0
)
stations["latitude"] = stations["coordinates"].apply(
    lambda c: c.get("latitude", 0) if isinstance(c, dict) else 0
)

# Keep stations with valid coordinates
valid = stations[
    (stations["longitude"] != 0) &
    (stations["latitude"] != 0)
].copy()

# Convert stations into geographic points
points = gpd.GeoDataFrame(
    valid,
    geometry=gpd.points_from_xy(
        valid["longitude"], valid["latitude"]
    ),
    crs="EPSG:4326"
)

# Load and combine state boundaries
boundaries = gpd.read_file(
    "data/raw/india_states.geojson"
)[["st_nm", "geometry"]]

boundaries = boundaries.dissolve(
    by="st_nm"
).reset_index()

# Match stations with state boundaries
joined = gpd.sjoin(
    points,
    boundaries,
    how="left",
    predicate="within"
)

# Prepare complete station mapping
mapping = pd.DataFrame({
    "station_code": railway_codes
})

mapping = mapping.merge(
    stations[["code", "name", "state"]],
    left_on="station_code",
    right_on="code",
    how="left"
)

mapping = mapping.drop(columns=["code"])

# Add geographically identified states
mapping["state"] = mapping["state"].replace(
    r"^\s*$", pd.NA, regex=True
)
spatial_states = joined[["code", "st_nm"]].rename(
    columns={"code": "station_code", "st_nm": "mapped_state"}
)

mapping = mapping.merge(
    spatial_states,
    on="station_code",
    how="left"
)

# Prefer geographic mapping; use existing state as fallback
mapping["state"] = (
    mapping["mapped_state"]
    .fillna(mapping["state"])
)

# Correct three verified boundary cases
overrides = {
    "MAHE": "Puducherry",
    "BNY": "Uttar Pradesh",
    "VSG": "Goa"
}

mapping["state"] = mapping["station_code"].map(
    overrides
).fillna(mapping["state"])

# Remove temporary column
mapping = mapping.drop(columns=["mapped_state"])

# Save final mapping
mapping.to_csv(
    "data/processed/station_state_mapping.csv",
    index=False
)

# Display results
print("Station-state mapping completed!")
print("Total station codes:", len(mapping))
print("Mapped stations:", mapping["state"].notna().sum())
print("Unmapped stations:", mapping["state"].isna().sum())

print("\nTop states by station count:")
print(mapping["state"].value_counts().to_string())

print("\nSaved to data/processed/station_state_mapping.csv")
