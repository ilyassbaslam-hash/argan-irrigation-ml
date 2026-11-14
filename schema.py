"""Shared column definitions for the argan irrigation dataset.

Keep this file in sync with the data-generation schema.
"""

FEATURES = [
    "day_of_year",
    "month",
    "temp_max",
    "temp_min",
    "rain_mm",
    "et0_mm",
    "rain_forecast_3d_mm",
    "argan_stage",
    "tree_age_years",
    "trees_per_ha",
    "canopy_cover_pct",
    "soil_type",
    "soil_depth_cm",
    "soil_moisture_pct",
]

CATEGORICAL = ["argan_stage", "soil_type"]

LABELS = ["irrigation_L_per_tree", "irrigate_yes_no"]

# The order does not matter to the model; it only makes reports readable.
STAGES = ["winter_rest", "flowering", "fruit_growth", "fruit_ripening"]
SOILS = ["sandy", "sandy_loam", "loam", "clay_loam", "clay"]

# Rough physical ranges used to reject obviously broken input rows.
RANGES = {
    "day_of_year": (1, 366),
    "month": (1, 12),
    "temp_max": (-10.0, 60.0),
    "temp_min": (-15.0, 50.0),
    "rain_mm": (0.0, 300.0),
    "et0_mm": (0.0, 20.0),
    "rain_forecast_3d_mm": (0.0, 900.0),
    "tree_age_years": (0, 120),
    "trees_per_ha": (10, 2000),
    "canopy_cover_pct": (0.0, 100.0),
    "soil_depth_cm": (10, 400),
    "soil_moisture_pct": (0.0, 65.0),
}
