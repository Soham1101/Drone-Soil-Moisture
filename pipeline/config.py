"""Shared constants for the soil moisture pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Coordinate reference system used for all lat/lon inputs (WGS84).
LATLON_CRS = "EPSG:4326"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_DAILY_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "soil_moisture_0_to_7cm_mean",
]

# How many days of precipitation history to sum leading up to the flight date.
PRECIP_LOOKBACK_DAYS = 7

ODM_DOCKER_IMAGE = "opendronemap/odm"
