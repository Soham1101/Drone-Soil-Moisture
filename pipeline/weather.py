"""Open-Meteo client: pulls weather features for a single site lat/lon and date.

Soil moisture is not spatially uniform-free, but for this project the field is
small enough that one manually-entered site coordinate is sufficient for weather
(per project decision) -- unlike the TDR probe coordinates used elsewhere, which
still vary per point for the kriging step.
"""

from __future__ import annotations

import datetime as dt

import requests

from pipeline.config import (
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    PRECIP_LOOKBACK_DAYS,
    WEATHER_DAILY_VARS,
)

# Open-Meteo's ERA5-Land reanalysis (used by the archive endpoint) lags behind
# real time by about 5 days; more recent dates must go through the forecast
# endpoint instead, which serves recent-past + forecast data but has no
# soil-moisture reanalysis field yet.
ARCHIVE_LAG_DAYS = 5


def _parse_date(date: str | dt.date) -> dt.date:
    if isinstance(date, dt.date):
        return date
    return dt.datetime.strptime(date, "%Y-%m-%d").date()


def _choose_endpoint(target_date: dt.date, today: dt.date | None = None) -> str:
    today = today or dt.date.today()
    if (today - target_date).days > ARCHIVE_LAG_DAYS:
        return OPEN_METEO_ARCHIVE_URL
    return OPEN_METEO_FORECAST_URL


def get_weather(
    lat: float,
    lon: float,
    date: str | dt.date,
    lookback_days: int = PRECIP_LOOKBACK_DAYS,
    today: dt.date | None = None,
) -> dict:
    """Return weather features for `date` at (lat, lon), plus summed precipitation
    over the `lookback_days` leading up to it.
    """
    target_date = _parse_date(date)
    start_date = target_date - dt.timedelta(days=lookback_days)
    url = _choose_endpoint(target_date, today=today)

    response = requests.get(
        url,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": target_date.isoformat(),
            "daily": ",".join(WEATHER_DAILY_VARS),
            "timezone": "auto",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    daily = payload["daily"]

    target_idx = daily["time"].index(target_date.isoformat())
    precip_values = [v for v in daily["precipitation_sum"] if v is not None]

    return {
        "site_lat": lat,
        "site_lon": lon,
        "date": target_date.isoformat(),
        "temperature_2m_mean_c": daily["temperature_2m_mean"][target_idx],
        "relative_humidity_2m_mean_pct": daily["relative_humidity_2m_mean"][target_idx],
        "wind_speed_10m_mean_kmh": daily["wind_speed_10m_mean"][target_idx],
        "precipitation_sum_lookback_mm": sum(precip_values) if precip_values else None,
        "precipitation_lookback_days": lookback_days,
        "era5_soil_moisture_0_7cm_reference": daily["soil_moisture_0_to_7cm_mean"][target_idx],
    }
