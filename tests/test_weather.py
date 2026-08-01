import datetime as dt

from pipeline import weather


def _fake_response(mocker, daily_time, **daily_vars):
    resp = mocker.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"daily": {"time": daily_time, **daily_vars}}
    return resp


def test_chooses_archive_endpoint_for_old_date():
    today = dt.date(2026, 8, 1)
    old_date = dt.date(2026, 7, 1)
    assert weather._choose_endpoint(old_date, today=today) == weather.OPEN_METEO_ARCHIVE_URL


def test_chooses_forecast_endpoint_for_recent_date():
    today = dt.date(2026, 8, 1)
    recent_date = dt.date(2026, 7, 30)
    assert weather._choose_endpoint(recent_date, today=today) == weather.OPEN_METEO_FORECAST_URL


def test_get_weather_extracts_target_date_and_sums_precip(mocker):
    fake = _fake_response(
        mocker,
        daily_time=["2026-07-25", "2026-07-26", "2026-07-27"],
        temperature_2m_mean=[20.0, 21.0, 22.0],
        relative_humidity_2m_mean=[50, 55, 60],
        wind_speed_10m_mean=[5.0, 6.0, 7.0],
        precipitation_sum=[1.0, 0.0, 2.0],
        soil_moisture_0_to_7cm_mean=[0.2, 0.21, 0.22],
    )
    mocker.patch.object(weather.requests, "get", return_value=fake)

    result = weather.get_weather(35.71, -78.85, "2026-07-27", lookback_days=2)

    assert result["temperature_2m_mean_c"] == 22.0
    assert result["relative_humidity_2m_mean_pct"] == 60
    assert result["precipitation_sum_lookback_mm"] == 3.0
    assert result["era5_soil_moisture_0_7cm_reference"] == 0.22


def test_get_weather_handles_missing_soil_moisture(mocker):
    fake = _fake_response(
        mocker,
        daily_time=["2026-07-30"],
        temperature_2m_mean=[24.0],
        relative_humidity_2m_mean=[65],
        wind_speed_10m_mean=[8.0],
        precipitation_sum=[0.0],
        soil_moisture_0_to_7cm_mean=[None],
    )
    mocker.patch.object(weather.requests, "get", return_value=fake)

    result = weather.get_weather(35.71, -78.85, "2026-07-30", lookback_days=0)

    assert result["era5_soil_moisture_0_7cm_reference"] is None
