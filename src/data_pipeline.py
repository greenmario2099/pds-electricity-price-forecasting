"""
Data pipeline for the project "Forecasting German Day-Ahead Electricity Prices".

All data comes from SMARD (Bundesnetzagentur, CC BY 4.0), downloaded from
https://www.smard.de/en/downloadcenter/download-market-data/ with
country = Germany, resolution = Hour, period = 01.10.2018 - 17.11.2026.

Usage (from a notebook in the notebooks/ folder):

    import sys; sys.path.append('..')
    from src.data_pipeline import load_data, build_features, FEATURES

    data = load_data('../data/')
    features = build_features(data)

Leakage rule: the day-ahead auction for day D closes at 12:00 on day D-1.
Every feature for an hour of day D must therefore be known at that time:
  - SMARD's day-ahead forecasts for day D (load, residual load, wind, solar)
  - prices of day D-1 and earlier (they were set in the auction on day D-2)
  - calendar information
Caveat: the official wind and solar forecasts only have to be published by 18:00 on day D-1
(EU Regulation 543/2013, Art. 14). We use them as a proxy for the weather-based forecasts
that traders prepare themselves before the auction (see report, Section 4.1).
"""

import numpy as np
import pandas as pd
import holidays

FILES = {
    'prices': 'Day-ahead_prices_201810010000_202611180000_Hour.csv',
    'consumption_fc': 'Forecasted_consumption_201810010000_202611180000_Hour.csv',
    'generation_fc': 'Forecasted_generation_Day-Ahead_201810010000_202611180000_Hour.csv',
    'generation_actual': 'Actual_generation_201810010000_202611180000_Hour.csv',
}

# first row of all SMARD files: 01.10.2018 00:00 German time = 30.09.2018 22:00 UTC
START_UTC = '2018-09-30 22:00'

# features used by the models (target = 'price')
FUNDAMENTAL_FEATURES = ['load_fc', 'residual_load_fc', 'wind_onshore_fc', 'wind_offshore_fc', 'solar_fc',
                        'renewable_share', 'residual_load_ramp', 'residual_load_rel_day',
                        'residual_load_day_max', 'residual_load_day_min', 'residual_load_change_24h']
CALENDAR_FEATURES = ['hour', 'weekday', 'month', 'day_of_year', 'weekend', 'holiday']
PRICE_FEATURES = ['lag24', 'lag48', 'lag168', 'same_hour_mean_7d', 'prev_day_mean', 'prev_day_min',
                  'prev_day_max', 'mean_7d', 'mean_30d', 'std_7d']
FEATURES = FUNDAMENTAL_FEATURES + CALENDAR_FEATURES + PRICE_FEATURES


def read_smard_csv(path):
    """Read one SMARD CSV file (';' separator, ',' thousands separator, '-' = missing)."""
    df = pd.read_csv(path, sep=';', thousands=',', na_values='-', encoding='utf-8-sig')
    df.columns = [c.replace(' Calculated resolutions', '').strip() for c in df.columns]
    return df


def make_time_index(n_rows):
    """Hourly time index in German local time, built from UTC.

    The 'Start date' column of SMARD is local time and therefore ambiguous at the
    daylight saving switches. All files have one row per hour, so we rebuild the
    index from the first timestamp instead.
    """
    return pd.date_range(START_UTC, periods=n_rows, freq='h', tz='UTC').tz_convert('Europe/Berlin')


def load_raw(data_dir):
    """Load all four SMARD files, check that they are aligned and give them a time index."""
    raw = {name: read_smard_csv(data_dir + file) for name, file in FILES.items()}
    n = len(raw['prices'])
    for name, df in raw.items():
        assert len(df) == n, f'{name} has a different number of rows'
        assert (df['Start date'] == raw['prices']['Start date']).all(), f'{name} is not aligned'
    index = make_time_index(n)
    for name in raw:
        raw[name] = raw[name].drop(columns=['Start date', 'End date'])
        raw[name].index = index
    return raw


def load_data(data_dir):
    """Return one clean hourly DataFrame with the target and the forecast variables."""
    raw = load_raw(data_dir)
    data = pd.DataFrame(index=raw['prices'].index)
    data['price'] = raw['prices']['Germany/Luxembourg [€/MWh]']
    data['load_fc'] = raw['consumption_fc']['grid load [MWh]']
    data['residual_load_fc'] = raw['consumption_fc']['Residual load [MWh]']
    data['wind_onshore_fc'] = raw['generation_fc']['Wind onshore [MWh]']
    data['wind_offshore_fc'] = raw['generation_fc']['Wind offshore [MWh]']
    data['solar_fc'] = raw['generation_fc']['Photovoltaics [MWh]']
    # Data quality: the consumption forecast is missing for one full day (31.01.2020, 24 hours).
    # We fill it with the forecast of the same hour one week earlier (same weekday profile).
    for col in ['load_fc', 'residual_load_fc']:
        missing = data[col].isna()
        data.loc[missing, col] = data[col].shift(168)[missing]
    assert data.isna().sum().sum() == 0, 'unexpected missing values'
    return data


def load_actual_generation(data_dir):
    """Actual generation by source (MWh). Only used for EDA: it is not known before the auction."""
    raw = load_raw(data_dir)
    gen = raw['generation_actual']
    gen.columns = [c.replace(' [MWh]', '') for c in gen.columns]
    return gen


def german_holidays(index):
    """1 if the day is a nationwide public holiday in Germany (official dates), else 0."""
    years = range(index.year.min(), index.year.max() + 1)
    de_holidays = holidays.Germany(years=years)
    return pd.Series([int(d in de_holidays) for d in index.date], index=index)


def build_features(data):
    """Create all model features. Rows with missing lags (first 30 days) are dropped."""
    f = data.copy()
    price = f['price']

    # --- fundamentals (SMARD day-ahead forecasts, known for the whole day D) ---
    f['renewable_share'] = (f['wind_onshore_fc'] + f['wind_offshore_fc'] + f['solar_fc']) / f['load_fc']
    f['residual_load_ramp'] = f['residual_load_fc'].diff()
    day = f.index.date
    day_mean = f.groupby(day)['residual_load_fc'].transform('mean')
    f['residual_load_rel_day'] = f['residual_load_fc'] - day_mean
    f['residual_load_day_max'] = f.groupby(day)['residual_load_fc'].transform('max')
    f['residual_load_day_min'] = f.groupby(day)['residual_load_fc'].transform('min')
    f['residual_load_change_24h'] = f['residual_load_fc'] - f['residual_load_fc'].shift(24)

    # --- calendar ---
    f['hour'] = f.index.hour
    f['weekday'] = f.index.dayofweek
    f['month'] = f.index.month
    f['day_of_year'] = f.index.dayofyear
    f['weekend'] = (f['weekday'] >= 5).astype(int)
    f['holiday'] = german_holidays(f.index)

    # --- past prices (all known at noon of day D-1) ---
    f['lag24'] = price.shift(24)
    f['lag48'] = price.shift(48)
    f['lag168'] = price.shift(168)
    f['same_hour_mean_7d'] = sum(price.shift(24 * k) for k in range(1, 8)) / 7
    daily = price.groupby(day).agg(['mean', 'min', 'max'])
    prev_daily = daily.shift(1)  # statistics of the previous calendar day
    f['prev_day_mean'] = prev_daily['mean'].reindex(day).values
    f['prev_day_min'] = prev_daily['min'].reindex(day).values
    f['prev_day_max'] = prev_daily['max'].reindex(day).values
    f['mean_7d'] = price.shift(24).rolling(24 * 7).mean()
    f['mean_30d'] = price.shift(24).rolling(24 * 30).mean()
    f['std_7d'] = price.shift(24).rolling(24 * 7).std()

    # --- classification target for the extension ---
    f['negative'] = (price < 0).astype(int)

    return f.dropna()


def split_by_time(features, train=('2023-01', '2024-12'), val=('2025-01', '2025-12'), test=('2026-01', '2026-11')):
    """Chronological split - never shuffle time series data."""
    return (features.loc[train[0]:train[1]], features.loc[val[0]:val[1]], features.loc[test[0]:test[1]])


def rolling_forecast(features, make_model, months, feature_cols, window_days=730, target='price', fit_kwargs=None):
    """Rolling-origin evaluation: for every month, train on the previous `window_days` and predict that month.

    make_model: function without arguments that returns a fresh (unfitted) model.
    Returns a Series of predictions for all hours of the given months.
    """
    predictions = []
    for month in months:
        month_data = features[features.index.strftime('%Y-%m') == str(month)]
        start = month_data.index[0]
        train_window = features[(features.index < start) & (features.index >= start - pd.Timedelta(days=window_days))]
        model = make_model()
        model.fit(train_window[feature_cols], train_window[target], **(fit_kwargs or {}))
        predictions.append(pd.Series(model.predict(month_data[feature_cols]), index=month_data.index))
    return pd.concat(predictions)
