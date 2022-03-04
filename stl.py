# coding: utf-8
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram, detrend


def detect_ts(ts, sigma=2.7, period=None):
    """This function gets anomalies from stand-alone time-Series.

    Parameters
    -----------
    ts : pandas.Series with pd.Datetime index
        Univariate time_series.
    sigma : int
        Multiply coeficient of standart deviation used, defaults to 3.
    period : int, default=None
        Number of points, used to define both smoothing window and seasonal
        period.

    Returns
    -----------
    (anomalies, smoothed_trend, seasonality, residuals) : tuple
    """

    period = period if period else get_season_period(ts)
    if not period:
        raise ValueError(
            'Couldn\'t automatically define season period'
            'and no period provided.'
        )

    k, b = np.polyfit(range(len(ts)), ts.values, 1)
    trend = pd.Series(k*np.array(range(len(ts))) + b, index=ts.index)
    ts_detrended = ts - trend
    smoothing_window = period // 3
    season = ts_detrended.rolling(smoothing_window).mean()
    resid = ts - trend - season

    threshold = sigma * resid.std()
    indexes = np.where(abs(resid) > threshold)[0]
    anomalies = ts[indexes]

    return anomalies, trend, season, resid


def extract_trend(ts: pd.Series):
    k, b = np.polyfit(range(len(ts)), ts.values, 1)
    trend = pd.Series(k * np.array(range(len(ts))) + b, index=ts.index)
    return trend, k, b


def extract_seasonality(ts_detrended: pd.Series, period=6):
    smoothing_window = period // 3
    season = ts_detrended.rolling(smoothing_window).mean()
    return season


def get_season_period(ts):
    ts = pd.Series(detrend(ts), ts.index)
    f, Pxx = periodogram(ts)
    Pxx = list(map(lambda x: x.real, Pxx))
    ziped = list(zip(f, Pxx))
    ziped.sort(key=lambda x: x[1])
    highest_freqs = [x[0] for x in ziped[-100:]]
    season_periods = [round(1/(x+0.001)) for x in highest_freqs]
    for period in reversed(season_periods):
        if 4 < period < 100:
            return int(period)
