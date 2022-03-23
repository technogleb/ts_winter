import numpy as np
import pandas as pd
from datetime import timedelta

import isodate


def to_relative(multi_ts: pd.DataFrame):
    """
    Transforms multi-dimensional time series into relative values
    """
    return multi_ts.apply(lambda x: x / x.sum(), axis=1)


def calculate_si(real, reference, fill_val=0.5):
    """
    Calculates population stability index using following formula
    SI = Σ (pA i - pB i ) * ( ln(pA i ) - ln(pB i ) )

    Parameters
    ----------
    real : 1d np.array
        Distribution for
    reference : 1d np.array
        Reference distribution.
    fill_val : float
        Value to replace NANs with

    Returns
    -------
    stability_index : float
    """

    si = (real - reference) * (np.log(real) - np.log(reference))

    return np.sum(si)


def _get_reference_time_delta(reference_type, granularity):
    """Gets timedelta for one of the following reference types: {'day', 'hour', 'instant'}"""
    if reference_type == 'day':
        delta = timedelta(days=1)
    elif reference_type == 'hour':
        delta = timedelta(hours=1)
    else:
        delta = isodate.parse_duration(granularity)
    return delta


def calculate_ts_stability_index(
        ts, incoming_point, reference_type='day', granularity='PT1H', fill_val=0.5):
    """
    Calculates population stability index for n-dimensional time-series.

    Parameters
    ----------
    ts : pd.Dataframe (n_samples, n_dim) with datetime index
        N-dimensional time-series with granularity of the following format "dd:hh:mm"
    incoming_point : pd.Dataframe of shape (1, n_dim_new) with datetime index
        New point, for which to make decision about stability. N_dim_new doesn't have
        to be equal to n_dim. In case when it's either less or higher all missing
        previous points for that dimension are filled with fill_val (usually quasi-null) value.
    reference_type : {'instant', 'day', 'week'}
        Point which distribution gets compared to the incoming point.
        'instant' stand for previous point, 'day' for point 24 hours ago and
        'week' stands for 24*7 hours ago.
    granularity: str
        Frequency of time-series of the following format "dd::hh::mm"
    fill_val : float
        Value to use when synchronizing batches with different number of dimensions

    Returns
    -------
    si : float
        Stability index for incoming point calculated relative to reference point.
    """
    if not isinstance(ts, pd.DataFrame):
        raise TypeError(
            'ts argument must be of type pandas Dataframe, {} provided.'.format(type(ts)))

    if reference_type not in {'instant', 'day', 'week'}:
        raise TypeError('reference_point must be one of [instant, day, week]}')

    current_time = incoming_point.index[-1]

    time_delta = _get_reference_time_delta(reference_type, granularity)

    reference_time = current_time - time_delta
    if reference_time not in ts.index:
        raise KeyError(f"There is no point with time {reference_time} in the time-series")
    reference_point = ts.loc[[reference_time]]  # use double brackets to slice point as dataframe instead of series

    si = calculate_si(incoming_point.values, reference_point.values, fill_val=fill_val)

    return si


def calculate_ts_stability_index_batch(ts, **kwargs):
    """Returns series with stability index calculated for every point of ts, where possible"""
    si_values = []
    timestamps = []
    for time, row in ts.iterrows():
        ts_history = ts[:time].iloc[:-1]
        try:
            si_value = calculate_ts_stability_index(ts_history, row.to_frame().T, **kwargs)
        except KeyError:  # means either not enouqh points at the start or not standart granularity (like 5 hours)
            si_value = None
        si_values.append(si_value)
        timestamps.append(time)

    return pd.Series(data=si_values, index=timestamps)


def make_si_predictor(ts: pd.DataFrame, granularity='PT1H', reference_type='day'):
    # calculate_ts_stability_index_batch
    # fit model
    # return predictor, si_history
    pass


def detect_with_dynamic_threshold(
        ts,
        incoming_point,
        si_history,
        predictor,
):
    pass