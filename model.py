import inspect
from typing import Dict, Callable, List, Any
from collections import defaultdict
from copy import deepcopy

import isodate
import pandas as pd
import numpy as np
from monthdelta import monthdelta
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample


def get_timedelta_from_granularity(granularity: str):
    datetime_interval = isodate.parse_duration(granularity)
    if isinstance(datetime_interval, isodate.duration.Duration):
        years, months = datetime_interval.years, datetime_interval.months
        total_months = int(years * 12 + months)
        datetime_interval = monthdelta(months=total_months)
    return datetime_interval


class BaseEstimator:
    """
    Implements get/set parameters logic, validating estimator and other
    methods, common for all estimators. This is improved version of sklearn get/set params logic,
    that also checks all parent's class parameters in addition to self
    """

    @classmethod
    def _get_param_names(cls, deep=True) -> List[str]:
        """Get parameter names for the estimator"""

        def get_param_names_for_class(cls):
            # fetch the constructor or the original constructor before
            # deprecation wrapping if any
            init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
            if init is object.__init__:
                # No explicit constructor to introspect
                return []

            # introspect the constructor arguments to find the model parameters
            # to represent
            init_signature = inspect.signature(init)
            # Consider the constructor parameters excluding 'self'
            parameters = [
                p for p in init_signature.parameters.values()
                if p.name != 'self' and p.kind != p.VAR_KEYWORD
                and p.kind != p.VAR_POSITIONAL
            ]

            # Add extra_params - params, that are not listed in signature (is_fitted e.q.)
            extra_params = getattr(cls, 'extra_params', [])

            # Extract and sort argument names excluding 'self'
            return sorted([p.name for p in parameters] + extra_params)

        # get self params
        parameters = get_param_names_for_class(cls)

        # if deep get all parents params
        if deep:
            for parent_class in cls.__bases__:
                parameters.extend(get_param_names_for_class(parent_class))

        return parameters

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators

        Returns
        -------
        params
            Parameter names mapped to their values
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)

            out[key] = value

        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects.
        The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object

        Parameters
        ----------
        **params
            Estimator parameters

        Returns
        -------
        self
            Estimator instance
        """

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')

            if key not in valid_params:
                raise ValueError('Invalid parameter %s for predictor %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


class TimeSeriesPredictor(BaseEstimator):
    def __init__(
            self,
            granularity: str,
            num_lags: int,
            model: Callable = LinearRegression,
            mappers: Dict[str, Callable] = {},
            **kwargs
    ):

        self.granularity = granularity
        self.num_lags = num_lags
        self.model = model(**kwargs)
        self.mappers = mappers
        self.fitted = False
        self.std = None

    def transform_into_matrix(self, ts: pd.Series) -> pd.DataFrame:
        """
        Transforms time series into lags matrix to allow
        applying supervised learning algorithms

        Parameters
        ------------
        ts
            Time series to transform

        Returns
        --------
        lags_matrix
            Dataframe with transformed values
        """

        ts_values = ts.values
        data = {}
        for i in range(self.num_lags + 1):
            data[f'lag_{self.num_lags - i}'] = np.roll(ts_values, -i)

        lags_matrix = pd.DataFrame(data)[:-self.num_lags]
        lags_matrix.index = ts.index[self.num_lags:]

        return lags_matrix

    def enrich(
            self,
            lags_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adds external features to time series

        Parameters
        ------------
        lags_matrix
            Pandas dataframe with transformed time-series values
        mappers
            Dictionary of functions to map each timestamp of lags matrix.
            Each function should take timestamp as the only positional parameter
            and return value of your additional feature for that timestamp
        """

        mappers = self.mappers
        for name, mapper in mappers.items():
            feature = pd.Series(lags_matrix.index.map(mapper), lags_matrix.index, name=name)
            lags_matrix[name] = feature

        return lags_matrix

    def fit(self, ts: pd.Series, *args, **kwargs):
        lag_matrix = self.transform_into_matrix(ts)
        feature_matrix = self.enrich(lag_matrix)

        X, y = feature_matrix.drop('lag_0', axis=1), feature_matrix['lag_0']
        self.model.fit(X, y, *args, **kwargs)
        self.fitted = True

    def predict_next(self, ts_lags, n_steps=1):
        if not self.model:
            raise ValueError('Model is not fitted yet')

        predict = {}

        ts = deepcopy(ts_lags)
        for _ in range(n_steps):
            next_row = self.generate_next_row(ts)
            next_timestamp = next_row.index[-1]
            value = self.model.predict(next_row)[0]
            predict[next_timestamp] = value
            ts[next_timestamp] = value
        return pd.Series(predict)

    def predict_batch(self, ts: pd.Series, ts_batch: pd.Series = pd.Series()):
        if not self.model:
            raise ValueError('Model is not fitted yet')

        unite_ts = ts.append(ts_batch)
        matrix = self.enrich(self.transform_into_matrix(unite_ts))

        data_batch = matrix[-len(ts_batch):]
        preds = self.model.predict(data_batch.drop('lag_0', axis=1))

        return pd.Series(index=data_batch.index, data=preds)

    def generate_next_row(self, ts):
        """
        Takes time-series as an input and returns next row, that is fed to the fitted model,
        when predicting next value.

        Parameters
        ----------
        ts : pd.Series(values, timestamps)
            Time-series to detect on

        Returns
        ---------
        feature_matrix : pd.DataFrame
            Pandas dataframe, which contains feature lags of
            shape(1, num_lags+len(external_feautres))
        """

        delta = get_timedelta_from_granularity(self.granularity)
        next_timestamp = pd.to_datetime(ts.index[-1]) + delta
        lag_dict = {'lag_{}'.format(i): [ts[-i]] for i in range(1, self.num_lags + 1)}
        df = pd.DataFrame.from_dict(lag_dict)
        df.index = [next_timestamp]
        df = self.enrich(df)

        return df


class TimeSeriesDetector(TimeSeriesPredictor):
    def __init__(
            self, sigma=2.7, bootstrapping=False, bootstrapping_quantile=0.97, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.bootstrapping = bootstrapping
        self.bootstrapping_quantile = bootstrapping_quantile

    def fit_statistics(self, ts):
        preds = self.predict_batch(ts)
        residuals = ts - preds
        if self.bootstrapping:
            samples_statistics = []
            for i in range(100):
                sample = resample(residuals, n_samples=len(residuals))
                sample.dropna(inplace=True)
                statistic = np.quantile(sample, self.bootstrapping_quantile)
                samples_statistics.append(statistic)

            statistic_estimate = sum(samples_statistics) / len(samples_statistics)
            self.std = statistic_estimate
        else:
            std = residuals.std()
            self.std = std

    def get_prediction_intervals(self, y_pred, season=False):
        if season:
            std_series = pd.Series(
                map(lambda x: self.std.get(x.hour), y_pred.index), index=y_pred.index
            )
            lower, upper = y_pred - self.sigma * std_series, y_pred + self.sigma * std_series
        elif self.bootstrapping:
            lower, upper = y_pred - self.std, y_pred + self.std
        else:
            lower, upper = y_pred - self.sigma * self.std, y_pred + self.sigma * self.std
        return lower, upper

    def detect(self, ts_true, ts_pred, season=False):
        lower, upper = self.get_prediction_intervals(ts_pred, season=season)
        return ts_true[(ts_true < lower) | (ts_true > upper)]

    def fit_seasonal_statistics(self, ts_train, n_splits=3, period=24):
        def split(period, n_splits):
            avg = period // n_splits
            seq = range(period)
            out = []
            last = 0.0

            while last < len(seq):
                out.append(tuple(seq[int(last):int(last + avg)]))
                last += avg

            return out

        ranges = split(period, n_splits)

        seasonal_std = {}
        for range_ in ranges:
            range_std = ts_train[ts_train.index.map(lambda x: x.hour in range_)].std()
            for i in range_:
                seasonal_std[i] = range_std

        self.std = seasonal_std