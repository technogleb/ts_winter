import pandas as pd
import numpy as np

from model import BaseEstimator


class RollingEstimator(BaseEstimator):
    """
    Simple rolling statistics estimator with either mean or median filter, applied to lag_matrix

    Parameters
    -----------
    rolling_filter
        Can be either 'mean' or 'median'
    """
    def __init__(self, rolling_filter: str = 'mean'):
        if rolling_filter not in ['mean', 'median']:
            raise TypeError(f'Filter type {rolling_filter} is not supported')
        self.rolling_filter = rolling_filter

    def fit(self, X, y, **kwargs):
        # rolling estimators have nothing to fit
        pass

    def predict(self, X):
        if not isinstance(X, pd.core.frame.DataFrame):
            X = pd.DataFrame(X)
        return getattr(X, self.rolling_filter)(axis=1)


class ExponentialSmoothingEstimator(BaseEstimator):
    """
    Simple exponential smoothing (SES) estimator

    Parameters
    -----------
    alpha_coef
        Alpha coefficient in SES. If close to 1, sets bigger weights to most recent lags, otherwise
        equally distributes weights between all lags
    """

    extra_params = ['weights']

    def __init__(self, alpha_coef: float = 0.5):
        self.alpha_coef = alpha_coef
        self.weights = None

    def fit(self, X, y, **kwargs):
        self.weights = self.get_ses_weights(self.alpha_coef, X.shape[1])

    def predict(self, X):
        y = X.dot(self.weights)
        return y

    @staticmethod
    def get_ses_weights(alpha: float, n_features: int) -> np.array:
        weights = np.array(list(reversed([alpha * (1 - alpha) ** i for i in range(n_features)])))
        return weights
