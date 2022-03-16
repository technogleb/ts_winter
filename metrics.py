import numpy as np
from sklearn.metrics import mean_absolute_error as mae


def mean_absolute_percentage_error(y_true, y_pred):
    """Naive implementation of MAPE error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mase(y_pred, y_true, method='naive', X_test=None, constant=None):
    """
    Mean absolute scaled error. MAE error of your predictions, normalized by
    MAE error of different methods predictions.

    Parameters
    -----------
    y_pred : sequence
        Predictions you want to compare to with different methods.
    y_true: sequence
        True values
    method: {'naive', 'mean', 'median', 'constant'}
        The method used to generate y_method which is predictions to compare to
        predictions of your method
    X_test: pd.Dataframe object, optional
        Must be provided when using all methods but naive and constant
    constant: int, optional
        Must be provided if method arg is set to constant

    Returns
    --------
    mase_score : range(0,)
        The score, that is computed as following -
        mae(y_true, y_pred)/mae(y_true, y_method). For example if method
        is 'naive' and mase score is 0.25, that means that your method is 4
        times more accurate, then the naive one.
    """

    y_method = y_pred
    if method == 'naive':
        y_method = y_true.shift()
        y_method.fillna(y_method.mean(), inplace=True)
    if method != 'naive':
        if X_test is None:
            print('You should provide X_test to evaluate predict')
        X_test.drop([label for label in X_test.columns if 'lag_' in label],
                    inplace=True, axis=1)
    if method == 'mean':
        y_method = X_test.mean(axis=1).values
    if method == 'median':
        y_method = X_test.mean(axis=1).values
    if method == 'constant':
        y_method = np.full(y_true.shape, constant)
    return mae(y_true, y_pred) / mae(y_true, y_method)  # todo fix division by zero
