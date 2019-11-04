import numpy as np
import pandas as pd
import statsmodels as sm
import sklearn as sk


def get_hedge_ratio(X, y):
    """
    Calculate the Hedge Ratio of 2 series.

    Parameters
    ----------
    X : pandas.Series
        Price series of the independent series

    y : pandas.Series
        Price series fo the dependent series

    Returns
    -------
    beta : numpy.float64
        Hedge Ratio

    alpha : numpy.float64
        Intercept

    """

    assert isinstance(X, pd.Series), "First argument (X) must be a pandas series"
    assert isinstance(y, pd.Series), "Second argument (y) must be a pandas series"

    lr = sk.linear_model.LinearRegression()

    X = X.values.reshape(-1, 1)

    lr.fit(X, y)

    beta = lr.coef_[0]
    alpha = lr.intercept_

    return beta, alpha



def get_spread(X, y, beta, alpha):
    """
    Calculate the spread of 2 series.

    Parameters
    ----------
    X : pandas.Series
        Price series of the independent series

    y : pandas.Series
        Price series of the dependent series

    beta : numpy.float64
        Hedge Ratio

    alpha : numpy.float64
        Intercept

    Returns
    -------
    spread : pandas.Series
        Spread of 2 series

    """

    assert isinstance(X, pd.Series), "First argument (X) must be a pandas series"
    assert isinstance(y, pd.Series), "Second argument (y) must be a pandas series"
    assert isinstance(beta, np.float64), "Third argument (beta) must be a numpy float64"
    assert isinstance(alpha, np.float64), "Fourth argument (alpha) must be a numpy float64"



    spread = y - (beta * X + alpha)

    return spread

def check_stationarity(series):
    """
    Check stationarity using Augmented Dickey Fuller test.

    Parameters
    ----------
    series : pandas.Series
        Time series

    Returns
    -------
    Boolean
        True is stationary, False otherwise

    """

    assert isinstance(series, pd.Series), "First argument (series) must be a pandas series"

    result = sm.tsa.stattools.adfuller(series)

    p = result[1]
    p_test = result[4]['5%']

    if p <= p_test:
        return True
    else:
        return False


def check_cointegration(X, y):
    """
    Check if 2 series are cointegrated.
    Two-Step Engle-Granger test

    Parameters
    ----------
    X : pandas.Series
        Price series of the independent series

    y : pandas.Series
        Price series fo the dependent series

    Returns
    -------
    beta : numpy.float64
        Hedge Ratio

    alpha : numpy.float64
        Intercept

    spread : pandas.Series
        Spread of 2 series

    is_cointegrated : Boolean
        True is stationary, False otherwise

    """

    beta, alpha = get_hedge_ratio(X, y)
    spread = get_spread(X, y, beta, alpha)
    is_cointegrated = check_stationarity(spread)

    return beta, alpha, spread, is_cointegrated
