import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from report.utils import random_string
from django.conf import settings
import os
import math
from datetime import datetime as dt
from asset.utilities import bar_query_id
from asset.models import Symbol


plt.switch_backend('Agg')


def complete_missing_values(df):
    """
    Complete missing data of dataframe.

    Replace NaN values using ffill and bfill methods.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    Returns
    -------
    pandas.DataFrame
        Return completed data.

    """
    df_data = df.copy()
    df_data.dropna(how='all', inplace=True)
    df_data.fillna(method='ffill', inplace=True)
    df_data.fillna(method='bfill', inplace=True)
    return df_data


def normalize_data(df):
    """
    Normalize data in dataframe.

    Rescale data in order to make comparisons.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    Returns
    -------
    pandas.DataFrame
        Return rescaled data.

    """
    df_data = df.copy()
    df_data = df_data/df_data.iloc[0]
    return df_data


def remove_infs(df):
    """
    Remove np.inf and -np.inf.

    Replace np.inf and -np.inf with np.nan

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    Returns
    -------
    pandas.DataFrame
        Return data without infs.

    """
    df_data = df.copy()
    return df_data.replace([np.inf, -np.inf], np.nan)


def get_log_returns(df):
    """
    Get log returns of data.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    Returns
    -------
    pandas.DataFrame
        Return log returns.

    """
    df_data = df.copy()
    df_data = np.log(df_data / df_data.shift(1))

    return df_data.dropna()


def get_raw_returns(df):
    """
    Get raw returns of data.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    Returns
    -------
    pandas.DataFrame
        Return raw returns.

    """
    df_data = df.copy()
    df_data = df_data / df_data.shift(1) - 1

    return df_data.dropna()


def get_mean(df, sampling='daily'):
    """
    Calculate mean of dataframe columns.

    Dataframe of returns as input.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    Returns
    -------
    pandas.Series
        Return series of means.

    """
    df_data = df.copy()

    if sampling == 'daily':
        k = 1
    elif sampling == 'weekly':
        k = 5
    elif sampling == 'monthly':
        k = 12
    elif sampling == 'annual':
        k = 252
    else:
        raise ValueError('Sampling must be: daily, weekly', 'monthly', 'annual')

    return k * df_data.mean(axis=0)


def get_std(df, sampling='daily'):
    """
    Calculate standard deviation of dataframe columns.

    Dataframe of returns as input.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    Returns
    -------
    pandas.Series
        Standard deviation series.

    """
    df_data = df.copy()

    if sampling == 'daily':
        k = 1
    elif sampling == 'weekly':
        k = 5
    elif sampling == 'monthly':
        k = 12
    elif sampling == 'annual':
        k = 252
    else:
        raise ValueError('Sampling must be: daily, weekly', 'monthly', 'annual')

    return math.sqrt(k) * df_data.mean(axis=0)


def get_covar(df):
    """
    Calculate covariance among columns of dataframe.

    Dataframe of returns as input.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    Returns
    -------
    pandas.DataFrame
        Return covariance dataframe.

    """
    df_data = df.copy()
    return df_data.cov()


def get_correlation_matrix(df, method='pearson'):
    """
    Calculate correlation among columns of dataframe.

    Dataframe of returns as input.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe

    Returns
    -------
    pandas.DataFrame
        Return correlation dataframe.

    """
    df_data = df.copy()
    return df_data.corr(method=method).round(2)


def create_dataframe(tickers, con, start_date,
                     end_date=dt.now().strftime('%Y-%m-%d'),
                     bar='adj_close', freq='B'):
    index = pd.date_range(start=start_date, end=end_date, freq=freq)
    df = pd.DataFrame(index=index)

    for ticker in tickers:
        symbol = Symbol.objects.get(pk=ticker)
        query = bar_query_id('adj_close', ticker)
        df_tmp = pd.read_sql(
            query,
            con=con,
            index_col='date',
            parse_dates=True
        )
        df_bar = df_tmp[[bar]]
        df_bar.columns = [symbol.ticker]
        df = df.join(df_bar, how='left')

    return df


def plot_histogram(df,
                   user_id,
                   show=False,
                   uri=settings.REPORT_PLOTS_URI):

    mean = df.mean().values[0]
    std = df.std().values[0]

    ticker = list(df)[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(df.values, 50)
    ax.set_title("Histogram Returns of {}".format(ticker), fontsize=15)
    ax.set_xlabel("Returns")
    ax.set_ylabel("Frequency")

    ax.axvline(mean, color='g', linestyle='dashed', linewidth=2)
    ax.axvline(mean - 2 * std, color='r', linestyle='dashed', linewidth=2)
    ax.axvline(mean + 2 * std, color='r', linestyle='dashed', linewidth=2)

    if show:
        plt.show()
        return None
    else:
        path = os.path.join(uri, user_id)
        if not os.path.exists(path):
            os.makedirs(path)

        fig = random_string(10)
        fig_uri = "{}/{}.png".format(path, fig)
        plt.savefig(fig_uri)
        return path


def plot_heatmap(df,
                 user_id,
                 show=False,
                 uri=settings.REPORT_PLOTS_URI):

    symbols = list(df)

    data = df.values

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(symbols)))
    ax.set_yticks(np.arange(len(symbols)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(symbols)
    ax.set_yticklabels(symbols)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="black")

    ax.set_title("Correlation heatmap")
    fig.tight_layout()

    if show:
        plt.show()
        return None
    else:
        path = os.path.join(uri, user_id)
        if not os.path.exists(path):
            os.makedirs(path)

        fig = random_string(10)
        fig_uri = "{}/{}.png".format(path, fig)
        plt.savefig(fig_uri)
        return path
