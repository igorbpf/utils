# from finance.capm import plot_security_market_line

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt


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


def check_negativity(val):
    if val < 0:
        return 1
    else:
        return 0


def count_days_negative_returns(df):

    df_data = df.copy()

    df_data['negative days'] = df_data['Adj Close'].apply(check_negativity)

    num_negative_days = df_data['negative days'].sum()
    total_days = df_data['negative days'].shape

    return num_negative_days, total_days


def count_neg(array):
    array_bool = array < 0
    num_neg = array_bool.sum()
    size = array.shape[0]

    if size == num_neg:
        return 1
    else:
        return 0


def convert_rate(rate, periods):
    return (1 + rate)**(1/periods) - 1


def get_annualization_parameter(sampling_rate):
    """
    Calculate the annualization parameter.

    Parameters
    ----------
    sampling_rate : string
        String of the sampling rate chosen

    Returns
    -------
    float
        Return annualization parameter.

    """

    if sampling_rate == 'daily':
        return np.sqrt(252)
    elif sampling_rate == 'weekly':
        return np.sqrt(52)
    elif sampling_rate == 'monthly':
        return np.sqrt(12)
    else:
        return 1.0


def sharpe_ratio(array):
    mean = array.mean()
    std = array.std()

    risk_free_rate = convert_rate(0.05, 252)

    ann_param = get_annualization_parameter('daily')

    ShR = ann_param * (mean - risk_free_rate)/std
    return ShR


df = pd.read_csv('tickers/FB.csv', index_col='Date', parse_dates=True)

print(df.index)


print(df.head())
print("###########")
df_ret = get_log_returns(df)

series_ret = df_ret['Adj Close']

# print(count_days_negative_returns(df_ret))

series_count = series_ret.rolling(3).apply(count_neg, raw=True)

print(series_count.sum())
print(series_count.shape)

moving_sharpe = series_ret.rolling(20).apply(sharpe_ratio, raw=True)

print(moving_sharpe.tail())

moving_sharpe.plot()
plt.savefig('./fig.png')


# def create_dataframe(tickers, con, start_date,
#                      end_date=dt.now().strftime('%Y-%m-%d'),
#                      bar='adj_close', freq='B'):
#     index = pd.date_range(start=start_date, end=end_date, freq=freq)
#     df = pd.DataFrame(index=index)
#
#     for ticker in tickers:
#         symbol = Symbol.objects.get(pk=ticker)
#         query = bar_query_id('adj_close', ticker)
#         df_tmp = pd.read_sql(
#             query,
#             con=con,
#             index_col='date',
#             parse_dates=True
#         )
#         df_bar = df_tmp[[bar]]
#         df_bar.columns = [symbol.ticker]
#         df = df.join(df_bar, how='left')
#
#     return df
#
#
# plot_security_market_line()
