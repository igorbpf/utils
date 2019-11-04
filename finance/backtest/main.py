import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import bt
from finance.indicator import get_sma


def above_sma(df, sma_per=50, name='above_sma'):
    """
    Long securities that are above their n period
    Simple Moving Averages with equal weights.
    """
    df_data = df.copy()

    # calc sma
    sma = get_sma(df_data, window=sma_per)

    # create strategy
    s = bt.Strategy(name, [bt.algos.SelectWhere(df_data > sma),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])

    # now we create the backtest
    return bt.Backtest(s, df_data)


# first let's create a helper function to create a ma cross backtest
def sma_cross(df, short_ma=50, long_ma=200, name='ma_cross'):
    # these are all the same steps as above
    df_data = df.copy()

    short_sma = get_sma(df_data, window=short_ma)
    long_sma = get_sma(df_data, window=long_ma)

    # target weights
    tw = long_sma.copy()

    tw[short_sma > long_sma] = 1.0
    tw[short_sma <= long_sma] = -1.0
    tw[long_sma.isnull()] = 0.0

    # here we specify the children (3rd) arguemnt to make sure the strategy
    # has the proper universe. This is necessary in strategies of strategies
    s = bt.Strategy(name, [bt.algos.RunDaily(), bt.algos.WeighTarget(tw), bt.algos.Rebalance()])

    return bt.Backtest(s, df_data)


# simple backtest to test long-only allocation
def long_only(df, name='long_only'):

    df_data = df.copy()

    s = bt.Strategy(name, [bt.algos.RunOnce(),
                           bt.algos.SelectAll(),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])

    return bt.Backtest(s, df_data)


def run_strategy(*strategy):
    return bt.run(*strategy)
