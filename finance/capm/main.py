import numpy as np
import matplotlib.pyplot as plt
from finance.handler import complete_missing_values
from collections import OrderedDict

from report.utils import random_string
import os
from django.conf import settings


def get_beta_alpha(df, df_benchmark):
    """
    Calculate beta and alpha of assets.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of assets returns

    df_benchmark : pandas.DataFrame
        DataFrame of benchmark returns

    Returns
    -------
    beta: numpy.ndarray
        array of beta values of assets

    alpha: numpy.ndarray
        array of alpha values of assets

    """

    m1 = df_benchmark.shape[0]
    df_benchmark = np.reshape(df_benchmark.values, m1)

    beta, alpha = np.polyfit(df_benchmark, df, 1)
    return beta, alpha


def security_market_line(df_benchmark, beta, risk_free_rate=0.05):
    """
    Calculate the annualized expected return according to capm.

    Parameters
    ----------
    df_benchmark : pandas.DataFrame
        DataFrame of benchmark returns

    beta : float
        beta value of the asset

    risk_free_rate : float
        risk free rate investment rate

    Returns
    -------
    float
        expected return of the asset.

    """

    exp_mark = 252 * df_benchmark.mean().values[0]

    return risk_free_rate + beta * (exp_mark - risk_free_rate)


def capital_allocation_line(df, vol_comp, risk_free_rate=0.05):
    """
    Calculate the capital allocation line (CAL).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of risky returns

    vol_comp : float
        Annualized volatility of a mixed portfolio (risk free and risky assets)

    Returns
    -------
    float
        Expected return of a mixed portfolio
    """

    exp_port = 252 * df.mean()
    vol_port = 252 * df.vol()

    return risk_free_rate + vol_comp * (exp_port - risk_free_rate)/vol_port


def get_moving_beta(df, ticker_base, window=20, complete_data=True):
    df_data = df.copy()
    cov = df_data.rolling(window=window).cov()
    cov = cov[ticker_base].unstack(level=1)

    benchmark = cov[[ticker_base]].copy()

    cov = cov.drop(ticker_base, 1)

    if complete_data:
        cov['moving beta'] = complete_missing_values(cov/benchmark.values)
    else:
        cov['moving beta'] = cov/benchmark.values

    return cov[['moving beta']]


def plot_security_market_line(df,
                              df_benchmark,
                              user_id=None,
                              risk_free_rate=0.05,
                              show=False,
                              uri=settings.REPORT_PLOTS_URI):

    beta, alpha = get_beta_alpha(df, df_benchmark)

    beta_max = np.amax(beta)
    beta_min = np.amin(beta)

    if beta_min > 0 and beta_min < 0.5:
        beta_min = 0
    elif beta_min >= 0.5 and beta_min < 1.0:
        beta_min = 0.6

    beta_max = beta_max + 0.02

    betas = np.arange(beta_min, beta_max, 0.01)
    expected = security_market_line(df_benchmark, betas, risk_free_rate)

    real = []
    situation = []
    valued = []

    for i, b in enumerate(beta):
        exp = security_market_line(df_benchmark, b, risk_free_rate)
        r_val = exp + alpha[i]
        real.append(r_val)
        if r_val > exp:
            situation.append('g')
            valued.append('Undervalued')
        elif r_val < exp:
            situation.append('r')
            valued.append('Overvalued')
        else:
            situation.append('y')
            valued.append('Just Right')

    fig, ax = plt.subplots(figsize=(10,5))

    for i, b in enumerate(beta):
        ax.scatter(b, real[i], c=situation[i], label=valued[i])

    ax.set_ylim(bottom=0.0, top=max(real) + min(real)/2)
    ax.set_xlabel("Beta (Systematic Risk)")
    ax.set_ylabel("Return per Risk")
    ax.set_title("Security Market Line", fontsize=10)
    ax.legend()
    ax.grid(True)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.3)

    names = list(df)

    for i, name in enumerate(names):
        t = ax.annotate(name, (beta[i] - 0.005, real[i] + 0.005 + min(real)/25), bbox=bbox_props)
        bb = t.get_bbox_patch()
        bb.set_boxstyle("square", pad=0.6)

    ax.plot(betas, expected, c='blue')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

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


def plot_correlation(df1,
                     df2,
                     user_id,
                     show=False,
                     uri=settings.REPORT_PLOTS_URI):

    beta, alpha = get_beta_alpha(df1, df2)

    name1 = list(df1)[0]
    name2 = list(df2)[0]

    rets = df2.values
    max = np.amax(rets)
    min = np.amin(rets)

    max = max + 0.02
    min = min - 0.02

    ret2_vals = np.arange(min, max, 0.005)
    ret1_vals = beta[0] * ret2_vals + alpha[0]

    fig, ax = plt.subplots(figsize=(10,6))

    ax.scatter(df2.values, df1.values, c='blue')
    ax.plot(ret2_vals, ret1_vals, c='red')

    ax.set_title("Returns", fontsize=15)
    ax.set_ylabel("{} Returns".format(name1))
    ax.set_xlabel("{} Returns".format(name2))
    ax.grid(True)

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
