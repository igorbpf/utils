import numpy as np
import matplotlib.pyplot as plt

from finance.ratio import sharpe_ratio
from report.utils import random_string

import os
from django.conf import settings


plt.switch_backend('Agg')


def initialize_weights(num_assets):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    return weights


def expected_portfolio_returns(df, weights):
    return np.sum(df.mean() * weights) * 252


def expected_portfolio_std(df, weights):
    return np.sqrt(np.dot(weights.T, np.dot(df.cov() * 252, weights)))


def monte_carlo_portfolios(df, risk_free_rate):

    port_returns = []
    port_std = []
    sharpe = []
    port_weights = []

    stocks = list(df)

    # Monte-Carlo simulation
    for i in range(7000):
        weights = initialize_weights(len(stocks))
        port_returns.append(expected_portfolio_returns(df, weights))
        port_std.append(expected_portfolio_std(df, weights))
        data = (weights * df)
        data = data.sum(axis=1)
        sharpe.append(sharpe_ratio(data, risk_free_rate))
        port_weights.append(weights)

    max_sharpe_index = np.argmax(np.array(sharpe))
    max_sharpe = sharpe[max_sharpe_index]
    weigh = port_weights[max_sharpe_index]
    vol_sharpe_max = port_std[max_sharpe_index]

    port_returns = np.array(port_returns)
    port_std = np.array(port_std)

    return port_returns, port_std, max_sharpe, weigh, vol_sharpe_max


def plot_bullet(df_returns,
                df_std,
                user_id,
                max_sharpe=None,
                sharpe_vol=None,
                risk_free_rate=0.05,
                show=False,
                uri=settings.REPORT_PLOTS_URI):

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(df_std, df_returns,
                         c=(df_returns - risk_free_rate)/df_std, marker='o')

    ax.set_xlabel("Yearly Expected Volatility")
    ax.set_ylabel("Yearly Expected Returns")
    ax.set_title("Portfolios", fontsize=10)
    # ax.set_ylim(bottom=0.0)
    ax.grid(True)
    fig.colorbar(scatter, label="Sharpe Ratio")

    if max_sharpe and sharpe_vol:
        vols = np.arange(0, sharpe_vol + 0.07, 0.01)
        exps = risk_free_rate + vols * max_sharpe

        ax.set_xlim(left=0.0)
        ax.plot(vols, exps, c='red')
        ax.set_title("Portfolios and Capital Allocation Line")

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
