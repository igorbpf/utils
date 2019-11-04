import math


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
        return math.sqrt(252)
    elif sampling_rate == 'weekly':
        return math.sqrt(52)
    elif sampling_rate == 'monthly':
        return math.sqrt(12)
    else:
        return 1.0


def convert_rate(rate, periods):
    return (1 + rate)**(1/periods) - 1


def treynor_ratio(df, beta, risk_free_rate=0.05, sampling_rate='daily'):

    df_data = df.copy()

    risk_free_rate = convert_rate(risk_free_rate, 252)

    ann_param = get_annualization_parameter(sampling_rate)

    TR = ann_param * (df_data.mean() - risk_free_rate)/beta
    TR = TR.iloc[0]
    return TR


def sharpe_ratio(df, risk_free_rate=0.05, sampling_rate='daily'):

    df_data = df.copy()

    risk_free_rate = convert_rate(risk_free_rate, 252)

    ann_param = get_annualization_parameter(sampling_rate)

    ShR = ann_param * (df_data.mean() - risk_free_rate)/df_data.std()
    ShR = ShR.iloc[0]
    return ShR


def information_ratio(df, df_benchmark, risk_free_rate=0.05, sampling_rate='daily'):

    df_data = df.copy()
    df_benchmark_data = df_benchmark.copy()

    ann_param = get_annualization_parameter(sampling_rate)

    den = df_data.std() - df_benchmark_data.std()

    IR = ann_param * (df_data.mean() - df_benchmark_data.mean())/den
    IR = IR.iloc[0]
    return IR


def sortino_ratio(df, risk_free_rate=0.05, sampling_rate='daily'):

    df_data = df.copy()
    ann_param = get_annualization_parameter(sampling_rate)

    risk_free_rate = convert_rate(risk_free_rate, 252)

    num = ann_param * (df_data.mean() - risk_free_rate)
    den = df_data[df_data.values < 0].std()

    SoR = num/den
    SoR = SoR.iloc[0]
    return SoR
