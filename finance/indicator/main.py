from finance.handler import complete_missing_values


def get_sma(df, window=20, complete_data=True):
    df_sma = df.rolling(window).mean()

    if complete_data:
        return complete_missing_values(df_sma)

    return df_sma


def get_volatility(df, window=20, complete_data=True):
    vol = df.rolling(window).std()

    if complete_data:
        vol = complete_missing_values(vol)

    return vol


def get_max(df, window=20, complete_data=True):
    max = df.rolling(window).max()

    if complete_data:
        max = complete_missing_values(max)

    return max


def get_min(df, window=20, complete_data=True):
    min = df.rolling(window).min()

    if complete_data:
        min = complete_missing_values(min)

    return min


def get_bollinger_bands(df, window=20, complete_data=True):
    df_sma = get_sma(df, complete_data=True)
    up_bb = df_sma + 2 * get_volatility(df, complete_data=complete_data)
    do_bb = df_sma - 2 * get_volatility(df, complete_data=complete_data)

    headers = list(df)

    up_headers = [h + '_upper_band' for h in headers]
    do_headers = [h + '_lower_band' for h in headers]

    up_bb.columns = up_headers
    do_bb.columns = do_headers

    return up_bb, do_bb


def get_min_max_bands(df, window=20, complete_data=True):
    up_max = get_max(df, complete_data=complete_data)
    do_min = get_min(df, complete_data=complete_data)

    headers = list(df)

    max_headers = [h + '_max_band' for h in headers]
    min_headers = [h + '_min_band' for h in headers]

    up_max.columns = max_headers
    do_min.columns = min_headers

    return up_max, do_min


def get_ewa(df, window=20):
    return df.ewm(span=window).mean()


def get_ppo(df):
    ppo = get_ewa(df, window=9) - get_ewa(df, window=26)
    ppo = ppo/get_ewa(df, window=26)
    return ppo


def get_rsi(df, window=14, exponential=True):
    delta = df.diff()
    delta = delta[1:]

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    if exponential:
        roll_up = get_ewa(up, window=window)
        roll_down = get_ewa(down.abs(), window=window)

    else:
        roll_up = get_sma(up, window=window)
        roll_down = get_sma(down.abs(), window=window)

    rs = roll_up/roll_down
    rsi = 100.0 - (100.0/(1.0 + rs))

    return rsi


def get_macd(df):
    macd = get_ewa(df, window=12) - get_ewa(df, window=26)
    signal = get_ewa(macd, window=9)
    histogram = macd - signal
    return signal, macd, histogram


def max_drowdown():
    pass
