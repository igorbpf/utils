import numpy as np
from datetime import datetime as dt
from datetime import date, timedelta
import datetime
import calendar


def str2date(date):
    return dt.strptime(date, "%Y-%m-%d")


def date2date64(date):
    return np.datetime64(date)


def date2string(date):
    return date.strftime('%Y-%m-%d')


def get_today():
    return date2string(date.today())


def get_N_back(num_days):
    return date2string(date.today() - timedelta(days=num_days))


def date_range(numdays, base=dt.today(), drop_weekends=True):
    date_list = []
    delta = 0
    for x in range(0, numdays):

        date = base + datetime.timedelta(days=x + delta)

        if drop_weekends:
            week_day = calendar.day_name[date.weekday()]

            if week_day == 'Saturday':
                date = date + datetime.timedelta(days=2)
                delta += 2

            elif week_day == 'Sunday':
                date = date + datetime.timedelta(days=1)
                delta += 1

        date_list.append(date)

    return date_list


def datetime64_to_unix(datetime):
    date = datetime
    ts = (date - np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1, 's')
    return ts


def df2list_of_dict(df, column=''):
    data = df.T.to_dict().values()
    if column == '':
        return data
    return sorted(data, key=lambda k: k[column])
