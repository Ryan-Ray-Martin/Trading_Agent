from datetime import date, timedelta

import pandas_market_calendars as mcal


def trading_days(date_range):
    nyse = mcal.get_calendar('NYSE')
    count, days = 0, 0
    end_date = date.today() - timedelta(days = 1)
    while days < date_range:
        start_date = end_date - timedelta(days = count)
        schedule = nyse.schedule(start_date, end_date)
        days = len(schedule)
        count += 1
    end_date = str(schedule.market_open.iloc[-1]).split()[0]
    start_date = str(start_date)
    return start_date, end_date

print(trading_days(42))





