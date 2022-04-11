from datetime import date, timedelta

import pandas_market_calendars as mcal


def trading_days(date_range):
    nyse = mcal.get_calendar('NYSE')
    count = 0
    days = 0
    end_date = date.today() - timedelta(days = 1)
    while days < date_range:
        start_date = end_date - timedelta(days = count)
        schedule = nyse.schedule(start_date, end_date)
        days = len(schedule)
        count += 1
    return start_date

print(trading_days(40))





