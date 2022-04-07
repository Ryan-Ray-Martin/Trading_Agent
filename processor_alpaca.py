import alpaca_trade_api as tradeapi
import config
from collections import namedtuple
import numpy as np
import pandas as pd

Prices = namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])

class DataProcessor:
    def __init__(self, API_KEY=None, API_SECRET=None, APCA_API_BASE_URL=None, api=None):
        if api is None:
            try:
                self.api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, "v2")
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.api = api

    def download_bars(self, ticker, time_interval, start_date, end_date) -> pd.DataFrame:
        """A method that fetches the OHLC data from the alpaca API"""
        df = self.api.get_bars(ticker, time_interval, start_date, end_date, adjustment='raw').df
        return df

    def clean_data(self, df) -> namedtuple:
        """A method that forward fills the missing data after the first index
        and then backward fills the missing value at the front of the index.
        The features are then turned to arrays, and then passed into a Prices
        object of numpy arrays dtype float32 (Torch)"""
        df = df.ffill(axis=0).bfill(axis=0)
        o=df['open'].to_list()
        h=df['high'].to_list()
        l=df['low'].to_list()
        c=df['close'].to_list()
        v=df['volume'].to_list()
        return Prices(open=np.array(o, dtype=np.float32),
                  high=np.array(h, dtype=np.float32),
                  low=np.array(l, dtype=np.float32),
                  close=np.array(c, dtype=np.float32),
                  volume=np.array(v, dtype=np.float32))

    def prices_to_relative(self, prices) -> namedtuple:
        """
        Convert prices to relative in respect to open price
        :param ochl: tuple with open, close, high, low
        :return: tuple with open, rel_close, rel_high, rel_low
        """
        assert isinstance(prices, Prices)
        rh = (prices.high - prices.open) / prices.open
        rl = (prices.low - prices.open) / prices.open
        rc = (prices.close - prices.open) / prices.open
        return Prices(open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)

   

if __name__ == '__main__':
    alpaca = DataProcessor(config.API_KEY, config.API_SECRET, config.APCA_API_BASE_URL)
    df = alpaca.download_bars(["AAPL"], "1Min", "2022-02-08", "2022-04-05")
    print(df)
   # prices = alpaca.clean_data(df)
   # prices_rel = alpaca.prices_to_relative(prices)
   # print(prices_rel)"""