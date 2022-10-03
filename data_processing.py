from collections import namedtuple

import pandas as pd

import config
from processor_alpaca import DataProcessor


class Training_Data:
    def __init__(self, data_source, **kwargs) -> namedtuple:
        self.data_source = data_source
        if self.data_source == 'alpaca':
            try:
                API_KEY = config.API_KEY
                API_SECRET = config.API_SECRET
                APCA_API_BASE_URL = config.APCA_API_BASE_URL
                self.processor = DataProcessor(API_KEY, API_SECRET, APCA_API_BASE_URL)
                print('Alpaca successfully connected')
            except:
                raise ValueError('Please input correct account info for alpaca!')
        else:
            raise ValueError('Data source input is NOT supported yet.')
    
    def extract_data(self, ticker, time_interval, start_date, end_date) -> pd.DataFrame:
        df = self.processor.download_bars([ticker], time_interval, start_date, end_date)
        return df

    def transform_data(self, df: pd.DataFrame) -> namedtuple:
        df = self.processor.clean_data(df)
        return df

    def load_data(self, prices: namedtuple) -> namedtuple:
        df = self.processor.prices_to_relative(prices)
        return df
    
    def run_workflow(self, ticker, time_interval, start_date, end_date):
        data = self.extract_data(ticker, time_interval, start_date, end_date)
        data = self.transform_data(data)
        data = self.load_data(data)
        return data


if __name__ == "__main__":
    TD = Training_Data('alpaca')
    data = TD.run_workflow(ticker="TSLA", time_interval="1Day", start_date="2022-02-07", end_date="2022-09-20")
    print(data)