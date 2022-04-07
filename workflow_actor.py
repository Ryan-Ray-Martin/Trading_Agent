from collections import namedtuple

import pandas as pd
from ray import workflow

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
    
    @workflow.step
    def extract_data(processor, ticker, time_interval, start_date, end_date) -> pd.DataFrame:
        df = processor.download_bars([ticker], time_interval, start_date, end_date)
        return df

    @workflow.step
    def transform_data(processor, df: pd.DataFrame) -> namedtuple:
        df = processor.clean_data(df)
        return df

    @workflow.step
    def load_data(processor, prices: namedtuple) -> namedtuple:
        df = processor.prices_to_relative(prices)
        return df
    
    def run_workflow(self, ticker, time_interval, start_date, end_date):
        workflow.init()
        data = self.extract_data.step(self.processor, ticker, time_interval, start_date, end_date)
        data = self.transform_data.step(self.processor, data)
        data = self.load_data.step(self.processor, data)
        prices = data.run()
        return prices


if __name__ == "__main__":
    TD = Training_Data('alpaca')
    data = TD.run_workflow(ticker="TSLA", time_interval="1Min", start_date="2022-02-07", end_date="2022-04-06")
    print(data)
    





