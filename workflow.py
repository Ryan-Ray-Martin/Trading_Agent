from collections import namedtuple

import ray
from ray import workflow

import config
from processor_alpaca import DataProcessor

Prices = namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])
alpaca = DataProcessor(config.API_KEY, config.API_SECRET, config.APCA_API_BASE_URL)

@workflow.step
def extract() -> dict:
    df = alpaca.download_bars(["AAPL"], "1Min", "2021-06-08", "2021-06-20")
    return df


@workflow.step
def transform(df: dict) -> dict:
    df = alpaca.clean_data(df)
    return df


@workflow.step
def load(prices: dict) -> str:
    prices = alpaca.prices_to_relative(prices)
    return prices

def run_workflow():
    workflow.init()
    #ray.shutdown()
    #ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True)
    order_data = extract.step()
    order_summary = transform.step(order_data)
    etl = load.step(order_summary)
    prices = etl.run()
    return prices



"""if __name__ == "__main__":
    prices = run_workflow()
    print(prices)"""
