import ray
from ray import workflow
from processor_alpaca import DataProcessor
from collections import namedtuple
import config
import numpy as np


Prices = namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])


@workflow.step
def extract() -> dict:
    alpaca = DataProcessor(config.API_KEY, config.API_SECRET, config.APCA_API_BASE_URL)
    df = alpaca.download_bars(["AAPL"], "1Min", "2021-06-08", "2021-06-20")
    return df


@workflow.step
def transform(df: dict) -> dict:
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


@workflow.step
def load(prices: dict) -> str:
    assert isinstance(prices, Prices)
    rh = (prices.high - prices.open) / prices.open
    rl = (prices.low - prices.open) / prices.open
    rc = (prices.close - prices.open) / prices.open
    return Prices(open=prices.open, high=rh, low=rl, close=rc, volume=prices.volume)


if __name__ == "__main__":
    workflow.init()
    order_data = extract.step()
    order_summary = transform.step(order_data)
    etl = load.step(order_summary)
    print(etl.run())