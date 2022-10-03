import alpaca_trade_api as tradeapi
import config
import time

api = tradeapi.REST(config.API_KEY,
    config.API_SECRET, config.APCA_API_BASE_URL)

# Get a list of all active assets.
active_assets = api.list_assets(status='active')

# Filter the assets down to just those on NASDAQ.
start = time.time()
nasdaq_assets = dict(set([(a.name, a.symbol) for a in active_assets]))
end = time.time()
total_time = end - start
print(len(nasdaq_assets), total_time)

# TOD0: store this giant hashset in redis json to be queried in a sane way as a cache