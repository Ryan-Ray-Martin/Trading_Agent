import alpaca_trade_api as tradeapi
import config

api = tradeapi.REST(config.API_KEY,
    config.API_SECRET, config.APCA_API_BASE_URL)

# Get a list of all active assets.
active_assets = api.list_assets(status='active')

# Filter the assets down to just those on NASDAQ.
nasdaq_assets = dict(set([(a.name, a.symbol) for a in active_assets]))
print(nasdaq_assets)