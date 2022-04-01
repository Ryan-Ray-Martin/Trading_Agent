def env_creator(env_name):
    if env_name == "StocksEnv-v0":
        from stock_env import StocksEnv as env
    else:
        raise NotImplementedError
    return env