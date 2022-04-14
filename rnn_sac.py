import json
import os
from pathlib import Path
import numpy as np

import ray
from ray import tune
from ray.rllib.agents.registry import get_trainer_class

import matplotlib as mpl
import ray
from ray import tune

import config
import stock_env
from workflow_actor import Training_Data

mpl.use("Agg")
from functools import partial

import matplotlib.pyplot as plt
from ray.rllib.agents import ppo, sac
from ray.rllib.agents.callbacks import MultiCallbacks, RE3UpdateCallbacks
from ray.rllib.models import ModelCatalog

from create_env import env_creator



def train(
    start_date,
    end_date,
    time_interval,
    ticker,
    data_source, 
    env_name,
    **kwargs
):  
    """ A scalable, reusable training function containing configurations for rllib """

    cwd = kwargs.get("cwd", "./" + str(ticker))

    DP = Training_Data(data_source)
    data = DP.run_workflow(ticker, time_interval, start_date, end_date)

    stock_data = {"training_data": data}  

    # register the env
    env = env_creator("StocksEnv-v0")
    tune.register_env(env_name, lambda _: env(
        stock_data,
        bars_count=30,
        state_1d=False,
        reset_on_close=True))

    config_model = {
        "seed": 42,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": "tf",
        "env": env_name,
        "horizon": 1000,
        "gamma": 0.95,
        "batch_mode": "complete_episodes",
        "prioritized_replay": False,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "train_batch_size": 480,
        "target_network_update_freq": 480,
        "tau": 0.3,
        "burn_in": 4,
        "zero_init_states": False,
        "optimization": {
            "actor_learning_rate": 0.005,
            "critic_learning_rate": 0.005,
            "entropy_learning_rate": 0.0001,
        },
        "model": {
            "max_seq_len": 30,
        },
        "policy_model": {
            "use_lstm": True,
            "lstm_cell_size": 64,
            "fcnet_hiddens": [64, 64],
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
        },
        "Q_model": {
            "use_lstm": True,
            "lstm_cell_size": 64,
            "fcnet_hiddens": [64, 64],
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
        },
    }

    stop = {
            "training_iteration": 10,
            "timesteps_total": 100000,
            "episode_reward_mean": 0.25,
        }

    analysis = ray.tune.run(
        "RNNSAC",
        config=config_model,
        local_dir=cwd,
        stop=stop,
        checkpoint_at_end = True)

    # extract the best trial from tune run based on episode reward mean
    best_trial=analysis.get_best_trial(
            'episode_reward_mean',
            mode="max",
            scope="all",
            filter_nan_and_inf=True)

    # retreive the checkpoint path and config from the best trial
    return best_trial.checkpoint.value, best_trial.config

def test(
    start_date,
    end_date,
    ticker,
    time_interval,
    data_source,
    checkpoint_path,
    agent_config, 
    env_name,
    **kwargs,
):

    """ A scalable, resuable testing function for trained rllib agents """

    DP = Training_Data(data_source)
    test_data = DP.run_workflow(ticker, time_interval, start_date, end_date)

    stock_data = {"test_data": test_data}

    env = stock_env.StocksEnv(
        stock_data,
        bars_count=30,
        reset_on_close=False,
        commission=0.0,
        state_1d=False,
        random_ofs_on_reset=False,
        reward_on_close=False,
        volumes=False
    )

    agent = get_trainer_class("RNNSAC")(
        env=env_name, config=agent_config
    )
    agent.restore(checkpoint_path)

    init_prev_a = prev_a = 0
    init_prev_r = prev_r = 0.0

    # Set LSTM's initial internal state.
    lstm_cell_size = 64
    # range(2) b/c h- and c-states of the LSTM.
    state = [np.zeros([lstm_cell_size], np.float32) for _ in range(2)]

    state = agent.get_policy().get_initial_state()
    obs = env.reset()

    eps = 0
    ep_reward = 0
    rewards = []
    while True:
        action, state_out, info_trainer = agent.compute_action(
            obs,
            state=state,
            prev_action=prev_a,
            prev_reward=prev_r,
            full_fetch=True,
        )
        obs, reward, done, info = env.step(action)
        prev_a = action
        prev_r = reward
        ep_reward += reward
        eps += 1
        rewards.append(ep_reward)
        print("{}: reward={} action={}".format(eps, ep_reward, action))
        state = state_out
        if init_prev_a is not None:
            prev_a = action
        if init_prev_r is not None:
            prev_r = reward
        if done:
            break
    
    # plot rewards
    plt.clf()
    plt.plot(rewards, label = 'Agent', color = 'blue')
    plt.title("PnL for {} on {}".format(ticker, end_date))
    plt.ylabel("Reward, %")
    plt.savefig("{}_{}.png".format(ticker, end_date))

class Agent(object):
    def __init__(self, ticker, data_source, train_start_date, train_end_date, test_start_date, test_end_date, time_interval, **kwargs) -> None:
        self.ticker = ticker
        self.data_source = data_source
        self.time_interval = time_interval
        self.env_name = "{}_env".format(ticker)
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_end_date = test_end_date
        self.test_start_date = test_start_date

    def train(self):
        print("starting training")
        best_trial_checkpoint, best_trial_config = train(
            start_date = self.train_start_date,
            end_date = self.train_end_date,
            ticker = self.ticker,
            data_source = self.data_source,
            time_interval = self.time_interval,
            env_name = self.env_name)
        print("starting testing")
        test(
            start_date = self.test_start_date,
            end_date = self.test_end_date,
            ticker = self.ticker,
            data_source = self.data_source,
            time_interval = self.time_interval,
            checkpoint_path = best_trial_checkpoint,
            agent_config = best_trial_config,
            env_name = self.env_name)


if __name__ == '__main__':

    """ by using the following setup, we can effectively train multiple
     individual agents on a given portfolio; thus, addressing the concern
     of diversification and capacity in leveraged trades """

    """workflow will only run if dates are at least one day prior to current"""

    ray.shutdown()
    ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True)
    stock_universe = ['AAP', 'ORLY', 'AMPH','COP']

    for i in range(len(stock_universe)):
        Agent(
            ticker=stock_universe[i],
            data_source='alpaca',
            train_start_date='2022-02-11',
            train_end_date='2022-04-07',
            test_start_date='2022-04-08',
            test_end_date='2022-04-08',
            time_interval='1Min',
            API_KEY = config.API_KEY,
            API_SECRET = config.API_SECRET,
            APCA_API_BASE_URL = config.APCA_API_BASE_URL).train()