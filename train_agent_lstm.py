import matplotlib as mpl
import ray
from ray import tune

import config
import stock_env
from workflow_actor import Training_Data

mpl.use("Agg")
from functools import partial

import matplotlib.pyplot as plt
from ray.rllib.agents import ppo
from ray.rllib.agents.callbacks import MultiCallbacks, RE3UpdateCallbacks
from ray.rllib.models import ModelCatalog

from create_env import env_creator
from fc_net import CustomModel


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

    # register the custom model
    ModelCatalog.register_custom_model("fc_net", CustomModel)

    config_model = ppo.DEFAULT_CONFIG.copy()
    # Add a new RE3UpdateCallbacks
    config_model["model"]["custom_model"] = "fc_net"
    config_model["model"]["vf_share_layers"] = False
    config_model["callbacks"] = MultiCallbacks([
    config_model["callbacks"],
        partial(
            RE3UpdateCallbacks,
            embeds_dim=128,
            beta_schedule="linear decay",
            k_nn=600,
            ),
        ])

    config_model["env"] = env_name

    config_model["lr"] = 0.003
    config_model["seed"] = 12345
    config_model["sgd_minibatch_size"]= 512
    #config_model["lambda"]= 0.7
    config_model["framework"] = "tf"
    # Add type as RE3 in the exploration_config parameter
    config_model["exploration_config"] = {
            "type": "RE3",
            "sub_exploration": {
                "type": "StochasticSampling",
            }
        }

    stop = {
            "training_iteration": 1,
            "timesteps_total": 100000,
            "episode_reward_mean": 0.25,
        }

    analysis = ray.tune.run(
        ray.rllib.agents.ppo.PPOTrainer,
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

    agent = ray.rllib.agents.ppo.PPOTrainer(config=agent_config, env=env_name)
    agent.restore(checkpoint_path)

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

    episode_reward = 0
    total_steps = 0
    rewards = []

    obs = env.reset()
    while True:
        action = agent.compute_action(obs)
        obs, reward, done, _ = env.step(action)
        print("done", done)
        episode_reward += reward
        total_steps += 1
        rewards.append(episode_reward)
        print("{}: reward={} action={}".format(total_steps, episode_reward, action))
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