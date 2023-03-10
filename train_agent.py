import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo

import config
import stock_env
from create_env import env_creator
from workflow_actor import Training_Data


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
        bars_count=4,
        reset_on_close=True))

    config_model = ppo.DEFAULT_CONFIG.copy()
    config_model["model"]["use_attention"] = True
    config_model["model"]["attention_num_transformer_units"] = 1
    #config_model["model"]["attention_use_n_prev_actions"] = True
    #config_model["model"]["attention_use_n_prev_rewards"] = True
    config_model["model"]["attention_dim"] = 32
    config_model["model"]["attention_memory_inference"] = 10
    config_model["model"]["attention_memory_training"] = 10


    config_model["env"] = env_name

    #config_model["lr"] = 0.003
    config_model["seed"] = 12345
    #config_model["sgd_minibatch_size"]= 128
    #config_model["lambda"]= 0.7
    config_model["framework"] = "tf"
    config_model["num_workers"] = 0
    config_model["exploration_config"] = {
            "type": "Curiosity",  # <- Use the Curiosity module for exploring.
            "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
            "lr": .003,  # Learning rate of the curiosity (ICM) module. Also, .001
            "feature_dim": 32,  # Dimensionality of the generated feature vectors.
            # Setup of the feature net (used to encode observations into feature (latent) vectors).
            "feature_net_config": {
                "fcnet_hiddens": [],
                "fcnet_activation": "relu",
            },
            "inverse_net_hiddens": [32],  # Hidden layers of the "inverse" model.
            "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            "forward_net_hiddens": [32],  # Hidden layers of the "forward" model.
            "forward_net_activation": "relu",  # Activation of the "forward" model.
            "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            # Specify, which exploration sub-type to use (usually, the algo's "default"
            # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
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
        bars_count=4,
        reset_on_close=False,
        commission=0.0,
        random_ofs_on_reset=False,
        reward_on_close=False,
        volumes=False
    )

    # Set attention net's initial internal state.
    num_transformers = 1
    memory_inference = 10
    attention_dim = 32

    state = [
        np.zeros([memory_inference, attention_dim], np.float32)
        for _ in range(num_transformers)
    ]

    #state = agent.get_policy().get_initial_state()
    obs = env.reset()

    init_prev_a = prev_a = 0
    init_prev_r = prev_r = 0.0

    eps = 0
    ep_reward = 0
    rewards = []
    while True:
        action, state_out, _ = agent.compute_action(
            obs,
            state=state,
            prev_action=prev_a,
            prev_reward=prev_r,
            full_fetch=True,
        )
        obs, reward, done, _ = env.step(action)
        prev_a = action
        prev_r = reward
        ep_reward += reward
        eps += 1
        rewards.append(ep_reward)
        print("{}: reward={} action={}".format(eps, ep_reward, action))
        state = [
                np.concatenate([state[i], [state_out[i]]], axis=0)[1:]
                for i in range(num_transformers)
            ]
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
    stock_universe = ['SPY']

    for i in range(len(stock_universe)):
        Agent(
            ticker=stock_universe[i],
            data_source='alpaca',
            train_start_date='2022-02-11',
            train_end_date='2022-09-09',
            test_start_date='2022-09-12',
            test_end_date='2022-09-26',
            time_interval='1Day',
            API_KEY = config.API_KEY,
            API_SECRET = config.API_SECRET,
            APCA_API_BASE_URL = config.APCA_API_BASE_URL).train()
