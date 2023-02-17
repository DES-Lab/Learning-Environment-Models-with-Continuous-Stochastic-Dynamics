import random
from collections import defaultdict
from statistics import mean

import gym
import numpy as np
import torch
from gym import Env, Wrapper
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from tqdm import tqdm

from agents import load_agent
from utils import load, get_traces_from_policy, save


def get_q_values(model, obs):
    observation = obs.reshape((-1,) + model.observation_space.shape)
    observation = obs_as_tensor(observation, 'cpu')
    q_values = model.q_net(observation)
    return q_values.tolist()[0]


def get_cluster(obs, clustering_function, scaler=None):
    obs_copy = np.array(obs, dtype=np.double)
    if scaler:
        obs_copy = scaler.transform(obs_copy.reshape(1, -1))

    return f'c{clustering_function.predict(obs_copy.reshape(1, -1))[0]}'


class ClusterEnvWrapper(Wrapper):
    def __init__(self, env, target_cluster, clustering_function, scaler=None):
        super().__init__(env)
        self.target_cluster = target_cluster
        self.clustering_fun = clustering_function
        self.scaler = scaler

    def step(self, action):
        obs, rew, done, _ = super().step(action)

        cluster = get_cluster(obs, self.clustering_fun, self.scaler)
        if cluster == self.target_cluster:
            rew = 300
            # env.render()
            print('Cluster Reached')
            done = True
        else:
            # if rew >= 10:
            if rew >= 100:
                done = True
            rew = -0.01

        return obs, rew, done, _


def train_rl_agents(clusters_with_uncertainty_value, env, clustering_fun, scaler, top_and_bottom_n=3):
    sorted_clusters = dict(sorted(clusters_with_uncertainty_value.items(), key=lambda item: item[1]))
    sc = list(sorted_clusters.keys())
    clusters_to_train = sc[:top_and_bottom_n] + sc[-top_and_bottom_n:]
    # clusters_to_train = [sc[1]]

    print('Uncertainty values for selected clusters')
    for c in clusters_to_train:
        print(f'{c}: {clusters_with_uncertainty_value[c]}')

    prefix_agents = dict()
    for cluster_to_train in clusters_to_train:
        prefix_agents[cluster_to_train] = train_prefix_agent(cluster_to_train, env, clustering_fun, scaler)

    return prefix_agents


def train_prefix_agent(target_cluster, env, clustering_fun, scaler):
    cluster_env = ClusterEnvWrapper(env, target_cluster, clustering_fun, scaler)
    # eval_env = ClusterEnvWrapper(env, target_cluster, clustering_fun, scaler)

    constructor = lambda: cluster_env
    n_envs = 8

    train_env = make_vec_env(constructor, n_envs=n_envs)
    # Create the evaluation envs
    eval_env = make_vec_env(constructor, n_envs=5)

    # Adjust evaluation interval depending on the number of envs
    eval_freq = int(1e5)
    eval_freq = max(eval_freq // n_envs, 1)
    eval_freq = 1000

    # Create evaluation callback to save best model
    # and monitor agent performance
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./rl_testing_files/logs/",
        callback_on_new_best=callback_on_best,
        eval_freq=eval_freq,
        n_eval_episodes=10,
    )

    # Instantiate the agent
    # Hypermarkets from https://github.com/DLR-RM/rl-baselines3-zoo
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_starts=0,
        batch_size=128,
        buffer_size=100000,
        learning_rate=7e-4,
        target_update_interval=250,
        train_freq=1,
        gradient_steps=4,
        # Explore for 40_000 timesteps
        gamma=0.99,
        exploration_fraction=0.08,
        exploration_final_eps=0.1,
        policy_kwargs=dict(net_arch=[256, 128]),
        verbose=0,
    )

    # Train the agent (you can kill it before using ctrl+c)
    try:
        # 5e5
        model.learn(total_timesteps=int(100000), callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        pass

    # Load best model
    # model = DQN.load("rl_testing_files/logs/best_model.zip")
    # model.save(f"rl_testing_files/dqn_lunar_c{target_cluster}")

    # # Instantiate the agent
    # # https://huggingface.co/araffin/dqn-LunarLander-v2
    # model = DQN("MlpPolicy", env, verbose=0, learning_starts=0,
    #             batch_size=64,
    #             buffer_size=100000,
    #             learning_rate=7e-4,
    #             target_update_interval=250,
    #             train_freq=1,
    #             gradient_steps=4,
    #             # Explore for 40_000 timesteps
    #             exploration_fraction=0.08,
    #             exploration_final_eps=0.05,
    #             policy_kwargs=dict(net_arch=[128, 64]), )
    # # Train the agent and display a progress bar
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
    # eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)
    # model.learn(total_timesteps=int(5e5), progress_bar=True, callback=eval_callback)

    # Save the agent

    return model


def compute_clusters_and_uncertainties(model,
                                       traces_obtained_from_all_agents,
                                       n_clusters=64,
                                       include_reward=False,
                                       scale=False,
                                       clustering_type="k_means",
                                       model_name='rl_testing',
                                       clustering_samples=100000, ):
    observation_space = []
    for sampled_data in traces_obtained_from_all_agents:
        for trace in sampled_data:
            for x in trace:
                state = list(x[0][0])
                reward = x[2]
                if include_reward:
                    state.append(reward)

                observation_space.append(state)

    num_traces = sum([len(x) for x in traces_obtained_from_all_agents])

    observation_space = np.array(observation_space)
    observation_space = np.squeeze(observation_space)

    scaler = None
    if scale:
        scaler = make_pipeline(PowerTransformer())
        scaler.fit(observation_space)
        save(scaler, f'power_scaler_{model_name}_{num_traces}')

        observation_space = scaler.transform(observation_space)

    clustering_function = None
    cluster_labels = None
    if clustering_type == "k_means":
        clustering_function = KMeans(n_clusters=n_clusters, init="k-means++")
        reduced_obs_space = observation_space
        if clustering_samples:
            reduced_obs_space = random.choices(observation_space, k=clustering_samples)
        clustering_function.fit(reduced_obs_space)
        cluster_labels = clustering_function.predict(
            np.array(observation_space, dtype=float))  # list(clustering_function.labels_)
    elif clustering_type == "mean_shift":
        pass

    save(clustering_function, f'{model_name}_{clustering_type}_scale_{scale}_{n_clusters}_{num_traces}')
    print('Cluster labels computed')

    label_i = 0
    print('Obtaining cluster-q values pairs.')

    cluster_abs_diff = defaultdict(list)
    with torch.no_grad():

        for policy_samples in traces_obtained_from_all_agents:
            for sample in tqdm(policy_samples):
                for state, action, reward, done in sample:
                    cluster_label = f'c{cluster_labels[label_i]}'
                    label_i += 1

                    q_values = get_q_values(model, state)
                    cluster_abs_diff[cluster_label].append(abs(max(q_values) - min(q_values)))

    cluster_ambiguity_map = dict()
    for cluster_l, l in cluster_abs_diff.items():
        cluster_ambiguity_map[cluster_l] = mean(l)
    return cluster_ambiguity_map, clustering_function, scaler


def test_with_rl_prefixes(env, agent_under_test, prefix_agent, target_cluster, cluster_unmb, clustering_fun, scaler,
                          num_test_episodes=100):

    prefix_len_threshold = 300
    num_invalid_test_cases = 0
    num_crashes_invalid_runs = 0
    episode_rewards = []

    for _ in range(num_test_episodes):
        ep_reward = 0
        prefix_len = 0
        target_cluster_reached = False
        obs = env.reset()
        while True:
            #print('Executing prefix')
            while True:
                action, _ = prefix_agent.predict(obs)
                obs, rew, done, info = env.step(action)
                # env.render()
                ep_reward += rew
                cluster = get_cluster(obs, clustering_fun, scaler)

                if cluster == target_cluster:
                    target_cluster_reached = True
                    break

                prefix_len += 1
                if prefix_len == prefix_len_threshold:
                    break

            if not target_cluster_reached:
                num_invalid_test_cases += 1
                break

            #print('Executing suffix')
            episode_done = False
            while True:
                action, _ = agent_under_test.predict(obs)
                obs, rew, done, info = env.step(action)
                # env.render()
                ep_reward += rew
                if done:
                    if rew == -100:
                        num_crashes_invalid_runs += 1
                    episode_done = True
                    break

            if episode_done:
                episode_rewards.append(ep_reward)
                break

    print(f'Testing target: {target_cluster} : uncertainty ({cluster_unmb[target_cluster]})')
    print(f'Num testcases where target cluster was not reached: {num_invalid_test_cases}')
    print(f'Average Reward (in valid test cases): {mean(episode_rewards)}')
    print(f'Num of valid test cases: {num_test_episodes - num_invalid_test_cases}')
    print(f'Num Crashes (suffix): {num_crashes_invalid_runs}')
    print(f'-------------------------------------')


if __name__ == '__main__':
    # base enviroment
    environment = "LunarLander-v2"
    env = gym.make(environment)
    action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}

    # agent under test
    agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

    # parameters for uncertainty calculation
    num_traces = 1000
    num_clusters = 256
    scale = False
    include_reward_in_obs = False

    # Generation or loading of traces
    traces_file_name = f'rl_based_testing_{environment}_{num_traces}_traces'

    loaded_traces = load(f'{traces_file_name}')
    if loaded_traces:
        print('Traces loaded')
        all_data = loaded_traces
    else:
        print(f'Obtaining {num_traces} per agent')
        all_data = [get_traces_from_policy(agent, env, num_traces, action_map, stop_prob=0.0,
                                           randomness_probs=[0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25])]
        save(all_data, traces_file_name)
    traces = all_data

    # compute ambiguity value for each cluster and return corresponding clustering function and scaler
    cluster_ambiguity_map, clustering_function, scaler = \
        compute_clusters_and_uncertainties(agent, traces, include_reward=include_reward_in_obs)

    # train RL agents that execute the prefix of each test case
    prefix_agents = train_rl_agents(cluster_ambiguity_map, env, clustering_function, scaler, 2)

    # Test
    for c, prefix_agent in prefix_agents.items():
        test_with_rl_prefixes(env, agent, prefix_agent, c, cluster_ambiguity_map, clustering_function, scaler)
