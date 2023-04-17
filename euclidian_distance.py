from random import shuffle
from statistics import mean

import gym
import numpy as np
from stable_baselines3 import DQN

from agents import load_agent


def get_observation_action_pairs(env, num_ep=20):
    dqn_agent = load_agent("araffin/dqn-LunarLander-v2", 'dqn-LunarLander-v2.zip', DQN)

    sample_num = 0
    observation_actions_pairs = []
    for _ in range(num_ep):
        sample_num += 1
        if sample_num % 100 == 0:
            print(sample_num)
        obs = env.reset()
        while True:
            action, state = dqn_agent.predict(obs)
            observation_actions_pairs.append((obs, action))
            obs, reward, done, _ = env.step(action)
            if done:
                break

    return observation_actions_pairs


def chose_action_based_on_euclidean_distance(env_obs, obs_action_pairs, num_comparison=1000):
    chosen_action = None
    min_dist = None

    shuffle(obs_action_pairs)
    distances = []

    for obs, action in obs_action_pairs[:num_comparison]:
        ed = np.linalg.norm(env_obs - obs)

        distances.append(ed)
        if not min_dist or ed < min_dist:
            min_dist = ed
            chosen_action = action

    return chosen_action


def evaluate_on_environment(env, obs_action_pairs, num_episodes=100, render=False):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        ep_rew = 0
        while True:
            action = chose_action_based_on_euclidean_distance(obs, obs_action_pairs)
            obs, rew, done, info = env.step(action)

            if render:
                env.render()
            ep_rew += rew
            if done:
                print('Episode reward:', ep_rew)
                all_rewards.append(ep_rew)
                break

    print('Mean reward:', mean(all_rewards))



env_name = "LunarLander-v2"
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}

num_traces = 10
load_observations = True

env = gym.make(env_name)

print('Computing observation action pairs')
obs_action_pairs = get_observation_action_pairs(env, num_traces)
print(len(obs_action_pairs))

evaluate_on_environment(env, obs_action_pairs, render=False)

