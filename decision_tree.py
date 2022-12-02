import os
from statistics import mean

import gym
import numpy as np
from stable_baselines3 import DQN

from agents import load_agent
from utils import load, save
import graphviz
from sklearn import tree


def evaluate_on_environment(env, dt, num_episodes=100, render=False):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        ep_rew = 0
        while True:
            a = dt.predict(obs.reshape(1, -1))[0]
            obs, rew, done, info = env.step(a)
            if render:
                env.render()
            ep_rew += rew
            if done:
                all_rewards.append(ep_rew)
                break
    print('Mean reward:', mean(all_rewards))


def get_observation_action_pairs(env, num_ep=1000):
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


env_name = "LunarLander-v2"
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}

num_traces = 500
load_observations = True

env = gym.make(env_name)

if load_observations and os.path.exists(f'pickle_files/obs_actions_pairs_{num_traces}.pickle'):
    obs_action_pairs = load(f'obs_actions_pairs_{num_traces}')
    if obs_action_pairs:
        print('Observation actions pairs loaded')
else:
    print('Computing observation action pairs')
    obs_action_pairs = get_observation_action_pairs(env, num_traces)
    save(obs_action_pairs, path=f'obs_actions_pairs_{num_traces}')

x,y = [i[0] for i in obs_action_pairs], [i[1] for i in obs_action_pairs]
x = np.array(x)
y = np.array(y)

dt = tree.DecisionTreeClassifier(max_leaf_nodes=64)
dt.fit(x, y)

print(dt.get_n_leaves())
# to get a leaf if of leaf that meade the decision
# df.apply(obs)

evaluate_on_environment(env, dt, render=False)
# exit()
dot_data = tree.export_graphviz(dt)
graph = graphviz.Source(dot_data)
graph.render()
