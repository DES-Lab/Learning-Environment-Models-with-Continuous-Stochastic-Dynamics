import sys

import gym
from aalpy.learning_algs import run_RPNI
from aalpy.utils import convert_i_o_traces_for_RPNI
from stable_baselines3 import DQN

from abstraction import compute_clustering_function_and_map_to_traces
from agents import load_agent
from utils import load, get_traces_from_policy, save

environment = "LunarLander-v2"
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

# environment = "MountainCar-v0"
# action_map = {0: 'left', 1: 'no_action', 2: 'right'}
# agent = load_agent('DBusAI/DQN-MountainCar-v0-v2', 'DQN-MountainCar-v0.zip', DQN)

env = gym.make(environment)

# agent under test

# parameters for uncertainty calculation
num_traces = 10
num_clusters = 64
scale = False
include_reward_in_obs = False

# Generation or loading of traces
traces_file_name = f'rl_1based_testing_{environment}_{num_traces}_traces'

loaded_traces = load(f'{traces_file_name}')
if loaded_traces:
    print('Traces loaded')
    all_data = loaded_traces
else:
    print(f'Obtaining {num_traces} per agent')
    all_data = [get_traces_from_policy(agent, env, num_traces, action_map, stop_prob=0.0,
                                       randomness_probs=[0.1, 0.2, 0.15])]
    save(all_data, traces_file_name)
traces = all_data

alergia_traces = \
compute_clustering_function_and_map_to_traces(traces, action_map, n_clusters=64, scale=scale, env_name='LunarLander')[0]

rpni_traces = []
for i, trace in enumerate(alergia_traces):
    t = trace[1:]
    t.insert(0, tuple([f't{i}', 'Init O']))
    rpni_traces.extend(convert_i_o_traces_for_RPNI([t]))
sys.setrecursionlimit(15000)
rpni_model = run_RPNI(rpni_traces, 'mealy')
