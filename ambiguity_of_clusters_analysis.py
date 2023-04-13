import gym
from stable_baselines3 import DQN

from abstraction import compute_clusterin_and_map_basic
from agents import load_agent
from utils import load, get_traces_from_policy, save, map_clusters_to_actions, compute_ambiguity

environment = "LunarLander-v2"
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}

agent = None
if environment == 'LunarLander-v2':
    agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

assert agent
print('Agents loaded')

num_traces = 5000
num_clusters = 128
scale = True
include_reward_in_obs = False

env = gym.make(environment, )
# Todo add name
traces_file_name = f'rl_based_testing_{environment}_{num_traces}_traces'

loaded_traces = load(f'{traces_file_name}')
if loaded_traces:
    print('Traces loaded')
    all_data = loaded_traces
else:
    print(f'Obtaining {num_traces} per agent')
    all_data = [get_traces_from_policy(agent, env, num_traces, action_map, stop_prob=0.0,
                                       randomness_probabilities=[0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25])]
    save(all_data, traces_file_name)

alergia_traces, clustering_fun = compute_clusterin_and_map_basic(all_data,
                                                                 action_map,
                                                                 environment,
                                                                 num_clusters,
                                                                 return_clustering_fun=True,
                                                                 include_reward=include_reward_in_obs,
                                                                 scale=scale,
                                                                 reduce_dimensions=False)

agent_trace = alergia_traces[0]
cluster_action_map = map_clusters_to_actions(agent_trace)
avg_ambiguity, wamb, min_ambiguity, cluster_ambiguities = compute_ambiguity(cluster_action_map,
                                                                            weighted=True,
                                                                            return_amb_per_cluster=True)

print(wamb)
print(avg_ambiguity, wamb, min_ambiguity)
