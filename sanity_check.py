import random
from collections import Counter, OrderedDict

import gym
from aalpy.learning_algs import run_JAlergia
from stable_baselines3 import DQN
from tqdm import tqdm

from abstraction import compute_clustering_function_and_map_to_traces
from agents import get_lunar_lander_agents, load_agent
from utils import load, save, delete_file, save_samples_to_file, get_traces_from_policy

environment = "LunarLander-v2"
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}


agents = None
agent_names = None

if environment == 'LunarLander-v2':
    agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

# assert agents
# print('Agents loaded')

num_traces = 2500
num_clusters = 256
scale = True

env = gym.make(environment, )
traces_file_name = f'{environment}_{num_traces}_traces'

loaded_traces = load(f'{traces_file_name}')
if loaded_traces:
    print('Traces loaded')
    all_data = loaded_traces
else:
    print(f'Obtaining {num_traces} per agent')
    all_data = [
        get_traces_from_policy(agent, env, num_traces, action_map, stop_prob=0.0,
                               # randomness_probs=[0, 0.025, 0.05, 0.1, 0.15,0.2])),
                               randomness_probabilities=[0, 0.025, 0.05, 0.1])]
    # randomness_probs=[0, 0.025, 0.05, 0.1, 0.15,0.2,0.25])]
    save(all_data, traces_file_name)

clustering_type = 'k_means'
# alergia_traces = compute_clustering_function_and_map_to_traces(all_data,
#                                                                action_map,
#                                                                environment,
#                                                                num_clusters,
#                                                                clustering_type=clustering_type,
#                                                                scale=scale,
#                                                                reduce_dimensions=False)
# all_traces = alergia_traces[0]
#
# jalergia_samples = 'alergiaSamples.txt'
# save_samples_to_file(all_traces, jalergia_samples)
# mdp = run_JAlergia(jalergia_samples, 'mdp', 'alergia.jar', heap_memory='-Xmx4G', optimize_for='accuracy', eps=0.005)
# # delete_file(jalergia_samples)

scaler = load(f'power_scaler_{environment}_{num_traces}')
clustering_function = load(f'{environment}_{clustering_type}_scale_{scale}_{num_clusters}_{num_traces}')


def check_random_reachability_of_clusters(num_clusters, cf, scaler, num_episodes=500):
    cluster_reachability_counter = Counter()
    actions = list(action_map.keys())
    action_weight_map = [0.4, 0.2, 0.2, 0.2]

    print('Testing reachability of clusters')
    for _ in tqdm(range(num_episodes)):
        obs = env.reset()
        seen_cluster = set()
        while True:
            action = random.choices(actions, weights=action_weight_map, k=1)[0]

            obs, rew, done, info = env.step(action)

            if scaler:
                obs = scaler.transform(obs.reshape(1, -1))
            cluster = f'c{cf.predict(obs.reshape(1, -1))[0]}'
            seen_cluster.add(cluster)

            if done:
                for c in seen_cluster:
                    cluster_reachability_counter[c] += 1
                break

    sorted_cluster = OrderedDict(cluster_reachability_counter.most_common())

    for cluster, freq in sorted_cluster.items():
        print(f'{cluster}: {round(float(freq) / float(num_episodes) * 100, 2)} %')

    num_reached_clusters = len(list(sorted_cluster.keys()))
    print(f'Number of clusters not reached: {num_clusters - num_reached_clusters} of {num_reached_clusters}')


check_random_reachability_of_clusters(num_clusters, clustering_function, scaler)
