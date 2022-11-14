import gym
from aalpy.learning_algs import run_JAlergia

from abstraction import compute_clustering_function_and_map_to_traces
from agents import get_lunar_lander_agents
from utils import load, save, delete_file, save_samples_to_file, get_traces_from_policy

environment = "LunarLander-v2"
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}

agents = None
agent_names = None

if environment == 'LunarLander-v2':
    agents_and_names = get_lunar_lander_agents(evaluate=False)
    agents = [a[1] for a in agents_and_names]
    agent_names = '_'.join(a[0] for a in agents_and_names)

assert agents
print('Agents loaded')

num_traces = 8000
num_clusters = 64

env = gym.make(environment, )
traces_file_name = f'{environment}_{agent_names}_{num_traces}_traces'

loaded_traces = load(f'{traces_file_name}.pickle')
if loaded_traces:
    print('Traces loaded')
    all_data = loaded_traces
else:
    print(f'Obtaining {num_traces} per agent')
    all_data = []
    for agent in agents:
        all_data.append(
            get_traces_from_policy(agent, env, num_traces, action_map, randomness_probs=[0, 0.01, 0.025, 0.05]))
    save(all_data, traces_file_name)

# clustering_function, pca = compute_clustering_function(all_data, n_clusters=num_clusters, reduce_dimensions=False)
# compute_clustering_function_and_map_to_traces(all_data, num_clusters, reduce_dimensions=False)
# exit()
# print('Clustering function computed')

scale = True
alergia_traces = compute_clustering_function_and_map_to_traces(all_data,
                                                               action_map,
                                                               num_clusters,
                                                               scale=scale,
                                                               reduce_dimensions=False)

all_traces = alergia_traces[0]
for i in range(1, len(alergia_traces)):
    all_traces.extend(alergia_traces[i])

jalergia_samples = 'alergiaSamples.txt'
save_samples_to_file(all_traces, jalergia_samples)
mdp = run_JAlergia(jalergia_samples, 'mdp', 'alergia.jar', heap_memory='-Xmx4G', optimize_for='memory', eps=0.00005)
delete_file(jalergia_samples)

mdp.save(f'mdp_combined_scale_{scale}_{num_clusters}_{num_traces}')
