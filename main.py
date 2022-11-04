import gym
import numpy as np
from aalpy.learning_algs import run_Alergia
from aalpy.utils import statistical_model_checking
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from agents import get_lunar_lander_agents
from utils import load, save

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}


def get_traces_from_policy(agent, env, num_episodes):
    traces = []
    for _ in range(num_episodes):
        observation = env.reset()

        episode_trace = []
        while True:
            action, _ = agent.predict(observation)
            observation, reward, done, info = env.step(action)

            episode_trace.append((observation.reshape(1, -1), action, reward, done))
            if done:
                break

        traces.append(episode_trace)

    return traces


def compute_clustering_function(all_policy_traces, n_clusters=16, reduce_dimensions=False):
    observation_space = []
    for sampled_data in all_policy_traces:
        observation_space.extend([x[0] for trace in sampled_data for x in trace])

    observation_space = np.array(observation_space)
    observation_space = np.squeeze(observation_space)

    pca = None
    if reduce_dimensions:
        pca = PCA(n_components=16)
        pca.fit_transform(observation_space)
        print('PCA trained')

    clustering_function = KMeans(n_clusters=n_clusters)
    clustering_function.fit(observation_space)
    return clustering_function, pca


def compute_clustering_function_and_map_to_traces(all_policy_traces, n_clusters=16, reduce_dimensions=False,
                                                  include_reward_in_output=False):
    observation_space = []
    for sampled_data in all_policy_traces:
        observation_space.extend([x[0] for trace in sampled_data for x in trace])

    observation_space = np.array(observation_space)
    observation_space = np.squeeze(observation_space)

    pca = None
    if reduce_dimensions:
        pca = PCA(n_components=16)
        observation_space = pca.fit_transform(observation_space)
        print('Dimensions reduced with PCA')

    clustering_function = KMeans(n_clusters=n_clusters)
    clustering_function.fit(observation_space)
    cluster_labels = list(clustering_function.labels_)
    print('Cluster labels computed')

    alergia_datasets = []

    for policy_samples in all_policy_traces:
        dataset = []
        for sample in policy_samples:
            alergia_sample = ['INIT']
            for _, action, reward, done in sample:
                cluster_label = f'c{cluster_labels.pop(0)}'
                if include_reward_in_output:
                    cluster_label += f'_{round(reward, 2)}'
                alergia_sample.append(
                    (action_map[int(action)], cluster_label if not done else 'DONE'))  # action_map[int(action)]

            dataset.append(alergia_sample)
        alergia_datasets.append(dataset)

    assert len(cluster_labels) == 0
    print('Cluster labels replaced')
    return alergia_datasets


def compute_stochastic_model(traces, clustering_fun, include_reward_in_output=False, pca=None):
    alergia_traces = []
    print('Preparing traces for ALERGIA')
    for trace in traces:
        episode_trace = ['INIT']
        for obs, action, reward, done in trace:
            if pca:
                obs = pca.transform(obs.reshape(1, -1))
            output = f'C{clustering_fun.predict(obs.reshape(1, -1))[0]}'
            if include_reward_in_output:
                output += f'_{round(reward, 2)}'
            episode_trace.append((action_map[int(action)], output if not done else 'DONE'))  # action_map[int(action)]

        alergia_traces.append(episode_trace)

    model = run_Alergia(alergia_traces, automaton_type='mdp', print_info=True)
    model.make_input_complete()
    return model


environment = "LunarLander-v2"

agents = None
agent_names = None

if environment == 'LunarLander-v2':
    agents_and_names = get_lunar_lander_agents(evaluate=False)
    agents = [a[1] for a in agents_and_names]
    agent_names = '_'.join(a[0] for a in agents_and_names)

assert agents
print('Agents loaded')

num_traces = 2000
num_clusters = 16

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
        all_data.append(get_traces_from_policy(agent, env, num_traces))
    save(all_data, traces_file_name)

# clustering_function, pca = compute_clustering_function(all_data, n_clusters=num_clusters, reduce_dimensions=False)
# compute_clustering_function_and_map_to_traces(all_data, num_clusters, reduce_dimensions=False)
# exit()
# print('Clustering function computed')

alergia_traces = compute_clustering_function_and_map_to_traces(all_data, num_clusters, reduce_dimensions=False)

mdp_dqn = run_Alergia(alergia_traces[0], automaton_type='mdp', print_info=True)
mdp_a2c = run_Alergia(alergia_traces[1], automaton_type='mdp', print_info=True)

mdp_dqn.save('mdp_dqn')
mdp_a2c.save('mdp_a2c')
