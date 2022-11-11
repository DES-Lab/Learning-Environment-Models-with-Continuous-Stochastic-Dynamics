import pickle

import gym
import numpy as np
from aalpy.learning_algs import run_Alergia, run_JAlergia
from aalpy.utils import statistical_model_checking
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random

from agents import get_lunar_lander_agents
from utils import load, save, delete_file, save_samples_to_file

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}


def get_traces_from_policy(agent, env, num_episodes, randomness_probs=[0]):
    traces = []
    rand_i = 0
    for _ in range(num_episodes):
        curr_randomness = randomness_probs[rand_i]
        rand_i = (rand_i + 1) % len(randomness_probs)
        observation = env.reset()
        episode_trace = []
        while True:
            if random.random() < curr_randomness:
                action = random.randint(0,len(action_map)-1)
            else:
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


def compute_clustering_function_and_map_to_traces(all_policy_traces, n_clusters=16,
                                                  scale=False,
                                                  reduce_dimensions=False,
                                                  include_reward_in_output=False):
    observation_space = []
    for sampled_data in all_policy_traces:
        observation_space.extend([x[0] for trace in sampled_data for x in trace])

    observation_space = np.array(observation_space)
    observation_space = np.squeeze(observation_space)
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    scaler = StandardScaler()
    scaler.fit(observation_space)
    with open(f'standard_scaler_{num_traces}.pickle', 'wb') as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pca = None
    if reduce_dimensions:
        pca = PCA(n_components=4)
        observation_space = pca.fit_transform(observation_space)

        with open(f'pca_4.pickle', 'wb') as handle:
            pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Dimensions reduced with PCA')

    if scale:
        observation_space = scaler.transform(observation_space)
    clustering_function = KMeans(n_clusters=n_clusters)
    clustering_function.fit(observation_space)
    save(clustering_function, f'k_means_scale_{scale}_{num_clusters}_{num_traces}')
    cluster_labels = list(clustering_function.labels_)
    print('Cluster labels computed')

    alergia_datasets = []

    label_i = 0
    for policy_samples in all_policy_traces:
        dataset = []
        for sample in policy_samples:
            alergia_sample = ['INIT']
            for _, action, reward, done in sample:
                # cluster_label = f'c{cluster_labels.pop(0)}'
                cluster_label = f'c{cluster_labels[label_i]}'
                label_i+=1
                #if include_reward_in_output:
               # cluster_label += f'_{round(reward, 2)}'
                if reward == 100 and done:
                    alergia_sample.append(
                        (action_map[int(action)],f"succ__{cluster_label}"))
                if reward >= 1 and done:
                    alergia_sample.append(
                        (action_map[int(action)],f"pos__{cluster_label}"))
                else:
                    alergia_sample.append(
                        (action_map[int(action)], cluster_label if not done else 'DONE'))  # action_map[int(action)]

            dataset.append(alergia_sample)
        alergia_datasets.append(dataset)

    #assert len(cluster_labels) == 0
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

    # jalergia_samples = 'alergiaSamples.txt'
    # save_samples_to_file(alergia_traces, jalergia_samples)
    #
    # model = run_JAlergia(jalergia_samples, 'mdp', 'alergia.jar', heap_memory='-Xmx4G', optimize_for='memory')
    # model.save('reward_automaton')
    # delete_file(jalergia_samples)

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
        all_data.append(get_traces_from_policy(agent, env, num_traces,randomness_probs=[0,0.01,0.025,0.05]))
    save(all_data, traces_file_name)

# clustering_function, pca = compute_clustering_function(all_data, n_clusters=num_clusters, reduce_dimensions=False)
# compute_clustering_function_and_map_to_traces(all_data, num_clusters, reduce_dimensions=False)
# exit()
# print('Clustering function computed')
scale = True
alergia_traces = compute_clustering_function_and_map_to_traces(all_data,
                                                               num_clusters,
                                                               scale = scale,
                                                               reduce_dimensions=False)


# mdp_dqn = run_Alergia(alergia_traces[0],eps=0.05, automaton_type='mdp', print_info=True)
# mdp_a2c = run_Alergia(alergia_traces[1],eps=0.05, automaton_type='mdp', print_info=True)
all_traces = alergia_traces[0]
for i in range(1,len(alergia_traces)):
# all_traces = alergia_traces[0]
    all_traces.extend(alergia_traces[i])
# mdp = run_Alergia(all_traces,eps=0.0005, automaton_type='mdp', print_info=True)
jalergia_samples = 'alergiaSamples.txt'
save_samples_to_file(all_traces, jalergia_samples)

mdp = run_JAlergia(jalergia_samples, 'mdp', 'alergia.jar', heap_memory='-Xmx4G', optimize_for='memory', eps=0.00005)
delete_file(jalergia_samples)

mdp.save(f'mdp_combined_scale_{scale}_{num_clusters}_{num_traces}')
