import os
import pickle
import random

from sklearn.cluster import KMeans
from tqdm import tqdm


def compute_clusters(data, n_clusters):
    clustering_function = KMeans(n_clusters=n_clusters)
    clustering_function.fit(data)
    return clustering_function


def save(x, path):
    with open(f'pickle_files/{path}.pickle', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_name):
    full_file_name = f'pickle_files/{file_name}.pickle'
    if os.path.exists(full_file_name):
        with open(full_file_name, 'rb') as handle:
            return pickle.load(handle)
    else:
        return None


def append_samples_to_file(samples, filename='jAlergiaData.txt'):
    with open(filename, 'a') as f:
        for sample in samples:
            s = f'{str(sample[0])},'
            for i, o in sample[1:]:
                s += f'{str(i)},{str(o)},'
            f.write(s[:-1] + '\n')


def save_samples_to_file(samples, filename='jAlergiaData.txt'):
    with open(filename, 'w') as f:
        for sample in samples:
            s = f'{str(sample[0])},'
            for i, o in sample[1:]:
                s += f'{str(i)},{str(o)},'
            f.write(s[:-1] + '\n')


def delete_file(filename):
    import os
    if os.path.exists(filename):
        os.remove(filename)


def compress_trace(x):
    from itertools import groupby
    return [key for key, _group in groupby(x)]


def get_traces_from_policy(agent, env, num_episodes, randomness_probabilities=(0,)):
    traces = []
    rand_i = 0
    print(f'Getting demonstrations from an pretrained agent. Included randomness: {randomness_probabilities}')

    for _ in tqdm(range(num_episodes)):
        curr_randomness = randomness_probabilities[rand_i]

        observation = env.reset()
        episode_trace = []
        while True:
            if random.random() < curr_randomness:
                action = random.randint(0, env.action_space.n - 1)
            else:
                action, _ = agent.predict(observation)
            observation, reward, done, info = env.step(action)

            episode_trace.append((observation.reshape(1, -1), action, reward, done))

            if done:
                break

        traces.append(episode_trace)

    return traces


def create_abstract_traces(traces, cluster_labels):
    abstract_traces = []

    i = 0
    print('Creating Alergia Traces')
    for trace in tqdm(traces):
        at = ['Init']  # initial
        for _, action, rew, done in trace:
            if rew == 100:
                abstract_obs = 'GOAL'
            else:
                abstract_obs = f'c{cluster_labels[i].item(0)}'
            at.append((action.item(0), abstract_obs))
            i += 1
        abstract_traces.append(at)

    return abstract_traces
