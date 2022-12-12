import os
import pickle
import random

import numpy as np
from aalpy.base import SUL
from sklearn.cluster import KMeans
from tqdm import tqdm


class GymSUL(SUL):
    def __init__(self, env, clustering_fun):
        super().__init__()
        self.env = env
        self.clustering_fun = clustering_fun
        self.last_obs = None

    def pre(self):
        obs = self.env.reset()
        self.last_obs = obs
        return obs

    def post(self):
        pass

    def step(self, letter):
        if letter is None:
            return self.last_obs
        obs, reward, done, _ = self.env.step(letter)
        cluster = self.clustering_fun.predict(obs.reshape(1, -1))
        return f'c{cluster}' if not done else 'DONE'


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


def get_traces_from_policy(agent, env, num_episodes, action_map,stop_prob = 0, randomness_probs=(0,),
                           duplicate_action=False):
    traces = []
    rand_i = 0
    for _ in tqdm(range(num_episodes)):
        curr_randomness = randomness_probs[rand_i]
        rand_i = (rand_i + 1) % len(randomness_probs)
        observation = env.reset()
        episode_trace = []
        while True:
            if random.random() < curr_randomness:
                action = random.randint(0, len(action_map) - 1)
            else:
                action, _ = agent.predict(observation)
            observation, reward, done, info = env.step(action)
            if duplicate_action:
                observation, reward, done, info = env.step(action)
            episode_trace.append((observation.reshape(1, -1), action, reward, done))
            if stop_prob > 0 and random.random() < stop_prob:
                break
            if done:
                break

        traces.append(episode_trace)

    return traces
