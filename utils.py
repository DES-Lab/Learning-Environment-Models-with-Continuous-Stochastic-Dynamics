import math
import os
import pickle
import random

from sklearn.cluster import KMeans
from tqdm import tqdm

CARTPOLE_CUTOFF = 100
ACROBOT_GOAL = 100
def compute_clusters(data, n_clusters):
    clustering_function = KMeans(n_clusters=n_clusters)
    clustering_function.fit(data)
    return clustering_function


def save(x, path):
    with open(path, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as handle:
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


def get_traces_from_policy(agent, env, num_episodes, agent_name,
                           randomness_probabilities=(0,), load_traces=True):
    traces_name = f'pickles/traces/{env.unwrapped.spec.id}_{agent_name}' \
                  f'_num_ep_{num_episodes}_{str(randomness_probabilities)}.pk'

    if load_traces and os.path.exists(traces_name):
        traces = load(traces_name)
        print('Traces loaded.')
        assert traces
        return traces

    traces = []
    rand_i = 0
    print(f'Getting demonstrations from an pretrained agent. Included randomness: {randomness_probabilities}')

    for _ in tqdm(range(num_episodes)):
        curr_randomness = randomness_probabilities[rand_i]

        observation = env.reset()
        episode_trace = []
        step = 0
        while True:
            if random.random() < curr_randomness:
                action = random.randint(0, env.action_space.n - 1)
            else:
                action, _ = agent.predict(observation)
            observation, reward, done, info = env.step(action)
            if "CartPole" in env.unwrapped.spec.id and step >= CARTPOLE_CUTOFF:
                done = True
            step += 1
            episode_trace.append((observation.reshape(1, -1), action, reward, done))

            if done:
                break

        traces.append(episode_trace)

    print(f'Traces computed. Saving to {traces_name}')
    save(traces, traces_name)

    return traces




def remove_nan(mdp):
    changed = False
    for s in mdp.states:
        to_remove = []
        for input in s.transitions.keys():
            is_nan = False
            for t in s.transitions[input]:
                if math.isnan(t[1]):
                    is_nan = True
                    to_remove.append(input)
                    break
            if not is_nan:
                if abs(sum(map(lambda t: t[1], s.transitions[input])) - 1) > 1e-6:
                    to_remove.append(input)
        for input in to_remove:
            changed = True
            s.transitions.pop(input)
    return changed
