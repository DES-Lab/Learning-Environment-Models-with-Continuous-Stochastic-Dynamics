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

    print(f'Traces computed. Saving to {traces_name}')
    save(traces, traces_name)

    return traces


def create_abstract_traces(traces, cluster_labels, count_same_cluster=False):
    abstract_traces = []

    i = 0
    print('Creating Alergia Traces')
    for trace in tqdm(traces):
        at = ['Init']  # initial
        for _, action, rew, done in trace:
            abstract_obs = f'c{cluster_labels[i].item(0)}'
            if rew == 100:
                abstract_obs += '__succ'
            at.extend((f'i{action.item(0)}', abstract_obs))
            i += 1
        abstract_traces.append(at)

    if count_same_cluster:
        counting_abstract_traces = []

        for trace_id, trace in enumerate(abstract_traces):
            actions = trace[1::2]
            clusters = trace[2::2]  # get all clusters
            counted_clusters = []
            cc = 1
            for i in range(len(clusters)):
                if i == 0:
                    counted_clusters.append((clusters[i], cc))
                else:
                    counted_clusters.append((clusters[i], 1 if clusters[i] != counted_clusters[i - 1][0] else
                    counted_clusters[i - 1][1] + 1))

            new_trace = ['Init']
            for i, o in zip(actions, counted_clusters):
                new_trace.extend((i, f'{o[0]}_{o[1]}'))
            counting_abstract_traces.append(new_trace)

        return counting_abstract_traces

    return abstract_traces
