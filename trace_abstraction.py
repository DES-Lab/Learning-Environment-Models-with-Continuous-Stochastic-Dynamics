import numpy as np
from tqdm import tqdm

from utils import CARTPOLE_CUTOFF, ACROBOT_GOAL


def change_features_clustering(x):
    transformed = np.zeros((x.shape[0],4))
    # transformed[:x.shape[0], :x.shape[1]] = x
    transformed[:, 0] = x[:, 0] + x[:,2]
    transformed[:, 1] = x[:, 1] + x[:,3]
    transformed[:, 2] = x[:, 4] + x[:,5]
    transformed[:, 3] = x[:, 6] + x[:,7]
    return transformed

def create_abstract_traces(env_name, traces, cluster_labels, count_same_cluster=False):
    abstract_traces = []

    i = 0
    print('Creating Alergia Traces')
    for trace in tqdm(traces):
        at = ['Init']  # initial
        step = 0
        for _, action, rew, done in trace:
            abstract_obs = f'c{cluster_labels[i].item(0)}'
            if env_name == 'LunarLander-v2':
                if rew == 100:
                    abstract_obs += '__succ'
                if rew == -100:
                    abstract_obs = 'crash'
            if env_name == 'MountainCar-v0':
                if done and len(trace) < 200:
                    abstract_obs += '__succ'
            if env_name == 'Acrobot-v1':
                if done and len(trace) < ACROBOT_GOAL:
                    abstract_obs += '__succ'
            if env_name == 'CartPole-v1':
                if done and step >= CARTPOLE_CUTOFF:
                    abstract_obs += '__succ'
            step += 1
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
                new_trace.extend((i, f'{o[0]}_{o[1]}' if 'succ' not in o[0] else o[0]))
            counting_abstract_traces.append(new_trace)

        return counting_abstract_traces

    return abstract_traces
