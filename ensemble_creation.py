import os
from collections import defaultdict

import gym
from aalpy.learning_algs import run_JAlergia
from aalpy.utils import load_automaton_from_file
from stable_baselines3 import DQN

from abstraction import compute_clustering_function_and_map_to_traces
from agents import load_agent
from utils import get_traces_from_policy, save_samples_to_file, delete_file, compress_trace

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}


# gets suffixes of a trace, removes INIT placeholder
def get_trace_suffixes(trace):
    return [trace[len(trace) - i - 1:] for i in range(len(trace))]


# returns a dictionary where key is cluster label and value is corresponding mdp
def compute_ensemble_mdp(alergia_traces,
                         optimize_for='accuracy', alergia_eps=0.005,
                         input_completeness='sink_state', skip_sequential_outputs=False,
                         save_path_prefix='ensemble'):
    cluster_traces = defaultdict(list)
    ensemble_mdps = dict()

    for trace in alergia_traces:
        if skip_sequential_outputs:
            trace = compress_trace(trace)
        trace_suffixes = get_trace_suffixes(trace)
        for suffix in trace_suffixes[:-1]: # IGNORE INIT (Placeholder)
            cluster_label = suffix[0][1]
            if len(suffix) > 1:
                cluster_traces[cluster_label].append([cluster_label, ] + suffix[1:])

    for cluster_label, cluster_traces in cluster_traces.items():
        cluster_samples = 'cluster_samples.txt'
        save_samples_to_file(cluster_traces, cluster_samples)
        mdp = run_JAlergia(cluster_samples, 'mdp', 'alergia.jar', heap_memory='-Xmx4G',
                           optimize_for=optimize_for, eps=alergia_eps)
        delete_file(cluster_samples)
        if input_completeness:
            mdp.make_input_complete(input_completeness)
        ensemble_mdps[cluster_label] = mdp

        # save cluster mdp to file
        if save_path_prefix:
            mdp.save(f'learned_models/{save_path_prefix}_{cluster_label}')

    return ensemble_mdps


def load_ensemble(saved_path_prefix='ensemble', input_completeness='sink_state'):
    ensemble_files = []
    ensemble_maps = dict()
    for file in os.listdir('learned_models'):
        if file.startswith(saved_path_prefix):
            ensemble_files.append(file)
    for f in ensemble_files:
        mdp = load_automaton_from_file(f'learned_models/{f}', 'mdp')
        if not mdp.is_input_complete() and input_completeness is not None:
            mdp.make_input_complete(input_completeness)
        cluster_label = f[len(saved_path_prefix) + 1:][:-4]
        print(cluster_label)
        ensemble_maps[cluster_label] = mdp

    print('Ensemble MDPs loaded.')
    return ensemble_maps


if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

    traces = [get_traces_from_policy(dqn_agent, env, 10000, action_map)]
    alergia_traces = compute_clustering_function_and_map_to_traces(traces, action_map, n_clusters=32, scale=True, )[0]

    compute_ensemble_mdp(alergia_traces, )

    # ensemble_mdp = load_ensemble('ensemble')
