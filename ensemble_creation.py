import os
import random
from collections import defaultdict

import gym
from aalpy.learning_algs import run_JAlergia
from aalpy.utils import load_automaton_from_file
from stable_baselines3 import DQN

from abstraction import compute_clustering_function_and_map_to_traces
from agents import load_agent
from utils import get_traces_from_policy, save_samples_to_file, delete_file, compress_trace, save, load

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}


# gets suffixes of a trace, removes INIT placeholder
def get_trace_suffixes(trace):
    return [trace[len(trace) - i - 1:] for i in range(len(trace))]


# returns a dictionary where key is cluster label and value is corresponding mdp
def compute_ensemble_mdp(alergia_traces,
                         optimize_for='accuracy', alergia_eps=0.005,
                         input_completeness='self_loop', skip_sequential_outputs=False,
                         save_path_prefix='ensemble', suffix_strategy='longest', depth=1, nr_traces_limit = -1):
    cluster_traces = defaultdict(list)
    ensemble_mdps = dict()

    for trace in alergia_traces:
        if skip_sequential_outputs:
            trace = compress_trace(trace)
        if suffix_strategy == 'all':
            trace_suffixes = get_trace_suffixes(trace)
            for suffix in trace_suffixes[:-1]: # IGNORE INIT (Placeholder)
                cluster_label = suffix[0][1]
                if len(suffix) > 1:
                    cluster_traces[cluster_label].append([cluster_label, ] + suffix[1:])
        elif suffix_strategy == 'longest':
            clusters_in_trace = set(filter(lambda label: "c" in label and label.index("c") == 0,
                                           map(lambda inp_out: inp_out[1],trace[1:])))
            for c in clusters_in_trace:
                indexes = [i for i, inp_out in list(enumerate(trace)) if inp_out[1] == c]
                for i in range(min(depth, len(indexes))):
                    ith_first_index_of_c = indexes[i]
                    longest_suffix = trace[ith_first_index_of_c + 1:]
                    cluster_traces[c].append([c, ] + longest_suffix)
        elif suffix_strategy == 'shortest':
            clusters_in_trace = set(filter(lambda label: "c" in label and label.index("c") == 0,
                                           map(lambda inp_out: inp_out[1],trace[1:])))
            for c in clusters_in_trace:
                indexes = [i for i,inp_out in list(enumerate(trace))[::-1] if inp_out[1] == c]
                for i in range(min(depth,len(indexes))):
                    ith_last_index_of_c = indexes[i]
                    shortest_suffix = trace[ith_last_index_of_c+1:]
                    cluster_traces[c].append([c,] + shortest_suffix)
        elif suffix_strategy == 'all_nonconsec':
            clusters_in_trace = set(filter(lambda label: "c" in label and label.index("c") == 0,
                                           map(lambda inp_out: inp_out[1], trace[1:])))
            for c in clusters_in_trace:
                indexes = [i for i, inp_out in list(enumerate(trace)) if inp_out[1] == c]
                non_consective_indexes = []
                last_i = -10
                for i in indexes:
                    if last_i < i - 1:
                        non_consective_indexes.append(i)
                    last_i = i
                indexes = non_consective_indexes
                for i in range(len(indexes)):
                    ith_index_of_c = indexes[i]
                    suffix = trace[ith_index_of_c + 1:]
                    cluster_traces[c].append([c, ] + suffix)
                
    for cluster_label, one_cluster_traces in cluster_traces.items():
        cluster_samples = 'cluster_samples.txt'
        if nr_traces_limit > 0 and len(one_cluster_traces) > nr_traces_limit:
            one_cluster_traces = random.choices(one_cluster_traces,k=nr_traces_limit)
        save_samples_to_file(one_cluster_traces, cluster_samples)
        mdp = run_JAlergia(cluster_samples, 'mdp', 'alergia.jar', heap_memory='-Xmx6G',
                           optimize_for=optimize_for, eps=alergia_eps)
        #delete_file(cluster_samples)
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
        cluster_label = f[len(saved_path_prefix) + 1:][:-4]
        print(cluster_label)
        mdp = load_automaton_from_file(f'learned_models/{f}', 'mdp')
        if not mdp.is_input_complete() and input_completeness is not None:
            mdp.make_input_complete(input_completeness)
        ensemble_maps[cluster_label] = mdp

    print('Ensemble MDPs loaded.')
    return ensemble_maps


if __name__ == '__main__':
    env_name = "LunarLander-v2"
    num_traces = 45000
    env = gym.make(env_name)
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

    trace_file = f"{env_name}_{num_traces}_traces"
    traces = load(trace_file)
    if traces is None:
        # traces = [get_traces_from_policy(dqn_agent, env, num_traces, action_map, randomness_probs=[0, 0.05, 0.1, 0.15])]
        traces = [get_traces_from_policy(dqn_agent, env, num_traces, action_map, randomness_probs=[0, 0.025, 0.05, 0.1, 0.15,0.2])]
    save(traces,trace_file)
    alergia_traces = compute_clustering_function_and_map_to_traces(traces, action_map, n_clusters=128,clustering_type="k_means",
                                                                   scale=False,)[0]
    compute_ensemble_mdp(alergia_traces,suffix_strategy="longest", save_path_prefix="ensemble_45k_k_means",
                         depth=1, nr_traces_limit = 25000)
    # compute_ensemble_mdp(alergia_traces,suffix_strategy="longest", save_path_prefix="ensemble_14k",depth = 5, nr_traces_limit = 25000)

    # ensemble_mdp = load_ensemble('ensemble')
