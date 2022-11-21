from collections import defaultdict, Counter
from random import random

import gym
from stable_baselines3 import DQN
from aalpy.automata import Mdp, MdpState

from abstraction import compute_clustering_function_and_map_to_traces
from agents import load_agent
from utils import load, get_traces_from_policy, save

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}


def counter_to_mdp(counter, initial_state_config='first'):
    label_to_mdp_state_map = dict()
    state_counter = 0
    for label, _ in counter.keys():
        state_counter += 1
        label_to_mdp_state_map[label] = MdpState(f's{state_counter}', output=label)

    for (origin, action), destination_counter in counter.items():
        total_obs = sum(destination_counter.values())
        # Node(output), probability
        for destination_label, freq in destination_counter.items():
            # for succ states
            if destination_label not in label_to_mdp_state_map.keys():
                state_counter += 1
                label_to_mdp_state_map[destination_label] = MdpState(f's{state_counter}', output=destination_label)

            origin_state = label_to_mdp_state_map[origin]
            destination_state = label_to_mdp_state_map[destination_label]
            origin_state.transitions[action].append((destination_state, freq / total_obs))

    # does it even matter?
    if initial_state_config == 'first':
        initial_state = list(label_to_mdp_state_map.values())[0]
    else:
        initial_state = random.choice(list(label_to_mdp_state_map.values()))

    return Mdp(initial_state, list(label_to_mdp_state_map.values()))


def counting_mdp_from_cluster_labels(alergia_traces, max_history_len=5, initial_state_config='first'):
    counting_mdps_increasing_history = dict()
    for i in range(1, max_history_len + 1):
        print(f'Creating MDP with history size of {i}')
        # (cluster_label, action) : cluster_label : frequency
        observation_counter = defaultdict(Counter)

        for trace in alergia_traces:
            trace_splits = [(trace[j:j + i], trace[j + i:j + 2 * i]) for j in range(1, len(trace))]
            for x, y in trace_splits:
                if len(x) != len(y):
                    continue

                origin_label = '_'.join([l[1] for l in x])
                destination_label = '_'.join([l[1] for l in y])
                action = x[-1][0]
                observation_counter[(origin_label, action)][destination_label] += 1

        mdp = counter_to_mdp(observation_counter, initial_state_config)
        counting_mdps_increasing_history[i] = mdp

    return counting_mdps_increasing_history


env_name = "LunarLander-v2"

num_traces = 1000
scale = True
n_clusters = 16
history_size = 10

env = gym.make(env_name)

trace_file = f"{env_name}_{num_traces}_traces"
traces = load(trace_file)
if traces is None:
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
    traces = [get_traces_from_policy(dqn_agent, env, num_traces, action_map, randomness_probs=[0, 0.01, 0.025, 0.05])]
    save(traces, trace_file)
print('Traces obtained')

traces = [traces[0][:10]]
alergia_traces = \
    compute_clustering_function_and_map_to_traces(traces, action_map, n_clusters=n_clusters, scale=scale, )[0]

histroy_mdps = counting_mdp_from_cluster_labels(alergia_traces)
