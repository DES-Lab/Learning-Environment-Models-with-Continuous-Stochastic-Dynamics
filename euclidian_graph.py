from collections import defaultdict, Counter

import gym
from stable_baselines3 import DQN

from HistoryCountingMdpCreation import counter_to_mdp
from agents import load_agent
from utils import load, get_traces_from_policy, save

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}

env_name = "LunarLander-v2"

num_traces = 2000

env = gym.make(env_name)

trace_file = f"{env_name}_{num_traces}_traces"
traces = load(trace_file)
if traces is None:
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
    traces = [get_traces_from_policy(dqn_agent, env, num_traces, action_map, randomness_probs=[0, 0.01, 0.025, 0.05])]
    save(traces, trace_file)
print('Traces obtained')

def traces_to_graph(traces):
    counting_mdps_increasing_history = dict()
    # (cluster_label, action) : cluster_label : frequency
    observation_counter = defaultdict(Counter)

    for trace in traces:
        trace_splits = [(trace[j:j + i], trace[j + i:j + 2 * i]) for j in range(1, len(trace))]
        for x, y in trace_splits:
            origin_label = '_'.join([l[1] for l in x])
            destination_label = '_'.join([l[1] for l in y])

            if 'succ' not in destination_label and len(x) != len(y):
                continue

            if 'succ' in destination_label:
                destination_label = 'succ'

            action = x[-1][0]
            observation_counter[(origin_label, action)][destination_label] += 1

    mdp = counter_to_mdp(observation_counter, initial_state_config='initial')
    mdp.make_input_complete('sink_state')

    return mdp
