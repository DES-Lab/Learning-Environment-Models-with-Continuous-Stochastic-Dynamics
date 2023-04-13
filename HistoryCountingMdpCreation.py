from collections import defaultdict, Counter
from random import random, choice
from statistics import mean

import gym
import aalpy.paths
from sklearn.metrics import euclidean_distances
from stable_baselines3 import DQN
from aalpy.automata import Mdp, MdpState

from abstraction import compute_clustering_function_and_map_to_traces
from agents import load_agent
from prism_scheduler import PrismInterface
from utils import load, get_traces_from_policy, save

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}

aalpy.paths.path_to_prism = '/home/mtappler/Programs/prism-4.7-linux64/bin/prism'
cluster_center_cache = dict()

def counter_to_mdp(counter, initial_state_config='first', only_last_cluster=False):
    label_to_mdp_state_map = dict()
    state_counter = 0
    for label, _ in counter.keys():
        state_counter += 1
        output = label
        if label != "DONE" and label != "succ" and label != "pos" and \
                only_last_cluster:
            labels = output.split("_")
            non_cluster_labels = list(filter(lambda x: not x.startswith("c") and not len(x) == 0, labels))
            cluster_labels = list(filter(lambda x: x.startswith("c"), labels))
            last_cluster = cluster_labels[-1]
            output = "__".join([last_cluster] + non_cluster_labels)
        label_to_mdp_state_map[label] = MdpState(f's{state_counter}', output=output)

    for (origin, action), destination_counter in counter.items():
        total_obs = sum(destination_counter.values())
        # Node(output), probability
        for destination_label, freq in destination_counter.items():
            # for succ states
            if destination_label not in label_to_mdp_state_map.keys():
                state_counter += 1
                output = destination_label

                if destination_label != "DONE" and destination_label != "succ" and destination_label != "pos" and \
                        only_last_cluster:
                    labels = output.split("_")
                    non_cluster_labels = list(filter(lambda x : not x.startswith("c") and not len(x) == 0, labels))
                    cluster_labels = list(filter(lambda x : x.startswith("c"), labels))
                    last_cluster = cluster_labels[-1]
                    output = "__".join([last_cluster] + non_cluster_labels)
                label_to_mdp_state_map[destination_label] = MdpState(f's{state_counter}', output=output)

            origin_state = label_to_mdp_state_map[origin]
            destination_state = label_to_mdp_state_map[destination_label]
            origin_state.transitions[action].append((destination_state, freq / total_obs))

    # does it even matter?
    if initial_state_config == 'first':
        initial_state = list(label_to_mdp_state_map.values())[0]
    else:
        initial_state = random.choice(list(label_to_mdp_state_map.values()))

    return Mdp(initial_state, list(label_to_mdp_state_map.values()))


def counting_mdp_from_cluster_labels(alergia_traces, max_history_len=5, initial_state_config='first', only_last_cluster = True):
    counting_mdps_increasing_history = dict()
    for i in range(1, max_history_len + 1):
        print(f'Creating MDP with history size of {i}')
        # (cluster_label, action) : cluster_label : frequency
        observation_counter = defaultdict(Counter)

        for trace in alergia_traces:
            trace_splits = [(trace[j:j + i], trace[j + i:j + 2 * i]) for j in range(1, len(trace))]
            for x, y in trace_splits:

                if len(y) == 0:
                    continue
                origin_label = '_'.join([l[1] for l in x])
                destination_label = '_'.join([l[1] for l in y])

                if 'succ' not in destination_label and len(x) != len(y):
                    continue

                if 'succ' in destination_label:
                    destination_label = 'succ'

                action = x[-1][0]
                observation_counter[(origin_label, action)][destination_label] += 1

        mdp = counter_to_mdp(observation_counter, initial_state_config, only_last_cluster)
        mdp.make_input_complete('sink_state')
        counting_mdps_increasing_history[i] = mdp

    return counting_mdps_increasing_history


def evaluate_ensemble(counter_mdps, env, clustering_function, scaler=None, num_episodes=100, render=True):
    prism_schedulers = dict()
    for k, v in counter_mdps.items():
        print(f'Computing scheduler for MDP with history size of {k}')
        prism_schedulers[k] = PrismInterface(["succ"], v).scheduler

    max_history_size = max(counter_mdps.keys())
    input("Start")

    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        history = []
        ep_rew = 0
        while True:

            concrete_obs = obs.reshape(1, -1)
            if scaler:
                concrete_obs = scaler.transform(concrete_obs)
            obs = f'c{clustering_function.predict(concrete_obs)[0]}'

            history.append(obs)
            history = history[-max_history_size:]

            selected_actions = dict()

            for history_size, scheduler in prism_schedulers.items():
                label = '_'.join(history[-history_size:])
                # set current state of the scheduler
                scheduler.current_state = None
                for s, obs_set in scheduler.label_dict.items():
                    if label in obs_set:
                        scheduler.current_state = s
                        break
                if scheduler.current_state is None:
                    min_dist = 1e30
                    min_l = None
                    for i, corr_center in enumerate(clustering_function.cluster_centers_):
                        if i not in cluster_center_cache:
                            cluster_center_cache[i] = clustering_function.predict(corr_center.reshape(1, -1))[0]
                        distance = euclidean_distances(concrete_obs, corr_center.reshape(1, -1))
                        if min_dist is None or distance < min_dist:
                            min_dist = distance
                            min_l = f"c{i}"
                    for s, obs_set in scheduler.label_dict.items():
                        if min_l in obs_set:
                            scheduler.current_state = s
                            break
                else:
                   #if state is reached, get action
                    #if scheduler.current_state is not None:
                    action = scheduler.get_input()
                    if action is not None:
                        selected_actions[history_size] = action

            if selected_actions:
                action = selected_actions[max(selected_actions.keys())]
            else:
                print(history)
                action = choice(list(input_map.keys()))

            concrete_action = input_map[action]
            obs, rew, done, info = env.step(concrete_action)
            if render:
                env.render()
            ep_rew += rew
            if done:
                print('Episode reward', ep_rew)
                all_rewards.append(ep_rew)
                break
    print('Mean reward:', mean(all_rewards))


env_name = "LunarLander-v2"

num_traces = 2200
scale = False
n_clusters = 512
history_size = 2

env = gym.make(env_name)

trace_file = f"{env_name}_{num_traces}_traces"
traces = load(trace_file)
if traces is None:
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
    traces = [get_traces_from_policy(dqn_agent, env, num_traces, action_map, randomness_probabilities=[0, 0.01, 0.025, 0.05])]
    save(traces, trace_file)
print('Traces obtained')

alergia_traces = \
    compute_clustering_function_and_map_to_traces(traces, action_map, n_clusters=n_clusters, scale=scale, )[0]

histroy_mdps = counting_mdp_from_cluster_labels(alergia_traces, max_history_len=history_size)

#cf = load(f'k_means_scale_{scale}_{n_clusters}_{num_traces}')
#scaler = load(f'standard_scaler_{num_traces}') if scale else None

histroy_mdps[history_size].save(f"simple_mdp_{num_traces}_{scale}_{n_clusters}_{history_size}")
# evaluate_ensemble(histroy_mdps, env, cf, scaler, num_episodes=20)
