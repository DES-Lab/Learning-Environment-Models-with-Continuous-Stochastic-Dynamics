from collections import defaultdict, Counter
from random import choices
from statistics import mean

import gym
from stable_baselines3 import DQN

from abstraction import compute_clustering_function_and_map_to_traces
from agents import load_agent
from utils import load, get_traces_from_policy, save

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}


def compute_history_based_prediction(alergia_traces, max_history_len=5):
    action_dict = defaultdict(dict)
    for trace in alergia_traces:
        trace_splits = [trace[j:j + i] for i in range(1, max_history_len + 1) for j in range(1, len(trace))]
        # print(trace_splits)
        for split in trace_splits:
            clusters = tuple(c[1] for c in split)
            action = split[-1][0]
            if action not in action_dict[clusters].keys():
                action_dict[clusters][action] = 0
            action_dict[clusters][action] += 1

    return action_dict


def choose_action(obs, action_dict, based_on='confidence'):
    all_splits = [tuple(obs[len(obs) - i - 1:]) for i in range(len(obs))]
    all_splits.reverse()
    chosen_action = None
    if based_on == 'longest':
        for split in all_splits:
            if split in action_dict.keys():
                chosen_action = max(action_dict[split], key=action_dict[split].get)
                break
    if based_on == 'probabilistic_longest':
        for split in all_splits:
            if split in action_dict.keys():
                actions = list(action_dict[split].keys())
                weights = list(action_dict[split].values())
                chosen_action = choices(actions, weights=weights)[0]
                break

    elif based_on == 'confidence':
        split_action_lists = []  # compute frequqnct
        for split in all_splits:
            pass
    elif based_on == 'majority_vote':
        pass

    return chosen_action


def evaluate(env, action_dict, clustering_function, scaler, history_window_size=5, num_episodes=100,
             strategy='longest',
             render=True):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        history = []
        ep_rew = 0
        while True:
            conc_obs = obs.reshape(1, -1)
            if scaler:
                conc_obs = scaler.transform(conc_obs)
            obs = f'c{clustering_function.predict(conc_obs)[0]}'
            print(obs)

            history.append(obs)
            history = history[-history_window_size:]

            abstract_action = choose_action(tuple(history), action_dict, based_on=strategy)
            action = input_map[abstract_action]
            obs, rew, done, info = env.step(action)
            if render:
                env.render()
            ep_rew += rew
            if done:
                print('Episode reward', ep_rew)
                all_rewards.append(ep_rew)
                break
    print('Mean reward:', mean(all_rewards))


env_name = "LunarLander-v2"

num_traces = 100
scale = False
n_clusters = 512
history_size = 10

env = gym.make(env_name)

trace_file = f"{env_name}_{num_traces}_traces"
traces = load(trace_file)
if traces is None:
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
    traces = [get_traces_from_policy(dqn_agent, env, num_traces, action_map, randomness_probs=[0, 0.01, 0.025, 0.05])]
    save(traces, trace_file)
print('Traces obtained')

alergia_traces = compute_clustering_function_and_map_to_traces(traces, action_map, n_clusters=n_clusters, scale=scale, )[0]

action_dict = compute_history_based_prediction(alergia_traces, max_history_len=history_size)

cf = load(f'k_means_scale_{scale}_{n_clusters}_{num_traces}')
scaler = load(f'standard_scaler_{num_traces}') if scale else None

evaluate(env, action_dict, cf, scaler, history_window_size=history_size, strategy='probabilistic_longest')
