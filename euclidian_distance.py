from random import shuffle
from statistics import mean

import gym
import numpy as np
from stable_baselines3 import DQN

from agents import load_agent
from trace_abstraction import create_abstract_traces

def get_observation_action_pairs(env, num_ep=20):
    dqn_agent = load_agent("araffin/dqn-LunarLander-v2", 'dqn-LunarLander-v2.zip', DQN)

    sample_num = 0
    observation_actions_pairs = []
    episodes = []
    for _ in range(num_ep):
        sample_num += 1
        if sample_num % 100 == 0:
            print(sample_num)
        obs = env.reset()
        ep = []
        while True:
            action, state = dqn_agent.predict(obs)
            observation_actions_pairs.append((obs, action))
            obs, reward, done, _ = env.step(action)
            ep.append((obs.reshape(1, -1), action, reward, done))

            if done:
                break

        episodes.append(ep)

    return observation_actions_pairs, episodes


def chose_action_based_on_euclidean_distance(env_obs, obs_action_pairs, num_comparison=1000):
    chosen_action = None
    min_dist = None

    shuffle(obs_action_pairs)
    distances = []

    for obs, action in obs_action_pairs[:num_comparison]:
        ed = np.linalg.norm(env_obs - obs)

        distances.append(ed)
        if not min_dist or ed < min_dist:
            min_dist = ed
            chosen_action = action

    return chosen_action


def evaluate_on_environment(env, obs_action_pairs, num_episodes=100, render=False):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        ep_rew = 0
        while True:
            action = chose_action_based_on_euclidean_distance(obs, obs_action_pairs)
            obs, rew, done, info = env.step(action)

            if render:
                env.render()
            ep_rew += rew
            if done:
                print('Episode reward:', ep_rew)
                all_rewards.append(ep_rew)
                break

    print('Mean reward:', mean(all_rewards))


# def get_same_action_sequential_mean(traces):
#     same_obs_in_sequence = list()
#     for trace in traces:
#         prev_act = None
#         same_actions = list()
#
#         for obs, act in trace:
#             if prev_act is None:
#                 prev_act = act
#                 same_actions.append(obs)
#             else:
#                 if prev_act != act:
#                     same_obs_in_sequence.append((same_actions, prev_act))
#                     prev_act = act
#                     same_actions = [obs]
#                 else:
#                     same_actions.append(obs)
#
#     pruned_obs_act_pairs = []
#     for same_act_set, act in same_obs_in_sequence:
#         print(len(same_act_set))
#         pruned_obs_act_pairs.append((random.choice(same_act_set), act))
#
#     return pruned_obs_act_pairs
#
#     # for same_act_set, _ in same_obs_in_sequence:
#
#     # means_across_same_actions_in_seq = []
#     # for same_act_set in same_obs_in_sequence:
#     #     set_distances = []
#     #     for i in range(len(same_act_set)):
#     #         for j in range(i + 1, len(same_act_set)):
#     #             set_distances.append(np.linalg.norm(same_act_set[i] - same_act_set[j]))
#     #
#     #     means_across_same_actions_in_seq.append(mean(set_distances))
class CustomKMeans:
    def __init__(self, observations):
        self.obs = list(observations)

    def predict(self, X):
        distances = []
        min_dist = None
        chosen_obs = None
        for index, obs in enumerate(self.obs):
            ed = np.linalg.norm(X - obs)

            distances.append(ed)
            if not min_dist or ed < min_dist:
                min_dist = ed
                chosen_obs = index

        return np.array(chosen_obs)


env_name = "LunarLander-v2"
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}

num_traces = 10
load_observations = True

env = gym.make(env_name)

print('Computing observation action pairs')
obs_action_pairs, episodes = get_observation_action_pairs(env, num_traces)
print('Number of obs/action pairs', len(obs_action_pairs))
# obs_action_pairs = get_same_action_sequential_mean(episodes)
# print('Pruned', len(obs_action_pairs))

evaluate_on_environment(env, obs_action_pairs, num_episodes=10, render=False)

# k_means = CustomKMeans([x[0] for x in obs_action_pairs])
# cluster_labels = []
# for trace in episodes:
#     cl = []
#     for obs, _, _, _  in trace:
#         cl.append(k_means.predict(obs))
#     cluster_labels.extend(cl)
#
# at = create_abstract_traces(episodes, cluster_labels)
# model = run_JAlergia(at, automaton_type='mdp', path_to_jAlergia_jar='alergia.jar', optimize_for='accuracy')
#
#
# import aalpy.paths
#
# aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"
#
# ir = IterativeRefinement(env, model, at, None, k_means,
#                          scheduler_type='deterministic', count_observations=False)
#
# ir.iteratively_refine_model(10, 100)
