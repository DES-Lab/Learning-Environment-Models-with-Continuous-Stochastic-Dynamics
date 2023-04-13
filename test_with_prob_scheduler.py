import math

import gym
import aalpy.paths
import numpy as np
from aalpy.learning_algs import run_JAlergia
from stable_baselines3 import DQN

from agents import load_agent
from prism_scheduler import PrismInterface, ProbabilisticScheduler, compute_weighted_clusters
from aalpy.utils import load_automaton_from_file
from sklearn.metrics import euclidean_distances
from utils import load

# copied from other file
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
                if abs(sum(map(lambda t : t[1],s.transitions[input])) - 1) > 1e-6:
                    to_remove.append(input)
        for input in to_remove:
            changed = True
            s.transitions.pop(input)
    return changed

num_traces = 5000
num_clusters = 512
scale = True
include_reward = False
environment = 'LunarLander-v2'
aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.7-linux64/bin/prism"
agent_steps = 0
if agent_steps > 0:
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
else:
    dqn_agent = None

# model = load_automaton_from_file(f'mdp_combined_scale_{scale}_{num_clusters}_{num_traces}.dot', 'mdp')
# model.make_input_complete(missing_transition_go_to='sink_state')

jalergia_samples = 'alergiaSamples_lda_working.txt'
model = run_JAlergia(jalergia_samples, 'mdp', 'alergia.jar', heap_memory='-Xmx8G', optimize_for='accuracy', eps=0.00005)
remove_nan(model)
model.make_input_complete(missing_transition_go_to='sink_state')

prism_interface = PrismInterface(["succ"], model)
scheduler = ProbabilisticScheduler(prism_interface.scheduler,True)

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}

clustering_function = load(f'{environment}_k_means_scale_{scale}_{num_clusters}_{num_traces}')
scaler = load(f"pipeline_scaler_{environment}_{num_traces}")

env = gym.make(environment)
cluster_center_cache = dict()


def take_best_out(prism_interface, scaler, clustering, concrete_obs, action,
                  possible_outs, scale):
    first_out = possible_outs[0]
    min_dist = 1e30
    min_l = None

    for o in possible_outs:
        for i, corr_center in enumerate(clustering.cluster_centers_):
            if i not in cluster_center_cache:
                cluster_center_cache[i] = clustering_function.predict(corr_center.reshape(1, -1))[0]
            cluster = cluster_center_cache[i]
            if f"c{cluster}" in o:
                # print(f"out {cluster} {o}")
                distance = euclidean_distances(concrete_obs, corr_center.reshape(1, -1))
                if min_dist is None or distance < min_dist:
                    min_dist = distance
                    min_l = o
    prism_interface.scheduler.step_to(action, min_l)

nr_outputs = num_clusters

for _ in range(1000):
    obs = env.reset()
    if include_reward:
        conc_obs = np.append(obs,[0])
    else:
        conc_obs = obs
    conc_obs = conc_obs.reshape(1, -1)

    if scale:
        conc_obs = scaler.transform(conc_obs)
    # obs = f'c{clustering_function.predict(conc_obs)[0]}'
    weighted_clusters = compute_weighted_clusters(conc_obs, clustering_function, nr_outputs)
    scheduler.reset()
    # prism_interface.step_to('right_engine', obs)
    reward = 0
    curr_steps = 0
    while True:
        if agent_steps > 0 and curr_steps < agent_steps:
            concrete_action, _ = dqn_agent.predict(obs)
            curr_steps += 1
            action = action_map[concrete_action.item()]
        else:
            if curr_steps == agent_steps:
                print("Switching to MDP")
            curr_steps += 1
            action = scheduler.get_input()
            if action is None:
                print('Cannot schedule an action')
                break
            concrete_action = input_map[action]

        obs, rew, done, info = env.step(concrete_action)
        reward += rew
        if include_reward:
            conc_obs = np.append(obs,[rew])
        else:
            conc_obs = obs
        conc_obs = conc_obs.reshape(1, -1)

        if scale:
            conc_obs = scaler.transform(conc_obs)
        weighted_clusters = compute_weighted_clusters(conc_obs, clustering_function, nr_outputs)
        reached_state = scheduler.step_to(action, weighted_clusters)
        env.render()
        if not reached_state:
            done = True
            reward = -1000
            print('Run into state that is unreachable in the model.')
            # possible_outs = prism_interface.scheduler.poss_step_to(action)
            # take_best_out(prism_interface, scaler, clustering_function, conc_obs, action,
            #               possible_outs, scale)
        if done:
            print(env.game_over)
            if not env.game_over:
                print(rew)
                # import time
                # time.sleep(2)
            print(f'Episode reward: {reward} after {curr_steps} steps')
            if reward > 1:
                print('Success', reward)
            break
