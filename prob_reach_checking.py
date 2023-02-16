import math

import gym
import aalpy.paths
import numpy as np
from aalpy.learning_algs import run_JAlergia
from stable_baselines3 import DQN

from abstraction import create_label
from agents import load_agent
from prism_scheduler import PrismInterface, ProbabilisticScheduler, compute_weighted_clusters
from aalpy.utils import load_automaton_from_file
from sklearn.metrics import euclidean_distances
from utils import load, append_samples_to_file


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


num_traces = 2500
num_clusters = 256
scale = True
include_reward = False
environment = 'LunarLander-v2'
aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.7-linux64/bin/prism"
agent_steps = 0
if agent_steps > 0:
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
else:
    dqn_agent = None

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}

alergia_eps = 0.05
jalergia_samples = 'alergiaSamples.txt'
mdp = run_JAlergia(jalergia_samples, 'mdp', 'alergia.jar', heap_memory='-Xmx12G', optimize_for='accuracy', eps=alergia_eps)
remove_nan(mdp)
mdp.make_input_complete(missing_transition_go_to='sink_state')
goal = "c250"
prism_interface = PrismInterface([goal], mdp)
scheduler = ProbabilisticScheduler(prism_interface.scheduler,True)

# action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
# input_map = {v: k for k, v in action_map.items()}

clustering_function = load(f'{environment}_k_means_scale_{scale}_{num_clusters}_{num_traces}')
scaler = load(f"power_scaler_{environment}_{num_traces}")

env = gym.make(environment)
cluster_center_cache = dict()

def add_to_label(label,y_loc, reward,done):
    additional_label = "__low" if y_loc <= 0.2 else ""
    if  reward == 100 and done:
        return f"{label}__succ__pos{additional_label}"
    elif reward == -100 and done:
        return f"{label}__bad{additional_label}"
    elif reward >= 10 and done:
        return f"{label}__pos{additional_label}"
    else:
        return f"{label}{additional_label}"
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

model_refinements = 100
n_traces_per_ref = 200


def create_labels_in_traces(traces_in_ref, env_name):
    label_traces = []
    nr_steps = 0
    for trace in traces_in_ref:
        label_trace = ['INIT']
        nr_steps += 1
        last_cluster = 0
        for (i, (action, cluster_label_int, rew, done, state)) in enumerate(trace[1:]):
            next_cluster_label_int = trace[i+1][1] if i < len(trace) else None
            cluster_label = f'c{cluster_label_int}'
            label = create_label(nr_steps, cluster_label, cluster_label_int, done, env_name, last_cluster,
                                 next_cluster_label_int, rew, state)
            last_cluster = cluster_label_int
            label_trace.append((action, label))
            last_cluster = cluster_label_int
        label_traces.append(label_trace)
    return label_traces


for i in range(model_refinements):
    print(f"Model refinement {i}")
    traces_in_ref = []
    rewards = []
    goal_freq = 0
    for j in range(n_traces_per_ref):
        print(f"Trace {j}")
        curr_trace = ['INIT']
        obs = env.reset()
        if include_reward:
            conc_obs = np.append(obs,[0])
        else:
            conc_obs = obs
        conc_obs = conc_obs.reshape(1, -1).astype(float)

        if scale:
            conc_obs = scaler.transform(conc_obs)
        obs = f'c{clustering_function.predict(conc_obs)[0]}'
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
                # if curr_steps == agent_steps:
                #     print("Switching to MDP")
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
            conc_obs = conc_obs.reshape(1, -1).astype(float)
            before_scale = conc_obs
            if scale:
                conc_obs = scaler.transform(conc_obs)

            cluster_label_int = clustering_function.predict(conc_obs)[0]
            obs = f'c{cluster_label_int}'
            # obs = add_to_label(obs,before_scale[0][1],rew,done)
            # label = (obs,before_scale[0][1],rew,done)
            curr_trace.append((action,cluster_label_int,rew,done,before_scale))

            weighted_clusters = compute_weighted_clusters(conc_obs, clustering_function, nr_outputs)
            reached_state = scheduler.step_to(action, weighted_clusters)
            # env.render()
            if not reached_state:
                done = True
                reward = -1000
                print('Run into state that is unreachable in the model.')
                # possible_outs = prism_interface.scheduler.poss_step_to(action)
                # take_best_out(prism_interface, scaler, clustering_function, conc_obs, action,
                #               possible_outs, scale)
            if done or goal == obs:
                traces_in_ref.append(curr_trace)
                    # import time
                    # time.sleep(2)
                rewards.append(reward)
                print(f'Episode reward: {reward} after {curr_steps} steps')
                if goal == obs or (reward > 100 and goal == "succ"):
                    print("Reached goal")
                    goal_freq += 1
                if reward > 1:
                    print('Success', reward)
                break

    label_traces = create_labels_in_traces(traces_in_ref,environment)
    avg_reward = sum(rewards) / len(rewards)
    std_dev = math.sqrt((1. / len(rewards)) * sum([(r - avg_reward) ** 2 for r in rewards]))
    print(f"Average reward in iteration: {avg_reward} +/- {std_dev}")
    print(f"Goal frequency: {goal_freq/len(rewards)}")
    append_samples_to_file(label_traces,jalergia_samples)
    mdp = run_JAlergia(jalergia_samples, 'mdp', 'alergia.jar', heap_memory='-Xmx12G', optimize_for='accuracy', eps=alergia_eps)
    remove_nan(mdp)
    mdp.make_input_complete(missing_transition_go_to='sink_state')
    prism_interface = PrismInterface([goal], mdp)
    scheduler = ProbabilisticScheduler(prism_interface.scheduler, True)
    traces_in_ref.clear()
