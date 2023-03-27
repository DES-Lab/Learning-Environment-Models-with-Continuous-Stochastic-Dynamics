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

def compute_mdp(jalergia_samples, alergia_eps):
    mdp = run_JAlergia(jalergia_samples, 'mdp', 'alergia.jar', heap_memory='-Xmx12G', optimize_for='accuracy',
                       eps=alergia_eps)
    remove_nan(mdp)
    mdp.make_input_complete(missing_transition_go_to='sink_state')
    return mdp

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
# def take_best_out(cluster_center_cache,prism_interface, scaler, clustering, concrete_obs, action,
#                   possible_outs, scale):
#     first_out = possible_outs[0]
#     min_dist = 1e30
#     min_l = None
#
#     for o in possible_outs:
#         for i, corr_center in enumerate(clustering.cluster_centers_):
#             if i not in cluster_center_cache:
#                 cluster_center_cache[i] = clustering_function.predict(corr_center.reshape(1, -1))[0]
#             cluster = cluster_center_cache[i]
#             if f"c{cluster}" in o:
#                 # print(f"out {cluster} {o}")
#                 distance = euclidean_distances(concrete_obs, corr_center.reshape(1, -1))
#                 if min_dist is None or distance < min_dist:
#                     min_dist = distance
#                     min_l = o
#     prism_interface.scheduler.step_to(action, min_l)



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


def prob_reach_checking(env, scaler, clustering_function, n_traces_per_ref,
                        model_refinements, nr_reaches, input_map, jalergia_samples, nr_outputs, alergia_eps, goal,
                        environment_name, agent = None, mdp = None, refine_mdp = True):
    if mdp is None:
       mdp = compute_mdp(jalergia_samples, alergia_eps)
    prism_interface = PrismInterface([goal], mdp)
    scheduler = ProbabilisticScheduler(prism_interface.scheduler, True)
    crashes = 0
    curr_nr_reaches = 0
    nr_tries = 0
    for i in range(model_refinements):
        print(f"Model refinement {i}")
        traces_in_ref = []
        rewards = []
        goal_freq = 0
        for j in range(n_traces_per_ref):
            nr_tries += 1
            print(f"Trace {j}")
            curr_trace = ['INIT']
            obs = env.reset()
            conc_obs = obs
            conc_obs = conc_obs.reshape(1, -1).astype(float)
            unscaled_obs = conc_obs
            if scaler:
                conc_obs = scaler.transform(conc_obs)
            obs = f'c{clustering_function.predict(conc_obs)[0]}'
            weighted_clusters = compute_weighted_clusters(conc_obs, clustering_function, nr_outputs)
            scheduler.reset()
            # prism_interface.step_to('right_engine', obs)
            reward = 0
            curr_steps = 0
            reached = False
            while True:
                if reached:
                    action, _ = agent.predict(unscaled_obs)
                    unscaled_obs, rew, done, info = env.step(action.item())
                    if done and rew == -100:
                        crashes += 1
                    if done:
                        break
                else:
                    curr_steps += 1
                    action = scheduler.get_input()
                    if action is None:
                        print('Cannot schedule an action')
                        break
                    concrete_action = input_map[action]
                    obs, rew, done, info = env.step(concrete_action)
                    reward += rew
                    conc_obs = obs
                    conc_obs = conc_obs.reshape(1, -1).astype(float)
                    unscaled_obs = conc_obs
                    if scaler:
                        conc_obs = scaler.transform(conc_obs)

                    cluster_label_int = clustering_function.predict(conc_obs)[0]
                    obs = f'c{cluster_label_int}'
                    # obs = add_to_label(obs,unscaled_obs[0][1],rew,done)
                    # label = (obs,unscaled_obs[0][1],rew,done)
                    curr_trace.append((action, cluster_label_int, rew, done, unscaled_obs))

                    weighted_clusters = compute_weighted_clusters(conc_obs, clustering_function, nr_outputs)
                    reached_state = scheduler.step_to(action, weighted_clusters)
                    # env.render()
                    if not reached_state:
                        done = True
                        reward = -1000
                        print('Run into state that is unreachable in the model.')
                    if agent:
                        if not reached and goal == obs:
                            reached = True
                            print("Reached goal")
                            goal_freq += 1
                            curr_nr_reaches += 1
                            traces_in_ref.append(curr_trace)

                    if done:
                        traces_in_ref.append(curr_trace)
                        rewards.append(reward)
                        print(f'Episode reward: {reward} after {curr_steps} steps')
                        # if goal == obs or (reward > 100 and goal == "succ"):
                        #     print("Reached goal")
                        #     goal_freq += 1
                        if reward > 1:
                            print('Success', reward)
                        break
            if curr_nr_reaches >= nr_reaches:
                break
        label_traces = create_labels_in_traces(traces_in_ref, environment_name)
        if len(rewards) > 0:
            avg_reward = sum(rewards) / len(rewards)
            std_dev = math.sqrt((1. / len(rewards)) * sum([(r - avg_reward) ** 2 for r in rewards]))
            print(f"Average reward in iteration: {avg_reward} +/- {std_dev}")
        print(f"Goal frequency: {goal_freq / len(traces_in_ref)}")
        if refine_mdp:
            append_samples_to_file(label_traces, jalergia_samples)
        if curr_nr_reaches >= nr_reaches:
            break
        if refine_mdp:
            mdp = run_JAlergia(jalergia_samples, 'mdp', 'alergia.jar', heap_memory='-Xmx12G', optimize_for='accuracy',
                               eps=alergia_eps)
            remove_nan(mdp)
            mdp.make_input_complete(missing_transition_go_to='sink_state')
            prism_interface = PrismInterface([goal], mdp)
            scheduler = ProbabilisticScheduler(prism_interface.scheduler, True)
        traces_in_ref.clear()
    return crashes, nr_tries, curr_nr_reaches


