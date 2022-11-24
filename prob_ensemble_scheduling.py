from typing import Dict

import gym
import aalpy.paths

from ensemble_creation import load_ensemble
from prism_scheduler import ProbabilisticEnsembleScheduler, compute_weighted_clusters
from aalpy.utils import load_automaton_from_file
from aalpy.automata import Mdp
from sklearn.metrics import euclidean_distances
from collections import defaultdict
import random

from utils import load

def load_mdp_ensemble(environment,name, num_clusters, num_traces, scale) -> Dict[str,Mdp]:
    return load_ensemble(saved_path_prefix=name)

def cluster_from_out(out):
    # works if labels are "something_without_c__c\d+"
    return out[out.index("c"):]

def run_episode(env, input_map, ensemble_scheduler : ProbabilisticEnsembleScheduler,
                scale, scaler, clustering_function,nr_outputs):
    obs = env.reset()
    conc_obs = obs.reshape(1, -1)

    if scale:
        conc_obs = scaler.transform(conc_obs)
    obs = f'c{clustering_function.predict(conc_obs)[0]}'

    weighted_clusters = compute_weighted_clusters(conc_obs,clustering_function,nr_outputs)
    ensemble_scheduler.reset()
    ensemble_scheduler.activate_scheduler(obs,weighted_clusters)
    # prism_interface.step_to('right_engine', obs)
    reward = 0
    while True:
        action = ensemble_scheduler.get_input()
        if action is None:
            print('Cannot schedule an action')
            break
        concrete_action = input_map[action]

        obs, rew, done, info = env.step(concrete_action)
        reward += rew
        conc_obs = obs.reshape(1, -1)

        if scale:
            conc_obs = scaler.transform(conc_obs)
        obs = f'c{clustering_function.predict(conc_obs)[0]}'

        weighted_clusters = compute_weighted_clusters(conc_obs, clustering_function, nr_outputs)
        ensemble_scheduler.step_to(action, weighted_clusters, obs)
        env.render()
        if done:
            print(env.game_over)
            if not env.game_over:
                print(rew)
                # import time
                # time.sleep(2)
            print('Episode reward: ', reward)
            if reward > 1:
                print('Success', reward)
            break


num_clusters = 128
num_traces = 45000
scale = False
clustering_type = "k_means"
environment = 'LunarLander-v2'
aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.7-linux64/bin/prism"
target_label = "succ"
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}

max_state_size = 10
truly_probabilistic = True

mdp_ensemble = load_mdp_ensemble(environment,"ensemble_45k_k_means", num_clusters,num_traces,scale)
ensemble_scheduler = ProbabilisticEnsembleScheduler(mdp_ensemble,target_label,input_map,truly_probabilistic,
                                                    max_state_size)
scaler = load(f'standard_scaler_{num_traces}') if scale else None
clustering_function = load(f'{clustering_type}_scale_{scale}_{num_clusters}_{num_traces}')
env = gym.make(environment)

input("type")
for i in range(1000):
    run_episode(env,input_map, ensemble_scheduler, scale, scaler, clustering_function, nr_outputs= 16) #num_clusters)

