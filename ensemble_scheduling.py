from typing import Dict

import gym
import aalpy.paths

from ensemble_creation import load_ensemble
from prism_scheduler import PrismInterface, Scheduler
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

cluster_center_cache = dict()

def take_best_out(scheduler, clustering, concrete_obs, action, possible_outs):
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
    scheduler.step_to(action, min_l)

class EnsembleScheduler:
    def __init__(self,mdp_ensemble : Dict[str,Mdp], target_label):
        self.mdp_ensemble = mdp_ensemble
        self.target_label = target_label
        self.scheduler_ensemble = self.compute_schedulers(mdp_ensemble,target_label)
        self.active_schedulers = dict()
        self.initial_lives = 5 # 1 works okay

    def compute_schedulers(self,mdp_ensemble : Dict[str,Mdp], target_label) -> Dict[str,Scheduler]:
        schedulers = dict()
        for cluster_label in mdp_ensemble.keys():
            print(f"Initialized scheduler for {cluster_label}")
            mdp = mdp_ensemble[cluster_label]
            prism_interface = PrismInterface(self.target_label, mdp)
            schedulers[cluster_label] = prism_interface.scheduler
        return schedulers

    def reset(self):
        self.active_schedulers = dict()

    def activate_scheduler(self,cluster_label,conc_obs, clustering_function):
        if cluster_label not in self.active_schedulers.keys():
            if cluster_label not in self.scheduler_ensemble:
                # find closest
                activated_scheduler = find_closest_scheduler(conc_obs,clustering_function,self.scheduler_ensemble)
            else:
                activated_scheduler = self.scheduler_ensemble[cluster_label]
            self.active_schedulers[cluster_label] = (activated_scheduler, self.initial_lives)
            self.active_schedulers[cluster_label][0].reset()

    def step_to(self,inp,out, clustering_function, conc_obs):
        cluster_label = cluster_from_out(out)
        decrease_lives = []
        for label,(scheduler,lives) in self.active_schedulers.items():
            reached_state = scheduler.step_to(inp,out)
            if reached_state is None:
                # "deactivate"
                decrease_lives.append((label, lives))
                if lives > 1:
                    possible_outs = scheduler.poss_step_to(inp)
                    take_best_out(scheduler, clustering_function, conc_obs, inp,possible_outs)
        for (label,lives) in decrease_lives:
            if lives == 1:
                self.active_schedulers.pop(label)
            else:
                self.active_schedulers[label] = (self.active_schedulers[label][0], lives-1)
        self.activate_scheduler(cluster_label,conc_obs,clustering_function)
        #print(f"active schedulers: {len(self.active_schedulers)}")

    def get_input(self):

        input_preferences = defaultdict(int)
        for label,(scheduler,lives) in self.active_schedulers.items():
            input_pref = scheduler.get_input()
            if input_pref is None:
                print(f"Unknown input preferences for scheduler with label {label}")
            else:
                input_preferences[input_pref] += 1 / (self.initial_lives-lives +1)
        if len(input_preferences) == 0:
            print("Don't know any good input")
            return random.choice(list(input_map.keys()))
        (inputs, weights) = zip(*list(input_preferences.items()))
        # max_pref_cnt = 0
        # pref = None
        #print(f"Input preferences: {input_preferences}")
        # for inp,cnt in input_preferences.items():
        #     if cnt > max_pref_cnt:
        #         max_pref_cnt = cnt
        #         pref = inp

        return random.choices(inputs, weights=weights)[0]


def run_episode(env, input_map, ensemble_scheduler, scale, scaler, clustering_function):
    obs = env.reset()
    conc_obs = obs.reshape(1, -1)

    if scale:
        conc_obs = scaler.transform(conc_obs)
    obs = f'c{clustering_function.predict(conc_obs)[0]}'

    ensemble_scheduler.reset()
    ensemble_scheduler.activate_scheduler(obs,conc_obs,clustering_function)
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
        ensemble_scheduler.step_to(action, obs, clustering_function, conc_obs)
        env.render()
        #if not reached_state:
            # done = True
            # reward = -1000
            # print('Run into state that is unreachable in the model.')
        #    possible_outs = ensemble_scheduler.poss_step_to(action)
        #    take_best_out(prism_interface, scaler, clustering_function, conc_obs, action,
        #                  possible_outs, scale)
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
num_traces = 3600
scale = False
clustering_type = "mean_shift"
environment = 'LunarLander-v2'
aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.7-linux64/bin/prism"
target_label = "succ"

mdp_ensemble = load_mdp_ensemble(environment,"ensemble_3_6k_meanshift", num_clusters,num_traces,scale)
ensemble_scheduler = EnsembleScheduler(mdp_ensemble,target_label)
scaler = load(f'standard_scaler_{num_traces}')
clustering_function = load(f'{clustering_type}_scale_{scale}_{num_clusters}_{num_traces}')
env = gym.make(environment)

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}
input("type")
for i in range(1000):
    run_episode(env,input_map, ensemble_scheduler, scale, scaler, clustering_function)

