from typing import Dict

import gym
import aalpy.paths

from ensemble_creation import load_ensemble
from prism_scheduler import PrismInterface, Scheduler
from aalpy.utils import load_automaton_from_file
from aalpy.automata import Mdp
from sklearn.metrics import euclidean_distances
from collections import defaultdict
        
from utils import load

def load_mdp_ensemble(environment, num_clusters, num_traces, scale) -> Dict[str,Mdp]:
    return load_ensemble()

def cluster_from_out(out):
    # works if labels are "something_without_c__c\d+"
    return out[out.index("c"):]

class EnsembleScheduler:
    def __init__(self,mdp_ensemble : Dict[str,Mdp], target_label):
        self.mdp_ensemble = mdp_ensemble
        self.target_label = target_label
        self.scheduler_ensemble = self.compute_schedulers(mdp_ensemble,target_label)
        self.active_schedulers = dict()

    def compute_schedulers(self,mdp_ensemble : Dict[str,Mdp], target_label) -> Dict[str,Scheduler]:
        schedulers = dict()
        for cluster_label in mdp_ensemble.keys():
            mdp = mdp_ensemble[cluster_label]
            prism_interface = PrismInterface(self.target_label, mdp)
            schedulers[cluster_label] = prism_interface.scheduler
        return schedulers

    def reset(self):
        self.active_schedulers = dict()

    def activate_scheduler(self,cluster_label):
        if cluster_label not in self.active_schedulers.keys():
            self.active_schedulers[cluster_label] = self.scheduler_ensemble[cluster_label] 
            self.active_schedulers[cluster_label].reset()

    def step_to(self,inp,out):
        cluster_label = cluster_from_out(out)
        to_del = []
        for label,scheduler in self.active_schedulers.items():
            reached_state = scheduler.step_to(inp,out)
            if reached_state is None:
                # "deactivate"
                to_del.append(label)
        for label in to_del:
            self.active_schedulers.pop(label)
        
        self.activate_scheduler(cluster_label)
        print(f"active schedulers: {len(self.active_schedulers)}")

    def get_input(self):
        input_preferences = defaultdict(int)
        for label,scheduler in self.active_schedulers.items():
            input_pref = scheduler.get_input()
            if input_pref is None:
                print(f"Unknown input preferences for scheduler with label {label}")
            input_preferences[input_pref] += 1
        max_pref_cnt = 0
        pref = None
        print(f"Input preferences: {input_preferences}")
        for inp,cnt in input_preferences.items():
            if cnt > max_pref_cnt:
                max_pref_cnt = cnt
                pref = inp
        return pref
        
def run_episode(env,input_map, ensemble_scheduler, scale, scaler, clustering_function):
    obs = env.reset()
    conc_obs = obs.reshape(1, -1)

    if scale:
        conc_obs = scaler.transform(conc_obs)
    obs = f'c{clustering_function.predict(conc_obs)[0]}'

    ensemble_scheduler.reset()
    ensemble_scheduler.activate_scheduler(obs)
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
        ensemble_scheduler.step_to(action, obs)
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


num_clusters = 32
num_traces = 2000
scale = True
environment = 'LunarLander-v2'
aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.7-linux64/bin/prism"
target_label = "succ"

mdp_ensemble = load_mdp_ensemble(environment,num_clusters,num_traces,scale)
ensemble_scheduler = EnsembleScheduler(mdp_ensemble,target_label)
scaler = load(f'standard_scaler_{num_traces}')
clustering_function = load(f'k_means_scale_{scale}_{num_clusters}_{num_traces}')
env = gym.make(environment)

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}

for i in range(100):
    run_episode(env,input_map, ensemble_scheduler, scale, scaler, clustering_function)

