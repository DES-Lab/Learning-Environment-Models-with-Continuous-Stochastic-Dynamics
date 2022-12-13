from typing import Dict

import gym
import aalpy.paths
from stable_baselines3 import DQN

from agents import load_agent
from ensemble_creation import load_ensemble
from prism_scheduler import ProbabilisticEnsembleScheduler, compute_weighted_clusters
from aalpy.utils import load_automaton_from_file
from aalpy.automata import Mdp
from sklearn.metrics import euclidean_distances
from collections import defaultdict
import random

from utils import load, save


def load_mdp_ensemble(environment,name, num_clusters, num_traces, scale) -> Dict[str,Mdp]:
    return load_ensemble(saved_path_prefix=f"{name}_{num_traces}_scale_{scale}_k_means_{num_clusters}")

def cluster_from_out(out):
    # works if labels are "something_without_c__c\d+"
    return out[out.index("c"):]

def run_episode(env, agent, input_map, ensemble_scheduler : ProbabilisticEnsembleScheduler,
                scale, scaler, clustering_function,nr_outputs, duplicate_action = False):
    nr_agree = 0
    steps = 0
    orig_obs = env.reset()
    conc_obs = orig_obs.reshape(1, -1).astype(float)

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
        agent_action, _ = agent.predict(orig_obs)

        if action == action_map[agent_action.item()]:
            nr_agree += 1
        steps += 1
        #if steps % 2 == 0:
        #    action = action_map[agent_action.item()]
        # print(action,action_map[agent_action.item()])
        if action is None:
            print('Cannot schedule an action')
            break
        concrete_action = input_map[action]

        orig_obs, rew, done, info = env.step(concrete_action)
        if duplicate_action:
            orig_obs, rew, done, info = env.step(concrete_action)
        reward += rew
        conc_obs = orig_obs.reshape(1, -1).astype(float)

        if scale:
            conc_obs = scaler.transform(conc_obs)
        weighted_clusters = compute_weighted_clusters(conc_obs, clustering_function, nr_outputs)
        obs = f'c{clustering_function.predict(conc_obs)[0]}'
        ensemble_scheduler.step_to(action, weighted_clusters, obs)
        # env.render()
        if done:
            # print(env.game_over)
            # if not env.game_over:
                # print(rew)
                # import time
                # time.sleep(2)
            # print(f"Agreement: {nr_agree/steps}")
            # print('Episode reward: ', reward)
            # if reward > 1:
            #     print('Success', rew)
            return reward, nr_agree/steps, reward > 1


num_clusters = 400
num_traces = 2300
scale = True
clustering_type = "k_means"
environment = 'LunarLander-v2'
aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.7-linux64/bin/prism"
target_label = "succ"

if environment == "LunarLander-v2":
    dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
    action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
else:
    dqn_agent = load_agent('DBusAI/DQN-MountainCar-v0-v2', 'DQN-MountainCar-v0.zip', DQN)
    action_map = {0: 'left', 1: 'no_action', 2: 'right'}
input_map = {v: k for k, v in action_map.items()}


import sys
sys.setrecursionlimit(3000)
max_state_size = 2
truly_probabilistic = True
count_misses = False
duplicate_action = False
ensemble_name = f"ensemble_all_{environment}"
sched_name = f"prob_sched_{ensemble_name}_{clustering_type}_{num_traces}_{scale}_{num_clusters}"
ensemble_scheduler = load(sched_name)
if ensemble_scheduler is None:
    mdp_ensemble = load_mdp_ensemble(environment, ensemble_name, num_clusters, num_traces, scale)
    ensemble_scheduler = ProbabilisticEnsembleScheduler(mdp_ensemble,target_label,input_map,truly_probabilistic,
                                                        max_state_size,count_misses)
    save(ensemble_scheduler,f"prob_sched_{ensemble_name}_{clustering_type}_{num_traces}_{scale}_{num_clusters}")

ensemble_scheduler.set_max_state_size(max_state_size)
ensemble_scheduler.count_misses = count_misses
ensemble_scheduler.truly_probabilistic = truly_probabilistic
scaler = load(f'pipeline_scaler_{environment}_{num_traces}') if scale else None
clustering_function = load(f'{environment}_{clustering_type}_scale_{scale}_{num_clusters}_{num_traces}')
env = gym.make(environment)

nr_test_episodes = 2

ensemble_scheduler.max_schedulers = 2
ensemble_scheduler.max_misses = 300

    # for c_misses,n_outputs, state_size in [(False,6,2),(False,10,4)]:
for c_misses in [True,False]:
    # [10,20]:
    for n_outputs in range(2, 250, 20):
        for state_size in range(2, 250,20): #[6,10,18]:

            for truly_probabilistic in [True, False]:
                print(f"T-Prob:{truly_probabilistic}, c-misses:{c_misses}, nr-out: {n_outputs}, states:{state_size}")
                ensemble_scheduler.set_max_state_size(state_size)
                ensemble_scheduler.count_misses = c_misses
                ensemble_scheduler.truly_probabilistic = truly_probabilistic
                avg_reward = 0
                success_rate = 0
                avg_agreement = 0
                for i in range(nr_test_episodes):
                    reward, agreement, success = \
                        run_episode(env,dqn_agent,input_map, ensemble_scheduler, scale, scaler, clustering_function,
                                    nr_outputs=n_outputs, duplicate_action = duplicate_action) #num_clusters)
                    avg_reward += reward
                    success_rate += success
                    avg_agreement += agreement
                avg_reward /= nr_test_episodes
                success_rate /= nr_test_episodes
                avg_agreement /= nr_test_episodes
                print("Reward: {:.2f}, success: {:.2f}, agreement: {:.2f}".format(avg_reward,success_rate,avg_agreement))

