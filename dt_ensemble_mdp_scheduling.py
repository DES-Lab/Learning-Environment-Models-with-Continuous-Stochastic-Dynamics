from typing import Dict

import aalpy.paths
import gym
from aalpy.automata import Mdp
from aalpy.utils import load_automaton_from_file
from sklearn.preprocessing import FunctionTransformer
from stable_baselines3 import DQN

from agents import load_agent
from dt_mdp_learning import build_tree_copy, add_features
from ensemble_creation import load_ensemble
from prism_scheduler import PrismInterface, ProbabilisticScheduler, ProbabilisticEnsembleScheduler
from utils import load, save

def load_mdp_ensemble(environment,name, num_traces_env_learn, scale, max_leaf_nodes) -> Dict[str,Mdp]:
    return load_ensemble(saved_path_prefix=f"dt_{name}_{environment}_{num_traces_env_learn}_scale_{scale}_{max_leaf_nodes}")

def cluster_from_out(out):
    # works if labels are "something_without_c__c\d+"
    return out[out.index("c"):]

def compute_weighted_outs(tree_copy, obs, possible_outs, nr_outs):
    predicted_id = int(obs[1:])
    weighted_outputs = dict()
    # print(obs)
    for o in possible_outs:
        labels = o.split("__")
        for label in labels:
            if label.startswith("c"):
                o_leave_id = int(o[1:])
                path_to_root_predicted = tree_copy.paths_to_root[predicted_id]
                path_to_root_o = tree_copy.paths_to_root[o_leave_id]
                # print(path_to_root_predicted)
                # print(path_to_root_o)
                common_path_len = 0
                for i in range(min(len(path_to_root_predicted), len(path_to_root_o))):
                    if path_to_root_predicted[-i-1] == path_to_root_o[-i-1]:
                        common_path_len += 1
                    else:
                        break
                weighted_outputs[label] = common_path_len

    if nr_outs > 0:
        weighted_outputs = sorted(list(weighted_outputs.items()), key=lambda w : w[1], reverse=True)[0:nr_outs]
        weighted_outputs = dict(weighted_outputs)
    weight_sum = sum(weighted_outputs.values())

    shortest_common_path = min(weighted_outputs.values())
    weight_sum -= len(weighted_outputs) * shortest_common_path
    # print(weighted_outputs)
    if weight_sum == 0:
        for label in weighted_outputs:
            weighted_outputs[label] = 1/len(weighted_outputs)
    else:
        # print("HERE")
        for label in weighted_outputs:
            weighted_outputs[label] = (weighted_outputs[label] - shortest_common_path + 1)/ weight_sum
            # weighted_outputs[label] = (weighted_outputs[label] / weight_sum)
    return weighted_outputs

def run_episode(env, agent, input_map, tree_copy, ensemble_scheduler : ProbabilisticEnsembleScheduler,
                dt, transformer, nr_outputs):
    nr_agree = 0
    steps = 0
    orig_obs = env.reset()
    conc_obs = orig_obs.reshape(1, -1).astype(float)
    conc_obs = transformer.transform(conc_obs)
    obs = f'c{dt.apply(conc_obs)[0]}'

    weighted_clusters = {obs : 1} #compute_weighted_outs(conc_obs,dt,possible_outs)
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
        reward += rew
        conc_obs = orig_obs.reshape(1, -1).astype(float)
        conc_obs = transformer.transform(conc_obs)
        obs = f'c{dt.apply(conc_obs)[0]}'

        possible_outs = ensemble_scheduler.possible_outputs(action)
        weighted_clusters = compute_weighted_outs(tree_copy, obs,possible_outs, nr_outputs)
        ensemble_scheduler.step_to(action, weighted_clusters, obs)
        #env.render()
        if done:
            return reward, nr_agree/steps, reward > 1

num_traces_dt = 6000
max_leaves = 512
num_traces = 2300
scale = True
ensemble_name = "ensemble_all"
target_label = "succ"

if scale:
    transformer = FunctionTransformer(add_features)
else:
    transformer = FunctionTransformer()

import sys
sys.setrecursionlimit(2000)

environment = 'LunarLander-v2'
aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.7-linux64/bin/prism"
agent_steps = 0
truly_probabilistic = True
max_state_size = 10
count_misses = False
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}
dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

sched_name = f"prob_sched_{ensemble_name}_{num_traces_dt}_{num_traces}_{scale}_{max_leaves}"
ensemble_scheduler = load(sched_name)
if ensemble_scheduler is None:
    mdp_ensemble = load_mdp_ensemble(environment,ensemble_name, num_traces, scale, max_leaves)
    ensemble_scheduler = ProbabilisticEnsembleScheduler(mdp_ensemble,target_label,input_map,truly_probabilistic,
                                                        max_state_size,count_misses)
    save(ensemble_scheduler, sched_name)

ensemble_scheduler.set_max_state_size(max_state_size)
ensemble_scheduler.count_misses = count_misses
ensemble_scheduler.truly_probabilistic = truly_probabilistic

dt = load(f'dt_{environment}_{num_traces_dt}_{max_leaves}_{scale}')

env = gym.make(environment)
tree_copy_file_name = f"tree_copy_{num_traces_dt}_{max_leaves}_{num_traces}_{scale}"
tree_copy = load(tree_copy_file_name)
if tree_copy is None:
    tree_copy = build_tree_copy(dt.tree_)
    tree_copy.compute_aux_information()
    save(tree_copy,tree_copy_file_name)

nr_test_episodes = 5

for truly_probabilistic in [False,True]:
    # for c_misses,n_outputs, state_size in [(False,6,2),(False,10,4)]:
    for c_misses in [False,True]:
        for n_outputs in range(4,128,4): #[6,10,18]:
            for state_size in range(4,128,4): #[10,20]:
                print(f"T-Prob:{truly_probabilistic}, c-misses:{c_misses}, nr-out: {n_outputs}, states:{state_size}")
                ensemble_scheduler.set_max_state_size(state_size)
                ensemble_scheduler.count_misses = c_misses
                ensemble_scheduler.truly_probabilistic = truly_probabilistic
                avg_reward = 0
                success_rate = 0
                avg_agreement = 0
                for i in range(nr_test_episodes):
                    reward, agreement, success = \
                        run_episode(env,dqn_agent,input_map, tree_copy,ensemble_scheduler, dt, transformer,
                                    nr_outputs=n_outputs) #num_clusters)
                    avg_reward += reward
                    success_rate += success
                    avg_agreement += agreement
                avg_reward /= nr_test_episodes
                success_rate /= nr_test_episodes
                avg_agreement /= nr_test_episodes
                print(f"Reward: {avg_reward}, success: {success_rate}, agreement: {avg_agreement}")

