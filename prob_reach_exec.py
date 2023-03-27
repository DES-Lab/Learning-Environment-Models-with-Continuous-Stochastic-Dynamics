from collections import defaultdict

import gym

from agents import load_agent

import aalpy.paths

from aalpy.learning_algs import run_JAlergia

from prism_scheduler import PrismInterface, ProbabilisticScheduler
from prob_reach_checking import compute_mdp, prob_reach_checking
from stable_baselines3 import DQN, A2C, PPO

from utils import load

num_traces = 2500
num_clusters = 256
scale = True
include_reward = False
environment = 'LunarLander-v2'
aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.7-linux64/bin/prism"
agent_steps = 0
# if agent_steps > 0:
# dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
dqn_agents = dict()
# dqn_agents["A2C"] = load_agent('araffin/ppo-LunarLander-v2', 'ppo-LunarLander-v2.zip', A2C)
dqn_agents["AR-A2C"] = load_agent('araffin/ppo-LunarLander-v2', 'ppo-LunarLander-v2.zip', A2C)
# dqn_agents["DQN"] = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
dqn_agents["SB-PPO"] = load_agent('sb3/ppo-LunarLander-v2', 'ppo-LunarLander-v2.zip', PPO)
dqn_agents["SB-A2C"] = load_agent('sb3/a2c-LunarLander-v2', 'a2c-LunarLander-v2.zip', A2C)
dqn_agents["MC-PPO"] = load_agent('mcaoun/ppo-LunarLander-v2', 'ppo_lunar_agent.zip', PPO)

# else:
#     dqn_agent = None

action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
input_map = {v: k for k, v in action_map.items()}

alergia_eps = 0.05
jalergia_samples = 'alergiaSamples.txt'

# action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
# input_map = {v: k for k, v in action_map.items()}

clustering_function = load(f'{environment}_k_means_scale_{scale}_{num_clusters}_{num_traces}')
scaler = load(f"power_scaler_{environment}_{num_traces}")

env = gym.make(environment)
cluster_center_cache = dict()
nr_outputs = num_clusters

model_refinements = 5
n_traces_per_ref = 500
nr_reaches_target = 200
test_points = range(9,255,20)
test_info = defaultdict(list)
mdp = compute_mdp(jalergia_samples, alergia_eps)
for i in test_points: #num_clusters):
    for agent_name in ["AR-A2C","SB-PPO", "SB-A2C","MC-PPO"]:
        dqn_agent = dqn_agents[agent_name]
        goal = f"c{i}"
        crashes, nr_tries, nr_reaches_actual = prob_reach_checking(env, scaler, clustering_function, n_traces_per_ref,
                                                               model_refinements, nr_reaches_target, input_map,
                                                                   jalergia_samples, nr_outputs, alergia_eps, goal,
                                                               environment_name = environment, agent = dqn_agent,
                                                                   mdp = mdp, refine_mdp=False)
        # test_info[goal] = (crashes, nr_tries, nr_reaches_actual)
        test_info[agent_name].append((i,crashes, nr_tries, nr_reaches_actual))
        print(f"Testing {i} for {agent_name}: {crashes/nr_reaches_actual} after {nr_tries} tries found {nr_reaches_actual}")

for agent_name in ["AR-A2C","SB-PPO","SB-A2C","MC-PPO"]:
    print(f"Summary for {agent_name}")
    for result in test_info[agent_name]:
        (i, crashes, nr_tries, nr_reaches_actual) = result
        goal = f"c{i}"
        # (crashes, nr_tries, nr_reaches_actual) = test_info[goal]
        print(f"Testing {i}: {crashes/nr_reaches_actual} after {nr_tries} tries found {nr_reaches_actual}")