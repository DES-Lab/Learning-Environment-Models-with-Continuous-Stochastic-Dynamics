import os
from statistics import mean

import gym
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, FunctionTransformer, \
    PolynomialFeatures, SplineTransformer
from stable_baselines3 import DQN

from agents import load_agent
from utils import load, save
import graphviz
from sklearn import tree

def change_features(x):
    transformed = np.zeros((x.shape[0],3))
    # transformed[:x.shape[0], :x.shape[1]] = x
    transformed[:, 0] = x[:, 0] + x[:,2]
    transformed[:, 1] = x[:, 1] + x[:,3]
    transformed[:, 2] = x[:, 4] + x[:,5]
    return transformed

def add_more_features(x):
    transformed = np.zeros((x.shape[0],x.shape[1] + 6))
    transformed[:x.shape[0], :x.shape[1]] = x
    transformed[:,6] = transformed[:,6] + transformed[:,7]
    transformed[:,7] = transformed[:,0] + transformed[:,2]
    transformed[:,8] = transformed[:,1] + transformed[:,3]
    transformed[:,9] = transformed[:,4] + transformed[:,5]
    transformed[:,10] = transformed[:,2] * transformed[:,5] # x velocity * angular velocity
    transformed[:, 11] = transformed[:, 0] ** 2
    transformed[:, 12] = transformed[:, 1] ** 2
    transformed[:, 13] = (transformed[:, 0] - transformed[:, 1]) ** 2
    return transformed

def evaluate_on_environment(env, dt, scaler, num_episodes=100, render=False):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        ep_rew = 0
        while True:
            conc_obs = obs.reshape(1, -1)
            # print("==="*30)
            # print(dt.decision_path(conc_obs))
            # print(dt.apply(conc_obs))
            # print(dt.predict(conc_obs))
            if scaler is not None:
                conc_obs = scaler.transform(conc_obs)
            a = dt.predict(conc_obs)[0]
            obs, rew, done, info = env.step(a)
            if render:
                env.render()
            ep_rew += rew
            if done:
                all_rewards.append(ep_rew)
                break
    print('Mean reward:', mean(all_rewards))


def get_observation_action_pairs(env, num_ep=1000):
    dqn_agent = load_agent("araffin/dqn-LunarLander-v2", 'dqn-LunarLander-v2.zip', DQN)

    sample_num = 0
    observation_actions_pairs = []
    for _ in range(num_ep):
        sample_num += 1
        if sample_num % 100 == 0:
            print(sample_num)
        obs = env.reset()
        while True:
            action, state = dqn_agent.predict(obs)
            observation_actions_pairs.append((obs, action))
            obs, reward, done, _ = env.step(action)
            if done:
                break

    return observation_actions_pairs

if __name__ == "__main__":

    env_name = "LunarLander-v2"
    action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}

    num_traces = 6000
    load_observations = True

    env = gym.make(env_name)

    if load_observations and os.path.exists(f'pickle_files/obs_actions_pairs_{num_traces}.pickle'):
        obs_action_pairs = load(f'obs_actions_pairs_{num_traces}')
        if obs_action_pairs:
            print('Observation actions pairs loaded')
    else:
        print('Computing observation action pairs')
        obs_action_pairs = get_observation_action_pairs(env, num_traces)
        save(obs_action_pairs, path=f'obs_actions_pairs_{num_traces}')

    x,y = [i[0] for i in obs_action_pairs], [i[1] for i in obs_action_pairs]
    x = np.array(x)
    y = np.array(y)

    scaler = make_pipeline(
        FunctionTransformer(change_features)
        #FunctionTransformer(add_more_features)) # PowerTransformer() #PCA(n_components=6) #None # StandardScaler()
    )
    if scaler is not None:
        x = scaler.fit_transform(x)
    dt = tree.DecisionTreeClassifier(max_leaf_nodes=32) #, ccp_alpha=0.0001)
    dt.fit(x, y)

    # copy_root = build_tree_copy(dt.tree_)
    # print(copy_root.to_string())
    print(dt.get_n_leaves())
    # to get a leaf if of leaf that meade the decision
    # df.apply(obs)


    evaluate_on_environment(env, dt, scaler, render=False)
    # exit()
    dot_data = tree.export_graphviz(dt)
    graph = graphviz.Source(dot_data)
    #
    graph.render()
