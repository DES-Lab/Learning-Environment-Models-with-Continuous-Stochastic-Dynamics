import os
from collections import defaultdict
import random
from statistics import mean

import gym
import numpy as np
from aalpy.learning_algs import run_JAlergia
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, FunctionTransformer
from stable_baselines3 import DQN

from agents import load_agent
from ensemble_creation import compute_ensemble_mdp
from utils import load, save, save_samples_to_file, get_traces_from_policy
import graphviz
from sklearn import tree

def remove_features(x):
    transformed = np.zeros((x.shape[0],3))
    # transformed[:x.shape[0], :x.shape[1]] = x
    transformed[:, 0] = x[:, 0]
    transformed[:, 1] = x[:, 1]
    transformed[:, 2] = x[:, 4]

    return transformed

def change_features(x):
    transformed = np.zeros((x.shape[0],4))
    transformed[:, 0] = x[:, 0] + x[:,2]
    transformed[:, 1] = x[:, 1] + x[:,3]
    transformed[:, 2] = x[:, 4] + x[:,5]
    transformed[:, 3] = x[:, 6] + x[:,7]
    return transformed

def add_features(x):
    transformed = np.zeros((x.shape[0],x.shape[1] + 6))
    transformed[:x.shape[0], :x.shape[1]] = x
    transformed[:,6] = transformed[:,6] + transformed[:,7]
    transformed[:,7] = transformed[:,0] + transformed[:,2]
    transformed[:,8] = transformed[:,1] + transformed[:,3]
    transformed[:,9] = transformed[:,4] + transformed[:,5]
    # transformed[:,10] = transformed[:,2] + transformed[:,5] # x velocity + angular velocity
    transformed[:,10] = transformed[:,2] * transformed[:,5] # x velocity * angular velocity
    transformed[:, 11] = transformed[:, 0] ** 2
    transformed[:, 12] = transformed[:, 1] ** 2
    transformed[:, 13] = (transformed[:, 0] - transformed[:, 1]) ** 2
    return transformed
class Tree:
    def __init__(self, root):
        self.paths_to_root = dict()
        self.root = root
        self.id_to_node = dict()
        self.id_to_node[0] = root
        self.inv_depths_to_node_ids = None
        self.leaf_ids_below_nodes = dict()
        self.largest_depth = 1
        self.leaf_ids_to_class_probs = dict()

    def add(self,id, node):
        self.id_to_node[id] = node

    def compute_aux_information(self, base_tree):
        self.inv_depths_to_node_ids = defaultdict(list)
        for id,node in self.id_to_node.items():
            depth = node.inverse_depth()
            self.inv_depths_to_node_ids[depth].append(id)
            self.largest_depth = max(self.largest_depth,depth)
        for id, node in self.id_to_node.items():
            self.leaf_ids_below_nodes[id] = node.leaf_ids_below()
            self.paths_to_root[id] = node.path_to_root()
            if node.is_leaf():
                class_probs = list(base_tree.value[id][0])
                normalizer = sum(class_probs)
                for i in range(len(class_probs)):
                    class_probs[i] /= normalizer
                self.leaf_ids_to_class_probs[id] = class_probs
                print(f"Class probabilities in leaf {id}: {class_probs}")

class TreeNode:
    def __init__(self, id, parent):
        self.id = id
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.path_to_root_cache = None

    def path_to_root(self):
        if self.path_to_root_cache:
            return self.path_to_root_cache
        else:
            if self.parent is None:
                return []
            else:
                self.path_to_root_cache = [self.parent.id] + self.parent.path_to_root()
                return self.path_to_root_cache
    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    def inverse_depth(self):
        if self.is_leaf():
            return 0
        elif self.left_child is None:
            return 1 + self.right_child.inverse_depth()
        elif self.right_child is None:
            return 1 + self.left_child.inverse_depth()
        else:
            return 1 + max(self.left_child.inverse_depth(), self.right_child.inverse_depth())
    
    def leaf_ids_below(self):
        if self.is_leaf():
            return [self.id]
        elif self.left_child is None:
            return self.right_child.leaf_ids_below()
        elif self.right_child is None:
            return self.left_child.leaf_ids_below()
        else:
            return self.left_child.leaf_ids_below() + self.right_child.leaf_ids_below()

    def to_string(self):
        string_rep = ""
        if self.left_child is not None:
            string_rep += f"{self.id} -> {self.left_child.id}" + os.linesep
            string_rep += self.left_child.to_string()

        if self.right_child is not None:
            string_rep += f"{self.id} -> {self.right_child.id}" + os.linesep
            string_rep += self.right_child.to_string()
        return string_rep
def build_tree_copy_rec(tree, tree_copy, parent, parent_node_id):
    left_child_id = tree.children_left[parent_node_id]
    right_child_id = tree.children_right[parent_node_id]
    left_child = None
    right_child = None
    if left_child_id != -1:
        left_child = TreeNode(left_child_id,parent)
        tree_copy.add(left_child_id,left_child)
        left_left, left_right = build_tree_copy_rec(tree, tree_copy, left_child,left_child_id)
        left_child.left_child = left_left
        left_child.right_child = left_right
    if right_child_id != -1:
        right_child = TreeNode(right_child_id,parent)
        tree_copy.add(right_child_id,right_child)
        right_left, right_right = build_tree_copy_rec(tree, tree_copy, right_child,right_child_id)
        right_child.left_child = right_left
        right_child.right_child = right_right
    return left_child,right_child
def build_tree_copy(tree):
    root = TreeNode(0, None)
    tree_copy = Tree(root)
    left_child, right_child = build_tree_copy_rec(tree, tree_copy, root,0)
    root.left_child = left_child
    root.right_child = right_child
    return tree_copy
def map_dt_indexes_to_traces(traces,action_map,env_name,dt):
    print('Cluster labels computed')
    alergia_dataset = []
    print(f'Creating Alergia Samples. x {len(traces)}')
    for sample in traces:
        alergia_sample = ['INIT']
        for state, action, reward, done in sample:
            cluster_label = f'c{dt.apply(state.reshape(1,-1))[0]}'

            if "Lunar" in env_name and  reward == 100 and done:
                alergia_sample.append(
                    (action_map[int(action)], f"{cluster_label}__succ__pos"))
            elif "Lunar" in env_name and reward == -100 and done:
                alergia_sample.append(
                    (action_map[int(action)], f"{cluster_label}__bad"))
            elif "Lunar" in env_name and reward >= 10 and done:
                alergia_sample.append(
                    (action_map[int(action)], f"{cluster_label}__pos"))
            elif "Lunar" in env_name and reward >= 10 and done:
                alergia_sample.append(
                    (action_map[int(action)], f"{cluster_label}__pos"))
            elif "Mountain" in env_name and done and len(alergia_sample) < 200 and state[0][0] > 0:
                alergia_sample.append(
                    (action_map[int(action)], f"{cluster_label}__succ"))
            else:
                alergia_sample.append(
                    (action_map[int(action)], cluster_label if not done else 'DONE'))  # action_map[int(action)]

        alergia_dataset.append(alergia_sample)

    print('Cluster labels replaced')
    return alergia_dataset


def evaluate_on_environment(env, dt,transformer, num_episodes=100, render=False):
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        ep_rew = 0
        while True:
            conc_obs = obs.reshape(1, -1)
            conc_obs = transformer.transform(conc_obs)
            a = dt.predict(conc_obs)[0]
            obs, rew, done, info = env.step(a)
            if render:
                env.render()
            ep_rew += rew
            if done:
                all_rewards.append(ep_rew)
                break
    print('Mean reward:', mean(all_rewards))


def get_observation_action_pairs(env, dqn_agent, num_ep=1000, randomness_probs=[0]):

    sample_num = 0
    observation_actions_pairs = []
    random_i = 0
    for _ in range(num_ep):
        sample_num += 1
        curr_randomness = randomness_probs[random_i]
        random_i = (random_i + 1) % len(randomness_probs)
        if sample_num % 100 == 0:
            print(sample_num)
        obs = env.reset()
        while True:
            # action, state = dqn_agent.predict(obs)

            action, _ = dqn_agent.predict(obs)
            observation_actions_pairs.append((obs, action))
            # with some probability we override the action
            if random.random() < curr_randomness:
                action = random.randint(0, len(action_map) - 1)

            obs, reward, done, _ = env.step(action)
            if done:
                break

    return observation_actions_pairs
def learn_mdp(traces, input_completeness="sink_state"):
    cluster_samples_file_name = 'cluster_samples.txt'
    save_samples_to_file(traces, cluster_samples_file_name)
    mdp = run_JAlergia(cluster_samples_file_name, 'mdp', 'alergia.jar', heap_memory='-Xmx6G',
                       optimize_for="accuracy", eps=0.005)
    # delete_file(cluster_samples)
    mdp.make_input_complete(input_completeness)
    return mdp

def create_dt(obs_action_pairs,transformer, max_leaf_nodes):
    x, y = [i[0] for i in obs_action_pairs], [i[1] for i in obs_action_pairs]
    x = np.array(x)
    x = transformer.transform(x)
    y = np.array(y)
    dt = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    dt.fit(x, y)
    return dt

if __name__ == "__main__":
    env_name = "LunarLander-v2"
    num_traces_dt = 6000
    num_traces_env_learn = 2000
    load_observations = True
    max_leaf_nodes = 256
    random_obs_pairs = [0, 0.1, 0.2, 0.3,0.4]
    scale = True
    if scale:
        transformer = FunctionTransformer(change_features)
    else:
        transformer = FunctionTransformer()
    env = gym.make(env_name)

    if env_name == "LunarLander-v2":
        dqn_agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
        action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}
    else:
        dqn_agent = load_agent('DBusAI/DQN-MountainCar-v0-v2', 'DQN-MountainCar-v0.zip', DQN)
        action_map = {0: 'left', 1: 'no_action', 2: 'right'}

    if load_observations and os.path.exists(f'pickle_files/obs_actions_pairs_{env_name}_{num_traces_dt}.pickle'):
        obs_action_pairs = load(f'obs_actions_pairs_{env_name}_{num_traces_dt}')
        if obs_action_pairs:
            print('Observation actions pairs loaded')
    else:
        print('Computing observation action pairs')
        obs_action_pairs = get_observation_action_pairs(env, dqn_agent, num_traces_dt, randomness_probs=random_obs_pairs)
        save(obs_action_pairs, path=f'obs_actions_pairs_{env_name}_{num_traces_dt}')

    dt = create_dt(obs_action_pairs,transformer,max_leaf_nodes=max_leaf_nodes)
    evaluate_on_environment(env, dt, transformer, render=False)
    save(dt, f"dt_{env_name}_{num_traces_dt}_{max_leaf_nodes}_{scale}")

    trace_file = f"{env_name}_{num_traces_env_learn}_traces"
    traces = load(trace_file)
    if traces is None:
        # traces = [get_traces_from_policy(dqn_agent, env, num_traces, action_map, randomness_probs=[0, 0.05, 0.1, 0.15])]
        traces = [get_traces_from_policy(dqn_agent, env, num_traces_env_learn, action_map,
                                         # randomness_probs=[0, 0.05, 0.01, 0.02, 0.03,0.04], duplicate_action=False)]
                                         randomness_probs=[0, 0.05, 0.1, 0.15, 0.2, 0.25], duplicate_action=False)]
        save(traces,trace_file)
    traces = traces[0]
    transformed_traces = []
    for t in traces:
        transformed_trace = []
        for (obs, action, reward, done) in t:
            transformed_trace.append((transformer.transform(obs),action,reward,done))
        transformed_traces.append(transformed_trace)
    traces = transformed_traces
    alergia_traces = map_dt_indexes_to_traces(traces,action_map,env_name,dt)
    compute_ensemble_mdp(alergia_traces,suffix_strategy="all",optimize_for="accuracy",
                         input_completeness="sink_state",alergia_eps=0.005,
                         save_path_prefix=f"dt_ensemble_all_{env_name}_{num_traces_env_learn}_scale_{scale}_{max_leaf_nodes}",
                         depth=10, nr_traces_limit = 40000)
