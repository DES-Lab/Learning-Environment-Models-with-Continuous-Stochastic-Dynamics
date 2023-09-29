import argparse

import aalpy.paths
import gym
from aalpy.learning_algs import run_JAlergia
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PowerTransformer, StandardScaler
from stable_baselines3 import DQN, PPO

from agents import load_agent
from discretization_pipeline import get_observations_and_actions, PipelineWrapper, \
    get_k_means_clustering, LunarLanderManualDimReduction, AcrobotManualDimReduction
from iterative_refinement import IterativeRefinement
from trace_abstraction import create_abstract_traces
from utils import get_traces_from_policy
from random import seed

parser = argparse.ArgumentParser(
    prog='Experiment Runner: Automata Learning for Continuous Sotochastic Enviroments',
    # description='What the program does',
)

parser.add_argument('--path_to_prism', required=True, help='Path to prism installation '
                                                          '(bin/prism.bat for Win or bin/prism for Linux/Mac)')
parser.add_argument('--path_to_alergia', required=True, help='Path to alergia.jar')
parser.add_argument('--env_name', required=True,
                    help='Name of the enviroment. One of {"Acrobot", "LunarLander", "MountainCar", "Cartpole"}')
parser.add_argument('--dim_reduction', required=True,
                    help='For LunarLander: "lda" or "manual", '
                         'for acrobot "lda" or "manual", for MountaiCar and Cartopole "pt"')
parser.add_argument('--num_initial_traces', required=True, help='Number of traces obtrained by RL agent')
parser.add_argument('--num_clusters', required=True, help='Number of clusters')
parser.add_argument('--num_iterations', required=True, help='Number of refinement iterations')
parser.add_argument('--episodes_in_iter', required=True, help='Number of refinement iterations')
parser.add_argument('--exp_prefix', required=True, help='prefix which will be prepended to experiment results pickle')
parser.add_argument('--seed', required=True, help='Seed used to make experiments reproducible.')
args = parser.parse_args()

aalpy.paths.path_to_prism = args.path_to_prism

real_env_names_map = {'MountainCar' : 'MountainCar-v0', 'Acrobot': 'Acrobot-v1', 'LunarLander': 'LunarLander-v2', 'Cartpole':'Cartpole-v1'}
env_name = real_env_names_map[args.env_name]

seed(int(args.seed))

agents = None
agent_names = None

if env_name == 'Acrobot-v1':
    agent = load_agent('sb3/ppo-Acrobot-v1', 'ppo-Acrobot-v1.zip', PPO)
elif env_name == 'LunarLander-v2':
    agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
elif env_name == 'MountainCar-v0':
    agent = load_agent('sb3/dqn-MountainCar-v0', 'dqn-MountainCar-v0.zip', DQN)
elif env_name == 'CartPole-v1':
    agent = load_agent('sb3/ppo-CartPole-v1', 'ppo-CartPole-v1.zip', PPO)
else:
    print('Env not supported')
    assert False

num_clusters_per_env = args.num_clusters

num_traces = int(args.num_initial_traces)
num_clusters = int(args.num_clusters)
include_randomness_in_sampling = True
load_all = False

env = gym.make(env_name, )
traces_file_name = f'{env_name}_{num_traces}_traces'

randomness = (0, 0.05, 0.1, 0.15, 0.2) if include_randomness_in_sampling else (0,)

traces = get_traces_from_policy(agent, env, num_episodes=num_traces, agent_name='DQN',
                                randomness_probabilities=(0, 0.05, 0.1, 0.15, 0.2), load_traces=load_all)

obs, actions = get_observations_and_actions(traces)

dim_red_pipeline = None
if env_name == 'MountainCar-v0':
    assert args.dim_reduction == 'pt'
    dim_red_pipeline = PipelineWrapper(env_name, num_traces,
                                       [('powerTransformer', PowerTransformer()), ], load_pipeline=load_all)
if env_name == 'LunarLander-v2':
    if args.dim_reduction == 'manual':
        dim_red_pipeline = PipelineWrapper(env_name, num_traces, [
            ('manualMapper', LunarLanderManualDimReduction()),
            ('powerTransformer', PowerTransformer()), ], load_pipeline=load_all)
    elif args.dim_reduction == 'lda':
        dim_red_pipeline = PipelineWrapper(env_name, num_traces, [
            ('lda_2', LinearDiscriminantAnalysis(n_components=3)),
            ('powerTransformer', PowerTransformer()),],
            load_pipeline=load_all)
    else:
        print('Dimensionality Reduction Pipeline not supported. Check help message.')
        assert False

if env_name == 'CartPole-v1':
    assert args.dim_reduction == 'pt'
    dim_red_pipeline = PipelineWrapper(env_name, num_traces,
                                       [('powerTransformer', PowerTransformer()), ], load_pipeline=load_all)

if env_name == 'Acrobot-v1':
    if args.dim_reduction == 'manual':
        dim_red_pipeline = PipelineWrapper(env_name, num_traces,[
                                            ('manualMapper', AcrobotManualDimReduction()),],)
    elif args.dim_reduction == 'lda':
        dim_red_pipeline = PipelineWrapper(env_name, num_traces, [('powerTransformer', PowerTransformer()),
                                                                  ('lda_2', LinearDiscriminantAnalysis(n_components=2))],
                                           load_pipeline=load_all)
    else:
        print('Dimensionality Reduction Pipeline not supported. Check help message.')
        assert False

# fit and transform concrete traces
dim_red_pipeline.fit(obs, actions)
transformed = dim_red_pipeline.transform(obs)
# get clustering function
k_means_clustering, cluster_labels = get_k_means_clustering(transformed, num_clusters, dim_red_pipeline.pipeline_name,
                                                            load_fun=load_all)
# create abstract traces
abstract_traces = create_abstract_traces(env_name, traces, cluster_labels)
# get initial model
model = run_JAlergia(abstract_traces, automaton_type='mdp', path_to_jAlergia_jar=args.path_to_alergia, heap_memory='-Xmx12G',)

ir = IterativeRefinement(env, env_name, model, abstract_traces, dim_red_pipeline, k_means_clustering,
                         scheduler_type='probabilistic', experiment_name_prefix=args.exp_prefix)

# run iterative refinement
results = ir.iteratively_refine_model(int(args.num_iterations), int(args.episodes_in_iter))
