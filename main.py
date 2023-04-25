import aalpy.paths
import gym
from aalpy.learning_algs import run_JAlergia
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, PowerTransformer
from stable_baselines3 import DQN, PPO

from agents import load_agent
from discretization_pipeline import get_observations_and_actions, AutoencoderDimReduction, PipelineWrapper, \
    get_k_means_clustering, LunarLanderManualDimReduction
from iterative_refinement import IterativeRefinement
from utils import get_traces_from_policy
from trace_abstraction import create_abstract_traces

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"

env_name = "MountainCar-v0"

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

num_clusters_per_env = {'Acrobot-v1': 128, 'LunarLander-v2': 128,
                        'MountainCar-v0': 128, 'CartPole-v1':128}

num_traces = 2500
num_clusters = num_clusters_per_env[env_name]
include_randomness_in_sampling = True

env = gym.make(env_name, )
traces_file_name = f'{env_name}_{num_traces}_traces'

randomness = (0, 0.05, 0.1, 0.15, 0.2) if include_randomness_in_sampling else (0,)

traces = get_traces_from_policy(agent, env, num_episodes=num_traces, agent_name='DQN',
                                randomness_probabilities=(0, 0.05, 0.1, 0.15, 0.2))


obs, actions = get_observations_and_actions(traces)

dim_red_pipeline = None
if env_name == 'MountainCar-v0':
    dim_red_pipeline = PipelineWrapper(env_name, num_traces,
                                       [('powerTransformer', PowerTransformer()), ],)
if env_name == 'LunarLander-v2':
    dim_red_pipeline = PipelineWrapper(env_name, num_traces,
                                       [('scaler', StandardScaler),
                                        ('manualMapper', LunarLanderManualDimReduction()), ],)
if env_name == 'CartPole-v1':
    dim_red_pipeline = PipelineWrapper(env_name, num_traces,
                                       [('powerTransformer', PowerTransformer()), ],)
if env_name == 'Acrobot-v1':
    dim_red_pipeline = PipelineWrapper(env_name, num_traces,
                                       [('scaler', StandardScaler),
                                        ('lda2', LinearDiscriminantAnalysis(n_components=2))],)

# fit and transform concrete traces
dim_red_pipeline.fit(obs, actions)
transformed = dim_red_pipeline.transform(obs)
# get clustering function
k_means_clustering, cluster_labels = get_k_means_clustering(transformed, num_clusters, dim_red_pipeline.pipeline_name)
# create abstract traces
abstract_traces = create_abstract_traces(env_name, traces, cluster_labels)
# get initial model
model = run_JAlergia(abstract_traces, automaton_type='mdp', path_to_jAlergia_jar='alergia.jar', heap_memory='-Xmx12G',
                     optimize_for='accuracy')

ir = IterativeRefinement(env, env_name, model, abstract_traces, dim_red_pipeline, k_means_clustering,
                         scheduler_type='probabilistic')

# run iterative refinement
results = ir.iteratively_refine_model(50, 10)
