import gym
from aalpy.learning_algs import run_Alergia
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from stable_baselines3 import DQN

from agents import load_agent
from discretization_pipeline import get_observations_and_actions, AutoencoderDimReduction, PipelineWrapper, get_k_means_clustering
from iterative_refinement import IterativeRefinement
from utils import get_traces_from_policy, create_abstract_traces

environment = "LunarLander-v2"
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}

agents = None
agent_names = None

if environment == 'LunarLander-v2':
    agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
else:
    exit(1)


num_traces = 2500
num_clusters = 256
scale = 'ae'

env = gym.make(environment, )
traces_file_name = f'{environment}_{num_traces}_traces'

traces = get_traces_from_policy(agent, env, num_episodes=10, agent_name='DQN' )

obs, actions = get_observations_and_actions(traces)

ae = AutoencoderDimReduction(4, 20, 'test')

dim_red_pipeline = PipelineWrapper('test', 10, [('scaler', StandardScaler()), ('autoencode', ae), ])

dim_red_pipeline.fit(obs, actions)

transformed = dim_red_pipeline.transform(obs)

k_means_clustering, cluster_labels = get_k_means_clustering(transformed, 10, dim_red_pipeline.pipeline_name)

print(cluster_labels[:20])
exit()

data = create_abstract_traces(traces, cluster_labels)
model = run_Alergia(data, automaton_type='mdp')

ir = IterativeRefinement(env, model, traces, dim_red_pipeline, k_means_clustering, )

ir.iteratively_refine_model(10, 100)
