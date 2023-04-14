import gym
from aalpy.learning_algs import run_Alergia
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN
from clustering import get_k_means_clustering, get_mean_shift_clustering

from agents import load_agent
from dim_recution import get_observations_and_actions, ManualLunarLanderDimReduction, AutoencoderDimReduction, \
    PcaDimReduction, LdaDimReduction
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

# print('Agents loaded')

num_traces = 2500
num_clusters = 256
scale = 'ae'

env = gym.make(environment, )
traces_file_name = f'{environment}_{num_traces}_traces'

traces = get_traces_from_policy(agent, env, num_episodes=10, agent_name='DQN' )


exit()
obs, actions = get_observations_and_actions(traces)

manual = ManualLunarLanderDimReduction('LunarLander-v2')
lda = LdaDimReduction()
pca = PcaDimReduction(n_dim=4)
ae = AutoencoderDimReduction(4, 20)

scaler = make_pipeline(StandardScaler(), ae)
scaler.fit(obs, actions)

transformed = scaler.transform(obs)

k_means_clustering, cluster_labels = get_k_means_clustering(transformed, 10)

data = create_abstract_traces(traces, cluster_labels)
model = run_Alergia(data, automaton_type='mdp')

ir = IterativeRefinement(env, model, traces, scaler, k_means_clustering, )

ir.iteratively_refine_model(10, 100)
