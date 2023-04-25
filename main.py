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

env_name = "LunarLander-v2"

agents = None
agent_names = None

if env_name == 'Acrobot-v1':
    agent = load_agent('sb3/ppo-Acrobot-v1', 'ppo-Acrobot-v1.zip', PPO)
elif env_name == 'LunarLander-v2':
    agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
elif env_name == 'MountainCar-v0':  # power scaler, 48 clusters
    agent = load_agent('sb3/dqn-MountainCar-v0', 'dqn-MountainCar-v0.zip', DQN)
    # dqn_agent2 = load_agent('DBusAI/DQN-MountainCar-v0', 'DQN-MountainCar-v0.zip', DQN)
    # ppo_agent = load_agent('vukpetar/ppo-MountainCar-v0', 'ppo-mountaincar-v0.zip', PPO)
    # ppo_agent2 = load_agent('format37/PPO-MountainCar-v0', 'PPO-Mlp.zip', PPO)
elif env_name == 'CartPole-v1':
    agent = load_agent('sb3/ppo-CartPole-v1', 'ppo-CartPole-v1.zip', PPO)
    # dqn_agent = load_agent('sb3/dqn-CartPole-v1', 'dqn-CartPole-v1.zip', DQN)
else:
    print('Env not supported')
    assert False

num_traces = 2500
num_clusters = 64
count_observations = False

env = gym.make(env_name, )
traces_file_name = f'{env_name}_{num_traces}_traces'

traces = get_traces_from_policy(agent, env, num_episodes=num_traces, agent_name='DQN',)
                                #randomness_probabilities=(0, 0.05, 0.1, 0.15, 0.2))

prefix_size = 0
for i, t in enumerate(traces):
    traces[i] = t[prefix_size:]

obs, actions = get_observations_and_actions(traces)

# transformed = obs
# ae = AutoencoderDimReduction(4, 10,)
dim_red_pipeline = PipelineWrapper(env_name, num_traces,
                                   [('manual_dim_reduction', LunarLanderManualDimReduction()), ],
                                   # ('lda', LinearDiscriminantAnalysis(n_components=3),)],
                                   prefix_len=prefix_size)

dim_red_pipeline.fit(obs, actions)

transformed = dim_red_pipeline.transform(obs)

k_means_clustering, cluster_labels = get_k_means_clustering(transformed, num_clusters, dim_red_pipeline.pipeline_name)

abstract_traces = create_abstract_traces(env_name, traces, cluster_labels, count_same_cluster=count_observations)

model = run_JAlergia(abstract_traces, automaton_type='mdp', path_to_jAlergia_jar='alergia.jar', heap_memory='-Xmx12G',
                     optimize_for='accuracy')

ir = IterativeRefinement(env, env_name, model, abstract_traces, dim_red_pipeline, k_means_clustering,
                         scheduler_type='probabilistic', count_observations=count_observations)

results = ir.iteratively_refine_model(50, 200)

ir.model.save(f'final_model_{env_name}')
