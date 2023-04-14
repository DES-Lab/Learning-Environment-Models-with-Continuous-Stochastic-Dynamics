import gym
from aalpy.learning_algs import run_Alergia, run_JAlergia
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from stable_baselines3 import DQN

from agents import load_agent
from discretization_pipeline import get_observations_and_actions, AutoencoderDimReduction, PipelineWrapper, get_k_means_clustering
from iterative_refinement import IterativeRefinement
from utils import get_traces_from_policy, create_abstract_traces
import aalpy.paths

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"

env_name = "LunarLander-v2"


# test = ['a', 'a', 'a', 'a', 'b', 'b', 'a', 'a', 'a', 'c']
# res = []
# cc = 1
# for i in range(len(test)):
#     if i == 0:
#         res.append((test[i], cc))
#     else:
#         res.append((test[i], res[i-1][1]+1 if test[i] == res[i-1][0] else 1))
#
# print(test)
# print(res)
# exit()
agents = None
agent_names = None

agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)

num_traces = 2500
num_clusters = 128

env = gym.make(env_name, )
traces_file_name = f'{env_name}_{num_traces}_traces'

traces = get_traces_from_policy(agent, env, num_episodes=num_traces, agent_name='DQN' )

obs, actions = get_observations_and_actions(traces)

ae = AutoencoderDimReduction(4, 10,)

dim_red_pipeline = PipelineWrapper(env_name, num_traces, [('scaler', StandardScaler()), ('pca_4', PCA(n_components=4)),])
dim_red_pipeline.fit(obs, actions)

transformed = dim_red_pipeline.transform(obs)
k_means_clustering, cluster_labels = get_k_means_clustering(transformed, num_clusters, dim_red_pipeline.pipeline_name)

abstract_traces = create_abstract_traces(traces, cluster_labels, count_same_cluster=True)
for i in abstract_traces:
    i.pop(0)

model = run_JAlergia(abstract_traces, automaton_type='smm', path_to_jAlergia_jar='alergia.jar', optimize_for='accuracy')
print(model)
mdp = model.to_mdp()

ir = IterativeRefinement(env, model, abstract_traces, dim_red_pipeline, k_means_clustering, )

ir.iteratively_refine_model(10, 100)
