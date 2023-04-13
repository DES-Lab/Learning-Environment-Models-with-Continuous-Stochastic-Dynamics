import random

import gym
from aalpy.learning_algs import run_JAlergia, run_Alergia
from stable_baselines3 import DQN

from abstraction import compute_clustering_function_and_map_to_traces
from agents import get_lunar_lander_agents, load_agent
from dim_recution import get_observations_and_actions, ManualLunarLanderDimReduction, AutoencoderDimReduction, \
    PcaDimReduction, LdaDimReduction
from prism_scheduler import ProbabilisticScheduler, PrismInterface
from utils import load, save, delete_file, save_samples_to_file, get_traces_from_policy, create_abstract_traces

environment = "LunarLander-v2"
action_map = {0: 'no_action', 1: 'left_engine', 2: 'down_engine', 3: 'right_engine'}

agents = None
agent_names = None

if environment == 'LunarLander-v2':
    agent = load_agent('araffin/dqn-LunarLander-v2', 'dqn-LunarLander-v2.zip', DQN)
    # get_lunar_lander_agents(evaluate=False)
    # agents = [a[1] for a in agents_and_names]
    # agent_names = '_'.join(a[0] for a in agents_and_names)
else:
    exit(1)

# print('Agents loaded')

num_traces = 2500
num_clusters = 256
scale = 'ae'

env = gym.make(environment, )
traces_file_name = f'{environment}_{num_traces}_traces'

traces = get_traces_from_policy(agent, env, num_episodes=10, )

obs, actions = get_observations_and_actions(traces)

manual = ManualLunarLanderDimReduction('LunarLander-v2')
lda = LdaDimReduction()
pca = PcaDimReduction(n_dim=4)
ae = AutoencoderDimReduction(4, 20)

ae.fit(obs)

transformed = ae.transform(obs)

from clustering import get_k_means_clustering, get_mean_shift_clustering

cf, cl = get_k_means_clustering(transformed, 10)
cf2, cl2 = get_mean_shift_clustering(transformed, 0.5)

data = create_abstract_traces(traces, cl2)
model = run_Alergia(data, automaton_type='mdp')

import aalpy.paths

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.6/bin/prism.bat"


class IterativeRefinement:
    def __init__(self, env, initial_model, traces, dim_reduction_fun, clustering_fun, scale=False):
        self.env = env
        self.model = initial_model
        self.traces = traces
        self.dim_reduction_fun = dim_reduction_fun
        self.clustering_fun = clustering_fun
        self.use_scaler = scale

    def iteratively_refine_model(self, num_iterations, episodes_per_iteration):

        for refinement_iteration in range(num_iterations):
            model.make_input_complete('sink_state')
            scheduler = PrismInterface('GOAL', self.model).scheduler
            # scheduler = ProbabilisticScheduler(scheduler, truly_probabilistic=True)

            for _ in range(episodes_per_iteration):
                self.env.reset()
                ep_data = []
                while True:
                    action = scheduler.get_input()
                    observation, reward, done, _ = self.env.step(action)

                    abstract_obs = self.dim_reduction_fun.transform([observation])
                    abstract_obs = self.clustering_fun.predict(abstract_obs)
                    abstract_obs = f'c{abstract_obs}'

                    step_successful = scheduler.step_to(action, abstract_obs)
                    if not step_successful:
                        break

                    ep_data.append((observation.reshape(1, -1), action, reward, done))

                    if done:
                        break

                self.traces.append(ep_data)

            # refine model
            observation_space, action_space = get_observations_and_actions(self.traces)
            reduced_dim_obs_space = self.dim_reduction_fun.transform(observation_space)
            cluster_labels = self.clustering_fun.predict(reduced_dim_obs_space)

            extended_data = create_abstract_traces(self.traces, cluster_labels)
            self.model = run_Alergia(extended_data, automaton_type='mdp')
            print(f'Refinement {refinement_iteration} model size: {self.model.size} states')


ir = IterativeRefinement(env, model, traces, ae, cf2, )

ir.iteratively_refine_model(10, 100)
